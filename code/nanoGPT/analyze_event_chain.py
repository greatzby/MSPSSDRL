#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-task staged event-chain analysis script (S1→S3 / S1→S2 / S2→S3)

What this script does
- Iterates over checkpoints and runs autoregressive decoding for each (source, target) pair.
- Converts generated tokens into a node-path representation.
- Classifies behaviors (SUCCESS / INVALID_EDGE / OVER_SHOOT / ...).
- Extracts task-specific phase/event-chain indicators and aggregates summary metrics.
- Supports running multiple task types in a single execution.

Key features
1) --task-types supports comma-separated values or 'all'; keeps --task-type for backward compatibility.
2) One run analyzes multiple tasks; each task has its own per-checkpoint aggregation.
3) Output CSV rows include task_type; optional per-pair CSV per (checkpoint, task).
4) Event-chain definitions and aggregation logic are preserved from.

Important note (by design)
- The script treats the stop token as the vocab entry for '\\n' (meta["stoi"]["\\n"]).
- Decoding uses model.generate() with (temperature, top_k). temperature=0.0 relies on your generate() supporting greedy.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch

from model import GPT, GPTConfig


# -----------------------------------------------------------------------------
# Constants & data structures
# -----------------------------------------------------------------------------
VALID_TASK_TYPES: Tuple[str, ...] = ("s1s3", "s1s2", "s2s3")


@dataclass
class PairInfo:
    source: int
    target: int
    path_tokens: List[int]
    first_stage2: Optional[int]
    bridge_candidates: List[int]


@dataclass
class BehaviorRecord:
    step: int
    pair_index: int
    source: int
    target: int
    category: str
    stop_reached: bool
    path_length: int
    stage2_count: int
    first_action: Optional[int]
    target_index: Optional[int]
    tokens: List[int]
    raw_tokens: List[str]


@dataclass
class PhaseEventRecord:
    step: int
    pair_index: int
    source: int
    target: int

    first_action: Optional[int]
    first_valid: bool
    first_is_stage2: bool
    first_is_bridge: bool
    first_is_direct_target: bool
    first_is_invalid: bool

    hit_stage2: bool
    first_stage2: Optional[int]
    first_stage2_index: Optional[int]
    first_stage2_is_bridge: bool
    bridge_hit_any: bool
    stage2_available: bool
    bridge_candidates_count: int
    bridge_candidates: Tuple[int, ...]

    used_stage2: bool
    path_success: bool
    category: str

    legal_start: bool
    stage3_after_bridge: bool
    stage3_entry_index: Optional[int]
    bridge_to_stage3_hops: Optional[int]

    stop_reached: bool
    final_node: Optional[int]

    hit_stage3: bool
    first_stage3: Optional[int]
    first_stage3_index: Optional[int]
    stage3_stop_success: bool

    stage2_stop_token_emitted: bool
    stage2_stop_index: Optional[int]
    stage2_stop_clean: bool
    stage2_stop_success: bool
    stage2_stop_reason: str


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze multi-stage composition behavior with task-specific phase decomposition."
    )

    io = parser.add_argument_group("I/O")
    io.add_argument("--data-dir", type=str, required=True)
    io.add_argument("--checkpoints-dir", type=str, required=True)
    io.add_argument("--output-dir", type=str, required=True)
    io.add_argument("--run-type", type=str, choices=["sft", "pg", "ql"], required=True)
    io.add_argument("--ckpt-pattern", type=str, default=None)
    io.add_argument("--save-per-pair", action="store_true")

    task = parser.add_argument_group("Task selection")
    task.add_argument(
        "--task-type",
        type=str,
        default=None,
        help="(Deprecated) Single task type: s1s3, s1s2, or s2s3. Use --task-types instead.",
    )
    task.add_argument(
        "--task-types",
        type=str,
        default=None,
        help="Comma-separated list like 's1s3,s1s2' or 'all' for every task type.",
    )

    ckpt = parser.add_argument_group("Checkpoint sweep")
    ckpt.add_argument("--step-start", type=int, required=True)
    ckpt.add_argument("--step-end", type=int, required=True)
    ckpt.add_argument("--step-interval", type=int, required=True)

    sampling = parser.add_argument_group("Sampling / decoding")
    sampling.add_argument("--max-samples", type=int, default=0)
    sampling.add_argument("--sample-seed", type=int, default=2025)
    sampling.add_argument("--max-new-tokens", type=int, default=32)
    sampling.add_argument("--temperature", type=float, default=0.00001)
    sampling.add_argument("--top-k", type=int, default=1)

    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument("--device", type=str, default="cuda:0")
    runtime.add_argument("--quiet", action="store_true")
    runtime.add_argument("--progress", action="store_true")

    args = parser.parse_args()

    if args.step_interval <= 0:
        raise ValueError("--step-interval must be > 0")
    if args.step_end < args.step_start:
        raise ValueError("--step-end must be >= --step-start")

    return args


def resolve_task_types(single: Optional[str], multi: Optional[str]) -> List[str]:
    """
    Resolve requested task types.

    Rules (same intent as original):
    - If --task-types provided: use it; else use --task-type; else default to 's1s3'.
    - Accept 'all' to run every valid task.
    - Deduplicate while preserving order.
    """
    raw = multi or single or "s1s3"
    raw = raw.strip().lower()
    if raw == "all":
        return list(VALID_TASK_TYPES)

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    seen: Dict[str, None] = {}
    for part in parts:
        if part not in VALID_TASK_TYPES:
            raise ValueError(
                f"Unsupported task type: {part}. Valid: {', '.join(VALID_TASK_TYPES)} or 'all'."
            )
        if part not in seen:
            seen[part] = None
    return list(seen.keys())


# -----------------------------------------------------------------------------
# Data loading utilities
# -----------------------------------------------------------------------------
def _pick_existing_path(data_dir: Path, candidates: Sequence[str]) -> Path:
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist under {data_dir}: {', '.join(candidates)}")


def load_meta(meta_path: Path) -> dict:
    import pickle

    with open(meta_path, "rb") as f:
        return pickle.load(f)


def load_stage_info(stage_info_path: Path) -> List[List[int]]:
    """
    Load stage info from either:
    - pickle (.pkl): expected to have {"stages": [S1, S2, S3, ...]}
    - torch (.pt): same structure (torch.load)
    """
    stage_info = None
    if stage_info_path.suffix == ".pt":
        stage_info = torch.load(stage_info_path, map_location="cpu")
    else:
        import pickle

        with open(stage_info_path, "rb") as f:
            stage_info = pickle.load(f)

    return [list(map(int, stage)) for stage in stage_info.get("stages", [])]


def load_graph(graph_path: Path) -> nx.DiGraph:
    return nx.read_graphml(graph_path)


def build_successor_map(graph: nx.DiGraph) -> Dict[int, List[int]]:
    """
    Build a successor map with int node ids.
    Assumes GraphML nodes are digit-like strings (common in this pipeline).
    """
    succ_map: Dict[int, List[int]] = {}
    for node in graph.nodes:
        succ_map[int(node)] = [int(nbr) for nbr in graph.successors(node)]
    return succ_map


# -----------------------------------------------------------------------------
# Pair parsing & bridge candidate assignment
# -----------------------------------------------------------------------------
def parse_test_pairs(test_path: Path, stages: List[List[int]], task_type: str) -> List[PairInfo]:
    """
    Parse test pairs from test.txt and filter by task_type:
      - s1s3: source in S1, target in S3
      - s1s2: source in S1, target in S2
      - s2s3: source in S2, target in S3

    Each line format:
      src tgt [path_token_1 path_token_2 ...]
    """
    S1, S2, S3 = map(set, stages[:3])

    task_filters = {
        "s1s3": (S1, S3),
        "s1s2": (S1, S2),
        "s2s3": (S2, S3),
    }
    if task_type not in task_filters:
        raise ValueError(f"Unsupported task_type: {task_type}")

    src_set, tgt_set = task_filters[task_type]
    pairs: List[PairInfo] = []

    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            src, tgt = int(parts[0]), int(parts[1])
            if src not in src_set or tgt not in tgt_set:
                continue

            path_tokens = list(map(int, parts[2:])) if len(parts) > 2 else []
            stage2_nodes = [n for n in path_tokens if n in S2]

            pairs.append(
                PairInfo(
                    source=src,
                    target=tgt,
                    path_tokens=path_tokens,
                    first_stage2=stage2_nodes[0] if stage2_nodes else None,
                    bridge_candidates=[],
                )
            )

    return pairs


def assign_bridge_candidates(
    pairs: List[PairInfo],
    descendants_map: Dict[int, set[int]],
    stage_sets: Dict[str, set[int]],
    task_type: str,
) -> None:
    """
    Assign bridge candidates for each pair.

    NOTE: Logic preserved from the original:
    - Only for s1s3:
        candidate is a Stage-2 node reachable from source,
        and target is reachable from candidate.
    - For other tasks: empty list.
    """
    if task_type != "s1s3":
        for info in pairs:
            info.bridge_candidates = []
        return

    S2 = stage_sets["S2"]
    for info in pairs:
        reachable = descendants_map.get(info.source, set())
        stage2_reachable = reachable.intersection(S2)
        candidates: List[int] = []
        for node in stage2_reachable:
            if info.target in descendants_map.get(node, set()):
                candidates.append(node)
        info.bridge_candidates = sorted(set(candidates))


# -----------------------------------------------------------------------------
# Checkpoint/model helpers
# -----------------------------------------------------------------------------
def default_ckpt_pattern(run_type: str) -> str:
    if run_type == "sft":
        return "ckpt_{step}.pt"
    if run_type == "pg":
        return "ckpt_pg_{step}.pt"
    if run_type == "ql":
        return "ckpt_ql_{step}.pt"
    return "ckpt_{step}.pt"


def create_model_from_checkpoint(ckpt_path: Path, device: torch.device, vocab_size: int) -> GPT:
    """
    Load a GPT model from nanoGPT-style checkpoint dict.

    NOTE: Kept behavior:
    - Uses ckpt["model_args"] to build GPTConfig.
    - If current block_size > ckpt block_size, expands wpe and copies rows.
    - Removes attention bias/mask buffers from state_dict before loading (strict=False).
    - Sets model.eval() and returns.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_args = ckpt.get("model_args", {})
    _ = vocab_size  # kept for signature compatibility; not used intentionally

    config = GPTConfig(**model_args)
    model = GPT(config).to(device)

    state_dict = ckpt["model"]
    ckpt_block_size = model_args.get("block_size")
    model_block_size = config.block_size

    if ckpt_block_size and model_block_size > ckpt_block_size:
        wpe = state_dict.get("transformer.wpe.weight")
        if wpe is not None:
            new_wpe = model.transformer.wpe.weight.detach().clone()
            new_wpe[: wpe.size(0)] = wpe
            state_dict["transformer.wpe.weight"] = new_wpe

    keys_to_remove = [k for k in state_dict.keys() if k.endswith("attn.bias") or k.endswith("attn.mask")]
    for k in keys_to_remove:
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Decoding & path construction (logic preserved)
# -----------------------------------------------------------------------------
def run_greedy_generation(
    model: GPT,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    source: int,
    target: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    stop_token_id: int,
) -> Tuple[List[int], List[str], bool]:
    """
    Generate continuation tokens for prompt [src, tgt, src], then:
    - Convert each token into:
        - int(node) if token is digit
        - math.inf sentinel if non-digit (kept behavior)
    - Stop parsing when stop_token_id is emitted (excluded from returned sequences).
    """
    prompt_ids = [stoi[str(source)], stoi[str(target)], stoi[str(source)]]
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k if top_k > 0 else None,
    }
    with torch.no_grad():
        generated = model.generate(context, **gen_kwargs)[0].tolist()

    new_ids = generated[len(prompt_ids) :]
    raw_tokens: List[str] = []
    digits: List[int] = []
    stop_reached = False

    for tid in new_ids:
        if tid == stop_token_id:
            stop_reached = True
            break
        token = itos.get(tid, "[UNK]")
        raw_tokens.append(token)
        if token.isdigit():
            digits.append(int(token))
        else:
            digits.append(math.inf)

    return digits, raw_tokens, stop_reached


def build_path_from_digits(digits: List[int], source: int, stop_reached: bool) -> Tuple[List[int], str]:
    """
    Build a path_nodes list from the 'digits' stream.

    IMPORTANT: This keeps the original behavior exactly, including:
    - STOP_BEFORE_START / INVALID_TOKEN / SRC_MISMATCH / NO_EOS / OK statuses
    - Using math.inf as invalid-token sentinel
    """
    if not digits:
        return [source], "STOP_BEFORE_START"
    if digits[0] == math.inf:
        return [source], "INVALID_TOKEN"

    if digits[0] != source:
        clean = [source] + [d for d in digits if d != math.inf]
        return clean, "SRC_MISMATCH"

    clean_digits: List[int] = []
    for val in digits:
        if val == math.inf:
            return clean_digits, "INVALID_TOKEN"
        clean_digits.append(val)

    if not stop_reached:
        return clean_digits, "NO_EOS"

    return clean_digits, "OK"


def classify_behavior(
    path_nodes: List[int],
    base_status: str,
    source: int,
    target: int,
    stage_sets: Dict[str, set[int]],
    graph: nx.DiGraph,
) -> Tuple[str, int, int, Optional[int]]:
    """
    Assign a behavior category and a few auxiliary stats.

    NOTE: Logic preserved as-is from the original script.
    """
    S2 = stage_sets["S2"]

    if base_status == "STOP_BEFORE_START":
        return "STOP_BEFORE_START", 0, 0, None
    if base_status == "INVALID_TOKEN":
        return "INVALID_TOKEN", len(path_nodes), 0, None

    valid_edges = True
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if not graph.has_edge(str(u), str(v)):
            valid_edges = False
            break

    stage2_count = sum(1 for n in path_nodes if n in S2)
    target_index: Optional[int] = None
    for i, n in enumerate(path_nodes):
        if n == target:
            target_index = i
            break

    if base_status == "NO_EOS":
        if valid_edges and target_index is not None:
            return "OVER_SHOOT", len(path_nodes), stage2_count, target_index
        return "NO_EOS", len(path_nodes), stage2_count, target_index

    if base_status == "SRC_MISMATCH":
        return "SRC_MISMATCH", len(path_nodes), stage2_count, target_index

    if not valid_edges:
        return "INVALID_EDGE", len(path_nodes), stage2_count, target_index
    if target_index is None:
        return "MISSING_TARGET", len(path_nodes), stage2_count, target_index
    if target_index != len(path_nodes) - 1:
        return "OVER_SHOOT", len(path_nodes), stage2_count, target_index

    if stage2_count == 0:
        if len(path_nodes) >= 2 and path_nodes[1] == target:
            return "DIRECT_JUMP", len(path_nodes), stage2_count, target_index
        return "NO_STAGE2", len(path_nodes), stage2_count, target_index

    return "SUCCESS", len(path_nodes), stage2_count, target_index


# -----------------------------------------------------------------------------
# Aggregation (logic preserved)
# -----------------------------------------------------------------------------
def aggregate_behavior(records: List[BehaviorRecord]) -> Dict[str, float]:
    total = len(records)
    summary: Dict[str, float] = {
        "num_pairs": total,
        "avg_path_length": float(np.mean([r.path_length for r in records])) if records else 0.0,
        "avg_stage2_count": float(np.mean([r.stage2_count for r in records])) if records else 0.0,
    }

    counter = Counter(r.category for r in records)
    for cat in set(counter.keys()) | {"SUCCESS"}:
        summary[f"count_{cat}"] = counter[cat]
        summary[f"rate_{cat}"] = counter[cat] / total if total else 0.0

    return summary


def aggregate_phase_events(records: List[PhaseEventRecord], task_type: str) -> Dict[str, float]:
    # Entire event-chain logic is intentionally preserved (verbatim in spirit).
    if not records:
        return {}

    total = len(records)
    success_total = sum(1 for r in records if r.path_success)
    success_rate = success_total / total if total else 0.0

    stage2_available_total = sum(1 for r in records if r.stage2_available)
    stage2_available_rate = stage2_available_total / total if total else 0.0

    stage2_stop_token_total = sum(1 for r in records if r.stage2_stop_token_emitted)
    stage2_stop_token_rate = stage2_stop_token_total / total if total else 0.0

    stage2_stop_success_total = sum(1 for r in records if r.stage2_stop_success)
    stage2_stop_success_rate = stage2_stop_success_total / total if total else 0.0

    stage3_stop_success_total = sum(1 for r in records if r.stage3_stop_success)
    stage3_stop_success_rate = stage3_stop_success_total / total if total else 0.0

    hit_stage3_total = sum(1 for r in records if r.hit_stage3)
    hit_stage3_rate = hit_stage3_total / total if total else 0.0

    stage2_stop_reason_counts = Counter(r.stage2_stop_reason for r in records)

    summary: Dict[str, float] = {
        "total_pairs": total,
        "success_total": success_total,
        "success_rate": success_rate,
        "stage2_available_total": stage2_available_total,
        "stage2_available_rate": stage2_available_rate,
        "stage2_stop_token_total": stage2_stop_token_total,
        "stage2_stop_token_rate": stage2_stop_token_rate,
        "stage2_stop_success_total": stage2_stop_success_total,
        "stage2_stop_success_rate": stage2_stop_success_rate,
        "stage3_stop_success_total": stage3_stop_success_total,
        "stage3_stop_success_rate": stage3_stop_success_rate,
        "hit_stage3_total": hit_stage3_total,
        "hit_stage3_rate": hit_stage3_rate,
    }

    for reason, count in stage2_stop_reason_counts.items():
        summary[f"stage2_stop_reason_count_{reason}"] = count
        summary[f"stage2_stop_reason_rate_{reason}"] = count / total if total else 0.0

    if task_type == "s1s3":
        eventA_total = sum(1 for r in records if r.legal_start)
        eventA_rate = eventA_total / total if total else 0.0

        eventB_success = sum(1 for r in records if r.legal_start and r.bridge_hit_any)
        eventB_total = eventA_total
        eventB_rate_given_A = eventB_success / eventB_total if eventB_total else 0.0

        eventB_total_available = sum(1 for r in records if r.legal_start and r.stage2_available)
        eventB_success_available = sum(
            1 for r in records if r.legal_start and r.stage2_available and r.bridge_hit_any
        )
        eventB_rate_given_A_and_available = (
            eventB_success_available / eventB_total_available if eventB_total_available else 0.0
        )

        eventC_success = sum(1 for r in records if r.legal_start and r.bridge_hit_any and r.stage3_after_bridge)
        eventC_total = eventB_success
        eventC_rate_given_AB = eventC_success / eventC_total if eventC_total else 0.0

        eventD_success = sum(
            1
            for r in records
            if r.legal_start and r.bridge_hit_any and r.stage3_after_bridge and r.path_success
        )
        eventD_total = eventC_success
        eventD_rate_given_ABC = eventD_success / eventD_total if eventD_total else 0.0

        expected_success = eventA_rate * eventB_rate_given_A * eventC_rate_given_AB * eventD_rate_given_ABC
        success_gap = success_rate - expected_success

        bridge_hit_any_total = sum(1 for r in records if r.bridge_hit_any)
        bridge_hit_any_rate = bridge_hit_any_total / total if total else 0.0

        first_stage2_is_bridge_total = sum(1 for r in records if r.hit_stage2 and r.first_stage2_is_bridge)
        hit_stage2_total = sum(1 for r in records if r.hit_stage2)
        first_stage2_bridge_rate_given_A = (
            first_stage2_is_bridge_total / hit_stage2_total if hit_stage2_total else 0.0
        )

        first_stage2_indices = [r.first_stage2_index for r in records if r.first_stage2_index is not None]
        first_stage2_index_mean = float(np.mean(first_stage2_indices)) if first_stage2_indices else 0.0
        first_stage2_index_median = float(np.median(first_stage2_indices)) if first_stage2_indices else 0.0

        bridge_to_stage3_hops = [r.bridge_to_stage3_hops for r in records if r.bridge_to_stage3_hops is not None]
        bridge_to_stage3_hops_mean = float(np.mean(bridge_to_stage3_hops)) if bridge_to_stage3_hops else 0.0
        bridge_to_stage3_hops_median = float(np.median(bridge_to_stage3_hops)) if bridge_to_stage3_hops else 0.0

        first_valid_total = sum(1 for r in records if r.first_valid)
        first_valid_rate = first_valid_total / total if total else 0.0
        first_invalid_total = sum(1 for r in records if r.first_is_invalid)
        first_invalid_rate = first_invalid_total / total if total else 0.0
        first_stage2_total = sum(1 for r in records if r.first_is_stage2)
        first_stage2_rate = first_stage2_total / total if total else 0.0
        first_bridge_total = sum(1 for r in records if r.first_is_bridge)
        first_bridge_rate = first_bridge_total / total if total else 0.0
        first_direct_target_total = sum(1 for r in records if r.first_is_direct_target)
        first_direct_target_rate = first_direct_target_total / total if total else 0.0

        used_stage2_total = sum(1 for r in records if r.used_stage2)
        used_stage2_rate = used_stage2_total / total if total else 0.0

        success_without_stage2 = sum(1 for r in records if r.path_success and not r.hit_stage2)
        success_with_stage2_no_bridge = sum(1 for r in records if r.path_success and r.hit_stage2 and not r.bridge_hit_any)
        success_with_bridge = sum(1 for r in records if r.path_success and r.bridge_hit_any)

        avg_bridge_candidates_available = (
            float(np.mean([r.bridge_candidates_count for r in records if r.bridge_candidates_count > 0]))
            if stage2_available_total > 0
            else 0.0
        )

        stage3_after_bridge_total = sum(1 for r in records if r.stage3_after_bridge)
        stage3_after_bridge_rate = stage3_after_bridge_total / total if total else 0.0

        summary.update(
            {
                "eventA_total": eventA_total,
                "eventA_rate": eventA_rate,
                "eventB_total": eventB_total,
                "eventB_success": eventB_success,
                "eventB_rate_given_A": eventB_rate_given_A,
                "eventB_total_available": eventB_total_available,
                "eventB_success_available": eventB_success_available,
                "eventB_rate_given_A_and_available": eventB_rate_given_A_and_available,
                "eventC_total": eventC_total,
                "eventC_success": eventC_success,
                "eventC_rate_given_AB": eventC_rate_given_AB,
                "eventD_total": eventD_total,
                "eventD_success": eventD_success,
                "eventD_rate_given_ABC": eventD_rate_given_ABC,
                "expected_success_from_chain": expected_success,
                "success_gap_vs_expected": success_gap,
                "bridge_hit_any_total": bridge_hit_any_total,
                "bridge_hit_any_rate": bridge_hit_any_rate,
                "stage3_after_bridge_total": stage3_after_bridge_total,
                "stage3_after_bridge_rate": stage3_after_bridge_rate,
                "first_stage2_is_bridge_total": first_stage2_is_bridge_total,
                "first_stage2_bridge_rate_given_A": first_stage2_bridge_rate_given_A,
                "first_stage2_index_mean": first_stage2_index_mean,
                "first_stage2_index_median": first_stage2_index_median,
                "bridge_to_stage3_hops_mean": bridge_to_stage3_hops_mean,
                "bridge_to_stage3_hops_median": bridge_to_stage3_hops_median,
                "first_valid_total": first_valid_total,
                "first_valid_rate": first_valid_rate,
                "first_invalid_total": first_invalid_total,
                "first_invalid_rate": first_invalid_rate,
                "first_stage2_total": first_stage2_total,
                "first_stage2_rate": first_stage2_rate,
                "first_bridge_total": first_bridge_total,
                "first_bridge_rate": first_bridge_rate,
                "first_direct_target_total": first_direct_target_total,
                "first_direct_target_rate": first_direct_target_rate,
                "used_stage2_total": used_stage2_total,
                "used_stage2_rate": used_stage2_rate,
                "success_without_stage2": success_without_stage2,
                "success_with_stage2_no_bridge": success_with_stage2_no_bridge,
                "success_with_bridge": success_with_bridge,
                "avg_bridge_candidates_available": avg_bridge_candidates_available,
            }
        )

    elif task_type == "s1s2":
        eventA_prime_total = sum(1 for r in records if r.legal_start)
        eventA_prime_rate = eventA_prime_total / total if total else 0.0

        eventB_prime_total = sum(1 for r in records if r.legal_start and r.hit_stage2)
        eventB_prime_rate_given_A = eventB_prime_total / eventA_prime_total if eventA_prime_total else 0.0

        eventC_prime_total = sum(1 for r in records if r.legal_start and r.hit_stage2 and r.stage2_stop_token_emitted)
        eventC_prime_rate_given_AB = eventC_prime_total / eventB_prime_total if eventB_prime_total else 0.0

        eventD_prime_total = sum(1 for r in records if r.stage2_stop_success)
        eventD_prime_rate_given_ABC = eventD_prime_total / eventC_prime_total if eventC_prime_total else 0.0

        decomposition_chain_rate = (
            eventA_prime_rate * eventB_prime_rate_given_A * eventC_prime_rate_given_AB * eventD_prime_rate_given_ABC
        )

        summary.update(
            {
                "eventA_prime_total": eventA_prime_total,
                "eventA_prime_rate": eventA_prime_rate,
                "eventB_prime_total": eventB_prime_total,
                "eventB_prime_rate_given_A": eventB_prime_rate_given_A,
                "eventC_prime_total": eventC_prime_total,
                "eventC_prime_rate_given_AB": eventC_prime_rate_given_AB,
                "eventD_prime_total": eventD_prime_total,
                "eventD_prime_rate_given_ABC": eventD_prime_rate_given_ABC,
                "decomposition_chain_rate": decomposition_chain_rate,
            }
        )

    elif task_type == "s2s3":
        eventA_total = sum(1 for r in records if r.legal_start)
        eventA_rate = eventA_total / total if total else 0.0

        eventB_total = sum(1 for r in records if r.legal_start and r.hit_stage3)
        eventB_rate_given_A = eventB_total / eventA_total if eventA_total else 0.0

        eventC_total = sum(1 for r in records if r.legal_start and r.hit_stage3 and r.stage3_stop_success)
        eventC_rate_given_AB = eventC_total / eventB_total if eventB_total else 0.0

        execution_chain_rate = eventA_rate * eventB_rate_given_A * eventC_rate_given_AB

        summary.update(
            {
                "eventA_total": eventA_total,
                "eventA_rate": eventA_rate,
                "eventB_total": eventB_total,
                "eventB_rate_given_A": eventB_rate_given_A,
                "eventC_total": eventC_total,
                "eventC_rate_given_AB": eventC_rate_given_AB,
                "execution_chain_rate": execution_chain_rate,
            }
        )

    return summary


# -----------------------------------------------------------------------------
# CSV output
# -----------------------------------------------------------------------------
def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    """
    Write CSV with fieldnames = sorted union of all row keys.
    (Kept behavior: stable schema across steps/tasks by key union.)
    """
    if not rows:
        return

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    fieldnames = sorted(all_keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval=0)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    task_list = resolve_task_types(args.task_type, args.task_types)
    device = torch.device(args.device)

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.checkpoints_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg)

    # Resolve expected dataset files (prefer *.pkl if both exist).
    meta_path = _pick_existing_path(data_dir, ["meta.pkl"])
    stage_info_path = _pick_existing_path(data_dir, ["stage_info.pkl", "stage_info.pt"])
    graph_path = _pick_existing_path(data_dir, ["composition_graph.graphml"])
    test_path = _pick_existing_path(data_dir, ["test.txt"])

    meta = load_meta(meta_path)
    stages = load_stage_info(stage_info_path)
    stage_sets = {"S1": set(stages[0]), "S2": set(stages[1]), "S3": set(stages[2])}

    graph = load_graph(graph_path)
    succ_map = build_successor_map(graph)
    succ_set_map = {node: set(neighs) for node, neighs in succ_map.items()}

    # Descendants map for bridge candidate assignment (S1->S3).
    descendants_map: Dict[int, set[int]] = {
        int(node): {int(x) for x in nx.descendants(graph, node)} for node in graph.nodes
    }

    # Prepare pairs per task (with optional subsampling).
    pairs_by_task: Dict[str, List[PairInfo]] = {}
    for task in task_list:
        pairs_raw = parse_test_pairs(test_path, stages, task)
        assign_bridge_candidates(pairs_raw, descendants_map, stage_sets, task)

        if args.max_samples > 0 and len(pairs_raw) > args.max_samples:
            rng = random.Random(args.sample_seed)
            pairs = rng.sample(pairs_raw, args.max_samples)
        else:
            pairs = pairs_raw

        pairs_by_task[task] = pairs

    tasks_to_run = [t for t in task_list if pairs_by_task.get(t)]
    tasks_without_data = [t for t in task_list if not pairs_by_task.get(t)]
    if tasks_without_data:
        log("[Warn] No pairs found for tasks: " + ", ".join(t.upper() for t in tasks_without_data))

    if not tasks_to_run:
        raise RuntimeError("No task has available pairs to analyze. Check dataset filters and test.txt content.")

    log("Analyzing tasks -> " + ", ".join(f"{t.upper()}: {len(pairs_by_task[t])} pairs" for t in tasks_to_run))

    steps = range(args.step_start, args.step_end + 1, args.step_interval)
    pattern = args.ckpt_pattern or default_ckpt_pattern(args.run_type)

    behavior_rows: List[Dict[str, object]] = []
    phase_rows: List[Dict[str, object]] = []

    iterator_steps = steps
    if args.progress and not args.quiet:
        from tqdm import tqdm

        iterator_steps = tqdm(list(steps), desc="Checkpoints")

    stop_token_id = meta["stoi"]["\n"]

    for step in iterator_steps:
        ckpt_path = ckpt_dir / pattern.format(step=step)
        if not ckpt_path.exists():
            log(f"[Skip] checkpoint not found: {ckpt_path}")
            continue

        log(f"Processing checkpoint: {ckpt_path}")
        model = create_model_from_checkpoint(ckpt_path, device, meta["vocab_size"])

        for task in tasks_to_run:
            pairs = pairs_by_task[task]
            if not pairs:
                continue

            beh_recs: List[BehaviorRecord] = []
            phase_recs: List[PhaseEventRecord] = []

            pair_iterator: Sequence[Tuple[int, PairInfo]] = enumerate(pairs)
            if args.progress and not args.quiet:
                from tqdm import tqdm

                pair_iterator = enumerate(tqdm(pairs, leave=False, desc=f"{task.upper()} Pairs@{step}"))

            for idx, info in pair_iterator:
                digits, raw_tokens, stop = run_greedy_generation(
                    model=model,
                    stoi=meta["stoi"],
                    itos=meta["itos"],
                    source=info.source,
                    target=info.target,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=device,
                    stop_token_id=stop_token_id,
                )

                # Keep original behavior: prepend source into the "digits stream" before building the path.
                full_digits = [info.source] + digits
                path_nodes, status = build_path_from_digits(full_digits, info.source, stop)

                category, path_len, stage2_cnt, target_idx = classify_behavior(
                    path_nodes=path_nodes,
                    base_status=status,
                    source=info.source,
                    target=info.target,
                    stage_sets=stage_sets,
                    graph=graph,
                )

                first_action = path_nodes[1] if len(path_nodes) > 1 else None
                valid_neighbors = succ_set_map.get(info.source, set())

                first_valid = first_action is not None and first_action in valid_neighbors
                first_is_stage2 = first_action in stage_sets["S2"] if first_action is not None else False
                first_is_bridge = first_valid and (first_action in info.bridge_candidates)
                first_is_direct_target = (first_action == info.target) if first_action is not None else False
                first_is_invalid = not first_valid

                stage2_nodes_in_path = [n for n in path_nodes if n in stage_sets["S2"]]
                hit_stage2 = len(stage2_nodes_in_path) > 0
                first_stage2 = stage2_nodes_in_path[0] if hit_stage2 else None
                first_stage2_index = path_nodes.index(first_stage2) if hit_stage2 else None

                bridge_candidate_set = set(info.bridge_candidates)
                first_stage2_is_bridge = (first_stage2 in bridge_candidate_set) if hit_stage2 else False
                bridge_hit_any = any(n in bridge_candidate_set for n in stage2_nodes_in_path)
                stage2_available = len(info.bridge_candidates) > 0

                bridge_indices = [i for i, node in enumerate(path_nodes) if node in bridge_candidate_set]
                first_bridge_index = bridge_indices[0] if bridge_indices else None

                stage3_after_bridge = False
                stage3_entry_index: Optional[int] = None
                bridge_to_stage3_hops: Optional[int] = None

                if first_bridge_index is not None:
                    for j in range(first_bridge_index, len(path_nodes) - 1):
                        curr_node = path_nodes[j]
                        next_node = path_nodes[j + 1]
                        if curr_node in bridge_candidate_set and next_node in stage_sets["S3"]:
                            stage3_after_bridge = True
                            stage3_entry_index = j + 1
                            bridge_to_stage3_hops = stage3_entry_index - first_bridge_index
                            break

                final_node = path_nodes[-1] if path_nodes else None
                stage2_stop_token_emitted = stop and (final_node in stage_sets["S2"])
                stage2_stop_index = (len(path_nodes) - 1) if stage2_stop_token_emitted else None
                stage2_stop_clean = stage2_stop_token_emitted and category != "OVER_SHOOT"
                stage2_stop_success = stage2_stop_token_emitted and category == "SUCCESS"

                if stage2_stop_token_emitted:
                    if stage2_stop_success:
                        stage2_stop_reason = "STOP_CLEAN_SUCCESS"
                    else:
                        stage2_stop_reason = f"STOP_WITH_{category}"
                else:
                    if stop and (final_node in stage_sets["S3"]):
                        stage2_stop_reason = "STOP_AT_STAGE3"
                    elif stop:
                        stage2_stop_reason = "STOP_AT_OTHER"
                    else:
                        stage2_stop_reason = "NO_STOP"

                stage3_nodes_in_path = [n for n in path_nodes if n in stage_sets["S3"]]
                hit_stage3 = len(stage3_nodes_in_path) > 0
                first_stage3 = stage3_nodes_in_path[0] if hit_stage3 else None
                first_stage3_index = path_nodes.index(first_stage3) if hit_stage3 else None
                stage3_stop_success = stop and (final_node in stage_sets["S3"]) and category == "SUCCESS"

                beh_recs.append(
                    BehaviorRecord(
                        step=step,
                        pair_index=idx,
                        source=info.source,
                        target=info.target,
                        category=category,
                        stop_reached=stop,
                        path_length=path_len,
                        stage2_count=stage2_cnt,
                        first_action=first_action,
                        target_index=target_idx,
                        tokens=path_nodes,
                        raw_tokens=raw_tokens,
                    )
                )

                phase_recs.append(
                    PhaseEventRecord(
                        step=step,
                        pair_index=idx,
                        source=info.source,
                        target=info.target,
                        first_action=first_action,
                        first_valid=first_valid,
                        first_is_stage2=first_is_stage2,
                        first_is_bridge=first_is_bridge,
                        first_is_direct_target=first_is_direct_target,
                        first_is_invalid=first_is_invalid,
                        hit_stage2=hit_stage2,
                        first_stage2=first_stage2,
                        first_stage2_index=first_stage2_index,
                        first_stage2_is_bridge=first_stage2_is_bridge,
                        bridge_hit_any=bridge_hit_any,
                        stage2_available=stage2_available,
                        bridge_candidates_count=len(info.bridge_candidates),
                        bridge_candidates=tuple(info.bridge_candidates),
                        used_stage2=(stage2_cnt > 0),
                        path_success=(category == "SUCCESS"),
                        category=category,
                        legal_start=first_valid,
                        stage3_after_bridge=stage3_after_bridge,
                        stage3_entry_index=stage3_entry_index,
                        bridge_to_stage3_hops=bridge_to_stage3_hops,
                        stop_reached=stop,
                        final_node=final_node,
                        hit_stage3=hit_stage3,
                        first_stage3=first_stage3,
                        first_stage3_index=first_stage3_index,
                        stage3_stop_success=stage3_stop_success,
                        stage2_stop_token_emitted=stage2_stop_token_emitted,
                        stage2_stop_index=stage2_stop_index,
                        stage2_stop_clean=stage2_stop_clean,
                        stage2_stop_success=stage2_stop_success,
                        stage2_stop_reason=stage2_stop_reason,
                    )
                )

            # Per-(step, task) summaries
            beh_summary = aggregate_behavior(beh_recs)
            beh_summary["step"] = step
            beh_summary["task_type"] = task
            behavior_rows.append(beh_summary)

            phase_summary = aggregate_phase_events(phase_recs, task)
            if phase_summary:
                phase_summary["step"] = step
                phase_summary["task_type"] = task
                phase_rows.append(phase_summary)

            # Optional per-pair output (kept columns/values as original intent).
            if args.save_per_pair:
                per_pair_rows: List[Dict[str, object]] = []
                for beh, phase in zip(beh_recs, phase_recs):
                    per_pair_rows.append(
                        {
                            "step": step,
                            "task_type": task,
                            "pair_index": beh.pair_index,
                            "source": beh.source,
                            "target": beh.target,
                            "category": beh.category,
                            "path_success": int(phase.path_success),
                            "path_length": beh.path_length,
                            "stage2_count": beh.stage2_count,
                            "stop_reached": int(beh.stop_reached),
                            "final_node": "" if phase.final_node is None else phase.final_node,
                            "first_action": "" if phase.first_action is None else phase.first_action,
                            "first_valid": int(phase.first_valid),
                            "first_is_stage2": int(phase.first_is_stage2),
                            "first_is_bridge": int(phase.first_is_bridge),
                            "first_is_direct_target": int(phase.first_is_direct_target),
                            "first_is_invalid": int(phase.first_is_invalid),
                            "hit_stage2": int(phase.hit_stage2),
                            "first_stage2": "" if phase.first_stage2 is None else phase.first_stage2,
                            "first_stage2_index": "" if phase.first_stage2_index is None else phase.first_stage2_index,
                            "first_stage2_is_bridge": int(phase.first_stage2_is_bridge),
                            "bridge_hit_any": int(phase.bridge_hit_any),
                            "stage2_available": int(phase.stage2_available),
                            "bridge_candidates_count": phase.bridge_candidates_count,
                            "bridge_candidates": " ".join(map(str, phase.bridge_candidates)),
                            "used_stage2": int(phase.used_stage2),
                            "legal_start": int(phase.legal_start),
                            "stage3_after_bridge": int(phase.stage3_after_bridge),
                            "stage3_entry_index": "" if phase.stage3_entry_index is None else phase.stage3_entry_index,
                            "bridge_to_stage3_hops": "" if phase.bridge_to_stage3_hops is None else phase.bridge_to_stage3_hops,
                            "hit_stage3": int(phase.hit_stage3),
                            "first_stage3": "" if phase.first_stage3 is None else phase.first_stage3,
                            "first_stage3_index": "" if phase.first_stage3_index is None else phase.first_stage3_index,
                            "stage3_stop_success": int(phase.stage3_stop_success),
                            "stage2_stop_token_emitted": int(phase.stage2_stop_token_emitted),
                            "stage2_stop_index": "" if phase.stage2_stop_index is None else phase.stage2_stop_index,
                            "stage2_stop_clean": int(phase.stage2_stop_clean),
                            "stage2_stop_success": int(phase.stage2_stop_success),
                            "stage2_stop_reason": phase.stage2_stop_reason,
                            "path_tokens": " ".join(map(str, beh.tokens)),
                            "raw_tokens": " ".join(beh.raw_tokens),
                        }
                    )

                per_pair_path = out_dir / f"per_pair_step_{step}_{task}.csv"
                write_csv(per_pair_path, per_pair_rows)

    write_csv(out_dir / "behavior_summary.csv", behavior_rows)
    write_csv(out_dir / "phase_summary.csv", phase_rows)

    log("Analysis complete.")
    log(f"behavior_summary.csv saved to {out_dir}")
    log(f"phase_summary.csv saved to {out_dir}")
    if args.save_per_pair:
        log("Per-pair CSV files were generated per task and checkpoint.")


if __name__ == "__main__":
    main()