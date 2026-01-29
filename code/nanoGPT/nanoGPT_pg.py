#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Policy Gradient (REINFORCE) fine-tuning for GraphA.

This script initializes from an SFT checkpoint and performs REINFORCE updates using
a scalar success reward:
  reward = 1 if the decoded path is valid (under graph + stage constraints), else 0
It uses an EMA reward baseline to reduce variance and optional KL regularization to
penalize deviation from the SFT policy.

Prompts and stopping (aligned with evaluation):
  - prompt tokens: [src, tgt, src]
  - stop token id: meta["stoi"]["\\n"]
  - full path for validity checking: [src] + generated_nodes
  - for S1->S3 pairs, validity requires visiting at least one Stage-2 node

Required inputs (under --data_dir):
  - train_{K}.txt, test.txt
  - meta.pkl, stage_info.pkl
  - composition_graph.graphml
Also requires:
  - --sft_checkpoint (.pt) to initialize the policy (and KL reference if enabled)

Outputs (written to --log_dir/pg_{timestamp}/):
  - train_pg.log
  - metrics_pg.jsonl
  - ckpt_pg_{iter}.pt

Example:
  python nanoGPT/nanoGPT_pg.py \
    --data_dir data/datasets/graphA_full_P13_0 \
    --sft_checkpoint out/<your_sft_run_dir>/ckpt_5000.pt \
    --train_paths_per_pair 20 \
    --device cuda:0 \
    --max_iters 20000 \
    --eval_interval 1000 \
    --save_interval 2000
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from logger import get_logger
from model import GPT, GPTConfig

Node = int
PathList = List[int]
Pair = Tuple[int, int]
BucketName = str


# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy Gradient fine-tuning for GraphA.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dataset directory (contains train_K.txt, meta.pkl, stage_info.pkl, etc.).")
    parser.add_argument("--sft_checkpoint", type=str, required=True,
                        help="SFT checkpoint (.pt) used to initialize RL training.")
    parser.add_argument("--train_paths_per_pair", type=int, default=20,
                        help="Value of K for train_{K}.txt.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    # Optional model overrides
    parser.add_argument("--n_layer", type=int, default=None,
                        help="Override number of layers; default is loaded from checkpoint.")
    parser.add_argument("--n_head", type=int, default=None,
                        help="Override number of attention heads; default is loaded from checkpoint.")
    parser.add_argument("--n_embd", type=int, default=None,
                        help="Override embedding/hidden size; default is loaded from checkpoint.")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override dropout; default is loaded from checkpoint.")
    parser.add_argument("--bias", type=str, choices=["true", "false"], default=None,
                        help="Override whether linear layers use bias; default is loaded from checkpoint.")
    parser.add_argument("--block_size_override", type=int, default=None,
                        help="Override model block_size; default uses checkpoint or dataset meta.")

    # PG hyperparameters
    parser.add_argument("--max_iters", type=int, default=20000,
                        help="Number of PG update steps.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of (source, target) pairs sampled per PG update.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Policy learning rate.")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping threshold.")
    parser.add_argument("--adv_clip", type=float, default=5.0,
                        help="Advantage clipping absolute value (<=0 disables clipping).")

    parser.add_argument("--max_rollout_steps", type=int, default=32,
                        help="Max number of generated tokens per rollout (excluding prompt).")
    parser.add_argument("--rollout_temperature", type=float, default=1.2,
                        help="Sampling temperature for rollout; >=1 recommended for exploration.")
    parser.add_argument("--rollout_top_k", type=int, default=0,
                        help="Top-k truncation during rollout sampling (<=0 disables).")

    parser.add_argument("--kl_coef", type=float, default=3e-4,
                        help="KL regularization coefficient to penalize deviation from SFT (0 disables).")
    parser.add_argument("--baseline_beta", type=float, default=0.95,
                        help="EMA coefficient for reward baseline.")

    # Evaluation & logging
    parser.add_argument("--eval_interval", type=int, default=1000,
                        help="Run evaluation every N steps.")
    parser.add_argument("--save_interval", type=int, default=2000,
                        help="Save a checkpoint every N steps.")
    parser.add_argument("--max_eval_pairs", type=int, default=5000,
                        help="Max number of (s, t) pairs used for evaluation.")
    parser.add_argument("--eval_temperature", type=float, default=0.001,
                        help="Evaluation temperature; 0.0 means greedy decoding.")
    parser.add_argument("--eval_top_k", type=int, default=0,
                        help="Evaluation top-k (<=0 means greedy).")

    parser.add_argument("--log_dir", type=str, default="out_pg",
                        help="Output directory for logs and checkpoints.")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def decode_tokens(token_ids: Sequence[int],
                  itos: Dict[int, str],
                  stop_token_id: int) -> List[str]:
    """Decode token ids to string tokens, stopping at stop_token_id (excluded)."""
    tokens: List[str] = []
    for tid in token_ids:
        if tid == stop_token_id:
            break
        tokens.append(itos.get(int(tid), "[UNK]"))
    return tokens


def tokens_to_nodes(tokens: Sequence[str]) -> List[int]:
    """Parse digit tokens into node ids."""
    nodes: List[int] = []
    for tok in tokens:
        if tok.isdigit():
            nodes.append(int(tok))
    return nodes


def assemble_full_path(source: int, generated_nodes: Sequence[int]) -> List[int]:
    """
    Assemble the full node path for validation.

    NOTE: The prompt ends with `source`, and the model generates continuation nodes.
    For path validity checking, we prepend `source` to the generated node sequence.
    """
    full_path = [source]
    full_path.extend(generated_nodes)
    return full_path


def bucket_for_pair(source: int,
                    target: int,
                    stages: Sequence[Sequence[int]]) -> Optional[BucketName]:
    S1, S2, S3 = stages[:3]
    if source in S1 and target in S2:
        return "S1->S2"
    if source in S2 and target in S3:
        return "S2->S3"
    if source in S1 and target in S3:
        return "S1->S3"
    return None


def is_valid_path(path_nodes: List[int],
                  source: int,
                  target: int,
                  stages: Sequence[Sequence[int]],
                  graph: nx.DiGraph) -> bool:
    """Check if a node path is valid under graph edges and stage constraints."""
    if len(path_nodes) < 2:
        return False
    if path_nodes[0] != source or path_nodes[-1] != target:
        return False

    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if not graph.has_edge(str(u), str(v)):
            return False

    S1, S2, S3 = stages[:3]
    if source in S1 and target in S3:
        if not any(node in S2 for node in path_nodes[1:-1]):
            return False
    return True


def load_pairs(train_file: Path) -> List[Pair]:
    """Load unique (source, target) pairs from the given train file."""
    seen: set[Pair] = set()
    pairs: List[Pair] = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            s, t = int(parts[0]), int(parts[1])
            key = (s, t)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return pairs


def prepare_output_dir(base_dir: str) -> Path:
    """Create a timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base_dir) / f"pg_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def safe_max_new_tokens(block_size: int,
                        prompt_len: int,
                        desired: int) -> int:
    """
    Compute the maximum number of new tokens allowed by block_size (min 1).
    One position is reserved for the stop token.
    """
    available = block_size - prompt_len - 1
    if available <= 0:
        raise ValueError(
            f"Block size {block_size} is too small for prompt length {prompt_len}. "
            "Increase block_size (or use --block_size_override)."
        )
    return max(1, min(desired, available))


def load_state_dict_with_block_resize(model: GPT,
                                      raw_state_dict: Dict[str, Tensor],
                                      ckpt_block_size: int,
                                      logger) -> None:
    """
    Load a checkpoint when block_size is increased:
      1) expand positional embeddings and copy existing weights;
      2) do not load attention bias/mask (use the model's initialized buffers).
    """
    state_dict = dict(raw_state_dict)  # shallow copy; do not modify the original dict
    model_block_size = model.config.block_size

    if ckpt_block_size is None:
        ckpt_block_size = model_block_size

    if model_block_size == ckpt_block_size:
        model.load_state_dict(state_dict, strict=True)
        return

    if model_block_size < ckpt_block_size:
        raise ValueError(
            f"Checkpoint block_size={ckpt_block_size} is larger than current model block_size={model_block_size}; "
            "cannot shrink. Please use a larger --block_size_override."
        )

    logger.warning(
        "Expanding block_size: checkpoint=%d -> current=%d. New positional rows remain randomly initialized.",
        ckpt_block_size,
        model_block_size,
    )

    wpe_key = "transformer.wpe.weight"
    if wpe_key in state_dict:
        old_weight = state_dict[wpe_key]
        if old_weight.shape[0] != ckpt_block_size:
            logger.warning(
                "Checkpoint wpe.weight rows (%d) do not match recorded block_size (%d).",
                old_weight.shape[0], ckpt_block_size
            )
        new_weight = model.transformer.wpe.weight.detach().clone()
        new_weight[:old_weight.size(0)] = old_weight
        state_dict[wpe_key] = new_weight

    bias_like = [
        key for key in state_dict.keys()
        if key.endswith("attn.bias") or key.endswith("attn.mask")
    ]
    for key in bias_like:
        state_dict.pop(key)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    allowed_missing = set(
        key for key in missing_keys
        if key.endswith("attn.bias") or key.endswith("attn.mask")
    )
    leftover_missing = [k for k in missing_keys if k not in allowed_missing]
    if leftover_missing:
        logger.warning("Missing keys when loading checkpoint: %s", leftover_missing)

    if unexpected_keys:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected_keys)


# -----------------------------------------------------------------------------
# Evaluation (bucketed accuracies; aligned with SFT protocol)
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model: GPT,
                   pairs: List[Pair],
                   stages: Sequence[Sequence[int]],
                   stoi: Dict[str, int],
                   itos: Dict[int, str],
                   graph: nx.DiGraph,
                   device: torch.device,
                   temperature: float,
                   top_k: int,
                   max_steps: int,
                   max_pairs: int = 5000) -> Dict[str, Dict[str, float]]:
    model.eval()
    stop_token_id = stoi["\n"]
    block_size = model.config.block_size

    buckets: Dict[BucketName, List[Pair]] = {
        "S1->S2": [],
        "S2->S3": [],
        "S1->S3": [],
    }
    for s, t in pairs[:max_pairs]:
        bucket = bucket_for_pair(s, t, stages)
        if bucket:
            buckets[bucket].append((s, t))

    results: Dict[str, Dict[str, float]] = {}
    total_correct = 0
    total_cases = 0

    for bucket_name, bucket_pairs in buckets.items():
        correct = 0
        for source, target in bucket_pairs:
            prompt_tokens = [
                stoi[str(source)],
                stoi[str(target)],
                stoi[str(source)],
            ]
            max_new_tokens = safe_max_new_tokens(block_size, len(prompt_tokens), max_steps)

            x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

            generated = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
            )[0].tolist()

            new_tokens = generated[len(prompt_tokens):]
            decoded_tokens = decode_tokens(new_tokens, itos, stop_token_id)
            generated_nodes = tokens_to_nodes(decoded_tokens)
            full_path_nodes = assemble_full_path(source, generated_nodes)  # <-- fixed

            if is_valid_path(full_path_nodes, source, target, stages, graph):
                correct += 1

        total_correct += correct
        total_cases += len(bucket_pairs)
        acc = correct / len(bucket_pairs) if bucket_pairs else 0.0
        results[bucket_name] = {
            "correct": correct,
            "total": len(bucket_pairs),
            "accuracy": acc,
        }

    overall_acc = total_correct / total_cases if total_cases else 0.0
    results["overall"] = {
        "correct": total_correct,
        "total": total_cases,
        "accuracy": overall_acc,
    }

    model.train()
    return results


# -----------------------------------------------------------------------------
# Log-prob and KL
# -----------------------------------------------------------------------------
def compute_logprob_and_kl(model: GPT,
                           base_model: Optional[GPT],
                           traj_ids: List[int],
                           action_start: int,
                           device: torch.device) -> Tuple[Tensor, Tensor]:
    if len(traj_ids) <= action_start:
        zero = torch.tensor(0.0, device=device)
        return zero, zero

    x_ids = torch.tensor(traj_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y_ids = torch.tensor(traj_ids[1:], dtype=torch.long, device=device).unsqueeze(0)

    logits, _ = model(x_ids, y_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs.gather(-1, y_ids.unsqueeze(-1)).squeeze(-1)  # [1, T]
    logprob_sum = selected[:, action_start - 1:].sum()

    if base_model is None:
        kl_sum = torch.tensor(0.0, device=device)
    else:
        with torch.no_grad():
            base_logits, _ = base_model(x_ids, y_ids)
            base_log_probs = F.log_softmax(base_logits, dim=-1)
        probs = log_probs.exp()
        kl_per_token = (probs * (log_probs - base_log_probs)).sum(dim=-1)
        kl_sum = kl_per_token[:, action_start - 1:].sum()

    return logprob_sum, kl_sum


def build_prompt(source: int,
                 target: int,
                 stoi: Dict[str, int]) -> List[int]:
    return [
        stoi[str(source)],
        stoi[str(target)],
        stoi[str(source)],
    ]


def build_traj_ids(prompt_ids: List[int],
                   sampled_ids: List[int],
                   stop_token_id: int,
                   block_size: int) -> List[int]:
    traj = prompt_ids + sampled_ids
    if len(traj) >= block_size:
        # Defensive guard; safe_max_new_tokens should already prevent overflow.
        traj = traj[:block_size - 1]
    if not sampled_ids or sampled_ids[-1] != stop_token_id:
        if len(traj) < block_size:
            traj.append(stop_token_id)
    return traj


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_dir = Path(args.data_dir).resolve()
    train_txt = data_dir / f"train_{args.train_paths_per_pair}.txt"
    test_txt = data_dir / "test.txt"

    if not train_txt.exists():
        raise FileNotFoundError(f"Training text file not found: {train_txt}")
    if not test_txt.exists():
        raise FileNotFoundError(f"Test text file not found: {test_txt}")

    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    stoi: Dict[str, int] = meta["stoi"]
    itos: Dict[int, str] = meta["itos"]
    vocab_size = meta["vocab_size"]
    dataset_block_size = meta["block_size"]

    with open(data_dir / "stage_info.pkl", "rb") as f:
        stage_info = pickle.load(f)
    stages: List[List[int]] = stage_info["stages"]

    graph = nx.read_graphml(data_dir / "composition_graph.graphml")

    out_dir = prepare_output_dir(args.log_dir)
    logger = get_logger(os.path.join(out_dir, "train_pg.log"))
    logger.info("Policy Gradient training started.")
    logger.info("Output directory: %s", out_dir)
    logger.info("KL coefficient: %.6f", args.kl_coef)

    ckpt = torch.load(args.sft_checkpoint, map_location="cpu")
    ckpt_model_args = ckpt.get("model_args", {})

    def resolve_numeric(attr_name: str, ckpt_key: str, default_value):
        cli_value = getattr(args, attr_name, None)
        if cli_value is not None:
            return cli_value
        if ckpt_model_args and ckpt_key in ckpt_model_args:
            return ckpt_model_args[ckpt_key]
        return default_value

    def resolve_bias(default_value: bool) -> bool:
        if args.bias is not None:
            return args.bias.lower() == "true"
        if ckpt_model_args and "bias" in ckpt_model_args:
            return bool(ckpt_model_args["bias"])
        return default_value

    resolved_block_size = resolve_numeric("block_size_override", "block_size", dataset_block_size)
    if resolved_block_size != dataset_block_size:
        logger.warning("Using block_size=%d (override/checkpoint) while dataset meta reports %d.",
                       resolved_block_size, dataset_block_size)

    model_args = dict(
        vocab_size=vocab_size,
        block_size=resolved_block_size,
        n_layer=resolve_numeric("n_layer", "n_layer", 1),
        n_head=resolve_numeric("n_head", "n_head", 1),
        n_embd=resolve_numeric("n_embd", "n_embd", 120),
        dropout=resolve_numeric("dropout", "dropout", 0.0),
        bias=resolve_bias(False),
    )

    logger.info("Resolved model configuration: %s", json.dumps(model_args))

    ckpt_block_size = ckpt_model_args.get("block_size", dataset_block_size)

    model = GPT(GPTConfig(**model_args)).to(device)
    load_state_dict_with_block_resize(
        model=model,
        raw_state_dict=ckpt["model"],
        ckpt_block_size=ckpt_block_size,
        logger=logger,
    )

    if args.kl_coef > 0:
        base_model = GPT(GPTConfig(**model_args)).to(device)
        load_state_dict_with_block_resize(
            model=base_model,
            raw_state_dict=ckpt["model"],
            ckpt_block_size=ckpt_block_size,
            logger=logger,
        )
        for p in base_model.parameters():
            p.requires_grad = False
        base_model.eval()
    else:
        base_model = None

    optimizer = model.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type="cuda" if device.type == "cuda" else "cpu",
    )

    train_pairs = load_pairs(train_txt)
    logger.info("Loaded %d unique (source, target) pairs for PG training.", len(train_pairs))

    eval_pairs: List[Pair] = []
    with open(test_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                eval_pairs.append((int(parts[0]), int(parts[1])))

    stop_token_id = stoi["\n"]
    prompt_len = 3  # build_prompt always returns 3 tokens
    action_start = prompt_len
    baseline_reward = 0.0
    metrics_path = out_dir / "metrics_pg.jsonl"

    # Rollout cap under block_size
    rollout_cap = safe_max_new_tokens(model.config.block_size, prompt_len, args.max_rollout_steps)
    if rollout_cap < args.max_rollout_steps:
        logger.warning(
            "Requested max_rollout_steps=%d exceeds block_size=%d capacity; truncated to %d.",
            args.max_rollout_steps, model.config.block_size, rollout_cap
        )

    model.train()
    bucket_names = ["S1->S2", "S2->S3", "S1->S3"]

    for iteration in range(1, args.max_iters + 1):
        batch_pairs = random.choices(train_pairs, k=args.batch_size)

        pg_losses: List[Tensor] = []
        kl_losses: List[Tensor] = []
        rewards: List[float] = []
        path_lengths: List[int] = []

        bucket_reward_sum = defaultdict(float)
        bucket_counts = defaultdict(int)

        successes = 0

        for source, target in batch_pairs:
            prompt_ids = build_prompt(source, target, stoi)
            x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

            max_new_tokens = safe_max_new_tokens(
                model.config.block_size, len(prompt_ids), args.max_rollout_steps
            )

            generated_full = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=args.rollout_temperature,
                top_k=args.rollout_top_k if args.rollout_top_k > 0 else None,
            )[0].tolist()

            sampled_ids = generated_full[len(prompt_ids):]
            decoded_tokens = decode_tokens(sampled_ids, itos, stop_token_id)
            generated_nodes = tokens_to_nodes(decoded_tokens)
            full_path_nodes = assemble_full_path(source, generated_nodes)  # <-- fixed

            reward = 1.0 if is_valid_path(full_path_nodes, source, target, stages, graph) else 0.0
            rewards.append(reward)
            if reward > 0.5:
                successes += 1

            bucket = bucket_for_pair(source, target, stages)
            if bucket:
                bucket_reward_sum[bucket] += reward
                bucket_counts[bucket] += 1

            # Keep the original metric behavior (length of generated node sequence).
            path_lengths.append(len(generated_nodes))

            traj_ids = build_traj_ids(prompt_ids, sampled_ids, stop_token_id, model.config.block_size)
            logprob_sum, kl_sum = compute_logprob_and_kl(
                model=model,
                base_model=base_model,
                traj_ids=traj_ids,
                action_start=action_start,
                device=device,
            )

            advantage = reward - baseline_reward
            if args.adv_clip > 0:
                advantage = float(np.clip(advantage, -args.adv_clip, args.adv_clip))

            pg_losses.append(-advantage * logprob_sum)
            kl_losses.append(kl_sum)

            baseline_reward = args.baseline_beta * baseline_reward + (1 - args.baseline_beta) * reward

        mean_pg_loss = torch.stack(pg_losses).mean()
        mean_kl_loss = torch.stack(kl_losses).mean() if base_model is not None else torch.tensor(0.0, device=device)
        total_loss = mean_pg_loss + args.kl_coef * mean_kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        avg_path_len = float(np.mean(path_lengths)) if path_lengths else 0.0
        success_rate = successes / len(batch_pairs)

        if iteration % 50 == 0:
            logger.info(
                "Iter %6d | reward=%.3f | success=%.3f | avg_path=%.2f | pg_loss=%.4f | kl_loss=%.4f",
                iteration, avg_reward, success_rate, avg_path_len,
                mean_pg_loss.item(), mean_kl_loss.item()
            )

        record = {
            "iter": iteration,
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "avg_path_len": avg_path_len,
            "pg_loss": float(mean_pg_loss.item()),
            "kl_loss": float(mean_kl_loss.item()),
            "total_loss": float(total_loss.item()),
            "baseline": float(baseline_reward),
        }
        for bucket in bucket_names:
            cnt = bucket_counts.get(bucket, 0)
            total = float(cnt) if cnt > 0 else 1.0
            record[f"train_reward/{bucket}"] = bucket_reward_sum.get(bucket, 0.0) / total
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if iteration % args.eval_interval == 0 or iteration == args.max_iters:
            eval_results = evaluate_model(
                model=model,
                pairs=eval_pairs,
                stages=stages,
                stoi=stoi,
                itos=itos,
                graph=graph,
                device=device,
                temperature=args.eval_temperature,
                top_k=args.eval_top_k,
                max_steps=args.max_rollout_steps,
                max_pairs=args.max_eval_pairs,
            )
            logger.info("---- Evaluation at iter %d ----", iteration)
            for bucket, stats in eval_results.items():
                logger.info(
                    "  %s: %.2f%% (%d / %d)",
                    bucket,
                    stats["accuracy"] * 100.0,
                    stats["correct"],
                    stats["total"],
                )
            eval_record = {"iter": iteration, "eval": eval_results}
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(eval_record) + "\n")

        if iteration % args.save_interval == 0 or iteration == args.max_iters:
            ckpt_path = out_dir / f"ckpt_pg_{iteration}.pt"
            torch.save(
                {
                    "iter_num": iteration,
                    "model": model.state_dict(),
                    "model_args": model_args,
                    "config": vars(args),
                },
                ckpt_path,
            )
            logger.info("Saved PG checkpoint to %s", ckpt_path)

    logger.info("PG training finished.")


if __name__ == "__main__":
    main()