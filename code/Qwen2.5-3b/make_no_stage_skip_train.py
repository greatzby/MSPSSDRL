#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter training set by removing stage-skip pairs (e.g., S1->S3, S2->S5) while keeping eval unchanged.

Typical use:
  - First generate a full dataset (train contains all types).
  - Then run this script to create a new dataset variant where training only includes adjacent-stage pairs.

Example:
  python make_no_stage_skip_train.py \
    --src-dir data/datasets/.../tier3_full \
    --dest-dir data/datasets/.../tier3_train_adjacent_only \
    --paths-per-pair 20 \
    --max-jump 1 \
    --seed 42 \
    --pair-report pair_report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pickle


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remove stage-skip pairs from training set; keep eval unchanged."
    )

    io = p.add_argument_group("I/O")
    io.add_argument("--src-dir", type=Path, required=True, help="Full dataset directory (source).")
    io.add_argument("--dest-dir", type=Path, required=True, help="Destination directory (variant).")

    train = p.add_argument_group("Train file selection")
    train.add_argument("--paths-per-pair", type=int, default=20, help="Used to locate train_{K}.txt.")
    train.add_argument(
        "--train-file-pattern",
        type=str,
        default="train_{paths}.txt",
        help="Format string for the training file (must include '{paths}').",
    )

    stage = p.add_argument_group("Stage info")
    stage.add_argument(
        "--stage-info-name",
        type=str,
        default="stage_info.pkl",
        help="Stage info file name inside src-dir.",
    )

    fmt = p.add_argument_group("Line parsing format")
    fmt.add_argument(
        "--source-index",
        type=int,
        default=0,
        help="Token index for source node in each line (supports negative index).",
    )
    fmt.add_argument(
        "--target-index",
        type=int,
        default=1,
        help=(
            "Token index for target node in each line (supports negative index). "
            "NOTE: generator format is typically: [src tgt ...path...], so default target-index=1."
        ),
    )

    rule = p.add_argument_group("Filtering rules")
    rule.add_argument(
        "--max-jump",
        type=int,
        default=1,
        help=(
            "Keep pairs only if (dst_stage - src_stage) <= max_jump. "
            "For adjacent-only training, use 1."
        ),
    )
    rule.add_argument(
        "--keep-intra-stage",
        action="store_true",
        help="Also keep intra-stage pairs where (dst_stage - src_stage) == 0. Default: False.",
    )
    rule.add_argument(
        "--keep-unknown-stage",
        action="store_true",
        help="If a node is missing from stage_info, keep it as type=Unknown. Default: False (drop).",
    )

    misc = p.add_argument_group("Misc")
    misc.add_argument("--seed", type=int, default=2025, help="Shuffle seed for output training file.")
    misc.add_argument(
        "--pair-report",
        type=Path,
        default=None,
        help="Optional CSV path dumping per-pair stats & keep/drop decision.",
    )
    misc.add_argument(
        "--copy-patterns",
        nargs="*",
        default=[
            "composition_graph.graphml",
            "dataset_summary.json",
            "stage_info.pkl",
            "test.txt",
        ],
        help="Files/dirs to copy from src to dest if they exist (test.txt is copied unchanged).",
    )

    args = p.parse_args()
    if args.max_jump < 0:
        raise ValueError("--max-jump must be >= 0")
    return args


# -----------------------------------------------------------------------------
# Stage info helpers
# -----------------------------------------------------------------------------
def normalize_node_id(x) -> str:
    """Normalize stage_info node ids into comparable string keys."""
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float) and float(x).is_integer():
        return str(int(x))
    return str(x)


def load_node_to_stage(stage_info_path: Path) -> Tuple[Dict[str, int], int]:
    """
    Load stage_info.pkl which must contain:
      {"stages": [stage0_nodes, stage1_nodes, ...]}
    Returns:
      node_to_stage: {node_id_str -> stage_index_0_based}
      K: number of stages
    """
    with open(stage_info_path, "rb") as f:
        info = pickle.load(f)
    if not isinstance(info, dict) or "stages" not in info:
        raise ValueError("stage_info.pkl must be a dict containing key 'stages'.")

    stages = info["stages"]
    node_to_stage: Dict[str, int] = {}
    for si, nodes in enumerate(stages):
        for n in nodes:
            nid = normalize_node_id(n)
            if nid in node_to_stage:
                raise ValueError(f"Node {nid} appears in multiple stages.")
            node_to_stage[nid] = si

    return node_to_stage, len(stages)


# -----------------------------------------------------------------------------
# Train line parsing helpers
# -----------------------------------------------------------------------------
def canonical_token(tok: str) -> str:
    return tok.strip()


def resolve_index(tokens: List[str], index: int) -> str:
    """Resolve positive/negative token index."""
    if index >= 0:
        if index >= len(tokens):
            raise IndexError(f"Token index {index} out of range for tokens={tokens}")
        return tokens[index]

    resolved = len(tokens) + index
    if resolved < 0:
        raise IndexError(f"Token index {index} out of range for tokens={tokens}")
    return tokens[resolved]


def split_pairs(lines: List[str], source_idx: int, target_idx: int) -> Dict[Tuple[str, str], List[str]]:
    """
    Group raw lines by (src, tgt). The line itself is kept verbatim.
    """
    bucket: Dict[Tuple[str, str], List[str]] = {}
    for line in lines:
        parts = [canonical_token(t) for t in line.split()]
        if not parts:
            continue
        src = resolve_index(parts, source_idx)
        tgt = resolve_index(parts, target_idx)
        bucket.setdefault((src, tgt), []).append(line)
    return bucket


# -----------------------------------------------------------------------------
# Filtering logic (kept identical)
# -----------------------------------------------------------------------------
def pair_type(src: str, tgt: str, node_to_stage: Dict[str, int]) -> Optional[str]:
    s = node_to_stage.get(src)
    t = node_to_stage.get(tgt)
    if s is None or t is None:
        return None
    return f"S{s+1}->S{t+1}"


def should_keep_pair(
    src: str,
    tgt: str,
    node_to_stage: Dict[str, int],
    max_jump: int,
    keep_intra: bool,
    keep_unknown: bool,
) -> Tuple[bool, str, Optional[int]]:
    """
    Decide keep/drop for a (src, tgt) pair.

    Returns:
      (keep, ptype, jump)
        - keep: whether to keep
        - ptype: "Sx->Sy" or "Unknown"
        - jump: stage(target)-stage(source), None if Unknown
    """
    s = node_to_stage.get(src)
    t = node_to_stage.get(tgt)
    if s is None or t is None:
        return (keep_unknown, "Unknown", None)

    jump = t - s
    ptype = f"S{s+1}->S{t+1}"

    if jump < 0:
        # backward; usually not expected
        return (False, ptype, jump)

    if jump == 0:
        return (keep_intra, ptype, jump)

    # jump >= 1
    return (jump <= max_jump, ptype, jump)


def safe_ratio(a: int, b: int) -> float:
    return float(a) / b if b else 0.0


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def copy_side_files(src_dir: Path, dest_dir: Path, patterns: List[str]) -> None:
    """
    Copy optional side files/dirs from src to dest if they exist.
    """
    for name in patterns:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dest_dir / name
        if src.is_file():
            shutil.copy2(src, dst)
        elif src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def maybe_dump_pair_report(
    report_path: Optional[Path],
    pair_to_lines: Dict[Tuple[str, str], List[str]],
    keep_pairs: set,
    node_to_stage: Dict[str, int],
    max_jump: int,
    keep_intra: bool,
    keep_unknown: bool,
) -> None:
    if report_path is None:
        return

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src", "tgt", "pair_type", "jump", "num_paths", "kept"])
        for (src, tgt), lines in sorted(pair_to_lines.items()):
            _keep, ptype, jump = should_keep_pair(src, tgt, node_to_stage, max_jump, keep_intra, keep_unknown)
            w.writerow([src, tgt, ptype, "" if jump is None else jump, len(lines), 1 if (src, tgt) in keep_pairs else 0])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    train_name = args.train_file_pattern.format(paths=args.paths_per_pair)
    src_train = args.src_dir / train_name
    if not src_train.exists():
        raise FileNotFoundError(f"Training file not found: {src_train}")

    stage_path = args.src_dir / args.stage_info_name
    if not stage_path.exists():
        raise FileNotFoundError(f"stage_info not found: {stage_path}")

    node_to_stage, K = load_node_to_stage(stage_path)

    raw_lines = [ln.rstrip("\n") for ln in src_train.read_text(encoding="utf-8").splitlines() if ln.strip()]
    pair_to_lines = split_pairs(raw_lines, args.source_index, args.target_index)

    # Original stats
    orig_pairs_total = len(pair_to_lines)
    orig_paths_total = len(raw_lines)

    orig_pairs_by_type = defaultdict(int)
    orig_paths_by_type = defaultdict(int)
    for (src, tgt), lines in pair_to_lines.items():
        ptype = pair_type(src, tgt, node_to_stage) or "Unknown"
        orig_pairs_by_type[ptype] += 1
        orig_paths_by_type[ptype] += len(lines)

    # Select keep/drop pairs (logic preserved)
    keep_pairs = set()
    drop_pairs = set()

    kept_pairs_by_type = defaultdict(int)
    kept_paths_by_type = defaultdict(int)

    for (src, tgt), lines in pair_to_lines.items():
        keep, ptype, _jump = should_keep_pair(
            src,
            tgt,
            node_to_stage=node_to_stage,
            max_jump=args.max_jump,
            keep_intra=args.keep_intra_stage,
            keep_unknown=args.keep_unknown_stage,
        )
        if keep:
            keep_pairs.add((src, tgt))
            kept_pairs_by_type[ptype] += 1
            kept_paths_by_type[ptype] += len(lines)
        else:
            drop_pairs.add((src, tgt))

    # Flatten kept pairs into lines, then shuffle
    kept_lines: List[str] = []
    for pair in keep_pairs:
        kept_lines.extend(pair_to_lines[pair])

    rng = random.Random(args.seed)
    rng.shuffle(kept_lines)

    # Write dest training file
    dest_train = args.dest_dir / train_name
    with open(dest_train, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")

    # Copy side files (including test.txt unchanged)
    copy_side_files(args.src_dir, args.dest_dir, args.copy_patterns)

    # Optional per-pair report
    maybe_dump_pair_report(
        report_path=args.pair_report,
        pair_to_lines=pair_to_lines,
        keep_pairs=keep_pairs,
        node_to_stage=node_to_stage,
        max_jump=args.max_jump,
        keep_intra=args.keep_intra_stage,
        keep_unknown=args.keep_unknown_stage,
    )

    # Summary JSON
    kept_pairs_total = len(keep_pairs)
    kept_paths_total = len(kept_lines)

    summary = {
        "k_stages": K,
        "train_file": train_name,
        "max_jump": args.max_jump,
        "keep_intra_stage": bool(args.keep_intra_stage),
        "keep_unknown_stage": bool(args.keep_unknown_stage),
        "seed": args.seed,
        "counts": {
            "original": {
                "pairs_total": orig_pairs_total,
                "paths_total": orig_paths_total,
                "pairs_by_type": dict(orig_pairs_by_type),
                "paths_by_type": dict(orig_paths_by_type),
            },
            "kept": {
                "pairs_total": kept_pairs_total,
                "paths_total": kept_paths_total,
                "pairs_by_type": dict(kept_pairs_by_type),
                "paths_by_type": dict(kept_paths_by_type),
            },
            "dropped": {
                "pairs_total": len(drop_pairs),
                "paths_total": orig_paths_total - kept_paths_total,
            },
        },
        "ratios": {
            "kept_pairs_ratio": safe_ratio(kept_pairs_total, orig_pairs_total),
            "kept_paths_ratio": safe_ratio(kept_paths_total, orig_paths_total),
        },
        "pair_report": str(args.pair_report) if args.pair_report else None,
    }

    with open(args.dest_dir / "stage_skip_filter_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console report
    print("=" * 70)
    print(f"Created dataset variant in: {args.dest_dir}")
    print(f"  Copied eval (test.txt) unchanged from: {args.src_dir}")
    print(f"  Filtered train: {train_name}")
    print("- ORIGINAL")
    print(f"  pairs={orig_pairs_total:8d} | paths={orig_paths_total:8d}")
    print("- KEPT")
    print(
        f"  pairs={kept_pairs_total:8d} | paths={kept_paths_total:8d} "
        f"| kept_paths_ratio={summary['ratios']['kept_paths_ratio']:.3f}"
    )
    print(
        f"Rule: keep jump<= {args.max_jump} "
        f"{'(and intra-stage)' if args.keep_intra_stage else '(no intra-stage)'} "
        f"{'(keep unknown)' if args.keep_unknown_stage else '(drop unknown)'}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()