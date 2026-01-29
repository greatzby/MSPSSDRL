#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate an ALPINE-style dataset from a DAG composition graph with K stages.

Key behavior (ALPINE rule implemented here):
  - Consider all reachable (src, tgt) pairs where stage(src) < stage(tgt) by default.
    (Optionally include intra-stage reachable pairs via --include_intra_stage_pairs.)
  - Stratified train/test split by pair type "Si->Sj".
  - Direct-edge pairs (src->tgt exists in the DAG) are ALWAYS included in training
    (as an additional "direct path" sample).
  - Non-direct pairs are split by train_ratio.

Output dataset format:
  Each line is a sequence of integers:
      src dst path_node_0 path_node_1 ... path_node_L
  where path_node_* is a sampled path from src to dst (typically includes src and dst).
  Note: This means src/dst appear twice if the path includes them; we keep this format
  to avoid breaking existing consumers.

Example:
  python data/simple_graph/generate_alpine_allpairs.py \
      --input_graph data/graphs/.../composition_graph.graphml \
      --stage_info data/graphs/.../stage_info.pkl \
      --output_dir data/datasets/.../tier3_full \
      --train_paths_per_pair 20 \
      --eval_paths_per_pair 1 \
      --train_ratio 0.85 \
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple, Set

import networkx as nx
import numpy as np
from tqdm import tqdm

Node = str
Pair = Tuple[Node, Node]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ALPINE-style dataset (Tier-3) for K stages.")
    p.add_argument("--input_graph", type=str, required=True, help="Path to composition_graph.graphml")
    p.add_argument("--stage_info", type=str, required=True, help="Path to stage_info.pkl")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory")

    p.add_argument("--train_paths_per_pair", type=int, default=20,
                   help="Number of random paths per training pair (in addition to the direct-edge sample).")
    p.add_argument("--eval_paths_per_pair", type=int, default=1,
                   help="Number of random paths per evaluation pair.")
    p.add_argument("--train_ratio", type=float, default=0.85,
                   help="Train ratio for non-direct-edge pairs within each type (0~1).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--max_path_attempts", type=int, default=50,
                   help="Max retries when failing to sample a random path.")

    p.add_argument("--include_intra_stage_pairs", action="store_true",
                   help="If set, also include reachable pairs where stage(src)==stage(tgt). Default: False.")
    p.add_argument("--allow_unknown_stage_nodes", action="store_true",
                   help="If set, nodes missing from stage_info are allowed under type 'Unknown'. Default: False.")

    p.add_argument("--verbose_examples", type=int, default=0,
                   help="Print N example pairs per type for sanity check.")

    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.train_ratio <= 1.0):
        raise ValueError("--train_ratio must be in [0, 1]")
    if args.train_paths_per_pair < 0:
        raise ValueError("--train_paths_per_pair must be >= 0")
    if args.eval_paths_per_pair < 0:
        raise ValueError("--eval_paths_per_pair must be >= 0")
    if args.max_path_attempts <= 0:
        raise ValueError("--max_path_attempts must be > 0")


def load_graph(graph_path: Path) -> nx.DiGraph:
    G = nx.read_graphml(graph_path)
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Input graph must be a DAG for Tier-3 generation.")
    return G


def load_stage_info(stage_info_path: Path) -> dict:
    with open(stage_info_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"stage_info.pkl must contain a dict, got {type(obj).__name__}")
    if "stages" not in obj:
        raise KeyError("stage_info must contain key 'stages' (list of stage node lists).")
    return obj


def normalize_node_id(x) -> str:
    """GraphML nodes often become strings; stage info might store ints."""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float) and float(x).is_integer():
        return str(int(x))
    return str(x)


def build_node_to_stage(stage_info: dict) -> Tuple[List[List[str]], Dict[str, int]]:
    """
    Returns:
      stages_norm: List of stages, each a list[str] node ids
      node_to_stage: map node_id(str) -> stage_index (0-based)
    """
    stages_raw = stage_info["stages"]
    if not isinstance(stages_raw, (list, tuple)) or len(stages_raw) < 2:
        raise ValueError("stage_info['stages'] must be a list with >= 2 stages.")

    stages_norm: List[List[str]] = []
    node_to_stage: Dict[str, int] = {}
    for si, stage_nodes in enumerate(stages_raw):
        if not isinstance(stage_nodes, (list, tuple, set)):
            raise ValueError(f"stage {si} must be list/tuple/set, got {type(stage_nodes).__name__}")
        norm = [normalize_node_id(n) for n in stage_nodes]
        stages_norm.append(norm)
        for n in norm:
            if n in node_to_stage:
                raise ValueError(f"Node {n} appears in multiple stages ({node_to_stage[n]} and {si}).")
            node_to_stage[n] = si
    return stages_norm, node_to_stage


def precompute_ancestors_cache(G: nx.DiGraph) -> Dict[Node, Set[Node]]:
    """
    For each target node, compute its ancestors set (including itself).
    This cache enables constrained random-walk sampling towards a given target.
    """
    cache: Dict[Node, Set[Node]] = {}
    for node in tqdm(G.nodes, desc="Pre-computing ancestors cache"):
        anc = nx.ancestors(G, node)
        anc.add(node)
        cache[str(node)] = set(str(x) for x in anc)
    return cache


def try_int(x: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(
            f"Node id '{x}' is not an int. This generator writes integer node IDs. "
            f"Please rename nodes to ints or adjust serialization."
        ) from e


def generate_random_path(
    G: nx.DiGraph,
    source: Node,
    target: Node,
    ancestors_cache: Dict[Node, Set[Node]],
    rng: random.Random,
    max_attempts: int = 50,
) -> Optional[List[str]]:
    """
    Sample a random path from source to target using a constrained random walk:
    at each step, we only move to a successor that is an ancestor of the target
    (i.e., can still reach the target).
    """
    target_anc = ancestors_cache[target]

    for _ in range(max_attempts):
        path: List[Node] = [source]
        current = source
        visited = {source}

        while current != target:
            successors = list(G.successors(current))
            valid = [n for n in successors if (n in target_anc and n not in visited)]
            if not valid:
                # fallback: allow revisits (still a DAG, but may help in some edge cases)
                valid = [n for n in successors if (n in target_anc)]
            if not valid:
                break

            current = rng.choice(valid)
            path.append(current)
            visited.add(current)

            if len(path) > G.number_of_nodes():  # safety
                break

        if path and path[-1] == target:
            return [str(n) for n in path]

    return None


def classify_pair_by_stage(
    src: Node,
    dst: Node,
    node_to_stage: Dict[str, int],
    include_intra: bool,
    allow_unknown: bool,
) -> Optional[str]:
    s = node_to_stage.get(src, None)
    t = node_to_stage.get(dst, None)

    if s is None or t is None:
        if allow_unknown:
            return "Unknown"
        missing = []
        if s is None:
            missing.append(f"src={src}")
        if t is None:
            missing.append(f"dst={dst}")
        raise KeyError("Node missing from stage_info: " + ", ".join(missing))

    if s == t and not include_intra:
        return None
    if t < s:
        # Generally shouldn't happen for a layered DAG; skip for safety.
        return None
    return f"S{s+1}->S{t+1}"


def stratified_split_pairs(
    pairs_by_type: Dict[str, List[Pair]],
    G: nx.DiGraph,
    train_ratio: float,
    rng: random.Random,
) -> Tuple[List[Pair], List[Pair], Dict[str, Dict[str, int]]]:
    """
    Split pairs stratified by type "Si->Sj".

    Returns:
      train_pairs, test_pairs, split_stats_by_type[type] = {total, direct, non_direct, train, test}
    """
    train_pairs: List[Pair] = []
    test_pairs: List[Pair] = []
    stats: Dict[str, Dict[str, int]] = {}

    for pair_type, pair_list in pairs_by_type.items():
        pair_list = list(pair_list)
        rng.shuffle(pair_list)

        direct_pairs = [pair for pair in pair_list if G.has_edge(*pair)]
        non_direct_pairs = [pair for pair in pair_list if not G.has_edge(*pair)]

        # ALPINE rule: direct pairs always in training
        train_pairs.extend(direct_pairs)

        cutoff = int(len(non_direct_pairs) * train_ratio)
        train_non_direct = non_direct_pairs[:cutoff]
        test_non_direct = non_direct_pairs[cutoff:]

        train_pairs.extend(train_non_direct)
        test_pairs.extend(test_non_direct)

        stats[pair_type] = {
            "total_pairs": len(pair_list),
            "direct_pairs": len(direct_pairs),
            "non_direct_pairs": len(non_direct_pairs),
            "train_pairs": len(direct_pairs) + len(train_non_direct),
            "test_pairs": len(test_non_direct),
        }

    return train_pairs, test_pairs, stats


def create_dataset(
    G: nx.DiGraph,
    node_to_stage: Dict[str, int],
    include_intra: bool,
    allow_unknown: bool,
    ancestors_cache: Dict[Node, Set[Node]],
    train_pairs: Sequence[Pair],
    test_pairs: Sequence[Pair],
    train_paths_per_pair: int,
    eval_paths_per_pair: int,
    max_path_attempts: int,
    rng: random.Random,
) -> Tuple[List[List[int]], List[List[int]], Dict[str, Dict[str, int]]]:
    """
    Sample random paths for each pair and build train/test samples.

    Returns:
      train_samples, test_samples, counts_plain
    """
    train_samples: List[List[int]] = []
    test_samples: List[List[int]] = []

    counts: Dict[str, DefaultDict[str, int]] = {
        "train": defaultdict(int),
        "test": defaultdict(int),
    }

    pair_type_cache: Dict[Pair, str] = {}

    def get_pair_type(pair: Pair) -> str:
        if pair not in pair_type_cache:
            src, dst = pair
            t = classify_pair_by_stage(src, dst, node_to_stage, include_intra, allow_unknown)
            pair_type_cache[pair] = t or "Other"
        return pair_type_cache[pair]

    def record(container: List[List[int]], pair: Pair, path_nodes: Sequence[str], split: str) -> None:
        src, dst = pair
        sample = [try_int(src), try_int(dst)] + [try_int(x) for x in path_nodes]
        container.append(sample)
        pt = get_pair_type(pair)
        counts[split][pt] += 1
        counts[split]["__total__"] += 1

    # Train: add a "direct path" sample if edge exists, plus train_paths_per_pair random samples.
    for source, target in tqdm(train_pairs, desc="Generating training samples"):
        pair = (source, target)
        if G.has_edge(source, target):
            record(train_samples, pair, [source, target], "train")

        for _ in range(train_paths_per_pair):
            path = generate_random_path(
                G, source, target, ancestors_cache, rng=rng, max_attempts=max_path_attempts
            )
            if path is not None:
                record(train_samples, pair, path, "train")

    # Test: only random sampled paths
    for source, target in tqdm(test_pairs, desc="Generating eval samples"):
        pair = (source, target)
        for _ in range(eval_paths_per_pair):
            path = generate_random_path(
                G, source, target, ancestors_cache, rng=rng, max_attempts=max_path_attempts
            )
            if path is not None:
                record(test_samples, pair, path, "test")

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)

    counts_plain = {k: dict(v) for k, v in counts.items()}
    return train_samples, test_samples, counts_plain


def write_dataset(lines: Iterable[List[int]], file_path: Path) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(" ".join(map(str, line)) + "\n")


def main() -> None:
    args = parse_args()
    _validate_args(args)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    graph_path = Path(args.input_graph).resolve()
    stage_path = Path(args.stage_info).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    G = load_graph(graph_path)
    stage_info = load_stage_info(stage_path)
    stages_norm, node_to_stage = build_node_to_stage(stage_info)
    K = len(stages_norm)

    # Precompute cache used by random path sampler.
    ancestors_cache = precompute_ancestors_cache(G)

    # Collect reachable pairs by stage type.
    pairs_by_type: Dict[str, List[Pair]] = defaultdict(list)

    nodes = [str(n) for n in G.nodes]
    for src in tqdm(nodes, desc="Collecting reachable pairs (via descendants)"):
        for dst in nx.descendants(G, src):
            dst = str(dst)
            t = classify_pair_by_stage(
                src, dst,
                node_to_stage=node_to_stage,
                include_intra=args.include_intra_stage_pairs,
                allow_unknown=args.allow_unknown_stage_nodes,
            )
            if t is not None:
                pairs_by_type[t].append((src, dst))

    train_pairs, test_pairs, split_stats_by_type = stratified_split_pairs(
        pairs_by_type, G, args.train_ratio, rng
    )

    # Optional sanity prints
    if args.verbose_examples > 0:
        print("=" * 70)
        print("Example pairs by type:")
        for t, pairs in sorted(pairs_by_type.items()):
            ex = pairs[: args.verbose_examples]
            print(f"  {t}: {ex}")
        print("=" * 70)

    print("=" * 70)
    print(f"Stages: K={K}")
    print("Reachable pair statistics (before path sampling):")
    for t in sorted(pairs_by_type.keys()):
        st = split_stats_by_type[t]
        print(
            f"  {t:8s} total={st['total_pairs']:7d} | "
            f"direct={st['direct_pairs']:7d} | "
            f"train={st['train_pairs']:7d} | test={st['test_pairs']:7d}"
        )
    print("=" * 70)

    train_samples, test_samples, sample_counts = create_dataset(
        G=G,
        node_to_stage=node_to_stage,
        include_intra=args.include_intra_stage_pairs,
        allow_unknown=args.allow_unknown_stage_nodes,
        ancestors_cache=ancestors_cache,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        train_paths_per_pair=args.train_paths_per_pair,
        eval_paths_per_pair=args.eval_paths_per_pair,
        max_path_attempts=args.max_path_attempts,
        rng=rng,
    )

    train_file = out_dir / f"train_{args.train_paths_per_pair}.txt"
    test_file = out_dir / "test.txt"
    write_dataset(train_samples, train_file)
    write_dataset(test_samples, test_file)

    # Copy graph & stage info for provenance
    nx.write_graphml(G, out_dir / "composition_graph.graphml")
    with open(out_dir / "stage_info.pkl", "wb") as f:
        pickle.dump(stage_info, f)

    summary = {
        "k_stages": K,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "train_paths_per_pair": args.train_paths_per_pair,
        "eval_paths_per_pair": args.eval_paths_per_pair,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "include_intra_stage_pairs": bool(args.include_intra_stage_pairs),
        "allow_unknown_stage_nodes": bool(args.allow_unknown_stage_nodes),
        "pair_counts_by_type": {k: len(v) for k, v in pairs_by_type.items()},
        "pair_split_stats_by_type": split_stats_by_type,
        "pair_split_counts": {
            "train_pairs": len(train_pairs),
            "test_pairs": len(test_pairs),
        },
        "sample_counts_by_type": sample_counts,
        "format_note": "Each line: src dst path_nodes... (ints). Path may include src/dst.",
    }

    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print(f"Dataset saved to: {out_dir}")
    print(f"  Train: {train_file} (samples={len(train_samples)})")
    print(f"  Test : {test_file} (samples={len(test_samples)})")
    print("=" * 70)


if __name__ == "__main__":
    main()