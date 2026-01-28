#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphA generator for multi-stage layered DAG experiments.

This script creates a directed acyclic graph (DAG) with K stages (layers).
Nodes are arranged as:
  stage 1: 0 .. nodes_per_stage-1
  stage 2: nodes_per_stage .. 2*nodes_per_stage-1
  ...
Edges:
  - Intra-stage edges: within each stage, only from lower index to higher index (to keep a DAG),
    sampled with probability p_intra.
  - Inter-stage edges: from earlier stage to later stage, sampled with probability p_global.
    If --allow_stage_skip is NOT set, only edges from stage i to i+1 are allowed.

Outputs (in a dedicated folder):
  - composition_graph.graphml   (node IDs are strings, e.g., "0", "1", ...)
  - stage_info.pkl              (python dict with stage lists, etc.)
  - metadata.json               (basic statistics + parameters)

Example:
  python data/simple_graph/create_graph.py \
      --nodes_per_stage 30 \
      --num_stages 5 \
      --p_global 0.20 \
      --seed 42 \
      --experiment_name graphA \
      --output_root data/graphs
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import networkx as nx
import numpy as np


def _validate_args(args: argparse.Namespace) -> None:
    if args.nodes_per_stage <= 0:
        raise ValueError("--nodes_per_stage must be > 0")
    if args.num_stages < 2:
        raise ValueError("--num_stages must be >= 2")
    if not (0.0 <= args.p_global <= 1.0):
        raise ValueError("--p_global must be in [0, 1]")
    if args.p_intra is not None and not (0.0 <= args.p_intra <= 1.0):
        raise ValueError("--p_intra must be in [0, 1]")


def build_graph_a(
    nodes_per_stage: int,
    num_stages: int,
    p_global: float,
    p_intra: float,
    allow_stage_skip: bool,
    rng: random.Random,
) -> Tuple[nx.DiGraph, List[List[int]]]:
    """
    Construct a layered DAG following the GraphA rules.

    Returns:
        G: networkx.DiGraph with node IDs as strings ("0", "1", ...)
        stages: list of stages, each stage is a list of int node indices
    """
    total_nodes = nodes_per_stage * num_stages
    G = nx.DiGraph()

    # Use string node IDs to be consistent with GraphML behavior.
    G.add_nodes_from(str(i) for i in range(total_nodes))

    stages: List[List[int]] = []
    for stage_idx in range(num_stages):
        start = stage_idx * nodes_per_stage
        stop = (stage_idx + 1) * nodes_per_stage
        stage_nodes = list(range(start, stop))
        stages.append(stage_nodes)

        # Store stage index as a node attribute for convenience.
        for n in stage_nodes:
            G.nodes[str(n)]["stage"] = stage_idx  # 0-based stage id

    # Intra-stage edges (i < j to keep DAG)
    for stage_nodes in stages:
        for i_idx, src in enumerate(stage_nodes[:-1]):
            for dst in stage_nodes[i_idx + 1 :]:
                if rng.random() < p_intra:
                    G.add_edge(str(src), str(dst))

    # Inter-stage edges
    for src_stage_idx in range(num_stages - 1):
        for dst_stage_idx in range(src_stage_idx + 1, num_stages):
            if (not allow_stage_skip) and (dst_stage_idx != src_stage_idx + 1):
                continue
            src_nodes = stages[src_stage_idx]
            dst_nodes = stages[dst_stage_idx]
            for src in src_nodes:
                for dst in dst_nodes:
                    if rng.random() < p_global:
                        G.add_edge(str(src), str(dst))

    # Safety check (should always be a DAG by construction)
    if not nx.is_directed_acyclic_graph(G):
        raise RuntimeError("Graph construction produced a non-DAG. Please check edge rules.")

    return G, stages


def summarize_graph(G: nx.DiGraph, stages: List[List[int]]) -> Dict[str, Any]:
    """
    Compute simple graph statistics for logging and metadata.

    Includes:
      - num_nodes, num_edges, num_stages
      - stage_i_size
      - reachable pair counts for every stage pair Si->Sj (i<j):
        count of reachable (u in Si, v in Sj) pairs
    """
    summary: Dict[str, Any] = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_stages": len(stages),
    }

    for idx, nodes in enumerate(stages, start=1):
        summary[f"stage_{idx}_size"] = len(nodes)

    # Pre-build stage sets with string node IDs for fast intersection.
    stage_sets = [set(str(n) for n in stage_nodes) for stage_nodes in stages]

    for i in range(len(stages)):
        for j in range(i + 1, len(stages)):
            summary[f"S{i+1}->S{j+1}"] = 0

    # For each src node compute descendants once, then count by stage.
    for i, stage_nodes in enumerate(stages):
        for src in stage_nodes:
            reachable = nx.descendants(G, str(src))
            if not reachable:
                continue
            for j in range(i + 1, len(stages)):
                summary[f"S{i+1}->S{j+1}"] += len(reachable & stage_sets[j])

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a multi-stage layered DAG (GraphA).")
    parser.add_argument("--nodes_per_stage", type=int, default=30,
                        help="Number of nodes per stage (default: 30).")
    parser.add_argument("--num_stages", type=int, default=5,
                        help="Total number of stages (default: 5).")
    parser.add_argument("--p_global", type=float, required=True,
                        help="Edge probability for inter-stage edges.")
    parser.add_argument("--p_intra", type=float, default=None,
                        help="Edge probability within a stage (default: same as --p_global).")
    parser.add_argument("--allow_stage_skip", action="store_true",
                        help="Allow edges that skip stages, e.g., S1->S3. Default: only S(i)->S(i+1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--experiment_name", type=str, default="graphA",
                        help="Experiment prefix for folder naming (default: graphA).")
    parser.add_argument("--output_root", type=str, default="data/graphs",
                        help="Root directory where the graph folder will be created.")
    return parser.parse_args()


def make_output_dir(args: argparse.Namespace) -> Path:
    # Include p_intra and allow_stage_skip in folder name to avoid collisions.
    pglob = int(round(args.p_global * 100))
    pintra = int(round(args.p_intra * 100))
    skip_flag = "skip1" if args.allow_stage_skip else "skip0"
    suffix = (
        f"pg{pglob:03d}_"
        f"pi{pintra:03d}_"
        f"nps{args.nodes_per_stage}_"
        f"ns{args.num_stages}_"
        f"{skip_flag}_"
        f"seed{args.seed}"
    )
    folder_name = f"{args.experiment_name}_{suffix}"
    out_dir = Path(args.output_root).resolve() / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    args = parse_args()
    _validate_args(args)

    if args.p_intra is None:
        args.p_intra = args.p_global

    # Reproducibility: use a dedicated RNG instead of global random state.
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    G, stages = build_graph_a(
        nodes_per_stage=args.nodes_per_stage,
        num_stages=args.num_stages,
        p_global=args.p_global,
        p_intra=args.p_intra,
        allow_stage_skip=args.allow_stage_skip,
        rng=rng,
    )

    out_dir = make_output_dir(args)

    graph_path = out_dir / "composition_graph.graphml"
    nx.write_graphml(G, graph_path)

    stage_info = {
        "stages": stages,  # list[list[int]]
        "nodes_per_stage": args.nodes_per_stage,
        "num_stages": args.num_stages,
        "allow_stage_skip": bool(args.allow_stage_skip),
        "node_id_format": "stringified_ints",
    }
    with open(out_dir / "stage_info.pkl", "wb") as f:
        pickle.dump(stage_info, f)

    summary = summarize_graph(G, stages)
    summary.update(
        {
            "p_global": float(args.p_global),
            "p_intra": float(args.p_intra),
            "allow_stage_skip": bool(args.allow_stage_skip),
            "seed": int(args.seed),
            "experiment_name": args.experiment_name,
        }
    )

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print(f"GraphA DAG saved to: {out_dir}")
    print(f"GraphML: {graph_path}")
    print("Summary:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")
    print("=" * 70)


if __name__ == "__main__":
    main()