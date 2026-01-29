#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-learning fine-tuning for GraphA (token-level RL on graph path generation).

This script initializes a GPT policy from an SFT checkpoint and optimizes a
Q-learning objective over next-token actions. It supports:
  - rollout temperature scheduling and epsilon-greedy exploration
  - process-shaped rewards (valid-edge reward, stage-2 bridge reward, penalties, etc.)
  - optional target network (EMA or periodic hard sync)
  - optional KL regularization to stay close to the SFT reference policy

Prompts and path interpretation (aligned with evaluation):
  - prompt tokens: [src, tgt, src]
  - stop token id: meta["stoi"]["\\n"]
  - the generated continuation is parsed as node ids; the full path is:
      [src] + generated_nodes
  - for S1->S3 pairs, validity additionally requires visiting at least one Stage-2 node

Required inputs:
  - --data_dir must contain: train_{K}.txt, test.txt, meta.pkl, stage_info.pkl,
    composition_graph.graphml
  - --sft_checkpoint is a `.pt` checkpoint used for initialization (and KL reference)

Outputs (written to --log_dir/ql_{timestamp}/):
  - train_qlearning.log
  - metrics_ql.jsonl
  - ckpt_ql_{iter}.pt

Example:
python nanoGPT/nanoGPT_Qlearning.py \
    --data_dir data/datasets/graphA_full_P13_0 \
    --sft_checkpoint out/<your_sft_run_dir>/ckpt_5000.pt \
    --train_paths_per_pair 20 \
    --device cuda:0 \
    --max_iters 20000 \
    --batch_size 32 \
    --max_rollout_steps 32 \
    --reward_type process \
    --reward_valid_transition 0.10 \
    --reward_invalid_transition 0.25 \
    --reward_invalid_token 1.0 \
    --reward_hit_target 1.5
    --kl_coef 0.05
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
    parser = argparse.ArgumentParser(description="Q-Learning fine-tuning for GraphA.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dataset directory (contains train_K.txt, meta.pkl, stage_info.pkl, etc.).")
    parser.add_argument("--sft_checkpoint", type=str, required=True,
                        help="SFT checkpoint (.pt) for initialization and KL reference.")
    parser.add_argument("--train_paths_per_pair", type=int, default=20,
                        help="Value of K for train_{K}.txt.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    # Optional model overrides (not recommended to change arbitrarily)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--bias", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--block_size_override", type=int, default=None,
                        help="If you need to shrink block_size, set this; expansion is not allowed.")

    # Q-learning hyperparameters
    parser.add_argument("--max_iters", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)

    parser.add_argument("--max_rollout_steps", type=int, default=32)
    parser.add_argument("--rollout_top_k", type=int, default=1,
                        help="Top-k truncation during rollout sampling; default 1 pairs well with temperature schedule.")

    # Rollout temperature schedule
    parser.add_argument("--rollout_temperature", type=float, default=None,
                        help="If set, temperature is fixed to this value (backwards compatible).")
    parser.add_argument("--rollout_temp_start", type=float, default=0.0)
    parser.add_argument("--rollout_temp_end", type=float, default=1.0)
    parser.add_argument("--temp_warmup_iters", type=int, default=8000)

    # Epsilon-greedy schedule
    parser.add_argument("--epsilon_start", type=float, default=0.0)
    parser.add_argument("--epsilon_end", type=float, default=0.0)
    parser.add_argument("--epsilon_warmup_iters", type=int, default=0)

    # Invalid transition handling
    parser.add_argument("--allow_invalid_continue", action="store_true",
                        help="If set, allow continuing after an invalid edge; otherwise terminate immediately.")
    parser.add_argument("--max_invalid_transitions", type=int, default=2,
                        help="Max consecutive invalid edges before forced termination (requires --allow_invalid_continue).")

    # Rewards
    parser.add_argument("--reward_type", choices=["process", "outcome"], default="process")
    parser.add_argument("--reward_hit_target", type=float, default=1.5)
    parser.add_argument("--reward_valid_transition", type=float, default=0.1,
                        help="Positive reward for each valid edge transition.")
    parser.add_argument("--reward_stage_bridge", type=float, default=0.2,
                        help="For S1->S3 tasks: reward for first reaching a Stage-2 node.")
    parser.add_argument("--reward_stage_bridge_only_once", action="store_true",
                        help="If set, reward stage-2 bridge only once (default true).")
    parser.add_argument("--reward_invalid_transition", type=float, default=0.25)
    parser.add_argument("--reward_invalid_token", type=float, default=1.0)
    parser.add_argument("--reward_stop", type=float, default=-0.1,
                        help="Penalty for generating stop token early.")
    parser.add_argument("--penalty_stage2_detour", type=float, default=0.2,
                        help="For S1->S2 tasks: penalty for entering non-target Stage-2 nodes.")
    parser.add_argument("--penalty_stage3_detour", type=float, default=0.2,
                        help="For S2->S3 tasks: penalty for entering non-target Stage-3 nodes.")
    parser.add_argument("--penalty_repeat_node", type=float, default=0.1,
                        help="Penalty for revisiting non-target nodes.")
    parser.add_argument("--step_penalty", type=float, default=0.0,
                        help="Constant per-step penalty (encourages shorter paths).")

    # Target network
    parser.add_argument("--target_ema", type=float, default=0,
                        help="EMA coefficient for target network (0 disables EMA).")
    parser.add_argument("--target_sync_interval", type=int, default=0,
                        help="If EMA=0, hard-sync target network every N iters (0 disables).")

    # KL regularization
    parser.add_argument("--kl_coef", type=float, default=0.05,
                        help="KL coefficient (0 disables KL).")
    parser.add_argument("--kl_warmup_iters", type=int, default=0,
                        help="Number of iterations to keep KL constant before annealing.")
    parser.add_argument("--kl_anneal_iters", type=int, default=12000,
                        help="Linearly anneal KL to 0 over this many iterations (0 disables annealing).")

    # Evaluation & logging
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--max_eval_pairs", type=int, default=5000)
    parser.add_argument("--eval_temperature", type=float, default=0.001)
    parser.add_argument("--eval_top_k", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="out_qlearning")

    args = parser.parse_args()

    # Backward compatibility: if rollout_temperature is explicitly set, override the schedule.
    if args.rollout_temperature is not None:
        args.rollout_temp_start = args.rollout_temperature
        args.rollout_temp_end = args.rollout_temperature
        args.temp_warmup_iters = 0

    return args


# -----------------------------------------------------------------------------
# Utilities & schedules
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def current_temperature(iteration: int, args: argparse.Namespace) -> float:
    if args.temp_warmup_iters <= 0 or args.rollout_temp_start == args.rollout_temp_end:
        return args.rollout_temp_end
    ratio = min(1.0, iteration / max(1, args.temp_warmup_iters))
    return args.rollout_temp_start + ratio * (args.rollout_temp_end - args.rollout_temp_start)


def current_epsilon(iteration: int, args: argparse.Namespace) -> float:
    if args.epsilon_warmup_iters <= 0 or args.epsilon_start == args.epsilon_end:
        return args.epsilon_end
    ratio = min(1.0, iteration / max(1, args.epsilon_warmup_iters))
    return args.epsilon_start + ratio * (args.epsilon_end - args.epsilon_start)


def current_kl_coef(iteration: int, args: argparse.Namespace) -> float:
    if args.kl_coef <= 0.0:
        return 0.0
    if iteration <= args.kl_warmup_iters:
        return args.kl_coef
    if args.kl_anneal_iters <= 0:
        return args.kl_coef
    progress = min(1.0, (iteration - args.kl_warmup_iters) / max(1, args.kl_anneal_iters))
    return max(0.0, args.kl_coef * (1.0 - progress))


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


def assemble_full_path(source: int,
                       generated_nodes: Sequence[int]) -> List[int]:
    """Prepend `source` to the model-generated node sequence to form the full path."""
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
    out_dir = Path(base_dir) / f"ql_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def safe_max_new_tokens(block_size: int,
                        prompt_len: int,
                        desired: int) -> int:
    """Compute the maximum number of new tokens allowed by block_size (min 1)."""
    available = block_size - prompt_len - 1
    if available <= 0:
        raise ValueError(
            f"Block size {block_size} is too small for prompt length {prompt_len}. "
            "Please increase block_size or shorten the prompt."
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
    state_dict = dict(raw_state_dict)
    model_block_size = model.config.block_size
    if ckpt_block_size is None:
        ckpt_block_size = model_block_size

    if model_block_size == ckpt_block_size:
        model.load_state_dict(state_dict, strict=True)
        return

    if model_block_size < ckpt_block_size:
        raise ValueError(
            f"Checkpoint block_size={ckpt_block_size} is larger than current model block_size={model_block_size}."
        )

    logger.warning(
        "Expanding block_size: checkpoint=%d -> current=%d. New positional rows remain randomly initialized.",
        ckpt_block_size, model_block_size,
    )

    wpe_key = "transformer.wpe.weight"
    if wpe_key in state_dict:
        old_weight = state_dict[wpe_key]
        if old_weight.shape[0] != ckpt_block_size:
            logger.warning("Checkpoint wpe.weight rows (%d) do not match recorded block_size (%d).",
                           old_weight.shape[0], ckpt_block_size)
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

    allowed_missing = {key for key in missing_keys
                       if key.endswith("attn.bias") or key.endswith("attn.mask")}
    leftover_missing = [k for k in missing_keys if k not in allowed_missing]
    if leftover_missing:
        logger.warning("Missing keys when loading checkpoint: %s", leftover_missing)
    if unexpected_keys:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected_keys)


# -----------------------------------------------------------------------------
# Evaluation
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
            full_path = assemble_full_path(source, generated_nodes)

            if is_valid_path(full_path, source, target, stages, graph):
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
# Q-learning core
# -----------------------------------------------------------------------------
def forward_logits_and_actions(model: GPT,
                               traj_ids: List[int],
                               device: torch.device) -> Tuple[Tensor, Tensor]:
    """
    Return the logits sequence (pre-softmax) and the corresponding y_ids (teacher-forced targets).
    """
    if len(traj_ids) < 2:
        raise ValueError("Trajectory is too short to compute logits.")
    x_ids = torch.tensor(traj_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y_ids = torch.tensor(traj_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(x_ids, y_ids)
    return logits.squeeze(0), y_ids.squeeze(0)


def compute_q_learning_loss(model: GPT,
                            target_model: Optional[GPT],
                            traj_ids: List[int],
                            actions: List[int],
                            rewards: List[float],
                            dones: List[bool],
                            action_start: int,
                            device: torch.device,
                            gamma: float,
                            return_logits: bool = False) -> Optional[Dict[str, Tensor]]:
    """
    Compute the per-trajectory Q-learning loss.

    Returns a dict containing:
      - loss
      - td_error
      - (optional) policy_logits, actions_tensor, start_idx
    """
    if len(actions) == 0:
        return None

    logits, y_ids = forward_logits_and_actions(model, traj_ids, device)
    start_idx = action_start - 1
    num_steps = len(actions)

    if start_idx + num_steps > logits.size(0):
        raise ValueError("Action sequence exceeds logits length; check trajectory construction.")

    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones_tensor = torch.tensor(dones, dtype=torch.bool, device=device)

    expected_actions = y_ids[start_idx:start_idx + num_steps]
    if expected_actions.shape != actions_tensor.shape or not torch.equal(expected_actions, actions_tensor):
        raise ValueError("Action sequence does not match teacher-forced targets; possible truncation/alignment issue.")

    policy_segment = logits[start_idx:start_idx + num_steps, :]
    q_selected = policy_segment.gather(-1, actions_tensor.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        if target_model is None:
            target_logits = logits.detach()
        else:
            target_logits, _ = forward_logits_and_actions(target_model, traj_ids, device)
            target_logits = target_logits.detach()

        target_segment = target_logits[start_idx:start_idx + num_steps, :]
        next_max = torch.zeros_like(q_selected)

        if num_steps > 1:
            next_max[:-1] = target_segment[1:].max(dim=-1).values
        next_max[-1] = 0.0
        next_max = next_max * (~dones_tensor)

        targets = rewards_tensor + gamma * next_max

    loss = F.mse_loss(q_selected, targets, reduction="mean")
    td_error = (q_selected - targets).detach()

    result: Dict[str, Tensor] = {
        "loss": loss,
        "td_error": td_error,
    }
    if return_logits:
        result["policy_logits"] = policy_segment
        result["actions_tensor"] = actions_tensor
        result["start_idx"] = torch.tensor(start_idx, device=device)
    return result


def select_next_token(logits: Tensor,
                      temperature: float,
                      top_k: int,
                      epsilon: float) -> int:
    """
    Sample the next token id using temperature, top-k truncation, and epsilon-greedy exploration.
    """
    logits = logits.detach().clone()
    vocab_size = logits.size(-1)
    topk_indices = None

    if top_k > 0 and top_k < vocab_size:
        top_vals, topk_indices = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(0, topk_indices, top_vals)
        logits = mask

    if epsilon > 0.0 and random.random() < epsilon:
        if topk_indices is not None:
            idx = topk_indices[torch.randint(0, topk_indices.size(0), (1,), device=logits.device)]
        else:
            idx = torch.randint(0, vocab_size, (1,), device=logits.device)
        return int(idx.item())

    if temperature <= 1e-6:
        return int(torch.argmax(logits).item())

    scaled_logits = logits / max(temperature, 1e-6)
    probs = F.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


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
        traj = traj[:block_size - 1]
    if not sampled_ids or sampled_ids[-1] != stop_token_id:
        if len(traj) < block_size:
            traj.append(stop_token_id)
    return traj


def sample_trajectory(model: GPT,
                      source: int,
                      target: int,
                      prompt_ids: List[int],
                      max_new_tokens: int,
                      args: argparse.Namespace,
                      stoi: Dict[str, int],
                      itos: Dict[int, str],
                      graph: nx.DiGraph,
                      stages: Sequence[Sequence[int]],
                      stage_sets: Dict[str, set[int]],
                      pair_bucket: Optional[str],
                      stop_token_id: int,
                      device: torch.device,
                      block_size: int,
                      temperature: float,
                      epsilon: float) -> Dict[str, object]:
    """
    Manually roll out a single trajectory and return training fields.
    """
    model_was_training = model.training
    if model_was_training:
        model.eval()

    sampled_ids: List[int] = []
    decoded_tokens: List[str] = []
    rewards: List[float] = []
    dones: List[bool] = []

    traj_ids = prompt_ids.copy()
    current_node = source
    hit_target = False
    invalid_transition = False
    invalid_token = False
    visited_stage2 = False
    stage_bridge_rewarded = False
    valid_transition_steps = 0
    invalid_transition_steps = 0
    visited_nodes = {source}

    allow_continue = args.allow_invalid_continue
    max_invalid = max(1, args.max_invalid_transitions) if allow_continue else 1

    for step in range(max_new_tokens):
        if len(traj_ids) >= block_size - 1:
            break

        context = torch.tensor(traj_ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(context)
        step_logits = logits[0, -1, :]

        next_token_id = select_next_token(
            logits=step_logits,
            temperature=temperature,
            top_k=args.rollout_top_k,
            epsilon=epsilon,
        )

        traj_ids.append(next_token_id)
        sampled_ids.append(next_token_id)

        token_str = itos.get(int(next_token_id), "[UNK]")
        decoded_tokens.append(token_str)

        reward = -args.step_penalty if args.step_penalty != 0.0 else 0.0
        done = False

        if next_token_id == stop_token_id:
            reward += args.reward_stop
            done = True
        elif token_str.isdigit():
            next_node = int(token_str)
            adjacency = graph.has_edge(str(current_node), str(next_node))

            if adjacency:
                valid_transition_steps += 1
                reward += args.reward_valid_transition

                if next_node in stage_sets.get("S2", set()):
                    visited_stage2 = True

                if pair_bucket == "S1->S3" and args.reward_stage_bridge > 0.0:
                    if (not stage_bridge_rewarded) and next_node in stage_sets.get("S2", set()):
                        reward += args.reward_stage_bridge
                        if args.reward_stage_bridge_only_once:
                            stage_bridge_rewarded = True

                if pair_bucket == "S1->S2" and args.penalty_stage2_detour > 0.0:
                    if next_node != target and next_node in stage_sets.get("S2", set()):
                        reward -= args.penalty_stage2_detour

                if pair_bucket == "S2->S3" and args.penalty_stage3_detour > 0.0:
                    if next_node != target and next_node in stage_sets.get("S3", set()):
                        reward -= args.penalty_stage3_detour

                if args.penalty_repeat_node > 0.0 and next_node in visited_nodes and next_node != target:
                    reward -= args.penalty_repeat_node
                visited_nodes.add(next_node)

                current_node = next_node
                if next_node == target:
                    reward += args.reward_hit_target
                    hit_target = True
                    done = True
            else:
                invalid_transition = True
                invalid_transition_steps += 1
                reward -= args.reward_invalid_transition
                if not allow_continue or invalid_transition_steps >= max_invalid:
                    done = True
        else:
            invalid_token = True
            reward -= args.reward_invalid_token
            done = True

        rewards.append(float(reward))
        dones.append(bool(done))

        if done:
            break

    if model_was_training:
        model.train()

    if not sampled_ids or sampled_ids[-1] != stop_token_id:
        sampled_ids.append(stop_token_id)
        extra_reward = (-args.step_penalty if args.step_penalty != 0.0 else 0.0) + args.reward_stop
        rewards.append(float(extra_reward))
        dones.append(True)

    traj_ids = build_traj_ids(prompt_ids, sampled_ids, stop_token_id, block_size)
    decoded_tokens = decode_tokens(sampled_ids, itos, stop_token_id)
    generated_nodes = tokens_to_nodes(decoded_tokens)
    full_path_nodes = assemble_full_path(source, generated_nodes)

    first_step_node = generated_nodes[0] if generated_nodes else None
    first_step_is_source = first_step_node == source
    first_step_is_valid = bool(first_step_node is not None and
                               graph.has_edge(str(source), str(first_step_node)))

    success = is_valid_path(full_path_nodes, source, target, stages, graph)

    if args.reward_type == "outcome":
        adjusted_rewards = [0.0 for _ in rewards]
        if success and hit_target:
            target_token = str(target)
            for idx, token_id in enumerate(sampled_ids):
                token_str = itos.get(token_id, "[UNK]")
                if token_str == target_token:
                    adjusted_rewards[idx] = args.reward_hit_target
                    break
        rewards = adjusted_rewards

    episode_reward = float(sum(rewards))
    step_reward_mean = float(np.mean(rewards)) if rewards else 0.0

    return {
        "traj_ids": traj_ids,
        "actions": sampled_ids,
        "rewards": rewards,
        "dones": dones,
        "decoded_tokens": decoded_tokens,
        "generated_nodes": generated_nodes,
        "path_nodes": full_path_nodes,
        "success": success,
        "first_step_is_source": first_step_is_source,
        "first_step_is_valid": first_step_is_valid,
        "episode_reward": episode_reward,
        "step_reward_mean": step_reward_mean,
        "hit_target": hit_target and success,
        "invalid_transition": invalid_transition,
        "invalid_token": invalid_token,
        "visited_stage2": visited_stage2 or stage_bridge_rewarded,
        "valid_transition_steps": valid_transition_steps,
        "invalid_transition_steps": invalid_transition_steps,
        "bucket": pair_bucket,
    }


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
    stage_sets = {
        "S1": set(stages[0]) if len(stages) > 0 else set(),
        "S2": set(stages[1]) if len(stages) > 1 else set(),
        "S3": set(stages[2]) if len(stages) > 2 else set(),
    }

    graph = nx.read_graphml(data_dir / "composition_graph.graphml")

    out_dir = prepare_output_dir(args.log_dir)
    logger = get_logger(os.path.join(out_dir, "train_qlearning.log"))
    logger.info("Q-learning training started.")
    logger.info("Output directory: %s", out_dir)
    logger.info("Reward type: %s", args.reward_type)

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

    resolved_block_size = dataset_block_size
    if args.block_size_override is not None:
        if args.block_size_override > dataset_block_size:
            raise ValueError(
                f"Block size expansion is not allowed (override={args.block_size_override}, dataset={dataset_block_size})."
            )
        resolved_block_size = args.block_size_override
    elif ckpt_model_args and "block_size" in ckpt_model_args:
        resolved_block_size = int(ckpt_model_args["block_size"])

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

    if args.target_ema > 0 or args.target_sync_interval > 0:
        target_model = GPT(GPTConfig(**model_args)).to(device)
        load_state_dict_with_block_resize(
            model=target_model,
            raw_state_dict=ckpt["model"],
            ckpt_block_size=ckpt_block_size,
            logger=logger,
        )
        for p in target_model.parameters():
            p.requires_grad = False
        target_model.eval()
        logger.info("Target network enabled: EMA=%.4f, sync_interval=%d",
                    args.target_ema, args.target_sync_interval)
    else:
        target_model = None

    if args.kl_coef > 0.0:
        sft_ref_model = GPT(GPTConfig(**model_args)).to(device)
        load_state_dict_with_block_resize(
            model=sft_ref_model,
            raw_state_dict=ckpt["model"],
            ckpt_block_size=ckpt_block_size,
            logger=logger,
        )
        sft_ref_model.eval()
        for p in sft_ref_model.parameters():
            p.requires_grad = False
        logger.info("KL enabled: coef=%.4f, warmup=%d, anneal=%d",
                    args.kl_coef, args.kl_warmup_iters, args.kl_anneal_iters)
    else:
        sft_ref_model = None
        logger.info("KL disabled.")

    optimizer = model.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type="cuda" if device.type == "cuda" else "cpu",
    )

    train_pairs = load_pairs(train_txt)
    logger.info("Loaded %d unique (source, target) pairs for Q-learning training.", len(train_pairs))

    eval_pairs: List[Pair] = []
    with open(test_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                eval_pairs.append((int(parts[0]), int(parts[1])))

    stop_token_id = stoi["\n"]
    prompt_len = 3
    action_start = prompt_len
    metrics_path = out_dir / "metrics_ql.jsonl"

    rollout_cap = safe_max_new_tokens(model.config.block_size, prompt_len, args.max_rollout_steps)
    if rollout_cap < args.max_rollout_steps:
        logger.warning("max_rollout_steps=%d exceeds block_size=%d; truncated to %d.",
                       args.max_rollout_steps, model.config.block_size, rollout_cap)

    model.train()
    bucket_names = ["S1->S2", "S2->S3", "S1->S3"]

    for iteration in range(1, args.max_iters + 1):
        batch_pairs = random.choices(train_pairs, k=args.batch_size)

        q_losses: List[Tensor] = []
        td_errors: List[float] = []
        kl_losses: List[float] = []
        episode_rewards: List[float] = []
        step_rewards: List[float] = []
        path_lengths: List[int] = []
        valid_transition_list: List[int] = []
        invalid_transition_steps_list: List[int] = []
        first_step_source_list: List[float] = []
        first_step_valid_list: List[float] = []

        bucket_success_sum = defaultdict(float)
        bucket_counts = defaultdict(int)

        success_count = 0
        hit_target_count = 0
        invalid_transition_count = 0
        invalid_token_count = 0
        stage2_visit_count = 0

        temperature = current_temperature(iteration, args)
        epsilon = current_epsilon(iteration, args)
        kl_coef_current = current_kl_coef(iteration, args)

        for source, target in batch_pairs:
            prompt_ids = build_prompt(source, target, stoi)
            bucket = bucket_for_pair(source, target, stages)

            traj_info = sample_trajectory(
                model=model,
                source=source,
                target=target,
                prompt_ids=prompt_ids,
                max_new_tokens=rollout_cap,
                args=args,
                stoi=stoi,
                itos=itos,
                graph=graph,
                stages=stages,
                stage_sets=stage_sets,
                pair_bucket=bucket,
                stop_token_id=stop_token_id,
                device=device,
                block_size=model.config.block_size,
                temperature=temperature,
                epsilon=epsilon,
            )

            need_policy_logits = kl_coef_current > 0.0
            loss_result = compute_q_learning_loss(
                model=model,
                target_model=target_model,
                traj_ids=traj_info["traj_ids"],
                actions=traj_info["actions"],
                rewards=traj_info["rewards"],
                dones=traj_info["dones"],
                action_start=action_start,
                device=device,
                gamma=args.gamma,
                return_logits=need_policy_logits,
            )

            if loss_result is None:
                continue

            loss_i = loss_result["loss"]
            kl_value = 0.0

            if need_policy_logits and sft_ref_model is not None:
                start_idx = int(loss_result["start_idx"].item())
                policy_logits = loss_result["policy_logits"]
                seq_len = policy_logits.size(0)

                with torch.no_grad():
                    ref_logits, _ = forward_logits_and_actions(
                        sft_ref_model, traj_info["traj_ids"], device)

                ref_segment = ref_logits[start_idx:start_idx + seq_len, :]

                kl_loss = F.kl_div(
                    F.log_softmax(policy_logits, dim=-1),
                    F.softmax(ref_segment, dim=-1),
                    reduction="batchmean",
                )
                loss_i = loss_i + kl_coef_current * kl_loss
                kl_value = float(kl_loss.detach().item())

            q_losses.append(loss_i)
            kl_losses.append(kl_value)
            td_errors.append(float(loss_result["td_error"].abs().mean().item()))
            episode_rewards.append(traj_info["episode_reward"])
            step_rewards.append(traj_info["step_reward_mean"])
            path_lengths.append(len(traj_info["path_nodes"]))
            valid_transition_list.append(traj_info["valid_transition_steps"])
            invalid_transition_steps_list.append(traj_info["invalid_transition_steps"])
            first_step_source_list.append(1.0 if traj_info["first_step_is_source"] else 0.0)
            first_step_valid_list.append(1.0 if traj_info["first_step_is_valid"] else 0.0)

            success = bool(traj_info["success"])
            success_count += int(success)
            hit_target_count += int(traj_info["hit_target"])
            invalid_transition_count += int(traj_info["invalid_transition"])
            invalid_token_count += int(traj_info["invalid_token"])
            stage2_visit_count += int(traj_info.get("visited_stage2", False))

            bucket_name = traj_info["bucket"]
            if bucket_name:
                bucket_success_sum[bucket_name] += 1.0 if success else 0.0
                bucket_counts[bucket_name] += 1

        if not q_losses:
            logger.warning("Iter %d has no valid samples; skipping update.", iteration)
            continue

        total_loss = torch.stack(q_losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if target_model is not None:
            if args.target_ema > 0:
                with torch.no_grad():
                    for param, target_param in zip(model.parameters(), target_model.parameters()):
                        target_param.data.mul_(args.target_ema)
                        target_param.data.add_((1.0 - args.target_ema) * param.data)
            elif args.target_sync_interval > 0 and iteration % args.target_sync_interval == 0:
                target_model.load_state_dict(model.state_dict(), strict=True)

        avg_episode_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        avg_step_reward = float(np.mean(step_rewards)) if step_rewards else 0.0
        avg_path_len = float(np.mean(path_lengths)) if path_lengths else 0.0
        avg_valid_transitions = float(np.mean(valid_transition_list)) if valid_transition_list else 0.0
        avg_invalid_transition_steps = float(np.mean(invalid_transition_steps_list)) if invalid_transition_steps_list else 0.0
        first_step_source_rate = float(np.mean(first_step_source_list)) if first_step_source_list else 0.0
        first_step_valid_rate = float(np.mean(first_step_valid_list)) if first_step_valid_list else 0.0
        success_rate = success_count / len(batch_pairs)
        hit_rate = hit_target_count / len(batch_pairs)
        invalid_transition_rate = invalid_transition_count / len(batch_pairs)
        invalid_token_rate = invalid_token_count / len(batch_pairs)
        stage2_visit_rate = stage2_visit_count / len(batch_pairs)
        mean_td_error = float(np.mean(td_errors)) if td_errors else 0.0
        mean_kl_loss = float(np.mean(kl_losses)) if kl_losses else 0.0

        if iteration % 50 == 0:
            logger.info(
                "Iter %6d | loss=%.4f | td_err=%.4f | kl=%.4f | temp=%.3f | eps=%.3f | "
                "success=%.3f | hit=%.3f | stage2=%.3f | first_src=%.3f | first_valid=%.3f | "
                "invalid_edge=%.3f | invalid_tok=%.3f | avg_ep_reward=%.3f | avg_path=%.2f | valid_steps=%.2f",
                iteration,
                float(total_loss.item()),
                mean_td_error,
                mean_kl_loss,
                temperature,
                epsilon,
                success_rate,
                hit_rate,
                stage2_visit_rate,
                first_step_source_rate,
                first_step_valid_rate,
                invalid_transition_rate,
                invalid_token_rate,
                avg_episode_reward,
                avg_path_len,
                avg_valid_transitions,
            )

        record = {
            "iter": iteration,
            "loss": float(total_loss.item()),
            "td_error": mean_td_error,
            "kl_loss": mean_kl_loss,
            "temperature": temperature,
            "epsilon": epsilon,
            "kl_coef_current": kl_coef_current,
            "success_rate": success_rate,
            "hit_rate": hit_rate,
            "stage2_visit_rate": stage2_visit_rate,
            "first_step_source_rate": first_step_source_rate,
            "first_step_valid_rate": first_step_valid_rate,
            "invalid_transition_rate": invalid_transition_rate,
            "invalid_token_rate": invalid_token_rate,
            "avg_episode_reward": avg_episode_reward,
            "avg_step_reward": avg_step_reward,
            "avg_path_len": avg_path_len,
            "avg_valid_transitions": avg_valid_transitions,
            "avg_invalid_transition_steps": avg_invalid_transition_steps,
        }
        for bucket_name in bucket_names:
            cnt = bucket_counts.get(bucket_name, 0)
            record[f"train_success/{bucket_name}"] = (
                bucket_success_sum.get(bucket_name, 0.0) / cnt if cnt > 0 else 0.0
            )

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
            for bucket_name, stats in eval_results.items():
                logger.info(
                    "  %s: %.2f%% (%d / %d)",
                    bucket_name,
                    stats["accuracy"] * 100.0,
                    stats["correct"],
                    stats["total"],
                )
            eval_record = {"iter": iteration, "eval": eval_results}
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(eval_record) + "\n")

        if iteration % args.save_interval == 0 or iteration == args.max_iters:
            ckpt_path = out_dir / f"ckpt_ql_{iteration}.pt"
            torch.save(
                {
                    "iter_num": iteration,
                    "model": model.state_dict(),
                    "model_args": model_args,
                    "config": vars(args),
                },
                ckpt_path,
            )
            logger.info("Saved Q-learning checkpoint to %s", ckpt_path)

    logger.info("Q-learning training finished.")


if __name__ == "__main__":
    main()