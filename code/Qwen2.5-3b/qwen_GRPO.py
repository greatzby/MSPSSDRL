#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO/PPO-style fine-tuning for GraphA K-stage composition datasets (HF/Qwen tokenizer).

This script performs group-based on-policy RL updates using a PPO-style clipped objective.
For each sampled prompt (src,tgt), it draws `group_size` rollouts, computes advantages within the group,
and updates the policy using the ratio between new and old action log-probabilities.

Tokenization & rollout semantics:
  - STRICT action space: tokens must decode to ASCII digits, a single space, or EOS
  - a streaming parser converts the token stream into node ids; node transitions are committed on SPACE/EOS
  - prompt text: "src tgt src " (space-terminated)
  - success is computed by validating the predicted node path on the directed graph plus intermediate-stage
    coverage constraints for multi-stage pairs.

Important implementation detail:
  - The policy is kept in `eval()` mode during rollouts and log-prob recomputation to disable dropout so that
    old/new log-probabilities are comparable; parameters remain trainable (gradients still flow).

Inputs:
  - --data_dir: train_{K}.txt, test.txt, meta.pkl (block_size), stage_info.pkl, composition_graph.graphml
  - --sft_dir: HF model directory (or LoRA adapter directory) used to initialize the policy (and KL reference if enabled)

Outputs (written to --log_dir/grpo_{timestamp}/):
  - metrics_grpo.jsonl
  - periodic HF checkpoints (policy_model + tokenizer), plus copied data meta

Notes:
  - `--epsilon_*` options exist but epsilon-greedy breaks strict on-policy PPO assumptions; recommended eps=0.
  - Evaluation reports both `accuracy` and `accuracy_raw` depending on the selected success path mode.

Example:
python Qwen2.5-3b/qwen_GRPO.py \
  --data_dir data/datasets/graphA_train_maxjump1 \
  --train_paths_per_pair 20 \
  --sft_dir out/<your_qwen_sft_run_dir>/ckpt_<iter> \
  --base_model Qwen/Qwen2.5-3B \
  --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Set

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel, get_peft_model, LoraConfig, TaskType
except Exception:
    PeftModel = None
    get_peft_model = None
    LoraConfig = None
    TaskType = None


Node = int
Pair = Tuple[int, int]
BucketName = str

INF_DIST = 10**9


# -----------------------------------------------------------------------------
# utils
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_output_dir(base_dir: str) -> Path:
    out_dir = Path(base_dir) / f"grpo_{now_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def current_temperature(iteration: int, args: argparse.Namespace) -> float:
    if args.temp_warmup_iters <= 0 or args.rollout_temp_start == args.rollout_temp_end:
        return float(args.rollout_temp_end)
    ratio = min(1.0, iteration / max(1, args.temp_warmup_iters))
    return float(args.rollout_temp_start + ratio * (args.rollout_temp_end - args.rollout_temp_start))


def current_epsilon(iteration: int, args: argparse.Namespace) -> float:
    if args.epsilon_warmup_iters <= 0 or args.epsilon_start == args.epsilon_end:
        return float(args.epsilon_end)
    ratio = min(1.0, iteration / max(1, args.epsilon_warmup_iters))
    return float(args.epsilon_start + ratio * (args.epsilon_end - args.epsilon_start))


def current_kl_coef(iteration: int, args: argparse.Namespace) -> float:
    if args.kl_coef <= 0.0:
        return 0.0
    if iteration <= args.kl_warmup_iters:
        return float(args.kl_coef)
    if args.kl_anneal_iters <= 0:
        return float(max(args.kl_min_coef, args.kl_coef))
    progress = min(1.0, (iteration - args.kl_warmup_iters) / max(1, args.kl_anneal_iters))
    decayed = float(args.kl_coef * (1.0 - progress))
    return float(max(args.kl_min_coef, decayed))


def load_pairs_unique(train_file: Path) -> List[Pair]:
    seen = set()
    pairs: List[Pair] = []
    for line in train_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        s, t = int(parts[0]), int(parts[1])
        key = (s, t)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def sample_prompts(train_pairs: List[Pair], batch_size: int, with_replacement: bool) -> List[Pair]:
    if batch_size <= 0:
        return []
    if with_replacement:
        return random.choices(train_pairs, k=batch_size)
    if batch_size <= len(train_pairs):
        return random.sample(train_pairs, k=batch_size)
    return random.choices(train_pairs, k=batch_size)


def build_node_to_stage(stages: Sequence[Sequence[int]]) -> Dict[int, int]:
    node_to_stage: Dict[int, int] = {}
    for si, nodes in enumerate(stages, start=1):
        for n in nodes:
            n = int(n)
            if n in node_to_stage:
                raise ValueError(f"Node {n} appears in multiple stages.")
            node_to_stage[n] = si
    return node_to_stage


def bucket_for_pair_k(source: int, target: int, node_to_stage: Dict[int, int], K: int) -> Optional[BucketName]:
    si = node_to_stage.get(int(source))
    sj = node_to_stage.get(int(target))
    if si is None or sj is None:
        return None
    if not (1 <= si <= K and 1 <= sj <= K and si < sj):
        return None
    return f"S{si}->S{sj}"


def required_intermediate_stages(si: int, sj: int) -> List[int]:
    if sj <= si + 1:
        return []
    return list(range(si + 1, sj))


def safe_max_new_tokens(block_size: int, prompt_len: int, desired: int) -> int:
    available = block_size - prompt_len - 1
    if available <= 0:
        raise ValueError(f"block_size={block_size} too small for prompt_len={prompt_len}")
    return max(1, min(int(desired), int(available)))


def top_k_filtering_1d(logits_1d: Tensor, top_k: int) -> Tensor:
    if top_k <= 0 or top_k >= logits_1d.size(-1):
        return logits_1d
    vals, idx = torch.topk(logits_1d, top_k)
    out = torch.full_like(logits_1d, float("-inf"))
    out.scatter_(0, idx, vals)
    return out


# -----------------------------------------------------------------------------
# Graph helpers
# -----------------------------------------------------------------------------
def build_int_adjacency(G: nx.DiGraph) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = defaultdict(set)
    for u, v in G.edges():
        try:
            ui = int(str(u))
            vi = int(str(v))
        except Exception:
            continue
        adj[ui].add(vi)
    return dict(adj)


def build_successor_strings(adj: Dict[int, Set[int]]) -> Dict[int, Set[str]]:
    out: Dict[int, Set[str]] = {}
    for u, succs in adj.items():
        out[u] = set(str(v) for v in succs)
    return out


def build_reverse_adjacency(adj: Dict[int, Set[int]], node_min: int, node_max: int) -> Dict[int, List[int]]:
    rev: Dict[int, List[int]] = defaultdict(list)
    for u, succs in adj.items():
        if u < node_min or u > node_max:
            continue
        for v in succs:
            if v < node_min or v > node_max:
                continue
            rev[v].append(u)
    return dict(rev)


def bfs_dist_to_target(rev_adj: Dict[int, List[int]], target: int, node_min: int, node_max: int) -> Dict[int, int]:
    dist = {n: INF_DIST for n in range(node_min, node_max + 1)}
    if target < node_min or target > node_max:
        return dist
    dist[target] = 0
    q = deque([target])
    while q:
        x = q.popleft()
        dx = dist[x]
        for p in rev_adj.get(x, []):
            if dist[p] == INF_DIST:
                dist[p] = dx + 1
                q.append(p)
    return dist


def precompute_all_target_dists(adj: Dict[int, Set[int]], node_min: int, node_max: int) -> Dict[int, Dict[int, int]]:
    rev = build_reverse_adjacency(adj, node_min=node_min, node_max=node_max)
    all_d: Dict[int, Dict[int, int]] = {}
    for t in range(node_min, node_max + 1):
        all_d[t] = bfs_dist_to_target(rev, target=t, node_min=node_min, node_max=node_max)
    return all_d


def is_valid_path_k(
    path_nodes: List[int],
    source: int,
    target: int,
    node_to_stage: Dict[int, int],
    adj: Dict[int, Set[int]],
) -> bool:
    if len(path_nodes) < 2:
        return False
    if path_nodes[0] != source or path_nodes[-1] != target:
        return False

    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if v not in adj.get(int(u), set()):
            return False

    si = node_to_stage.get(int(source))
    sj = node_to_stage.get(int(target))
    if si is None or sj is None or not (si < sj):
        return False

    req = required_intermediate_stages(int(si), int(sj))
    if not req:
        return True

    present = set()
    for n in path_nodes[1:-1]:
        st = node_to_stage.get(int(n))
        if st is not None:
            present.add(int(st))
    return all(r in present for r in req)


# -----------------------------------------------------------------------------
# STRICT action mask
# -----------------------------------------------------------------------------
def _is_ascii_digit(ch: str) -> bool:
    return "0" <= ch <= "9"


def _is_allowed_piece_ascii_digit_or_single_space(piece: str) -> bool:
    if piece is None:
        return False
    if piece == " ":
        return True
    return (len(piece) == 1) and _is_ascii_digit(piece)


def _is_allowed_piece_ascii_digits_space_loose(piece: str) -> bool:
    if piece is None or piece == "":
        return False
    for ch in piece:
        if _is_ascii_digit(ch) or ch == " ":
            continue
        return False
    return True


def build_allowed_token_mask(tokenizer, device: torch.device, mode: str, allow_eos: bool = True) -> Tensor:
    V = len(tokenizer)
    allowed = torch.zeros(V, dtype=torch.bool)

    for tid in range(V):
        s = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if mode == "ascii_digits_space_single":
            if _is_allowed_piece_ascii_digit_or_single_space(s):
                allowed[tid] = True
        elif mode == "ascii_digits_space":
            if _is_allowed_piece_ascii_digits_space_loose(s):
                allowed[tid] = True
        else:
            raise ValueError(f"Unknown action mask mode: {mode}")

    if allow_eos and tokenizer.eos_token_id is not None:
        allowed[int(tokenizer.eos_token_id)] = True

    return allowed.to(device)


@dataclass
class ActionSpace:
    eos_id: int
    eos_ids: Tensor
    space_ids: Tensor
    digit_ids: Dict[str, Tensor]
    device: torch.device

    @property
    def all_static_ids(self) -> Tensor:
        parts = [self.eos_ids, self.space_ids] + [self.digit_ids[str(d)] for d in range(10)]
        return torch.cat(parts, dim=0)


def build_action_space_single_char(tokenizer, device: torch.device, allowed_token_mask: Tensor) -> ActionSpace:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id.")
    allowed_ids = torch.nonzero(allowed_token_mask, as_tuple=False).squeeze(-1).tolist()

    digits: Dict[str, List[int]] = {str(d): [] for d in range(10)}
    spaces: List[int] = []
    has_eos = False

    for tid in allowed_ids:
        s = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if s == " ":
            spaces.append(int(tid))
        elif len(s) == 1 and _is_ascii_digit(s):
            digits[s].append(int(tid))
        elif int(tid) == int(eos_id):
            has_eos = True

    if not has_eos:
        raise ValueError("EOS is not included in allowed_token_mask; unsafe.")
    if len(spaces) == 0:
        raise ValueError("No single-space token id found under current tokenizer/mask.")
    for d in range(10):
        if len(digits[str(d)]) == 0:
            raise ValueError(f"No token id found for digit '{d}' under current tokenizer/mask.")

    space_ids = torch.tensor(spaces, dtype=torch.long, device=device)
    digit_ids = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in digits.items()}
    return ActionSpace(
        eos_id=int(eos_id),
        eos_ids=torch.tensor([int(eos_id)], dtype=torch.long, device=device),
        space_ids=space_ids,
        digit_ids=digit_ids,
        device=device,
    )


# -----------------------------------------------------------------------------
# streaming node parser
# -----------------------------------------------------------------------------
@dataclass
class ParseResult:
    completed_nodes: List[int]
    invalid_char: bool


class NodeStreamParser:
    def __init__(self, node_max: int, node_min: int = 0) -> None:
        self.pending_digits: List[str] = []
        self.node_max = int(node_max)
        self.node_min = int(node_min)
        self.max_digits = max(1, len(str(max(0, self.node_max))))

    def _flush_pending(self) -> Optional[int]:
        if not self.pending_digits:
            return None
        s = "".join(self.pending_digits)
        self.pending_digits.clear()
        try:
            return int(s)
        except Exception:
            return None

    def consume_text(self, piece: str) -> ParseResult:
        completed: List[int] = []
        invalid = False

        for ch in piece:
            if _is_ascii_digit(ch):
                self.pending_digits.append(ch)

                if len(self.pending_digits) > self.max_digits:
                    invalid = True
                    break

                if len(self.pending_digits) == self.max_digits:
                    try:
                        v = int("".join(self.pending_digits))
                        if v > self.node_max:
                            invalid = True
                            break
                    except Exception:
                        invalid = True
                        break

            elif ch == " ":
                node = self._flush_pending()
                if node is not None:
                    completed.append(node)
            else:
                invalid = True
                break

        return ParseResult(completed_nodes=completed, invalid_char=invalid)

    def finalize(self) -> List[int]:
        node = self._flush_pending()
        return [node] if node is not None else []

    def pending_as_str(self) -> str:
        return "".join(self.pending_digits)


# -----------------------------------------------------------------------------
# node reward credit assignment
# -----------------------------------------------------------------------------
def apply_node_reward_credit_to_digits(
    rewards: List[float],
    digit_step_indices: List[int],
    node_reward: float,
    mode: str,
) -> float:
    if mode == "none":
        return float(node_reward)
    if mode == "uniform_digits":
        if not digit_step_indices:
            return float(node_reward)
        share = float(node_reward) / float(len(digit_step_indices))
        for idx in digit_step_indices:
            rewards[idx] += share
        return 0.0
    raise ValueError(f"Unknown node_reward_credit mode: {mode}")


# -----------------------------------------------------------------------------
# model forward helpers
# -----------------------------------------------------------------------------
def forward_logits(model, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.logits


# -----------------------------------------------------------------------------
# candidate builder (v2_fix keeps guards disabled)
# -----------------------------------------------------------------------------
def dynamic_candidate_token_ids(
    pending: str,
    current_node: int,
    action_space: Optional[ActionSpace],
    args: argparse.Namespace,
    succ_strs_by_node: Optional[Dict[int, Set[str]]] = None,
) -> Optional[Tensor]:
    if action_space is None:
        return None

    if (not args.dynamic_digit_guard) and (not args.dynamic_successor_prefix_guard):
        if not bool(args.mask_space_when_pending_empty):
            return action_space.all_static_ids
        parts: List[Tensor] = [action_space.eos_ids]
        for d in "0123456789":
            parts.append(action_space.digit_ids[d])
        if pending != "":
            parts.append(action_space.space_ids)
        return torch.cat(parts, dim=0)

    raise ValueError(
        "This v2_fix keeps guards disabled. "
        "Please run with --no-dynamic_digit_guard --no-dynamic_successor_prefix_guard."
    )


def select_next_token_from_candidates(
    logits_1d: Tensor,
    candidate_ids: Tensor,
    temperature: float,
    top_k: int,
    epsilon: float,
    epsilon_explore_top_k: int,
) -> int:
    cand_logits = logits_1d.detach().index_select(0, candidate_ids)

    if torch.isneginf(cand_logits).all():
        ridx = torch.randint(0, candidate_ids.numel(), (1,), device=candidate_ids.device)
        return int(candidate_ids[ridx].item())

    if epsilon > 0.0 and random.random() < epsilon:
        K = int(epsilon_explore_top_k)
        K = max(1, min(K, candidate_ids.numel()))
        _vals, idx = torch.topk(cand_logits, K)
        ridx = idx[torch.randint(0, idx.numel(), (1,), device=idx.device)]
        return int(candidate_ids[ridx].item())

    if top_k > 0:
        K = min(int(top_k), candidate_ids.numel())
        cand_logits = top_k_filtering_1d(cand_logits, K)

    if temperature <= 1e-6:
        ridx = torch.argmax(cand_logits)
        return int(candidate_ids[ridx].item())

    probs = F.softmax(cand_logits / max(float(temperature), 1e-6), dim=-1)
    ridx = torch.multinomial(probs, num_samples=1)
    return int(candidate_ids[ridx].item())


def masked_logprob_and_entropy_from_logits(
    logits_1d: Tensor,
    candidate_ids: Tensor,
    action_id: int,
    temperature: float = 1.0,
    top_k: int = 0,
) -> Tuple[Tensor, Tensor]:
    cl = logits_1d.index_select(0, candidate_ids).float()
    if top_k > 0:
        K = min(int(top_k), int(cl.numel()))
        cl = top_k_filtering_1d(cl, K)
    if temperature != 1.0:
        cl = cl / max(float(temperature), 1e-6)

    logp = F.log_softmax(cl, dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(dim=-1)

    pos = (candidate_ids == int(action_id)).nonzero(as_tuple=False)
    if pos.numel() == 0:
        return torch.tensor(float("-inf"), device=logits_1d.device), entropy
    lp_a = logp[pos[0, 0]]
    return lp_a, entropy


# -----------------------------------------------------------------------------
# Token-level trace
# -----------------------------------------------------------------------------
def compute_pending_and_env_states_for_actions(
    tokenizer,
    actions: List[int],
    source: int,
    eos_id: int,
    adj: Dict[int, Set[int]],
    allow_invalid_continue: bool,
    max_invalid_transitions: int,
) -> Tuple[List[str], List[int]]:
    pending = ""
    cur = int(source)
    invalid_cnt = 0

    pending_states: List[str] = []
    env_states: List[int] = []

    def _commit_node(node_str: str) -> None:
        nonlocal cur, invalid_cnt
        if node_str == "":
            return
        try:
            nxt = int(node_str)
        except Exception:
            invalid_cnt += 1
            return
        if nxt in adj.get(cur, set()):
            cur = nxt
        else:
            invalid_cnt += 1

    for aid in actions:
        pending_states.append(pending)
        env_states.append(cur)

        if int(aid) == int(eos_id):
            _commit_node(pending)
            pending = ""
            break

        piece = tokenizer.decode([aid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for ch in piece:
            if "0" <= ch <= "9":
                pending += ch
            elif ch == " ":
                _commit_node(pending)
                pending = ""
                if (not allow_invalid_continue) and invalid_cnt > 0:
                    break
                if allow_invalid_continue and invalid_cnt >= max(1, int(max_invalid_transitions)):
                    break
            else:
                invalid_cnt += 1
                break

        if (not allow_invalid_continue) and invalid_cnt > 0:
            break
        if allow_invalid_continue and invalid_cnt >= max(1, int(max_invalid_transitions)):
            break

    pending_states.append(pending)
    env_states.append(cur)
    return pending_states, env_states


# -----------------------------------------------------------------------------
# rollout
# -----------------------------------------------------------------------------
def build_prompt_text(source: int, target: int) -> str:
    return f"{source} {target} {source}"


@torch.no_grad()
def sample_trajectory_hf(
    model,
    tokenizer,
    source: int,
    target: int,
    adj: Dict[int, Set[int]],
    succ_strs_by_node: Dict[int, Set[str]],
    node_to_stage: Dict[int, int],
    K: int,
    args: argparse.Namespace,
    device: torch.device,
    block_size: int,
    temperature: float,
    epsilon: float,
    action_space: Optional[ActionSpace],
    dist_cache: Optional[Dict[int, Dict[int, int]]] = None,
) -> Dict[str, object]:
    model_was_training = model.training
    model.eval()  # rollout must be deterministic wrt dropout

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id; cannot run rollout safely.")

    si = node_to_stage.get(int(source))
    sj = node_to_stage.get(int(target))
    target_stage = node_to_stage.get(int(target))
    pair_bucket = bucket_for_pair_k(source, target, node_to_stage=node_to_stage, K=K)

    required_stages = (
        required_intermediate_stages(int(si), int(sj))
        if (si is not None and sj is not None and si < sj)
        else []
    )
    visited_required_stages: set[int] = set()

    prompt_text = build_prompt_text(source, target) + " "
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    prompt_len = len(prompt_ids)

    max_new = safe_max_new_tokens(block_size, prompt_len, args.max_rollout_steps)

    traj_ids: List[int] = list(prompt_ids)
    sampled_ids: List[int] = []
    rewards: List[float] = []
    dones: List[bool] = []
    old_logprobs: List[float] = []
    entropies: List[float] = []

    parser = NodeStreamParser(node_max=args.node_max, node_min=args.node_min)
    generated_nodes: List[int] = []
    applied_path_nodes: List[int] = [source]
    action_token_texts: List[str] = []

    pending_digit_step_indices: List[int] = []

    current_node = int(source)
    visited_nodes = {int(source)}

    hit_target = False
    invalid_transition = False
    invalid_token = False
    valid_transition_steps = 0
    invalid_transition_steps = 0

    allow_continue = bool(args.allow_invalid_continue)
    max_invalid = max(1, int(args.max_invalid_transitions)) if allow_continue else 1

    dist_to_target = None
    if args.reward_distance_shaping_alpha != 0.0:
        if dist_cache is None:
            raise ValueError("dist_cache must be provided when reward_distance_shaping_alpha != 0")
        dist_to_target = dist_cache.get(int(target), None)

    def _covered_all_required() -> bool:
        if not required_stages:
            return True
        return all(s in visited_required_stages for s in required_stages)

    def apply_node_transition(next_node: int) -> Tuple[float, bool]:
        nonlocal current_node, hit_target, invalid_transition
        nonlocal valid_transition_steps, invalid_transition_steps
        nonlocal invalid_token

        reward_delta = -float(args.step_penalty) if args.step_penalty != 0.0 else 0.0
        done = False

        if next_node < args.node_min or next_node > args.node_max:
            invalid_token = True
            reward_delta -= float(args.reward_invalid_token)
            return reward_delta, True

        next_stage = node_to_stage.get(int(next_node))
        if next_stage is None:
            invalid_token = True
            reward_delta -= float(args.reward_invalid_token)
            return reward_delta, True

        cur_stage = node_to_stage.get(int(current_node))

        if int(next_node) in adj.get(int(current_node), set()):
            valid_transition_steps += 1
            reward_delta += float(args.reward_valid_transition)

            if args.reward_stage_bridge > 0.0 and cur_stage is not None:
                if int(next_stage) == int(cur_stage) + 1:
                    reward_delta += float(args.reward_stage_bridge)

            if required_stages and int(next_stage) in required_stages:
                newly = int(next_stage) not in visited_required_stages
                visited_required_stages.add(int(next_stage))
                if args.reward_required_stage_cover > 0.0:
                    if (not args.reward_required_stage_cover_only_once) or newly:
                        reward_delta += float(args.reward_required_stage_cover)

            if (
                args.penalty_target_stage_detour > 0.0
                and target_stage is not None
                and int(next_stage) > int(target_stage)
            ):
                reward_delta -= float(args.penalty_target_stage_detour)
                if args.terminate_on_overshoot:
                    done = True

            if dist_to_target is not None and args.reward_distance_shaping_alpha != 0.0:
                d_cur = dist_to_target.get(int(current_node), INF_DIST)
                d_nxt = dist_to_target.get(int(next_node), INF_DIST)
                if d_cur < INF_DIST and d_nxt < INF_DIST:
                    shaping = float(args.reward_distance_shaping_alpha) * float(d_cur - d_nxt)
                    if args.distance_shaping_cap > 0:
                        shaping = float(np.clip(shaping, -args.distance_shaping_cap, args.distance_shaping_cap))
                    reward_delta += shaping

            if args.penalty_repeat_node > 0.0 and next_node in visited_nodes and next_node != target:
                reward_delta -= float(args.penalty_repeat_node)

            visited_nodes.add(int(next_node))
            current_node = int(next_node)
            applied_path_nodes.append(int(next_node))

            if int(next_node) == int(target):
                if args.reward_hit_target_requires_coverage and (not _covered_all_required()):
                    reward_delta += float(args.reward_hit_target_uncovered)
                    hit_target = False
                    done = True
                else:
                    reward_delta += float(args.reward_hit_target)
                    hit_target = True
                    done = True
        else:
            invalid_transition = True
            invalid_transition_steps += 1
            reward_delta -= float(args.reward_invalid_transition)
            if (not allow_continue) or (invalid_transition_steps >= max_invalid):
                done = True

        return reward_delta, done

    # prime cache
    with torch.inference_mode():
        input_ids0 = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        out0 = model(input_ids=input_ids0, use_cache=True)
        past = out0.past_key_values
        last_logits = out0.logits[0, -1, :]

    for _step in range(max_new):
        if len(traj_ids) >= block_size - 1:
            break

        cand_ids = dynamic_candidate_token_ids(
            pending=parser.pending_as_str(),
            current_node=current_node,
            action_space=action_space,
            args=args,
            succ_strs_by_node=succ_strs_by_node,
        )
        if cand_ids is None:
            raise RuntimeError("ActionSpace is required for ascii_digits_space_single.")

        next_id = select_next_token_from_candidates(
            logits_1d=last_logits,
            candidate_ids=cand_ids,
            temperature=temperature,
            top_k=args.rollout_top_k,
            epsilon=epsilon,
            epsilon_explore_top_k=args.epsilon_explore_top_k,
        )

        lp, ent = masked_logprob_and_entropy_from_logits(
            logits_1d=last_logits,
            candidate_ids=cand_ids,
            action_id=next_id,
            temperature=temperature if args.logprob_matches_rollout_temperature else 1.0,
            top_k=args.rollout_top_k if args.logprob_matches_rollout_top_k else 0,
        )
        old_logprobs.append(float(lp.item()))
        entropies.append(float(ent.item()))

        traj_ids.append(int(next_id))
        sampled_ids.append(int(next_id))

        piece = tokenizer.decode([int(next_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        action_token_texts.append(piece)

        if len(piece) == 1 and _is_ascii_digit(piece):
            pending_digit_step_indices.append(len(sampled_ids) - 1)

        reward = 0.0
        done = False

        if int(next_id) == int(eos_id):
            for node in parser.finalize():
                generated_nodes.append(int(node))
                r_delta, d2 = apply_node_transition(int(node))
                reward += apply_node_reward_credit_to_digits(
                    rewards=rewards,
                    digit_step_indices=pending_digit_step_indices,
                    node_reward=float(r_delta),
                    mode=str(args.node_reward_credit),
                )
                pending_digit_step_indices.clear()
                if d2:
                    done = True
                    break
            reward += float(args.reward_stop)
            done = True
        else:
            pr = parser.consume_text(piece)
            if pr.invalid_char:
                invalid_token = True
                reward -= float(args.reward_invalid_token)
                done = True
            else:
                for node in pr.completed_nodes:
                    generated_nodes.append(int(node))
                    r_delta, d2 = apply_node_transition(int(node))
                    reward += apply_node_reward_credit_to_digits(
                        rewards=rewards,
                        digit_step_indices=pending_digit_step_indices,
                        node_reward=float(r_delta),
                        mode=str(args.node_reward_credit),
                    )
                    pending_digit_step_indices.clear()
                    if d2:
                        done = True
                        break

        rewards.append(float(reward))
        dones.append(bool(done))
        if done:
            break

        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([[int(next_id)]], dtype=torch.long, device=device),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            last_logits = out.logits[0, -1, :]

    # ensure EOS (compute EOS old_logprob properly)
    if (not sampled_ids) or (int(sampled_ids[-1]) != int(eos_id)):
        x_full = torch.tensor(traj_ids, dtype=torch.long, device=device).unsqueeze(0)
        logits_full = forward_logits(model, x_full)[0]
        next_logits = logits_full[-1]

        cand_ids = dynamic_candidate_token_ids(
            pending=parser.pending_as_str(),
            current_node=current_node,
            action_space=action_space,
            args=args,
            succ_strs_by_node=succ_strs_by_node,
        )
        if cand_ids is None:
            raise RuntimeError("ActionSpace is required for ascii_digits_space_single.")

        lp_eos, ent_eos = masked_logprob_and_entropy_from_logits(
            logits_1d=next_logits,
            candidate_ids=cand_ids,
            action_id=int(eos_id),
            temperature=temperature if args.logprob_matches_rollout_temperature else 1.0,
            top_k=args.rollout_top_k if args.logprob_matches_rollout_top_k else 0,
        )
        old_logprobs.append(float(lp_eos.item()))
        entropies.append(float(ent_eos.item()))

        sampled_ids.append(int(eos_id))
        traj_ids.append(int(eos_id))
        piece = tokenizer.decode([int(eos_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        action_token_texts.append(piece)

        extra_reward = 0.0
        for node in parser.finalize():
            generated_nodes.append(int(node))
            r_delta, _ = apply_node_transition(int(node))
            extra_reward += apply_node_reward_credit_to_digits(
                rewards=rewards,
                digit_step_indices=pending_digit_step_indices,
                node_reward=float(r_delta),
                mode=str(args.node_reward_credit),
            )
            pending_digit_step_indices.clear()

        extra_reward += float(args.reward_stop)
        rewards.append(float(extra_reward))
        dones.append(True)

    raw_path_nodes = [int(source)] + list(map(int, generated_nodes))

    success_used = is_valid_path_k(
        raw_path_nodes if args.success_path_mode == "raw" else applied_path_nodes,
        int(source),
        int(target),
        node_to_stage=node_to_stage,
        adj=adj,
    )
    success_raw = is_valid_path_k(raw_path_nodes, int(source), int(target), node_to_stage=node_to_stage, adj=adj)
    path_for_success = raw_path_nodes if args.success_path_mode == "raw" else applied_path_nodes

    if args.reward_type == "outcome":
        adjusted = [0.0 for _ in rewards]
        if success_used and hit_target:
            adjusted[-1] = float(args.reward_hit_target)
        else:
            adjusted[-1] = -float(args.reward_miss_target)
        rewards = adjusted
    else:
        if (not success_used) and rewards:
            rewards[-1] -= float(args.reward_miss_target)

    required_cnt = len(required_stages)
    visited_required_cnt = len(visited_required_stages)
    covered_all_required = (visited_required_cnt == required_cnt) if required_cnt > 0 else True
    covered_ratio = (visited_required_cnt / required_cnt) if required_cnt > 0 else 1.0

    raw_completion_text = tokenizer.decode(sampled_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    raw_full_text = prompt_text + raw_completion_text

    if model_was_training:
        model.train()

    return {
        "prompt_text": prompt_text,
        "prompt_ids": prompt_ids,
        "traj_ids": traj_ids,
        "actions": sampled_ids,
        "rewards": rewards,
        "dones": dones,
        "episode_reward": float(sum(rewards)) if rewards else 0.0,
        "old_logprobs": old_logprobs,
        "rollout_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "temperature": float(temperature),
        "rollout_top_k": int(args.rollout_top_k),
        "action_token_texts": action_token_texts,
        "raw_completion_text": raw_completion_text,
        "raw_full_text": raw_full_text,
        "final_parser_pending_digits": parser.pending_as_str(),
        "generated_nodes": generated_nodes,
        "raw_path_nodes": raw_path_nodes,
        "path_nodes": path_for_success,
        "success": bool(success_used),
        "success_raw": bool(success_raw),
        "hit_target": bool(hit_target),
        "invalid_transition": bool(invalid_transition),
        "invalid_token": bool(invalid_token),
        "valid_transition_steps": int(valid_transition_steps),
        "invalid_transition_steps": int(invalid_transition_steps),
        "bucket": pair_bucket,
        "covered_all_required": bool(covered_all_required),
        "covered_required_ratio": float(covered_ratio),
        "node_reward_credit": str(args.node_reward_credit),
        "mask_space_when_pending_empty": bool(args.mask_space_when_pending_empty),
    }


# -----------------------------------------------------------------------------
# KL loss
# -----------------------------------------------------------------------------
def kl_loss_masked_full_dynamic_from_logits(
    tokenizer,
    seg_pi: Tensor,
    seg_ref: Tensor,
    actions: List[int],
    source: int,
    device: torch.device,
    args: argparse.Namespace,
    adj: Dict[int, Set[int]],
    succ_strs_by_node: Dict[int, Set[str]],
    action_space: ActionSpace,
) -> Tensor:
    eos_id = int(tokenizer.eos_token_id)

    pending_states, env_states = compute_pending_and_env_states_for_actions(
        tokenizer=tokenizer,
        actions=actions,
        source=int(source),
        eos_id=eos_id,
        adj=adj,
        allow_invalid_continue=bool(args.allow_invalid_continue),
        max_invalid_transitions=int(args.max_invalid_transitions),
    )

    T = seg_pi.size(0)
    kls: List[Tensor] = []
    for t in range(T):
        cand_ids = dynamic_candidate_token_ids(
            pending=pending_states[t],
            current_node=int(env_states[t]),
            action_space=action_space,
            args=args,
            succ_strs_by_node=succ_strs_by_node,
        )
        if cand_ids is None:
            cand_ids = action_space.all_static_ids

        lp = seg_pi[t].index_select(0, cand_ids).float()
        lr = seg_ref[t].index_select(0, cand_ids).float()

        logp = F.log_softmax(lp, dim=-1)
        logq = F.log_softmax(lr, dim=-1)
        p = logp.exp()
        kl_t = (p * (logp - logq)).sum(dim=-1)
        kls.append(kl_t)

    return torch.stack(kls, dim=0).mean()


# -----------------------------------------------------------------------------
# PPO helpers
# -----------------------------------------------------------------------------
def discounted_rtg(rewards: List[float], dones: List[bool], gamma: float) -> List[float]:
    T = len(rewards)
    out = [0.0] * T
    g = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            g = 0.0
        g = float(rewards[t]) + float(gamma) * g
        out[t] = g
    return out


def compute_advantages_for_group(
    trajs: List[Dict[str, object]],
    adv_type: str,
    adv_norm: str,
    adv_clip: float,
    gamma: float,
    eps: float,
) -> List[Tensor]:
    if adv_type not in ("sequence", "rtg"):
        raise ValueError(f"Unknown adv_type: {adv_type}")
    if adv_norm not in ("group", "none"):
        raise ValueError(f"Unknown adv_norm: {adv_norm}")

    if adv_type == "sequence":
        rets = torch.tensor([float(tr["episode_reward"]) for tr in trajs], dtype=torch.float32)
        if adv_norm == "group":
            adv_scalar = (rets - rets.mean()) / (rets.std(unbiased=False) + eps)
        else:
            adv_scalar = rets
        advs: List[Tensor] = []
        for i, tr in enumerate(trajs):
            T = len(tr["actions"])
            a = adv_scalar[i].repeat(T)
            if adv_clip > 0:
                a = a.clamp(-adv_clip, adv_clip)
            advs.append(a)
        return advs

    all_vals: List[float] = []
    rtgs: List[List[float]] = []
    for tr in trajs:
        r = list(map(float, tr["rewards"]))
        d = list(map(bool, tr["dones"]))
        g = discounted_rtg(r, d, gamma=gamma)
        rtgs.append(g)
        all_vals.extend(g)

    all_t = torch.tensor(all_vals, dtype=torch.float32)
    if adv_norm == "group":
        mu = all_t.mean()
        sd = all_t.std(unbiased=False) + eps
    else:
        mu = torch.tensor(0.0)
        sd = torch.tensor(1.0)

    advs2: List[Tensor] = []
    for g in rtgs:
        a = (torch.tensor(g, dtype=torch.float32) - mu) / sd
        if adv_clip > 0:
            a = a.clamp(-adv_clip, adv_clip)
        advs2.append(a)
    return advs2


def compute_new_logprobs_and_entropy_for_traj(
    tokenizer,
    logits_pi: Tensor,
    traj_ids: List[int],
    prompt_len: int,
    actions: List[int],
    source: int,
    args: argparse.Namespace,
    adj: Dict[int, Set[int]],
    succ_strs_by_node: Dict[int, Set[str]],
    action_space: ActionSpace,
    temperature: float,
    top_k: int,
) -> Tuple[Tensor, Tensor]:
    start_idx = prompt_len - 1
    T = len(actions)
    seg = logits_pi[start_idx : start_idx + T, :]  # [T,V]

    eos_id = int(tokenizer.eos_token_id)
    pending_states, env_states = compute_pending_and_env_states_for_actions(
        tokenizer=tokenizer,
        actions=actions,
        source=int(source),
        eos_id=eos_id,
        adj=adj,
        allow_invalid_continue=bool(args.allow_invalid_continue),
        max_invalid_transitions=int(args.max_invalid_transitions),
    )

    lps: List[Tensor] = []
    ents: List[Tensor] = []
    for t in range(T):
        cand_ids = dynamic_candidate_token_ids(
            pending=pending_states[t],
            current_node=int(env_states[t]),
            action_space=action_space,
            args=args,
            succ_strs_by_node=succ_strs_by_node,
        )
        if cand_ids is None:
            cand_ids = action_space.all_static_ids

        lp, ent = masked_logprob_and_entropy_from_logits(
            logits_1d=seg[t],
            candidate_ids=cand_ids,
            action_id=int(actions[t]),
            temperature=temperature if args.logprob_matches_rollout_temperature else 1.0,
            top_k=top_k if args.logprob_matches_rollout_top_k else 0,
        )
        lps.append(lp)
        ents.append(ent)

    return torch.stack(lps, dim=0), torch.stack(ents, dim=0)


# -----------------------------------------------------------------------------
# eval (NO applied accuracy fields)
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    pairs: List[Pair],
    node_to_stage: Dict[int, int],
    K: int,
    adj: Dict[int, Set[int]],
    succ_strs_by_node: Dict[int, Set[str]],
    device: torch.device,
    args: argparse.Namespace,
    block_size: int,
    max_pairs: int,
    action_space: ActionSpace,
    dist_cache: Optional[Dict[int, Dict[int, int]]],
) -> Dict[str, Dict[str, float]]:
    was_training = model.training
    model.eval()

    bucket_names = [f"S{i}->S{j}" for i in range(1, K + 1) for j in range(i + 1, K + 1)]
    buckets: Dict[BucketName, List[Pair]] = {bn: [] for bn in bucket_names}

    if max_pairs <= 0:
        res = {bn: {"correct": 0, "total": 0, "accuracy": 0.0, "accuracy_raw": 0.0} for bn in bucket_names}
        res["overall"] = {"correct": 0, "total": 0, "accuracy": 0.0, "accuracy_raw": 0.0}
        return res

    for s, t in pairs[:max_pairs]:
        b = bucket_for_pair_k(s, t, node_to_stage=node_to_stage, K=K)
        if b is not None:
            buckets[b].append((s, t))

    total_correct = 0
    total_correct_raw = 0
    total_cases = 0

    results: Dict[str, Dict[str, float]] = {}
    for bname in bucket_names:
        bpairs = buckets[bname]
        correct = 0
        correct_raw = 0

        for s, t in bpairs:
            traj = sample_trajectory_hf(
                model=model,
                tokenizer=tokenizer,
                source=s,
                target=t,
                adj=adj,
                succ_strs_by_node=succ_strs_by_node,
                node_to_stage=node_to_stage,
                K=K,
                args=args,
                device=device,
                block_size=block_size,
                temperature=args.eval_temperature,
                epsilon=0.0,
                action_space=action_space,
                dist_cache=dist_cache,
            )
            correct += int(bool(traj["success"]))
            correct_raw += int(bool(traj["success_raw"]))

        n = len(bpairs)
        total_cases += n
        total_correct += correct
        total_correct_raw += correct_raw

        results[bname] = {
            "correct": int(correct),
            "total": int(n),
            "accuracy": (correct / n if n else 0.0),
            "accuracy_raw": (correct_raw / n if n else 0.0),
        }

    results["overall"] = {
        "correct": int(total_correct),
        "total": int(total_cases),
        "accuracy": (total_correct / total_cases if total_cases else 0.0),
        "accuracy_raw": (total_correct_raw / total_cases if total_cases else 0.0),
    }

    if was_training:
        model.train()
    else:
        model.eval()
    return results


# -----------------------------------------------------------------------------
# model loading
# -----------------------------------------------------------------------------
def maybe_wrap_lora(model, args: argparse.Namespace):
    if args.lora_r <= 0:
        return model
    if get_peft_model is None:
        raise RuntimeError("peft not installed but lora_r>0 was set.")
    lconf = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=args.lora_target_modules.split(",") if args.lora_target_modules else None,
    )
    return get_peft_model(model, lconf)


def load_policy_model_and_ref(base_model: str, sft_dir: str, device: torch.device, args: argparse.Namespace, vocab_size: int):
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    def _device_map_single():
        if args.load_in_4bit:
            return "auto"
        if device.type == "cuda":
            return {"": int(device.index)}
        return {"": "cpu"}

    def _load_base():
        kwargs = dict(torch_dtype=torch_dtype, trust_remote_code=bool(args.trust_remote_code))
        if args.load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = _device_map_single()
        return AutoModelForCausalLM.from_pretrained(base_model, **kwargs)

    def _maybe_resize_to_tokenizer_vocab(m, name: str) -> None:
        if vocab_size is None or vocab_size <= 0:
            return
        cur = int(m.get_input_embeddings().weight.size(0))
        if cur == int(vocab_size):
            return
        print(f"[vocab] {name}: resizing token embeddings {cur} -> {int(vocab_size)} (to match tokenizer)")
        m.resize_token_embeddings(int(vocab_size))
        if hasattr(m, "tie_weights"):
            try:
                m.tie_weights()
            except Exception:
                pass

    sft_path = Path(sft_dir)
    has_adapter = (sft_path / "adapter_config.json").exists()

    if has_adapter:
        if PeftModel is None:
            raise RuntimeError("peft not installed but adapter_config.json exists.")
        base = _load_base()
        _maybe_resize_to_tokenizer_vocab(base, name="base(policy)")
        policy = PeftModel.from_pretrained(base, sft_dir, is_trainable=True)
    else:
        kwargs = dict(torch_dtype=torch_dtype, trust_remote_code=bool(args.trust_remote_code))
        if args.load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = _device_map_single()
        policy = AutoModelForCausalLM.from_pretrained(sft_dir, **kwargs)
        _maybe_resize_to_tokenizer_vocab(policy, name="policy(full)")
        policy = maybe_wrap_lora(policy, args)

    ref = None
    if args.kl_coef > 0.0:
        if has_adapter:
            base2 = _load_base()
            _maybe_resize_to_tokenizer_vocab(base2, name="base(ref)")
            ref = PeftModel.from_pretrained(base2, sft_dir, is_trainable=False)
        else:
            kwargs = dict(torch_dtype=torch_dtype, trust_remote_code=bool(args.trust_remote_code))
            if args.load_in_4bit:
                kwargs["load_in_4bit"] = True
                kwargs["device_map"] = "auto"
            else:
                kwargs["device_map"] = _device_map_single()
            ref = AutoModelForCausalLM.from_pretrained(sft_dir, **kwargs)
            _maybe_resize_to_tokenizer_vocab(ref, name="ref(full)")
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False

    return policy, ref


# -----------------------------------------------------------------------------
# args
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GRPO/PPO-style (HF/Qwen) for composition graphs (K stages) v2_fix")

    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--train_paths_per_pair", type=int, default=20)
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B")
    p.add_argument("--sft_dir", type=str, required=True)
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--eval_only", action="store_true")

    # training (default reduced)
    p.add_argument("--max_iters", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--prompt_sample_with_replacement", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--grad_clip", type=float, default=0.5)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--gamma", type=float, default=0.98)

    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--adv_type", choices=["sequence", "rtg"], default="sequence")
    p.add_argument("--adv_norm", choices=["group", "none"], default="group")
    p.add_argument("--adv_clip", type=float, default=5.0)
    p.add_argument("--adv_norm_eps", type=float, default=1e-8)

    p.add_argument("--max_rollout_steps", type=int, default=64)
    p.add_argument("--rollout_top_k", type=int, default=0)
    p.add_argument("--rollout_temp_start", type=float, default=1.0)
    p.add_argument("--rollout_temp_end", type=float, default=1.0)
    p.add_argument("--temp_warmup_iters", type=int, default=0)

    p.add_argument("--epsilon_start", type=float, default=0.0)
    p.add_argument("--epsilon_end", type=float, default=0.0)
    p.add_argument("--epsilon_warmup_iters", type=int, default=0)
    p.add_argument("--epsilon_explore_top_k", type=int, default=30)

    p.add_argument("--logprob_matches_rollout_temperature", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--logprob_matches_rollout_top_k", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument(
        "--action_mask",
        type=str,
        default="ascii_digits_space_single",
        choices=["none", "ascii_digits_space_single", "ascii_digits_space"],
    )

    p.add_argument("--node_reward_credit", type=str, default="none", choices=["none", "uniform_digits"])
    p.add_argument("--mask_space_when_pending_empty", action=argparse.BooleanOptionalAction, default=True)

    # keep interface, but v2_fix expects them OFF
    p.add_argument("--dynamic_digit_guard", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dynamic_successor_prefix_guard", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--forbid_any_guard", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--allow_invalid_continue", action="store_true")
    p.add_argument("--max_invalid_transitions", type=int, default=3)
    p.add_argument("--success_path_mode", choices=["applied", "raw"], default="applied")

    p.add_argument("--reward_type", choices=["process", "outcome"], default="process")
    p.add_argument("--reward_hit_target", type=float, default=3.0)
    p.add_argument("--reward_hit_target_requires_coverage", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reward_hit_target_uncovered", type=float, default=0.0)
    p.add_argument("--reward_valid_transition", type=float, default=0.10)
    p.add_argument("--reward_stage_bridge", type=float, default=0.5)
    p.add_argument("--reward_required_stage_cover", type=float, default=0.0)
    p.add_argument("--reward_required_stage_cover_only_once", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reward_invalid_transition", type=float, default=0.6)
    p.add_argument("--reward_invalid_token", type=float, default=1.0)
    p.add_argument("--reward_stop", type=float, default=0.0)
    p.add_argument("--reward_miss_target", type=float, default=1.5)

    p.add_argument("--penalty_target_stage_detour", type=float, default=0.2)
    p.add_argument("--terminate_on_overshoot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--penalty_repeat_node", type=float, default=0.25)
    p.add_argument("--step_penalty", type=float, default=0.002)

    p.add_argument("--node_min", type=int, default=0)
    p.add_argument("--node_max", type=int, default=149)

    p.add_argument("--kl_coef", type=float, default=0.2)
    p.add_argument("--kl_warmup_iters", type=int, default=0)
    p.add_argument("--kl_anneal_iters", type=int, default=0)
    p.add_argument("--kl_min_coef", type=float, default=0.05)

    p.add_argument("--reward_distance_shaping_alpha", type=float, default=0.35)
    p.add_argument("--distance_shaping_cap", type=float, default=1.0)

    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=200)
    p.add_argument("--max_eval_pairs", type=int, default=500)
    p.add_argument("--eval_temperature", type=float, default=1e-3)
    p.add_argument("--log_dir", type=str, default="out_qwen_grpo")
    p.add_argument("--log_interval", type=int, default=50)

    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.action_mask != "ascii_digits_space_single":
        raise ValueError("This script requires --action_mask ascii_digits_space_single")
    if args.dynamic_digit_guard or args.dynamic_successor_prefix_guard:
        raise ValueError("v2_fix expects NO guards. Use --no-dynamic_digit_guard --no-dynamic_successor_prefix_guard.")
    if args.group_size <= 0:
        raise ValueError("--group_size must be > 0")
    if args.epsilon_start != 0.0 or args.epsilon_end != 0.0:
        print("[WARN] epsilon-greedy breaks strict on-policy PPO. Recommended eps=0.")
    if args.adv_type == "rtg" and args.group_size < 2 and args.adv_norm == "group":
        raise ValueError("adv_norm=group needs group_size>=2")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    _validate_args(args)
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    data_dir = Path(args.data_dir).resolve()
    train_txt = data_dir / f"train_{args.train_paths_per_pair}.txt"
    test_txt = data_dir / "test.txt"
    if not train_txt.exists():
        raise FileNotFoundError(train_txt)
    if not test_txt.exists():
        raise FileNotFoundError(test_txt)

    meta_path = data_dir / "meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    block_size = int(meta.get("block_size", 63))

    sft_path = Path(args.sft_dir)
    tok_dir = sft_path if (sft_path / "tokenizer.json").exists() else (data_dir / "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, trust_remote_code=bool(args.trust_remote_code))
    print(f"[tokenizer] loaded from {tok_dir}")

    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer missing eos/pad token id.")
    if int(tokenizer.pad_token_id) == int(tokenizer.eos_token_id):
        raise ValueError("pad_token_id == eos_token_id; prepare step broken.")

    with open(data_dir / "stage_info.pkl", "rb") as f:
        stage_info = pickle.load(f)
    stages: List[List[int]] = stage_info["stages"]
    K = len(stages)
    node_to_stage = build_node_to_stage(stages)
    print(f"[stages] K={K}")

    G = nx.read_graphml(data_dir / "composition_graph.graphml")
    adj = build_int_adjacency(G)
    succ_strs_by_node = build_successor_strings(adj)

    dist_cache = None
    if args.reward_distance_shaping_alpha != 0.0:
        t0 = time.perf_counter()
        dist_cache = precompute_all_target_dists(adj, node_min=args.node_min, node_max=args.node_max)
        print(f"[dist] precomputed all target distances in {time.perf_counter() - t0:.2f}s")

    out_dir = prepare_output_dir(args.log_dir)
    metrics_path = out_dir / "metrics_grpo.jsonl"
    print(f"[out_dir] {out_dir}")
    print(f"[meta] block_size={block_size}, eos_id={tokenizer.eos_token_id}, pad_id={tokenizer.pad_token_id}, vocab={len(tokenizer)}")
    print(f"[credit] node_reward_credit={args.node_reward_credit}, mask_space_when_pending_empty={args.mask_space_when_pending_empty}")

    allowed_token_mask = build_allowed_token_mask(tokenizer, device=device, mode=args.action_mask, allow_eos=True)
    kept = int(allowed_token_mask.sum().item())
    print(f"[action_mask] mode={args.action_mask} enabled. allowed={kept}/{len(tokenizer)} tokens.")
    action_space = build_action_space_single_char(tokenizer, device=device, allowed_token_mask=allowed_token_mask)

    policy_model, ref_model = load_policy_model_and_ref(
        base_model=args.base_model,
        sft_dir=args.sft_dir,
        device=device,
        args=args,
        vocab_size=len(tokenizer),
    )

    # IMPORTANT: keep policy_model in eval mode to match rollout logprobs (dropout OFF),
    # but it is still trainable and gradients still work.
    policy_model.eval()
    policy_model.config.pad_token_id = tokenizer.pad_token_id
    policy_model.config.eos_token_id = tokenizer.eos_token_id
    policy_model.config.use_cache = True

    if args.kl_coef > 0.0 and ref_model is None:
        raise RuntimeError("ref_model is required when kl_coef>0")
    if ref_model is not None:
        ref_model.config.pad_token_id = tokenizer.pad_token_id
        ref_model.config.eos_token_id = tokenizer.eos_token_id
        ref_model.config.use_cache = False
        ref_model.eval()

    train_pairs = load_pairs_unique(train_txt)
    eval_pairs: List[Pair] = []
    for line in test_txt.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            eval_pairs.append((int(parts[0]), int(parts[1])))

    bucket_names = [f"S{i}->S{j}" for i in range(1, K + 1) for j in range(i + 1, K + 1)]
    print(f"[data] train unique pairs={len(train_pairs)}, eval pairs={len(eval_pairs)}, num_buckets={len(bucket_names)}")

    if args.eval_only:
        eval_res = evaluate_model(
            model=policy_model,
            tokenizer=tokenizer,
            pairs=eval_pairs,
            node_to_stage=node_to_stage,
            K=K,
            adj=adj,
            succ_strs_by_node=succ_strs_by_node,
            device=device,
            args=args,
            block_size=block_size,
            max_pairs=args.max_eval_pairs,
            action_space=action_space,
            dist_cache=dist_cache,
        )
        eval_record = {"iter": 0, "eval": eval_res}
        print(json.dumps(eval_record, ensure_ascii=False))
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
        return

    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    t0_all = time.perf_counter()

    for iteration in range(1, args.max_iters + 1):
        t_iter = time.perf_counter()

        temperature = current_temperature(iteration, args)
        epsilon = current_epsilon(iteration, args)
        kl_coef_cur = current_kl_coef(iteration, args)

        batch_prompts = sample_prompts(train_pairs, args.batch_size, with_replacement=bool(args.prompt_sample_with_replacement))
        rollouts_by_prompt: Dict[Pair, List[Dict[str, object]]] = defaultdict(list)

        success_used = 0
        success_raw = 0
        invalid_edge_count = 0
        invalid_tok_count = 0
        ep_rewards: List[float] = []
        ent_list: List[float] = []

        bucket_success_sum = defaultdict(float)
        bucket_counts = defaultdict(int)

        for (s, t) in batch_prompts:
            for _g in range(args.group_size):
                traj = sample_trajectory_hf(
                    model=policy_model,
                    tokenizer=tokenizer,
                    source=s,
                    target=t,
                    adj=adj,
                    succ_strs_by_node=succ_strs_by_node,
                    node_to_stage=node_to_stage,
                    K=K,
                    args=args,
                    device=device,
                    block_size=block_size,
                    temperature=temperature,
                    epsilon=epsilon,
                    action_space=action_space,
                    dist_cache=dist_cache,
                )
                rollouts_by_prompt[(s, t)].append(traj)

                success_used += int(bool(traj["success"]))
                success_raw += int(bool(traj["success_raw"]))
                invalid_edge_count += int(traj["invalid_transition"])
                invalid_tok_count += int(traj["invalid_token"])
                ep_rewards.append(float(traj["episode_reward"]))
                ent_list.append(float(traj["rollout_entropy"]))

                b = traj.get("bucket", None)
                if b:
                    bucket_success_sum[b] += 1.0 if bool(traj["success"]) else 0.0
                    bucket_counts[b] += 1

        losses: List[Tensor] = []
        kl_list: List[float] = []
        approx_kl_list: List[float] = []
        clip_frac_list: List[float] = []
        adv_mean_list: List[float] = []

        # IMPORTANT: keep dropout OFF for PPO logprob recomputation
        policy_model.eval()

        for (s, t), trajs in rollouts_by_prompt.items():
            advs = compute_advantages_for_group(
                trajs=trajs,
                adv_type=args.adv_type,
                adv_norm=args.adv_norm,
                adv_clip=float(args.adv_clip),
                gamma=float(args.gamma),
                eps=float(args.adv_norm_eps),
            )

            for tr, adv in zip(trajs, advs):
                traj_ids = tr["traj_ids"]
                prompt_len = len(tr["prompt_ids"])
                actions = tr["actions"]
                old_lp = torch.tensor(list(map(float, tr["old_logprobs"])), dtype=torch.float32, device=device)

                x_ids = torch.tensor(traj_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
                logits_pi = forward_logits(policy_model, x_ids)[0]

                new_lp, ent_t = compute_new_logprobs_and_entropy_for_traj(
                    tokenizer=tokenizer,
                    logits_pi=logits_pi,
                    traj_ids=traj_ids,
                    prompt_len=prompt_len,
                    actions=actions,
                    source=s,
                    args=args,
                    adj=adj,
                    succ_strs_by_node=succ_strs_by_node,
                    action_space=action_space,
                    temperature=float(tr["temperature"]),
                    top_k=int(tr["rollout_top_k"]),
                )
                new_lp = new_lp.to(device)

                adv_t = adv.to(device)
                if adv_t.numel() != new_lp.numel():
                    raise ValueError("Advantage length mismatch with actions/new_logprobs.")
                if old_lp.numel() != new_lp.numel():
                    raise ValueError("old_logprobs length mismatch with actions/new_logprobs.")

                ratio = torch.exp(new_lp - old_lp)
                clipped = torch.clamp(ratio, 1.0 - float(args.clip_range), 1.0 + float(args.clip_range))
                pg = torch.minimum(ratio * adv_t, clipped * adv_t)
                policy_loss = -pg.mean()

                clip_frac = (torch.abs(ratio - 1.0) > float(args.clip_range)).float().mean()
                approx_kl = (old_lp - new_lp).mean()

                loss_i = policy_loss
                if args.entropy_coef != 0.0:
                    loss_i = loss_i - float(args.entropy_coef) * ent_t.mean()

                kl_val = 0.0
                if kl_coef_cur > 0.0 and ref_model is not None:
                    with torch.no_grad():
                        logits_ref = forward_logits(ref_model, x_ids)[0]
                    start_idx = prompt_len - 1
                    T = len(actions)
                    seg_pi = logits_pi[start_idx : start_idx + T, :]
                    seg_ref = logits_ref[start_idx : start_idx + T, :]
                    kl = kl_loss_masked_full_dynamic_from_logits(
                        tokenizer=tokenizer,
                        seg_pi=seg_pi,
                        seg_ref=seg_ref,
                        actions=actions,
                        source=s,
                        device=device,
                        args=args,
                        adj=adj,
                        succ_strs_by_node=succ_strs_by_node,
                        action_space=action_space,
                    )
                    loss_i = loss_i + float(kl_coef_cur) * kl
                    kl_val = float(kl.detach().item())

                losses.append(loss_i)
                kl_list.append(kl_val)
                approx_kl_list.append(float(approx_kl.detach().item()))
                clip_frac_list.append(float(clip_frac.detach().item()))
                adv_mean_list.append(float(adv_t.mean().detach().item()))

        total_loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        opt.step()

        iter_time = time.perf_counter() - t_iter
        num_trajs = max(1, args.batch_size * args.group_size)

        if args.log_interval > 0 and (iteration % args.log_interval == 0):
            record = {
                "iter": iteration,
                "iter_time_sec": float(iter_time),
                "elapsed_min": float((time.perf_counter() - t0_all) / 60.0),
                "loss": float(total_loss.item()),
                "kl_loss": float(np.mean(kl_list)) if kl_list else 0.0,
                "kl_coef_current": float(kl_coef_cur),
                "approx_kl": float(np.mean(approx_kl_list)) if approx_kl_list else 0.0,
                "clip_frac": float(np.mean(clip_frac_list)) if clip_frac_list else 0.0,
                "adv_mean": float(np.mean(adv_mean_list)) if adv_mean_list else 0.0,
                "entropy_rollout": float(np.mean(ent_list)) if ent_list else 0.0,
                "temperature": float(temperature),
                "epsilon": float(epsilon),
                "rollout_top_k": int(args.rollout_top_k),
                "success_rate": success_used / num_trajs,
                "success_rate_raw": success_raw / num_trajs,
                "invalid_edge_rate": invalid_edge_count / num_trajs,
                "invalid_tok_rate": invalid_tok_count / num_trajs,
                "avg_episode_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
                "K": int(K),
                "batch_prompts": int(args.batch_size),
                "group_size": int(args.group_size),
                "adv_type": str(args.adv_type),
                "adv_norm": str(args.adv_norm),
                "clip_range": float(args.clip_range),
                "node_reward_credit": str(args.node_reward_credit),
                "mask_space_when_pending_empty": bool(args.mask_space_when_pending_empty),
                "prompt_sample_with_replacement": bool(args.prompt_sample_with_replacement),
            }
            for bn in bucket_names:
                cnt = bucket_counts.get(bn, 0)
                record[f"train_success/{bn}"] = (bucket_success_sum.get(bn, 0.0) / cnt) if cnt else 0.0

            print(json.dumps(record, ensure_ascii=False))
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if iteration % args.eval_interval == 0 or iteration == args.max_iters:
            eval_res = evaluate_model(
                model=policy_model,
                tokenizer=tokenizer,
                pairs=eval_pairs,
                node_to_stage=node_to_stage,
                K=K,
                adj=adj,
                succ_strs_by_node=succ_strs_by_node,
                device=device,
                args=args,
                block_size=block_size,
                max_pairs=args.max_eval_pairs,
                action_space=action_space,
                dist_cache=dist_cache,
            )
            eval_record = {"iter": iteration, "eval": eval_res}
            print(json.dumps(eval_record, ensure_ascii=False))
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(eval_record, ensure_ascii=False) + "\n")

        if iteration % args.save_interval == 0 or iteration == args.max_iters:
            ckpt_dir = out_dir / f"ckpt_grpo_{iteration}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            policy_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            with open(ckpt_dir / "train_args.json", "w", encoding="utf-8") as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=2)
            with open(ckpt_dir / "data_meta.pkl", "wb") as f:
                pickle.dump(meta, f)
            print(f"[save] {ckpt_dir}")


if __name__ == "__main__":
    main()