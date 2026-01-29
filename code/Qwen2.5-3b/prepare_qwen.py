#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare GraphA dataset for Qwen (HF) training.

Outputs (written into --data_dir):
  - train_{K}.bin (uint32)
  - val.bin       (uint32)  # from test.txt
  - meta.pkl      (hf_model, pad/eos ids, seq_len, block_size, etc.)
  - tokenizer/    (saved tokenizer with guaranteed PAD != EOS)

Important:
  We FORCE a real PAD token if:
    (a) tokenizer has no pad_token_id, OR
    (b) pad_token_id == eos_token_id
  to avoid masking EOS as padding during training.

Usage:
  python prepare_qwen.py \
    --data_dir data/datasets/graphA_pg020_tier3 \
    --train_paths_per_pair 20 \
    --hf_model Qwen/Qwen2.5-3B \
    --block_multiple 32 \
    --append_eos
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare composition dataset binaries (Qwen tokenizer).")

    io = p.add_argument_group("I/O")
    io.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing train_{K}.txt and test.txt.",
    )
    io.add_argument(
        "--train_paths_per_pair",
        type=int,
        default=20,
        help="Used to find train_{K}.txt.",
    )

    tok = p.add_argument_group("Tokenizer")
    tok.add_argument(
        "--hf_model",
        type=str,
        required=True,
        help="HF model name, e.g. Qwen/Qwen2.5-3B",
    )
    tok.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loaders (sometimes needed for Qwen).",
    )

    pack = p.add_argument_group("Packing")
    pack.add_argument(
        "--block_multiple",
        type=int,
        default=32,
        help="Round sequence length up to a multiple of this value.",
    )
    pack.add_argument(
        "--append_eos",
        action="store_true",
        help="Append eos_token_id to each line (recommended).",
    )

    args = p.parse_args()
    if args.block_multiple <= 0:
        raise ValueError("--block_multiple must be > 0")
    return args


# -----------------------------------------------------------------------------
# Helpers (logic preserved)
# -----------------------------------------------------------------------------
def round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def ensure_real_pad_token(tokenizer, pad_token_str: str = "<|pad|>") -> Tuple[bool, Optional[int], Optional[bool]]:
    """
    Ensure tokenizer has a real pad_token_id AND pad_token_id != eos_token_id.

    Returns:
      (changed, old_pad_id, old_pad_equal_eos)
        changed: whether we modified the tokenizer's pad token setting/vocab
        old_pad_id: previous pad_token_id (None if absent)
        old_pad_equal_eos: whether old pad == eos (None if eos/pad absent)
    """
    old_pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    old_equal: Optional[bool] = None
    if old_pad_id is not None and eos_id is not None:
        old_equal = (old_pad_id == eos_id)

    need_fix = (tokenizer.pad_token_id is None) or (
        tokenizer.eos_token_id is not None and tokenizer.pad_token_id == tokenizer.eos_token_id
    )
    if not need_fix:
        return False, old_pad_id, old_equal

    # Force a distinct PAD token string. If not in vocab, HF will add it.
    tokenizer.add_special_tokens({"pad_token": pad_token_str})

    # Hard verify
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Failed to set pad_token_id (still None) after add_special_tokens().")
    if tokenizer.eos_token_id is not None and tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise RuntimeError(
            f"PAD/EOS are still equal after fix! pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}. "
            f"Try changing pad_token_str."
        )

    return True, old_pad_id, old_equal


def encode_lines(tokenizer, lines: List[str], append_eos: bool) -> List[List[int]]:
    encoded: List[List[int]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        ids = tokenizer(line, add_special_tokens=False)["input_ids"]
        if append_eos:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has no eos_token_id; cannot append EOS.")
            ids = ids + [tokenizer.eos_token_id]

        encoded.append(ids)
    return encoded


def pad_to_len(seq: List[int], seq_len: int, pad_id: int) -> List[int]:
    if len(seq) > seq_len:
        raise ValueError(f"Sequence too long: {len(seq)} > {seq_len}. Increase --block_multiple.")
    return seq + [pad_id] * (seq_len - len(seq))


def write_bin(path: Path, sequences: List[List[int]], seq_len: int, pad_id: int) -> None:
    flat: List[int] = []
    for s in sequences:
        flat.extend(pad_to_len(s, seq_len, pad_id))
    arr = np.array(flat, dtype=np.uint32)
    arr.tofile(path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()

    train_txt = data_dir / f"train_{args.train_paths_per_pair}.txt"
    test_txt = data_dir / "test.txt"
    if not train_txt.exists():
        raise FileNotFoundError(f"Training file not found: {train_txt}")
    if not test_txt.exists():
        raise FileNotFoundError(f"Test file not found: {test_txt}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
    )

    if args.append_eos and tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; cannot append EOS safely.")

    changed_pad, old_pad_id, old_equal = ensure_real_pad_token(tokenizer)

    # Final invariants (kept)
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer has no pad_token_id after ensure_real_pad_token().")
    if tokenizer.eos_token_id is not None and tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise RuntimeError("Invariant violated: pad_token_id == eos_token_id.")

    train_lines = train_txt.read_text(encoding="utf-8").splitlines()
    test_lines = test_txt.read_text(encoding="utf-8").splitlines()

    train_ids = encode_lines(tokenizer, train_lines, append_eos=args.append_eos)
    test_ids = encode_lines(tokenizer, test_lines, append_eos=args.append_eos)
    if not train_ids or not test_ids:
        raise ValueError("Encoded dataset is empty. Check train/test txt files.")

    max_len = max(max(len(x) for x in train_ids), max(len(x) for x in test_ids))
    seq_len = round_up(max_len, args.block_multiple)

    # Convention: stored sequence length seq_len = block_size + 1
    block_size = seq_len - 1

    train_bin = data_dir / f"train_{args.train_paths_per_pair}.bin"
    val_bin = data_dir / "val.bin"

    write_bin(train_bin, train_ids, seq_len=seq_len, pad_id=int(tokenizer.pad_token_id))
    write_bin(val_bin, test_ids, seq_len=seq_len, pad_id=int(tokenizer.pad_token_id))

    meta = {
        "format": "hf_tokenized",
        "hf_model": args.hf_model,
        "trust_remote_code": bool(args.trust_remote_code),
        "append_eos": bool(args.append_eos),
        "seq_len": int(seq_len),
        "block_size": int(block_size),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
        "changed_pad_token": bool(changed_pad),
        "old_pad_token_id": int(old_pad_id) if old_pad_id is not None else None,
        "old_pad_equal_eos": bool(old_equal) if old_equal is not None else None,
        "train_paths_per_pair": int(args.train_paths_per_pair),
        "dtype": "uint32",
        "tokenizer_len": int(len(tokenizer)),
    }

    with open(data_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # Save tokenizer locally for reproducibility (IMPORTANT)
    tok_dir = data_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tok_dir)

    print("=" * 70)
    print("Prepared HF-tokenized dataset:")
    print(f"  data_dir: {data_dir}")
    print(f"  hf_model: {args.hf_model}")
    print(f"  train_bin: {train_bin.name} (uint32)")
    print(f"  val_bin  : {val_bin.name}   (uint32)")
    print(f"  tokenizer saved to: {tok_dir}")
    print(f"  seq_len={seq_len}, block_size={block_size}")
    print(f"  pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")
    print(f"  pad_token={tokenizer.pad_token!r}, eos_token={tokenizer.eos_token!r}")
    print(f"  changed_pad_token={changed_pad}, tokenizer_len={len(tokenizer)}")
    if old_pad_id is not None:
        print(f"  old_pad_token_id={old_pad_id}, old_pad_equal_eos={old_equal}")
    print("=" * 70)


if __name__ == "__main__":
    main()