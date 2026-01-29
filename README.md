
# Multi-Stage Planning from Single-Stage Data

---

## 1. Environment setup

From the project root:

```bash
# 1) Create the conda environment from the explicit spec
conda create -n MSPSSDRL --file spec-file.txt

# 2) Activate it
conda activate MSPSSDRL

# 3) Install/upgrade extra pip dependencies
python -m pip install -U pip
python -m pip install -r requirements-extra.txt
```

---

## 2. Task and data format

### 2.1 GraphA task (multi-stage DAG)
- Nodes are partitioned into stages: `S1, S2, ..., SK`.
- The graph is a DAG (directed acyclic graph).
- Typical setting (when stage skipping is disabled in data generation) allows:
  - directed edges within a stage (acyclic)
  - directed edges only from `Si` to `S(i+1)` between stages (no long jumps)

### 2.2 Dataset line format (`train_*.txt` / `test.txt`)
Each line is a space-separated integer sequence:

```text
src  dst  path_node_0  path_node_1 ... path_node_L
```

- `src dst`: the query / condition (reachability from `src` to `dst`)
- `path_node_*`: one valid path from `src` to `dst` (typically includes both endpoints)

### 2.3 Prompt convention
- **nanoGPT (token-level):** prompt tokens `[src, dst, src]`, then the model generates subsequent nodes until newline.
- **Qwen (HF tokenizer):** prompt text `"src dst src "` (note the trailing space), then the model generates subsequent nodes until EOS.

---

## 3. Step A — Generate a GraphA DAG

Script: `data/simple_graph/create_graph.py`  
Outputs (under `--output_root`):
- `composition_graph.graphml`
- `stage_info.pkl`
- `metadata.json`

### 3.1 Example: nanoGPT-style graph (K=3)
```bash
python data/simple_graph/create_graph.py \
  --nodes_per_stage 30 \
  --num_stages 3 \
  --p_global 0.30 \
  --seed 42 \
  --experiment_name graphA \
  --output_root data/graphs
```

### 3.2 Example: Qwen-style graph (K=5)
```bash
python data/simple_graph/create_graph.py \
  --nodes_per_stage 30 \
  --num_stages 5 \
  --p_global 0.10 \
  --seed 42 \
  --experiment_name graphA \
  --output_root data/graphs
```

After running, pick the generated folder path and save it as:

```bash
GRAPH_DIR="data/graphs/<your_generated_graph_folder>"
```

---

## 4. Step B — Generate all-pairs dataset (reachable pairs + sampled paths)

Script: `data/simple_graph/generate_alpine_allpairs.py`

You must provide:
- `--input_graph  ${GRAPH_DIR}/composition_graph.graphml`
- `--stage_info   ${GRAPH_DIR}/stage_info.pkl`
- `--output_dir   <where_to_write_dataset>`

Example:

```bash
GRAPH_DIR="data/graphs/<your_generated_graph_folder>"
DATASET_DIR="data/datasets/graphA_full"

python data/simple_graph/generate_alpine_allpairs.py \
  --input_graph "${GRAPH_DIR}/composition_graph.graphml" \
  --stage_info "${GRAPH_DIR}/stage_info.pkl" \
  --output_dir "${DATASET_DIR}" \
  --train_paths_per_pair 20 \
  --eval_paths_per_pair 1 \
  --train_ratio 0.85 \
  --seed 42
```

The dataset directory typically contains:
- `train_20.txt`
- `test.txt`
- `dataset_summary.json`
- `composition_graph.graphml`
- `stage_info.pkl`

---

## 5. Step C — Optional dataset variants

### 5.1 (K=3 only) P13 ratio control (mix P0 and P13 arbitrarily)

Script: `nanoGPT/make_p13_variant.py`

Scope/definition (K=3):
- **P13:** pairs from stage `S1` to stage `S3`
- **P0:** all other pairs (non-P13)

You can generate training sets with any mix ratio between P0 and P13.

Example: P13 ratio = 0% (pure P0)
```bash
python nanoGPT/make_p13_variant.py \
  --src-dir  data/datasets/graphA_full \
  --dest-dir data/datasets/graphA_full_P13_0 \
  --target-ratio 0.0 \
  --paths-per-pair 20 \
  --seed 1234
```

Example: P13 ratio = 20%
```bash
python nanoGPT/make_p13_variant.py \
  --src-dir  data/datasets/graphA_full \
  --dest-dir data/datasets/graphA_full_P13_20 \
  --target-ratio 0.2 \
  --paths-per-pair 20 \
  --seed 1234
```

Example: P13 ratio = 100% (pure P13)
```bash
python nanoGPT/make_p13_variant.py \
  --src-dir  data/datasets/graphA_full \
  --dest-dir data/datasets/graphA_full_P13_100 \
  --target-ratio 1.0 \
  --paths-per-pair 20 \
  --seed 1234
```

---

### 5.2 (Any K) Filter training pairs by maximum stage jump (max_jump)

Script: `Qwen2.5-3b/make_no_stage_skip_train.py`

This script is **not limited to adjacent-only training**.  
You can choose a maximum stage jump:

- `--max-jump 1`: keep only gap=1 (adjacent stages)
- `--max-jump 2`: keep both gap=1 and gap=2; remove all gap>2
- In general: keep pairs with `(dst_stage - src_stage) <= max_jump`

Example: keep only gap ≤ 1
```bash
python Qwen2.5-3b/make_no_stage_skip_train.py \
  --src-dir  data/datasets/graphA_full \
  --dest-dir data/datasets/graphA_train_maxjump1 \
  --paths-per-pair 20 \
  --max-jump 1 \
  --seed 42
```

Example: keep gap ≤ 2 (retain both gap=1 and gap=2; remove gap>2)
```bash
python Qwen2.5-3b/make_no_stage_skip_train.py \
  --src-dir  data/datasets/graphA_full \
  --dest-dir data/datasets/graphA_train_maxjump2 \
  --paths-per-pair 20 \
  --max-jump 2 \
  --seed 42
```

---

## 6. nanoGPT pipeline (commonly K=3)

Below uses `data/datasets/graphA_full_P13_0` as an example dataset directory.  
You can replace it with any dataset folder you generated.

### 6.1 Preprocess to `.bin` + `meta.pkl`
```bash
python nanoGPT/prepare_composition.py \
  --data_dir data/datasets/graphA_full_P13_0 \
  --train_paths_per_pair 20 \
  --total_nodes 90 \
  --block_multiple 32
```

Notes:
- `total_nodes = nodes_per_stage * num_stages` (e.g., 30 * 3 = 90)

### 6.2 SFT training
```bash
python nanoGPT/nanoGPT_sft.py \
  --data_dir data/datasets/graphA_full_P13_0 \
  --train_paths_per_pair 20 \
  --device cuda:0 \
  --max_iters 50000 \
  --test_interval 1000 \
  --checkpoint_interval 5000
```

### 6.3 Policy Gradient (PG)
```bash
python nanoGPT/nanoGPT_pg.py \
  --data_dir data/datasets/graphA_full_P13_0 \
  --sft_checkpoint out/<your_sft_run_dir>/ckpt_5000.pt \
  --train_paths_per_pair 20 \
  --device cuda:0 \
  --max_iters 20000 \
  --eval_interval 1000 \
  --save_interval 2000
```

### 6.4 Q-learning (example with process-shaping rewards)

The intended point here is to use **process shaping + KL regularization** (dense intermediate feedback), e.g.:
- small reward for a valid edge transition
- penalty for invalid transitions/tokens
- terminal reward for reaching the target
- KL regularization anchors the policy to the SFT reference and helps prevent goal-agnostic drift.

If your script exposes explicit shaping flags, an example invocation looks like:

```bash
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
```

If your local copy uses different argument names, check the supported options via:

```bash
python nanoGPT/nanoGPT_Qlearning.py -h
```

## 6.5 Behavior diagnostics (Event-Chain / Stage-Event-Chain analysis, **K = 3**)

This section explains the **event-chain decomposition** used to diagnose *how* and *where* a model succeeds or fails on GraphA when **K = 3**.

### 6.5.1 What this analysis is for

For **K = 3**, GraphA induces three evaluation task types (determined purely by the stages of the source/target nodes):

- **`s1s3`**: $$S_1 \rightarrow S_3$$ (multi-stage; typically requires passing through $$S_2$$)
- **`s1s2`**: $$S_1 \rightarrow S_2$$ (single-stage)
- **`s2s3`**: $$S_2 \rightarrow S_3$$ (single-stage)

The event-chain analyzer:

1. **Iterates over multiple checkpoints** (e.g., every 2k steps).
2. For each checkpoint, for each selected task type, it:
   - enumerates the corresponding **(source, target)** pairs from `test.txt`,
   - runs model **decoding** for each pair,
   - validates the decoded token sequence as a path on the directed graph,
   - assigns a **behavior label** (e.g., `SUCCESS`, `INVALID_EDGE`, `OVER_SHOOT`, …),
   - decomposes performance into a **stage event chain** (A/B/C/D or A′/B′/C′/D′ depending on task).
3. Writes **aggregate CSV summaries** (and optionally **per-pair CSVs**) to make failure modes easy to quantify and compare across checkpoints and training methods (SFT/PG/Q-learning).

---

### 6.5.2 What the script outputs

For each `(checkpoint_step, task_type)` it produces:

- **`behavior_summary.csv`**  
  Aggregate behavior statistics: how many pairs fall into each behavior class (`SUCCESS`, `INVALID_EDGE`, `OVER_SHOOT`, …), plus rates and basic decoded-path statistics.

- **`phase_summary.csv`**  
  Aggregate **event-chain** statistics for the corresponding task type:
  - common diagnostic fields (success rate, stop-token-related rates, stage-availability rates, etc.)
  - task-specific event-chain fields:
    - `s1s3`: Event **A/B/C/D**
    - `s1s2`: Event **A′/B′/C′/D′**
    - `s2s3`: Event **A/B/C**

Optionally (for case studies / visualization):

- **`per_pair_step_{step}_{task}.csv`** *(only when `--save-per-pair` is enabled)*  
  Per-(src,tgt) detailed decoding traces and annotations (legal/illegal first action, whether a bridge stage was visited, stop position/reason, etc.).

---

### 6.5.3 Data directory requirements (`--data-dir`)

The analyzer expects `--data-dir` to contain the following files:

1. **`meta.pkl`**  
   Produced by the nanoGPT preprocessing pipeline. Must include at least:
   - `stoi` / `itos`
   - `vocab_size`
   - (and any other fields your model loader expects)

   **Important assumption:** the script treats the **stop token** as the newline token:
   - stop token id = `meta["stoi"]["\n"]`

2. **`stage_info.pkl`**  
   Must contain a structure like:
   ```python
   {"stages": [S1, S2, S3, ...]}
   ```
   where each `Si` is a list of **node ids** belonging to that stage.

3. **`composition_graph.graphml`**  
   A directed GraphML graph for the environment.
   - Nodes are typically stored as **strings** (e.g., `"17"`), and the script will typically do `int(node)` internally.

4. **`test.txt`**  
   One test example per line, formatted as space-separated integers:
   ```text
   src  tgt  path_token_1  path_token_2  ... 
   ```
   where:
   - `src`, `tgt`, and `path_token_*` are all integer node ids
   - the analyzer primarily uses `(src, tgt)` as the evaluation condition (the provided path tokens are typically not required for decoding, but the file must be present in this expected format)

---

### 6.5.4 Checkpoint directory requirements (`--checkpoints-dir`)

`--checkpoints-dir` should contain the model weight files you want to analyze.

By default, the analyzer derives the filename pattern from `--run-type`:

- `--run-type sft`  → `ckpt_{step}.pt`
- `--run-type pg`   → `ckpt_pg_{step}.pt`
- `--run-type ql`   → `ckpt_ql_{step}.pt`

You can override this by specifying your own pattern:

- `--ckpt-pattern 'my_prefix_{step}.pt'`

where `{step}` will be replaced with an integer step value (from `--step-start` to `--step-end`).

---

### 6.5.5 How decoding + labeling works

For each `(src, tgt)` pair:

1. Build a prompt in the same convention as nanoGPT training (typically `[src, tgt, src]`).
2. Decode up to `--max-new-tokens` using the specified sampling controls:
   - `--temperature`
   - `--top-k`
3. Interpret the generated tokens as a proposed node sequence (terminated by the stop token `"\n"`).
4. Validate the sequence against:
   - **graph legality** (each consecutive transition must be an edge; otherwise `INVALID_EDGE`)
   - **task target** (must reach `tgt` for success)
   - **termination behavior** (stop too early / never stop / overshoot, etc.)
5. Record:
   - a **behavior class** (for `behavior_summary.csv`)
   - a **stage event-chain signature** (for `phase_summary.csv`)

Exact event definitions and behavior labels are implemented in the analyzer script, but conceptually:
- **behavior classes** capture *what went wrong overall* (illegal transition, stopped early, overshot target, etc.)
- **event-chain fields** break success into *conditional steps* (“did A happen?”, “given A, did B happen?”, …), allowing you to localize failure points (e.g., “model often reaches Stage 2 but fails to terminate correctly”).

---

### 6.5.6 How to run

Depending on your repo, the script may be named differently. To make commands copy-paste friendly, set:

```bash
ANALYZER="nanoGPT/analyze_event_chain.py"  
```

#### (1) Single task (default is often `s1s3`)

```bash
python "${ANALYZER}" \
  --data-dir data/datasets/graphA_full_P13_0 \
  --checkpoints-dir out/sft_run \
  --run-type sft \
  --step-start 10000 --step-end 50000 --step-interval 10000 \
  --output-dir out_event_chain/sft_s1s3 \
  --max-new-tokens 32 \
  --temperature 0.0 \
  --top-k 0 \
  --device cuda:0 \
```

#### (2) Run all task types in one pass (**recommended**)

```bash
python "${ANALYZER}" \
  --data-dir data/datasets/graphA_full_P13_0 \
  --checkpoints-dir out/qlearning_run \
  --run-type ql \
  --task-types all \
  --step-start 2000 --step-end 20000 --step-interval 2000 \
  --output-dir out_event_chain/ql_all \
  --max-new-tokens 32 \
  --temperature 0.0 \
  --top-k 0 \
  --device cuda:0 \
  --progress
```

#### (3) Run only a subset of tasks

```bash
python "${ANALYZER}" \
  --data-dir data/datasets/graphA_full_P13_0 \
  --checkpoints-dir out/pg_run \
  --run-type pg \
  --task-types s1s3,s2s3 \
  --step-start 2000 --step-end 20000 --step-interval 2000 \
  --output-dir out_event_chain/pg_s1s3_s2s3 \
  --progress
```

#### (4) Save per-pair CSVs (for case study / visualization)

This can generate large files; it is common to run it on a **single checkpoint**:

```bash
python "${ANALYZER}" \
  --data-dir data/datasets/graphA_pg020_tier3 \
  --checkpoints-dir out/sft_run \
  --run-type sft \
  --task-types s1s3 \
  --step-start 50000 --step-end 50000 --step-interval 1 \
  --output-dir out_event_chain/per_pair_demo \
  --save-per-pair \
  --progress
```

---

### 6.5.7 Output file details

#### A) `behavior_summary.csv`

Each row corresponds to one `(step, task_type)`.

Typical fields include:

- `step`: checkpoint step analyzed
- `task_type`: `s1s3`, `s1s2`, or `s2s3`
- `num_pairs`: number of evaluated (src,tgt) pairs for this task
- decoded path statistics such as:
  - `avg_path_length`
  - `avg_stage2_count` (for tasks where Stage 2 is relevant)
- per-class counts and rates, e.g.:
  - `count_SUCCESS`, `rate_SUCCESS`
  - `count_INVALID_EDGE`, `rate_INVALID_EDGE`
  - `count_OVER_SHOOT`, `rate_OVER_SHOOT`
  - (exact set depends on the analyzer implementation)

How to interpret:
- A decreasing `INVALID_EDGE` rate often indicates improved local transition modeling.
- A high “stop-related failure” rate indicates termination control issues (e.g., not emitting newline at the right time).

#### B) `phase_summary.csv`

Each row corresponds to one `(step, task_type)` and contains:

1. **Common diagnostic fields** (examples):
   - `success_rate`
   - `stage2_available_rate`
   - `stage2_stop_token_rate`
   - `stage3_stop_success_rate`
   - (exact column names are script-defined)

2. **Task-specific event-chain fields**
   - For `s1s3`: Event A/B/C/D chain statistics  
     (including marginal rates and conditional rates such as `eventB_rate_given_A`, etc.)
   - For `s1s2`: Event A′/B′/C′/D′ chain statistics
   - For `s2s3`: Event A/B/C chain statistics

How to interpret:
- Event-chain conditional rates tell you *where the funnel breaks*.  
  For example, a model may have high A-rate (starts legally) but low C-rate-given-B (fails to finish after reaching the intermediate stage).

#### C) `per_pair_step_{step}_{task}.csv` (optional)

When `--save-per-pair` is enabled, you get per example diagnostics such as:

- `(src, tgt)`
- decoded token sequence (as ids and/or raw string)
- whether the **first action** is legal
- whether the trajectory visits a **bridge stage** (e.g., Stage 2 for `s1s3`)
- whether/when it hits Stage 2 / Stage 3 / the target node
- whether the model emitted the **stop token**, and where
- failure reason annotation (illegal edge, early stop, etc.)

This file is intended for:
- manual inspection
- curated failure case studies
- plotting per-pair trajectories

---


## 7. Qwen2.5-3B (HF) pipeline (any K)

Examples below use a filtered dataset directory:
- `data/datasets/graphA_train_maxjump1`

### 7.1 Prepare HF binary data (and save tokenizer locally)
```bash
python Qwen2.5-3b/prepare_qwen.py \
  --data_dir data/datasets/graphA_train_maxjump1 \
  --train_paths_per_pair 20 \
  --hf_model Qwen/Qwen2.5-3B \
  --block_multiple 32 \
  --append_eos \
```

### 7.2 SFT
```bash
python Qwen2.5-3b/qwen_sft.py \
  --data_dir data/datasets/graphA_train_maxjump1 \
  --train_paths_per_pair 20 \
  --device cuda:0 \
  --dtype bf16 \
  --max_iters 2000 \
  --test_interval 100 \
  --checkpoint_interval 500
```

### 7.3 Q-learning
```bash
python Qwen2.5-3b/qwen_Qlearning.py \
  --data_dir data/datasets/graphA_train_maxjump1 \
  --train_paths_per_pair 20 \
  --sft_dir out/<your_qwen_sft_run_dir>/ckpt_<iter> \
  --base_model Qwen/Qwen2.5-3B \
  --device cuda:0
```

### 7.4 GRPO
```bash
python Qwen2.5-3b/qwen_GRPO.py \
  --data_dir data/datasets/graphA_train_maxjump1 \
  --train_paths_per_pair 20 \
  --sft_dir out/<your_qwen_sft_run_dir>/ckpt_<iter> \
  --base_model Qwen/Qwen2.5-3B \
  --device cuda:0
```

For additional reward shaping / masking / rollout controls, run:

```bash
python Qwen2.5-3b/qwen_Qlearning.py -h
python Qwen2.5-3b/qwen_GRPO.py -h
```
This will print the full list of available CLI options and their descriptions.

---

## 8. Suggested run order (quick checklist)

### nanoGPT (K=3, optional P13 mixing)
1. `data/simple_graph/create_graph.py` (K=3)
2. `data/simple_graph/generate_alpine_allpairs.py`
3. (optional) `nanoGPT/make_p13_variant.py`
4. `nanoGPT/prepare_composition.py`
5. `nanoGPT/nanoGPT_sft.py`
6. `nanoGPT/nanoGPT_pg.py` or `nanoGPT/nanoGPT_Qlearning.py`
7. (optional) `nanoGPT/analyze_event_chain.py`

### Qwen (any K, optional max-jump filtering)
1. `data/simple_graph/create_graph.py` (e.g., K=5)
2. `data/simple_graph/generate_alpine_allpairs.py`
3. (optional) `Qwen2.5-3b/make_no_stage_skip_train.py` (choose `--max-jump`)
4. `Qwen2.5-3b/prepare_qwen.py`
5. `Qwen2.5-3b/qwen_sft.py`
6. `Qwen2.5-3b/qwen_Qlearning.py` or `Qwen2.5-3b/qwen_GRPO.py`

---