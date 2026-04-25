# On-policy Search R1

This directory holds the on-policy Search R1 example. For search rollouts you need **preprocessed Search R1 Parquet files** (see below) and the **local dense retriever** HTTP service that ships under `examples/sglang_multiturn/search_r1_like/local_dense_retriever/`.

## verl Python environment

Create the Conda environment used by the data preprocessor and training entrypoints:

**1. Create and activate the environment**

```bash
conda create -n verl python=3.12
conda activate verl
```

**2. Install vLLM, SGLang, and optional Megatron**

From the **repository root**, with the `verl` Conda environment active:

```bash
# Make sure you have activated verl conda env

# If you need to run with megatron
bash scripts/install_vllm_sglang_mcore.sh

# Or if you simply need to run with FSDP
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

Run **one** of the `install_vllm_sglang_mcore.sh` variants above (Megatron stack vs. FSDP-only).

**3. Install verl in editable mode**

Still from the **repository root**, with the same environment active:

```bash
pip install --no-deps -e .
```

The preprocessor imports `verl.utils.hdfs_io` and expects `pandas` and `huggingface_hub` (usually already pulled in by the install script).

## Search R1 dataset (preprocess)

The multiturn launchers under `examples/sglang_multiturn/search_r1_like/` expect `train.parquet` and `test.parquet` on disk. By default they read from `~/data/searchR1_processed_direct/` (override with `TRAIN_DATA` and `VAL_DATA` if you use another path).

Use the **`verl` Conda environment** from the previous section so `import verl` succeeds.

From the **repository root**, run:

```bash
cd /path/to/verl_search
conda activate verl
python3 examples/data_preprocess/preprocess_search_r1_dataset.py
```

That script downloads `train.parquet` / `test.parquet` from Hugging Face (default repo `PeterJinGo/nq_hotpotqa_train`), converts each row to the Search-R1-like multiturn schema (system/user prompt, `extra_info`, `tools_kwargs` for search), and writes processed files to `--local_dir` (default `~/data/searchR1_processed_direct/`).

Useful options:

- `--hf_repo_id` — Hugging Face dataset repo id (default `PeterJinGo/nq_hotpotqa_train`).
- `--local_dir` — where to write `train.parquet` and `test.parquet` (default `~/data/searchR1_processed_direct`).
- `--hdfs_dir` — optional; after local write, copy the processed directory to this HDFS path via verl’s HDFS helpers.

## Local dense retriever

### Install

From the **repository root**:

```bash
cd /path/to/verl_search
bash examples/sglang_multiturn/on_policy_search_r1/install_local_retrieval.sh
```

That script:

- Installs Miniconda under `$HOME/miniconda3` (if not already present) and creates a Conda environment named `retriever` with Python 3.10.
- Installs PyTorch (CUDA 12.8 wheels), `transformers`, `datasets`, `pyserini`, `huggingface_hub`, GPU FAISS, `uvicorn`, and `fastapi`.
- Downloads the FAISS index and Wikipedia JSONL corpus via `examples/sglang_multiturn/search_r1_like/local_dense_retriever/download.py`, merges index shards, and decompresses the corpus.

The installer comments target **NVIDIA Blackwell (e.g. B200, sm_100)** with a recent driver and CUDA 12.x user-mode libraries. Adjust the stack if your GPU differs.

Both `install_local_retrieval.sh` and `start_local_retrieval.sh` use `save_path=/mnt/task_runtime/data` for the index and corpus. Change that path in both scripts if you want data stored elsewhere.

### Start the server

After installation completes, still from the **repository root**:

```bash
bash examples/sglang_multiturn/on_policy_search_r1/start_local_retrieval.sh
```

This activates the `retriever` Conda environment and runs `retrieval_server.py` with the E5 retriever, merged FAISS index, and decompressed corpus, with **FAISS on GPU** (`--faiss_gpu`).

The FastAPI app is served by Uvicorn at **http://0.0.0.0:8000** (reachable as `http://localhost:8000` from the same machine).

If Conda is missing, run `examples/sglang_multiturn/on_policy_search_r1/install_local_retrieval.sh` first or install Miniconda/Anaconda and create an environment that matches what `start_local_retrieval.sh` expects (`conda activate retriever`).

## Example: on-policy Search R1 (Qwen2.5-3B-Instruct) with local retrieval

The multiturn launcher uses `examples/sglang_multiturn/config/tool_config/search_tool_config.yaml`, which defaults to **`http://127.0.0.1:8000/retrieve`** — the same URL path the local dense retriever exposes when you run `start_local_retrieval.sh`. Keep that server running for the whole training job.

**1. One-time setup**

- **Training stack:** follow [verl Python environment](#verl-python-environment) and [Search R1 dataset (preprocess)](#search-r1-dataset-preprocess) so `train.parquet` / `test.parquet` exist (default `~/data/searchR1_processed_direct/`).
- **Retriever:** follow [Local dense retriever](#local-dense-retriever) — run `install_local_retrieval.sh` once, then use `start_local_retrieval.sh` whenever you train.

**2. Start the local retriever (leave this running)**

From the **repository root**, in a dedicated shell:

```bash
cd /path/to/verl_search
bash examples/sglang_multiturn/on_policy_search_r1/start_local_retrieval.sh
```

**3. Launch on-policy Search R1 (3B)**

In another shell, from the **repository root**, activate the **`verl`** Conda environment (not `retriever`), set Weights & Biases, then run the wrapper script (it starts `run_qwen2.5-3b_instruct_search_multiturn.sh` in the background with `nohup` and logs under `logs/`):

```bash
cd /path/to/verl_search
conda activate verl
export WANDB_API_KEY_SEARCH_R1="<your_wandb_api_key>"
bash examples/sglang_multiturn/on_policy_search_r1/search_r1_qwen_2_5_3b_instruct.sh
```

The script sets `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`; adjust the script if you use fewer GPUs. If your Parquet files are not under `~/data/searchR1_processed_direct/`, set `TRAIN_DATA` and `VAL_DATA` before running (they are read by `run_qwen2.5-3b_instruct_search_multiturn.sh`). If the retriever listens on another host or port, change `retrieval_service_url` in `examples/sglang_multiturn/config/tool_config/search_tool_config.yaml` (or supply an equivalent Hydra override) so it matches your server.

## Combine rollout JSONL files

Use `examples/sglang_multiturn/on_policy_search_r1/combine_rollout_jsonl.py` to merge rollout rows by the same `input` + `gts` while collecting different `output`, `score`, `step`, and `acc` values.

From the repository root:

```bash
cd /path/to/verl_search
python examples/sglang_multiturn/on_policy_search_r1/combine_rollout_jsonl.py \
  --input-dir ~/data/searchR1_processed_direct/<experiment>/rollout_data
```

This scans `--input-dir` recursively for `*.jsonl` files and writes:

- default output: `<input-dir>/combined_rollouts.jsonl`

You can also pass a specific list of JSONL files. If `--jsonl-files` is provided, it takes precedence over `--input-dir`:

```bash
python examples/sglang_multiturn/on_policy_search_r1/combine_rollout_jsonl.py \
  --jsonl-files \
  ~/data/searchR1_processed_direct/<experiment>/rollout_data/1.jsonl \
  ~/data/searchR1_processed_direct/<experiment>/rollout_data/5.jsonl
```

Optional custom output path:

```bash
python examples/sglang_multiturn/on_policy_search_r1/combine_rollout_jsonl.py \
  --input-dir ~/data/searchR1_processed_direct/<experiment>/rollout_data \
  --output-file ~/data/searchR1_processed_direct/<experiment>/combined_subset.jsonl
```

python examples/sglang_multiturn/on_policy_search_r1/combine_rollout_jsonl.py \
  --jsonl-files \
  ~/data/searchR1_processed_direct/qwen2.5-3b-it_rm-searchR1-like-sgl-multiturn-20260420_005250/rollout_data/1.jsonl \
  ~/data/searchR1_processed_direct/qwen2.5-3b-it_rm-searchR1-like-sgl-multiturn-20260420_005250/rollout_data/5.jsonl \
  --output-file ./outputs/combined_rollouts.jsonl \

## Evaluation-only script (multiple tasks/checkpoints)

Use `eval_r1_qwen_2_5_3b_instruct.sh` when you want to run evaluation only (no RL training).
The launcher loops over checkpoint/task-selector pairs and calls `examples/sglang_multiturn/on_policy_search_r1/eval.py`.
`eval.py --task` supports:

- `all` (one pass over the full validation parquet)
- single short/full task id (`nq` or `searchR1_nq`)
- comma list (`nq,hotpotqa,musique`)

From the repository root:

```bash
cd /path/to/verl_search
conda activate verl

# Optional: override defaults from eval_r1_qwen_2_5_3b_instruct.sh.
# Default task selector: "all" (single eval.py run per checkpoint across all data_source values in test.parquet).
# Default checkpoint: Qwen/Qwen2.5-3B-Instruct.
# export EVAL_TASKS="all"
# export EVAL_TASKS="nq,hotpotqa"
# export EVAL_CHECKPOINTS="/path/to/ckpt-100,/path/to/ckpt-200"

# Optional Hydra overrides forwarded to eval.py (same as training overrides)
# Example: validation sampling (see actor_rollout_ref.rollout.val_kwargs in rollout config)
# export EVAL_COMMON_ARGS="actor_rollout_ref.rollout.val_kwargs.temperature=0.7 actor_rollout_ref.rollout.val_kwargs.do_sample=true"
# Or: export EVAL_COMMON_ARGS="--extra-override data.val_batch_size=128 --extra-override trainer.nnodes=1"

bash examples/sglang_multiturn/on_policy_search_r1/eval_r1_qwen_2_5_3b_instruct.sh

EVAL_CHECKPOINTS=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo bash examples/sglang_multiturn/on_policy_search_r1/eval_r1_qwen_2_5_3b_instruct.sh
```

The script writes one log per checkpoint: `logs/eval_r1_qwen2.5_3b_<checkpoint_slug>_<timestamp>.log` (checkpoint path or HF id is sanitized for the filename).