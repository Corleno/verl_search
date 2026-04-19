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
