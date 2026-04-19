# On-policy Search R1

This directory holds the on-policy Search R1 example. For search rollouts you need the **local dense retriever** HTTP service that ships under `examples/sglang_multiturn/search_r1_like/local_dense_retriever/`.

## Local dense retriever

### Install

From the **repository root** (the directory that contains `install_local_retrieval.sh`):

```bash
cd /path/to/verl_search
bash install_local_retrieval.sh
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
bash start_local_retrieval.sh
```

This activates the `retriever` Conda environment and runs `retrieval_server.py` with the E5 retriever, merged FAISS index, and decompressed corpus, with **FAISS on GPU** (`--faiss_gpu`).

The FastAPI app is served by Uvicorn at **http://0.0.0.0:8000** (reachable as `http://localhost:8000` from the same machine).

If Conda is missing, run `install_local_retrieval.sh` first or install Miniconda/Anaconda and create an environment that matches what `start_local_retrieval.sh` expects (`conda activate retriever`).
