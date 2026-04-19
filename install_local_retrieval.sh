#!/usr/bin/env bash
# Local dense retriever setup (see start_local_retrieval.sh).
#
# Run this on an NVIDIA B200 (Blackwell, sm_100) machine with a recent driver
# (CUDA 12.x user-mode). PyTorch 2.4 + pytorch-cuda 12.1 wheels do not target
# Blackwell; below we install the Linux cu128 stack instead.

# Download the Miniconda installer script
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# Install to $HOME/miniconda3 in batch mode
bash ~/miniconda.sh -b -p $HOME/miniconda3

# Activate conda (only in the current shell)
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# (Optional) Add conda to your default shell startup
conda init

# Reload shell config
source ~/.bashrc

# Create and activate the retriever environment with Python 3.10
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n retriever python=3.10 -y
conda activate retriever

# Install PyTorch with CUDA 12.8 wheels (Blackwell / B200); bundles its own CUDA user libraries.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install other Python packages
pip install transformers datasets pyserini huggingface_hub

# GPU FAISS: conda-forge `faiss-gpu` is built against NumPy 1.x and breaks under NumPy 2 (`_ARRAY_API`).
# Meta publishes CUDA 12 wheels as `faiss-gpu-cu12` (imports as `faiss`); pair with NumPy 2.x + SciPy 1.14+.
# PyPI wheels do not yet ship SASS for Blackwell (compute capability 10.x); retrieval_server.py falls back to a CPU index there.
# If you are upgrading an env that already has conda FAISS, remove it first:
#   conda remove -y faiss-gpu faiss libfaiss
pip install "numpy>=2.0,<2.4" "scipy>=1.14" "faiss-gpu-cu12==1.14.1.post1"

# Install the API service framework
pip install uvicorn fastapi


save_path=/mnt/task_runtime/data
python examples/sglang_multiturn/search_r1_like/local_dense_retriever/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz


