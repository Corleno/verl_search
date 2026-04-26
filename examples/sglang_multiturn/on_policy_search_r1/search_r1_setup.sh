#!/usr/bin/env bash
# Install training deps and preprocess Search R1 data (run from anywhere).
# Prereq: `conda create -n verl python=3.12 -y` and `conda activate verl` — see README.md.
# Megatron stack: USE_MEGATRON=1 bash path/to/this/script.sh
# FSDP only (default): bash path/to/this/script.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${USE_MEGATRON:-0}" == "1" ]]; then
  bash scripts/install_vllm_sglang_mcore.sh
else
  USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
fi

pip install --no-deps -e .
python3 examples/data_preprocess/preprocess_search_r1_dataset.py
