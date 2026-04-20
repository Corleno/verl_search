#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# check if the wandb api key is set
if [ -z "$WANDB_API_KEY_SEARCH_R1" ]; then
    echo "WANDB_API_KEY_SEARCH_R1 is not set"
    exit 1
fi

# W&B: default cloud (no custom host). Unset avoids the literal string "None", which breaks the client.
unset WANDB_BASE_URL
unset WANDB_ENTITY
export WANDB_API_KEY="${WANDB_API_KEY_SEARCH_R1}"

# set the cuda visible devices
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

EXPERIMENT_TS="$(date +%Y%m%d_%H%M%S)"
echo "Running experiment at ${EXPERIMENT_TS}"

EXPERIMENT_SLUG="qwen2.5-3b-it_rm-searchR1-like-sgl-multiturn-${EXPERIMENT_TS}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${HOME}/data/searchR1_processed_direct/${EXPERIMENT_SLUG}/rollout_data}"
mkdir -p "${ROLLOUT_DATA_DIR}"
echo "Rollout data directory: ${ROLLOUT_DATA_DIR}"

# Quote each Hydra override as one argument so they pass through intact.
nohup bash examples/sglang_multiturn/search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh \
    "trainer.experiment_name=${EXPERIMENT_SLUG}" \
    "trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}" \
    >"logs/${EXPERIMENT_SLUG}.log" 2>&1 &
