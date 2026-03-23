#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# check if the wandb api key is set
if [ -z "$WANDB_API_KEY_SEARCH_R1" ]; then
    echo "WANDB_API_KEY_SEARCH_R1 is not set"
    exit 1
fi

# set the wandb api key
export WANDB_API_KEY=$WANDB_API_KEY_SEARCH_R1

# set the cuda visible devices
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

EXPERIMENT_TS="$(date +%Y%m%d_%H%M%S)"
echo "Running experiment at ${EXPERIMENT_TS}"

# Quote the Hydra override as one argument so it always passes through intact.
nohup bash examples/sglang_multiturn/search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh \
    "trainer.experiment_name=qwen2.5-3b-it_rm-searchR1-like-sgl-multiturn-${EXPERIMENT_TS}" \
    >"logs/searchR1-like${EXPERIMENT_TS}.log" 2>&1 &
