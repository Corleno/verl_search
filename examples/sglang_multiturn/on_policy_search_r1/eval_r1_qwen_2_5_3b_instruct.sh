#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

EVAL_PY="${SCRIPT_DIR}/eval.py"
if [[ ! -f "${EVAL_PY}" ]]; then
    echo "Missing eval script: ${EVAL_PY}" >&2
    exit 1
fi

# Task selector(s) for eval.py.
# Supports:
#   - all                           (single eval.py run over full val parquet)
#   - short/full id                 (e.g. nq or searchR1_nq)
#   - comma-separated list          (e.g. nq,hotpotqa,musique)
# Example:
#   export EVAL_TASKS="all"
#   export EVAL_TASKS="nq,hotpotqa,2wikimultihopqa"
EVAL_TASKS="${EVAL_TASKS:-all}"

# Comma-separated list of checkpoint paths/names.
# Example:
#   export EVAL_CHECKPOINTS="/path/to/ckpt-100,/path/to/ckpt-200"
EVAL_CHECKPOINTS="${EVAL_CHECKPOINTS:-Qwen/Qwen2.5-3B-Instruct}"

# Extra args forwarded to eval.py for every run.
# Example:
#   export EVAL_COMMON_ARGS="data.val_batch_size=128 actor_rollout_ref.rollout.temperature=0.0"
# Or use eval.py's repeated --extra-override flag:
#   export EVAL_COMMON_ARGS="--extra-override data.val_batch_size=128 --extra-override trainer.nnodes=1"
EVAL_COMMON_ARGS="${EVAL_COMMON_ARGS:-}"

IFS=',' read -r -a TASKS <<< "${EVAL_TASKS}"
IFS=',' read -r -a CHECKPOINTS <<< "${EVAL_CHECKPOINTS}"
read -r -a COMMON_ARGS <<< "${EVAL_COMMON_ARGS}"

if [[ "${#TASKS[@]}" -eq 0 ]]; then
    echo "No tasks found in EVAL_TASKS." >&2
    exit 1
fi
if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
    echo "No checkpoints found in EVAL_CHECKPOINTS." >&2
    exit 1
fi

# Filesystem-safe slug from checkpoint path or HF id (used in log filenames).
_checkpoint_slug() {
    local s="$1"
    s="${s//\//_}"
    s="${s// /_}"
    s="${s//:/_}"
    s="${s//\[/}"
    s="${s//\]/}"
    if [[ "${#s}" -gt 120 ]]; then
        s="${s:0:120}"
    fi
    echo "${s}"
}

RUN_TS="$(date +%Y%m%d_%H%M%S)"

for ckpt in "${CHECKPOINTS[@]}"; do
    CKPT_SLUG="$(_checkpoint_slug "${ckpt}")"
    for task in "${TASKS[@]}"; do
        # Filesystem-safe task slug for the log filename
        TASK_SLUG="${task//\//_}"
        TASK_SLUG="${TASK_SLUG// /_}"
        TASK_SLUG="${TASK_SLUG//:/_}"
        TASK_SLUG="${TASK_SLUG//\[/}"
        TASK_SLUG="${TASK_SLUG//\]/}"
        if [[ "${#TASK_SLUG}" -gt 60 ]]; then
            TASK_SLUG="${TASK_SLUG:0:60}"
        fi

        LOG_FILE="logs/eval_r1_qwen2.5_3b_${CKPT_SLUG}_${TASK_SLUG}_${RUN_TS}.log"
        echo "Writing eval logs for checkpoint '${ckpt}' task '${task}' to ${LOG_FILE}"
        
        echo "============================================================" | tee -a "${LOG_FILE}"
        echo "Evaluating checkpoint='${ckpt}' task='${task}'" | tee -a "${LOG_FILE}"
        echo "============================================================" | tee -a "${LOG_FILE}"

        python3 "${EVAL_PY}" \
            --checkpoint "${ckpt}" \
            --task "${task}" \
            "${COMMON_ARGS[@]}" \
            "$@" \
            2>&1 | tee -a "${LOG_FILE}"
    done
done
