#!/usr/bin/env python3
"""Evaluation entrypoint for on-policy Search R1 checkpoints.

This script mirrors the training launcher setup from:
examples/sglang_multiturn/search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh
but runs validation only by enabling:
  - trainer.val_before_train=True
  - trainer.val_only=True
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_data_paths() -> tuple[Path, Path]:
    train_data = Path(os.environ.get("TRAIN_DATA", "~/data/searchR1_processed_direct/train.parquet")).expanduser()
    val_data = Path(os.environ.get("VAL_DATA", "~/data/searchR1_processed_direct/test.parquet")).expanduser()
    return train_data, val_data


def _setup_wandb_env() -> None:
    """Mirror the W&B setup used by the Search R1 training launchers."""
    wandb_api_key_search_r1 = os.environ.get("WANDB_API_KEY_SEARCH_R1", "")
    if not wandb_api_key_search_r1:
        raise RuntimeError("WANDB_API_KEY_SEARCH_R1 is not set")

    # W&B default cloud (no custom host). Avoid literal "None" from inherited env.
    os.environ.pop("WANDB_BASE_URL", None)
    os.environ.pop("WANDB_ENTITY", None)
    os.environ["WANDB_API_KEY"] = wandb_api_key_search_r1


def _resolve_task_data_source(df: pd.DataFrame, task: str) -> str:
    """Match --task to parquet ``data_source`` (preprocessor uses ``searchR1_<dataset>``)."""
    task = task.strip()
    if "data_source" not in df.columns:
        raise ValueError("`data_source` column missing in validation parquet.")

    col = df["data_source"].dropna().astype(str)
    if col.eq(task).any():
        return task
    if not task.startswith("searchR1_"):
        prefixed = f"searchR1_{task}"
        if col.eq(prefixed).any():
            return prefixed
    available = sorted(col.unique().tolist())
    raise ValueError(
        f"No rows found for task '{task}' in validation data. "
        f"Use a short name (e.g. nq → searchR1_nq) or the full data_source string. "
        f"Available data_source values: {available}"
    )


def _is_all_task_selector(task_selector: str) -> bool:
    return task_selector.strip().lower() == "all"


def _filter_eval_data_by_task(val_path: Path, task_selector: str) -> Path:
    if not val_path.exists():
        raise FileNotFoundError(f"Validation parquet not found: {val_path}")

    if _is_all_task_selector(task_selector):
        return val_path

    requested_tasks = [t.strip() for t in task_selector.split(",") if t.strip()]
    if not requested_tasks:
        raise ValueError("`--task` cannot be empty. Use a task id, comma list, or 'all'.")

    df = pd.read_parquet(val_path)
    if "data_source" not in df.columns:
        raise ValueError(
            f"`data_source` column not found in {val_path}. "
            "Cannot filter by --task. Please provide a task-filtered parquet via --val-data."
        )

    resolved_tasks = [_resolve_task_data_source(df, task) for task in requested_tasks]
    task_df = df[df["data_source"].isin(resolved_tasks)]

    selector_slug = "__".join(resolved_tasks)
    tmp_dir = Path(tempfile.mkdtemp(prefix="search_r1_eval_subset_"))
    safe = selector_slug.replace("/", "_").replace(":", "_")
    out_path = tmp_dir / f"{safe}.parquet"
    task_df.to_parquet(out_path, index=False)
    return out_path


def _resolve_n_gpus_per_node(default: int = 8) -> int:
    """GPUs per node for ``trainer.n_gpus_per_node``.

    If ``CUDA_VISIBLE_DEVICES`` is set, use its length (so a restricted slice still matches Ray).
    Otherwise use ``default`` (8 for multi-GPU eval machines).
    """
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible_devices:
        return len([x for x in cuda_visible_devices.split(",") if x.strip()])
    return default


def _build_base_overrides(
    checkpoint: str,
    train_data: Path,
    val_data: Path,
    n_gpus: int,
    tool_config: Path,
) -> list[str]:
    return [
        "algorithm.adv_estimator=grpo",
        "data.train_batch_size=512",
        "data.val_batch_size=256",
        "data.max_prompt_length=4096",
        "data.max_response_length=3000",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.return_raw_chat=True",
        f"actor_rollout_ref.model.path={checkpoint}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.max_model_len=15000",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
        "actor_rollout_ref.rollout.n=5",
        "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4",
        "actor_rollout_ref.rollout.multi_turn.max_user_turns=4",
        "actor_rollout_ref.rollout.multi_turn.format=hermes",
        "actor_rollout_ref.rollout.multi_turn.max_tool_response_length=16384",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.val_before_train=True",
        "trainer.val_only=True",
        "trainer.test_freq=0",
        "trainer.save_freq=0",
        'trainer.logger=["console","wandb"]',
        "trainer.project_name=search_r1_like_async_rl_eval",
        f"trainer.experiment_name=eval-SearchR1-{Path(checkpoint).name}",
        f"trainer.n_gpus_per_node={n_gpus}",
        "trainer.nnodes=1",
        "trainer.log_val_generations=32",
        "trainer.validation_data_dir=/tmp/val_dump",
        f"data.train_files={train_data}",
        f"data.val_files={val_data}",
        f"actor_rollout_ref.rollout.multi_turn.tool_config_path={tool_config}",
        "trainer.total_epochs=0",
        "trainer.total_training_steps=0",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation-only eval for Search R1 checkpoints.")
    parser.add_argument("--checkpoint", required=True, help="Model/checkpoint path to evaluate.")
    parser.add_argument(
        "--task",
        required=True,
        help=(
            "Task selector for validation parquet filtering. "
            "Use a short name (nq), full data_source (searchR1_nq), comma list "
            "(nq,hotpotqa), or 'all' for one-pass evaluation over the full validation parquet."
        ),
    )
    parser.add_argument("--train-data", default=None, help="Optional train parquet path.")
    parser.add_argument("--val-data", default=None, help="Optional val parquet path to filter by task.")
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override (can be passed multiple times).",
    )
    args, unknown = parser.parse_known_args()
    _setup_wandb_env()

    repo_root = _repo_root()
    config_path = repo_root / "examples/sglang_multiturn/config"
    tool_config = config_path / "tool_config/search_tool_config.yaml"

    default_train, default_val = _default_data_paths()
    train_data = Path(args.train_data).expanduser() if args.train_data else default_train
    raw_val_data = Path(args.val_data).expanduser() if args.val_data else default_val

    filtered_val_data = _filter_eval_data_by_task(raw_val_data, args.task)

    # main_ppo always constructs train/val dataloaders; train_data must exist even for val_only.
    if not train_data.exists():
        print(
            f"Train parquet not found ({train_data}); using selected val parquet for train loader bootstrap.",
            file=sys.stderr,
        )
        train_data = filtered_val_data

    n_gpus = _resolve_n_gpus_per_node()
    overrides = _build_base_overrides(
        checkpoint=args.checkpoint,
        train_data=train_data,
        val_data=filtered_val_data,
        n_gpus=n_gpus,
        tool_config=tool_config,
    )
    overrides.extend(args.extra_override)
    overrides.extend(unknown)

    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={config_path}",
        "--config-name=search_multiturn_grpo",
        *overrides,
    ]

    print("Launching evaluation command:")
    print(" ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()