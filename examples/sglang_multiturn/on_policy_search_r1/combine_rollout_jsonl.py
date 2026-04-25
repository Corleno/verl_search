#!/usr/bin/env python3
"""Combine rollout JSONL rows by shared (input, gts).

Example:
    python combine_rollout_jsonl.py --input-dir <rollout_data_dir> --output-file ./outputs/combined_rollouts.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _stable_json_key(value: Any) -> str:
    """Serialize arbitrary objects into a stable grouping key."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return repr(value)


def _jsonl_sort_key(path: Path) -> tuple[int, Any, str]:
    """Sort numeric step files first (e.g. 1.jsonl, 10.jsonl), then others."""
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), str(path))
    return (1, stem, str(path))


def _iter_jsonl_records(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {file_path} at line {line_no}: {exc}") from exc


def combine_rollouts(jsonl_files: list[Path]) -> tuple[list[dict[str, Any]], int]:
    if not jsonl_files:
        raise FileNotFoundError("No JSONL files were provided.")

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    total_rows = 0

    for jsonl_file in jsonl_files:
        for row in _iter_jsonl_records(jsonl_file):
            total_rows += 1

            input_text = row.get("input")
            gts = row.get("gts")
            key = (str(input_text), _stable_json_key(gts))

            if key not in grouped:
                grouped[key] = {
                    "input": input_text,
                    "gts": gts,
                    "outputs": [],
                    "scores": [],
                    "steps": [],
                    "accs": [],
                    "rollouts": [],
                }

            output = row.get("output")
            score = row.get("score")
            step = row.get("step")
            acc = row.get("acc")

            grouped[key]["outputs"].append(output)
            grouped[key]["scores"].append(score)
            grouped[key]["steps"].append(step)
            grouped[key]["accs"].append(acc)
            grouped[key]["rollouts"].append(
                {
                    "output": output,
                    "score": score,
                    "step": step,
                    "acc": acc,
                }
            )

    merged = []
    for item in grouped.values():
        item["num_rollouts"] = len(item["rollouts"])
        merged.append(item)

    return merged, total_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load rollout JSONL files and combine rows that share "
            "the same input and gts, while collecting output/score/step/acc variations."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing rollout JSONL files (searches recursively).",
    )
    parser.add_argument(
        "--jsonl-files",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Optional explicit JSONL file list. If provided, these files are used "
            "instead of scanning --input-dir."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to <input-dir>/combined_rollouts.jsonl.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    explicit_files = args.jsonl_files or []

    if explicit_files:
        jsonl_files = []
        for file_path in explicit_files:
            resolved = file_path.expanduser().resolve()
            if not resolved.exists() or not resolved.is_file():
                raise FileNotFoundError(f"JSONL file does not exist: {resolved}")
            jsonl_files.append(resolved)
        jsonl_files = sorted(jsonl_files, key=_jsonl_sort_key)
        source_desc = f"{len(jsonl_files)} explicitly provided files"
    else:
        if not input_dir.exists() or not input_dir.is_dir():
            raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
        jsonl_files = sorted(input_dir.rglob("*.jsonl"), key=_jsonl_sort_key)
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found under: {input_dir}")
        source_desc = str(input_dir)

    output_file = args.output_file
    if output_file is None:
        output_file = input_dir / "combined_rollouts.jsonl"
    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    merged_rows, total_rows = combine_rollouts(jsonl_files=jsonl_files)

    with output_file.open("w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Loaded {total_rows} rows from {source_desc}")
    print(f"Combined into {len(merged_rows)} grouped samples")
    print(f"Wrote merged JSONL to: {output_file}")


if __name__ == "__main__":
    main()
