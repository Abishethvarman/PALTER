"""
Shared CLI, metadata, and result persistence helpers for DeceptBench.
"""

import argparse
import csv
import json
from datetime import date
from pathlib import Path


SCORER_VERSION = "1.0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARKS = {
    "factual_deception": (1, "factual_deception"),
    "sycophancy": (2, "sycophancy"),
    "strategic_deception": (3, "strategic_deception"),
    "alignment_faking": (4, "alignment_faking"),
    "omission": (5, "omission"),
    "obfuscation": (6, "obfuscation"),
    "reasoning_deception": (7, "reasoning_deception"),
}


def model_tag(model: str) -> str:
    return model.replace(":", "_").replace("/", "_").replace(".", "_")


def _benchmark_info(benchmark_type: str):
    if benchmark_type not in BENCHMARKS:
        known = ", ".join(sorted(BENCHMARKS))
        raise ValueError(f"Unknown benchmark type {benchmark_type!r}. Known: {known}")
    return BENCHMARKS[benchmark_type]


def _find_dataset(benchmark_type) -> str:
    number, type_name = _benchmark_info(benchmark_type)
    filename = f"_{number}_{type_name}_dataset.json"
    candidates = [
        Path("datasets") / filename,
        Path(filename),
        Path("..") / filename,
        Path("..") / "datasets" / filename,
        PROJECT_ROOT / "datasets" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Dataset for {benchmark_type} not found. Searched:\n  {searched}")


def parse_args(benchmark_type: str) -> argparse.Namespace:
    number, type_name = _benchmark_info(benchmark_type)
    today = date.today().isoformat()

    parser = argparse.ArgumentParser(description=f"Run DeceptBench {number}: {type_name}")
    parser.add_argument("--model", default="llama3.1", help="Ollama model name")
    parser.add_argument("--input", default=None, help="Dataset JSON path")
    parser.add_argument("--output-json", default=None, help="JSON output path")
    parser.add_argument("--output-csv", default=None, help="CSV output path")
    parser.add_argument("--date", default=today, help="Run date in YYYY-MM-DD format")
    args = parser.parse_args()

    args.model_tag = model_tag(args.model)
    if args.input is None:
        args.input = _find_dataset(benchmark_type)
    if args.output_json is None:
        args.output_json = str(
            PROJECT_ROOT / "results" / f"_{number}_{type_name}_{args.model_tag}_{args.date}.json"
        )
    if args.output_csv is None:
        args.output_csv = str(
            PROJECT_ROOT / "results" / f"_{number}_{type_name}_{args.model_tag}_{args.date}.csv"
        )
    return args


def build_metadata(args, benchmark_type, total_items) -> dict:
    return {
        "model": args.model,
        "model_tag": args.model_tag,
        "date": args.date,
        "benchmark_type": benchmark_type,
        "dataset_file": args.input,
        "total_items": total_items,
        "scorer_version": SCORER_VERSION,
    }


def save_results(results, csv_rows, args, benchmark_type, fieldnames):
    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": build_metadata(args, benchmark_type, len(results)),
        "results": results,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"JSON -> {json_path}")
    print(f"CSV -> {csv_path}")
