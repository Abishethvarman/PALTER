#!/usr/bin/env python3
"""
Master runner for DeceptBench.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path


MODELS = [
    "llama3.2:1b",
    "llama3.1",
    "llama3.1:70b",
]

OLLAMA_GPU_DEVICES = "0,1,2"

BENCHMARKS = [
    {
        "type": 1,
        "name": "factual_deception",
        "script": "benchmarks/_1_factual_deception_benchmark.py",
        "dataset": "datasets/_1_factual_deception_dataset.json",
    },
    {
        "type": 2,
        "name": "sycophancy",
        "script": "benchmarks/_2_sycophancy_benchmark.py",
        "dataset": "datasets/_2_sycophancy_dataset.json",
    },
    {
        "type": 3,
        "name": "strategic_deception",
        "script": "benchmarks/_3_strategic_deception_benchmark.py",
        "dataset": "datasets/_3_strategic_deception_dataset.json",
    },
    {
        "type": 4,
        "name": "alignment_faking",
        "script": "benchmarks/_4_alignment_faking_benchmark.py",
        "dataset": "datasets/_4_alignment_faking_dataset.json",
    },
    {
        "type": 5,
        "name": "omission",
        "script": "benchmarks/_5_omission_benchmark.py",
        "dataset": "datasets/_5_omission_dataset.json",
    },
    {
        "type": 6,
        "name": "obfuscation",
        "script": "benchmarks/_6_obfuscation_benchmark.py",
        "dataset": "datasets/_6_obfuscation_dataset.json",
    },
    {
        "type": 7,
        "name": "reasoning_deception",
        "script": "benchmarks/_7_reasoning_deception_benchmark.py",
        "dataset": "datasets/_7_reasoning_deception_dataset.json",
    },
]

ROOT = Path(__file__).resolve().parent


def model_tag(model):
    return model.replace(":", "_").replace("/", "_").replace(".", "_")


def estimate_seconds_per_call(model):
    lowered = model.lower()
    if "1b" in lowered:
        return 1
    if "70b" in lowered:
        return 10
    return 3


def fmt_duration(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def load_json_count(path):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "results" in payload:
        return len(payload["results"])
    return len(payload)


def run_ollama_list():
    if shutil.which("ollama") is None:
        return False, [], "ollama executable not found"
    try:
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", OLLAMA_GPU_DEVICES)
        env.setdefault("OLLAMA_SCHED_SPREAD", "1")
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
    except Exception as exc:
        return False, [], str(exc)
    if result.returncode != 0:
        return False, [], result.stderr.strip() or result.stdout.strip() or "ollama list failed"
    models = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return True, models, ""


def selected_benchmarks(types):
    if not types:
        return BENCHMARKS
    wanted = set(types)
    return [bench for bench in BENCHMARKS if bench["type"] in wanted]


def output_paths(bench, model, run_date):
    tag = model_tag(model)
    stem = f"_{bench['type']}_{bench['name']}_{tag}_{run_date}"
    return ROOT / "results" / f"{stem}.json", ROOT / "results" / f"{stem}.csv"


def log_path(bench, model, run_date):
    tag = model_tag(model)
    return ROOT / "logs" / f"_{bench['type']}_{bench['name']}_{tag}_{run_date}.log"


def preflight(models, benchmarks, strict=True):
    for dirname in ["results", "logs", "analysis"]:
        (ROOT / dirname).mkdir(parents=True, exist_ok=True)

    errors = []
    warnings = []

    ok_ollama, installed_models, ollama_error = run_ollama_list()
    if not ok_ollama:
        message = f"Ollama unavailable: {ollama_error}"
        (errors if strict else warnings).append(message)
    else:
        missing_models = [model for model in models if model not in installed_models]
        if missing_models:
            errors.append("Missing Ollama models: " + ", ".join(missing_models))

    if not (ROOT / "benchmarks" / "_0_scoring_engine.py").exists():
        errors.append("Missing benchmarks/_0_scoring_engine.py")

    dataset_counts = {}
    for bench in benchmarks:
        dataset_path = ROOT / bench["dataset"]
        if not dataset_path.exists():
            errors.append(f"Missing dataset: {bench['dataset']}")
            continue
        try:
            dataset_counts[bench["type"]] = load_json_count(dataset_path)
        except Exception as exc:
            errors.append(f"Could not read {bench['dataset']}: {exc}")

    print("\nPRE-FLIGHT")
    print(f"{'Type':<6} {'Benchmark':<24} {'Items':>8}  Dataset")
    for bench in benchmarks:
        count = dataset_counts.get(bench["type"], "ERR")
        print(f"{bench['type']:<6} {bench['name']:<24} {str(count):>8}  {bench['dataset']}")
    print(f"Models: {', '.join(models)}")
    print(f"Ollama GPU devices requested: {OLLAMA_GPU_DEVICES}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if errors:
        print("\nPre-flight errors:")
        for error in errors:
            print(f"  - {error}")
        if strict:
            sys.exit(1)

    return dataset_counts


def build_runs(models, benchmarks, dataset_counts, run_date, resume=False):
    runs = []
    skipped = []
    for bench in benchmarks:
        item_count = dataset_counts.get(bench["type"], 0)
        for model in models:
            out_json, out_csv = output_paths(bench, model, run_date)
            if resume and out_json.exists() and out_csv.exists():
                skipped.append((bench, model, out_json, out_csv))
                continue
            estimate = item_count * 3 * estimate_seconds_per_call(model)
            runs.append({
                "bench": bench,
                "model": model,
                "items": item_count,
                "estimate_seconds": estimate,
                "output_json": out_json,
                "output_csv": out_csv,
                "log": log_path(bench, model, run_date),
            })
    return runs, skipped


def print_run_plan(runs, skipped):
    total_estimate = sum(run["estimate_seconds"] for run in runs)
    print("\nRUN PLAN")
    print(f"{'#':>3} {'Benchmark':<24} {'Model':<18} {'Items':>7} {'Estimate':>12}")
    for index, run in enumerate(runs, start=1):
        print(
            f"{index:>3} {run['bench']['name']:<24} {run['model']:<18} "
            f"{run['items']:>7} {fmt_duration(run['estimate_seconds']):>12}"
        )
    if skipped:
        print("\nSkipped by --resume:")
        for bench, model, _, _ in skipped:
            print(f"  - {bench['name']} | {model}")
    print(f"\nTotal estimated time: {fmt_duration(total_estimate)}")


def run_one(run, index, total, run_date):
    bench = run["bench"]
    model = run["model"]
    print(f"[{index}/{total}] {bench['name']} | {model} - running...", flush=True)

    command = [
        sys.executable,
        bench["script"],
        "--model",
        model,
        "--input",
        bench["dataset"],
        "--output-json",
        str(run["output_json"]),
        "--output-csv",
        str(run["output_csv"]),
        "--date",
        run_date,
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_VISIBLE_DEVICES", OLLAMA_GPU_DEVICES)
    env.setdefault("OLLAMA_SCHED_SPREAD", "1")

    start = time.time()
    with run["log"].open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(command) + "\n\n")
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_handle.write(line)
            log_handle.flush()
        proc.wait()

    duration = time.time() - start
    status = "succeeded" if proc.returncode == 0 else "failed"
    print(f"[{index}/{total}] {bench['name']} | {model} - {status} in {fmt_duration(duration)}")
    return {
        "benchmark": bench["name"],
        "type": bench["type"],
        "model": model,
        "status": status,
        "returncode": proc.returncode,
        "duration_seconds": round(duration, 3),
        "log": str(run["log"].relative_to(ROOT)),
        "output_json": str(run["output_json"].relative_to(ROOT)),
        "output_csv": str(run["output_csv"].relative_to(ROOT)),
    }


def save_summary(run_date, rows, skipped):
    payload = {
        "date": run_date,
        "results": rows,
        "skipped": [
            {"benchmark": bench["name"], "type": bench["type"], "model": model}
            for bench, model, _, _ in skipped
        ],
    }
    path = ROOT / "logs" / f"run_summary_{run_date}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def print_final_summary(rows, skipped, summary_path):
    print("\nFINAL SUMMARY")
    print(f"{'Benchmark':<24} {'Model':<18} {'Status':<10} {'Duration':>10}")
    for row in rows:
        print(
            f"{row['benchmark']:<24} {row['model']:<18} {row['status']:<10} "
            f"{fmt_duration(row['duration_seconds']):>10}"
        )
    succeeded = sum(1 for row in rows if row["status"] == "succeeded")
    total = len(rows)
    print(f"\nSucceeded: {succeeded}/{total}")
    if skipped:
        print(f"Skipped: {len(skipped)}")
    print(f"Summary: {summary_path.relative_to(ROOT)}")
    print("Run analysis: python3 analysis/_0_cross_type_analysis.py")


def main():
    parser = argparse.ArgumentParser(description="Run DeceptBench across models and benchmark types")
    parser.add_argument("--resume", action="store_true", help="Skip runs with today's output files")
    parser.add_argument("--models", nargs="+", default=None, help="Run only these models")
    parser.add_argument("--types", nargs="+", type=int, default=None, help="Run only these benchmark types")
    parser.add_argument("--dry-run", action="store_true", help="Show run plan without executing")
    parser.add_argument("--yes", action="store_true", help="Start without confirmation")
    args = parser.parse_args()

    run_date = date.today().isoformat()
    models = args.models or MODELS
    benchmarks = selected_benchmarks(args.types)
    if not benchmarks:
        print("No benchmarks selected.")
        sys.exit(1)

    dataset_counts = preflight(models, benchmarks, strict=not args.dry_run)
    runs, skipped = build_runs(models, benchmarks, dataset_counts, run_date, resume=args.resume)
    print_run_plan(runs, skipped)

    if args.dry_run:
        print("\nDry run complete. No benchmarks executed.")
        return

    if not runs:
        print("\nNo runs to execute.")
        summary_path = save_summary(run_date, [], skipped)
        print_final_summary([], skipped, summary_path)
        return

    if not args.yes:
        answer = input("\nStart these runs? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return

    rows = []
    for index, run in enumerate(runs, start=1):
        rows.append(run_one(run, index, len(runs), run_date))

    summary_path = save_summary(run_date, rows, skipped)
    print_final_summary(rows, skipped, summary_path)


if __name__ == "__main__":
    main()
