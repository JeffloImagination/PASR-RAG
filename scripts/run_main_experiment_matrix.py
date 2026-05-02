from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the main PASR-RAG experiment matrix")
    parser.add_argument("--dataset", default="data/raw/hotpotqa_validation_small.jsonl")
    parser.add_argument("--metadata-path", default="data/agents_hotpot_small/agent_metadata.json")
    parser.add_argument("--flashrag-asset-dir", default="data/flashrag_assets/hotpot_small")
    parser.add_argument("--output-root", default="outputs/results/main_experiment_matrix")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retrieval-model-path", default="models/bge-base-en-v1.5")
    parser.add_argument("--include-random", action="store_true")
    return parser


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def run_command(command: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def extract_pasr_summary(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["method"] = summary_path.parent.name
    return summary


def extract_flashrag_summary(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["accuracy"] = summary.get("acc", summary.get("accuracy", 0.0))
    summary["soft_em"] = summary.get("soft_em", None)
    summary["relaxed_f1"] = summary.get("relaxed_f1", None)
    summary["support_fact_recall"] = summary.get("support_fact_recall", None)
    summary["avg_selected_privacy_cost"] = summary.get("avg_selected_privacy_cost", None)
    summary["hrhr"] = summary.get("hrhr", None)
    summary["bbr"] = summary.get("bbr", None)
    summary["avg_activated_sources"] = summary.get("avg_activated_sources", None)
    summary["latency_routing"] = summary.get("latency_routing", None)
    summary["latency_e2e"] = summary.get("latency_e2e", None)
    return summary


def export_matrix_summary(rows: list[dict], output_root: Path) -> tuple[Path, Path]:
    json_path = output_root / "matrix_summary.json"
    csv_path = output_root / "matrix_summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "method",
        "count",
        "em",
        "f1",
        "soft_em",
        "relaxed_f1",
        "accuracy",
        "support_fact_recall",
        "avg_selected_privacy_cost",
        "hrhr",
        "bbr",
        "avg_activated_sources",
        "latency_routing",
        "latency_e2e",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return json_path, csv_path


def main() -> int:
    args = build_parser().parse_args()
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    method_rows: list[dict] = []

    pasr_methods = ["pasr", "rel_only", "ma_rag_lite", "threshold"]
    if args.include_random:
        pasr_methods.append("random")

    for strategy in pasr_methods:
        method_output = output_root / strategy
        cmd = [
            sys.executable,
            "scripts/run_eval_batch.py",
            "--config",
            "configs/pasr_rag/base.yaml",
            "--metadata-path",
            args.metadata_path,
            "--dataset",
            args.dataset,
            "--output-dir",
            str(method_output),
            "--strategy",
            strategy,
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        proc = run_command(cmd)
        results[strategy] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        if proc.returncode == 0:
            summary_path = method_output / "summary.json"
            row = extract_pasr_summary(summary_path)
            row["method"] = strategy
            method_rows.append(row)

    suite_cmd = [
        sys.executable,
        "scripts/run_experiment_suite.py",
        "--dataset",
        args.dataset,
        "--metadata-path",
        args.metadata_path,
        "--flashrag-asset-dir",
        args.flashrag_asset_dir,
        "--output-root",
        str(output_root / "centralized"),
        "--run-bm25",
        "--run-dense",
        "--run-hybrid",
        "--retrieval-model-path",
        args.retrieval_model_path,
    ]
    if args.limit is not None:
        suite_cmd.extend(["--limit", str(args.limit)])

    suite_proc = run_command(suite_cmd)
    results["centralized_suite"] = {
        "returncode": suite_proc.returncode,
        "stdout": suite_proc.stdout,
        "stderr": suite_proc.stderr,
    }

    centralized_root = output_root / "centralized"
    for method_name in ["flashrag_bm25", "flashrag_bge", "flashrag_hybrid"]:
        summary_path = centralized_root / f"{method_name}_normalized" / "summary.json"
        if summary_path.exists():
            row = extract_flashrag_summary(summary_path)
            row["method"] = method_name
            method_rows.append(row)

    matrix_json, matrix_csv = export_matrix_summary(method_rows, output_root)
    results["matrix_summary"] = {
        "json": str(matrix_json),
        "csv": str(matrix_csv),
    }

    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "run_summary": str(summary_path),
                "matrix_json": str(matrix_json),
                "matrix_csv": str(matrix_csv),
                "methods": [row["method"] for row in method_rows],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
