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
    parser = argparse.ArgumentParser(description="Run H2 Pareto beta sweep for PASR-RAG")
    parser.add_argument("--config", default="configs/pasr_rag/base.yaml")
    parser.add_argument("--metadata-path", default="data/agents_hotpot_50/agent_metadata.json")
    parser.add_argument("--dataset", default="data/raw/hotpotqa_validation_50.jsonl")
    parser.add_argument("--output-root", default="outputs/results/h2_pareto_beta_sweep")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0])
    return parser


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
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


def export_rows(rows: list[dict], output_root: Path) -> tuple[Path, Path]:
    json_path = output_root / "pareto_summary.json"
    csv_path = output_root / "pareto_summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "beta",
        "count",
        "em",
        "f1",
        "soft_em",
        "relaxed_f1",
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

    run_log: dict[str, dict] = {}
    rows: list[dict] = []

    for beta in args.betas:
        run_dir = output_root / f"beta_{beta:g}"
        command = [
            sys.executable,
            "scripts/run_eval_batch.py",
            "--config",
            args.config,
            "--metadata-path",
            args.metadata_path,
            "--dataset",
            args.dataset,
            "--output-dir",
            str(run_dir),
            "--strategy",
            "pasr",
            "--beta",
            str(beta),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        proc = run_command(command)
        run_log[f"beta_{beta:g}"] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        if proc.returncode != 0:
            continue
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        summary["beta"] = beta
        rows.append(summary)

    rows.sort(key=lambda item: item["beta"])
    json_path, csv_path = export_rows(rows, output_root)
    run_summary_path = output_root / "run_summary.json"
    run_summary_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "summary_json": str(json_path),
                "summary_csv": str(csv_path),
                "run_summary": str(run_summary_path),
                "betas": args.betas,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
