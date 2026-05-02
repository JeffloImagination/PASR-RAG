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
    parser = argparse.ArgumentParser(description="Run PASR alpha sweep with fixed best-known settings")
    parser.add_argument("--config", default="configs/pasr_rag/base.yaml")
    parser.add_argument("--metadata-path", default="data/agents_hotpot_50/agent_metadata.json")
    parser.add_argument("--dataset", default="data/raw/hotpotqa_validation_50.jsonl")
    parser.add_argument("--output-root", default="outputs/results/alpha_sweep_full50_live")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--budget-ratio", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--m", type=int, default=5)
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
    json_path = output_root / "alpha_summary.json"
    csv_path = output_root / "alpha_summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "alpha",
        "beta",
        "budget_ratio",
        "k",
        "m",
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


def best_row(rows: list[dict]) -> dict:
    return max(
        rows,
        key=lambda row: (
            float(row.get("relaxed_f1", 0.0)),
            float(row.get("f1", 0.0)),
            float(row.get("em", 0.0)),
            -float(row.get("avg_selected_privacy_cost", 0.0)),
        ),
    )


def main() -> int:
    args = build_parser().parse_args()
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_log: dict[str, dict] = {}
    rows: list[dict] = []
    for alpha in args.alphas:
        run_dir = output_root / f"alpha_{alpha:g}"
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
            "--alpha",
            str(alpha),
            "--beta",
            str(args.beta),
            "--privacy-budget-ratio",
            str(args.budget_ratio),
            "--max-active-sources-k",
            str(args.k),
            "--top-m-per-source",
            str(args.m),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        proc = run_command(command)
        run_log[f"alpha_{alpha:g}"] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        if proc.returncode != 0:
            continue
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        summary["alpha"] = alpha
        summary["beta"] = args.beta
        summary["budget_ratio"] = args.budget_ratio
        summary["k"] = args.k
        summary["m"] = args.m
        rows.append(summary)

    rows.sort(key=lambda item: item["alpha"])
    if not rows:
        raise RuntimeError("Alpha sweep produced no successful runs.")

    best = best_row(rows)
    best_config = {
        "alpha": best["alpha"],
        "beta": best["beta"],
        "budget_ratio": best["budget_ratio"],
        "k": best["k"],
        "m": best["m"],
    }
    json_path, csv_path = export_rows(rows, output_root)
    (output_root / "best_alpha_config.json").write_text(
        json.dumps(best_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run_summary_path = output_root / "run_summary.json"
    run_summary_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "summary_json": str(json_path),
                "summary_csv": str(csv_path),
                "best_alpha_config": best_config,
                "run_summary": str(run_summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
