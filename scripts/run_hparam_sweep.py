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
    parser = argparse.ArgumentParser(description="Run staged PASR hyperparameter sweeps")
    parser.add_argument("--config", default="configs/pasr_rag/base.yaml")
    parser.add_argument("--metadata-path", default="data/agents_hotpot_50/agent_metadata.json")
    parser.add_argument("--dataset", default="data/raw/hotpotqa_validation_50.jsonl")
    parser.add_argument("--output-root", default="outputs/results/hparam_sweep_full50")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--budget-ratios", nargs="+", type=float, default=[0.2, 0.3])
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--m-values", nargs="+", type=int, default=[3, 5])
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


def export_rows(rows: list[dict], output_path: Path, fieldnames: list[str]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


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

    phase1_dir = output_root / "phase1_beta_budget"
    phase2_dir = output_root / "phase2_k_m"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    phase2_dir.mkdir(parents=True, exist_ok=True)

    run_log: dict[str, dict] = {}
    phase1_rows: list[dict] = []
    for beta in args.betas:
        for budget_ratio in args.budget_ratios:
            name = f"beta_{beta:g}_budget_{budget_ratio:g}"
            run_dir = phase1_dir / name
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
                "--privacy-budget-ratio",
                str(budget_ratio),
            ]
            if args.limit is not None:
                command.extend(["--limit", str(args.limit)])
            proc = run_command(command)
            run_log[name] = {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
            if proc.returncode != 0:
                continue
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            summary["beta"] = beta
            summary["budget_ratio"] = budget_ratio
            phase1_rows.append(summary)

    if not phase1_rows:
        raise RuntimeError("Phase 1 sweep produced no successful runs.")

    chosen_phase1 = best_row(phase1_rows)
    best_phase1_config = {
        "beta": chosen_phase1["beta"],
        "budget_ratio": chosen_phase1["budget_ratio"],
    }
    (output_root / "best_phase1_config.json").write_text(
        json.dumps(best_phase1_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    phase2_rows: list[dict] = []
    for k_value in args.k_values:
        for m_value in args.m_values:
            name = f"k_{k_value}_m_{m_value}"
            run_dir = phase2_dir / name
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
                str(best_phase1_config["beta"]),
                "--privacy-budget-ratio",
                str(best_phase1_config["budget_ratio"]),
                "--max-active-sources-k",
                str(k_value),
                "--top-m-per-source",
                str(m_value),
            ]
            if args.limit is not None:
                command.extend(["--limit", str(args.limit)])
            proc = run_command(command)
            run_log[name] = {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
            if proc.returncode != 0:
                continue
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            summary["beta"] = best_phase1_config["beta"]
            summary["budget_ratio"] = best_phase1_config["budget_ratio"]
            summary["k"] = k_value
            summary["m"] = m_value
            phase2_rows.append(summary)

    phase1_json = output_root / "phase1_summary.json"
    phase2_json = output_root / "phase2_summary.json"
    phase1_json.write_text(json.dumps(phase1_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    phase2_json.write_text(json.dumps(phase2_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    export_rows(
        phase1_rows,
        output_root / "phase1_summary.csv",
        [
            "beta",
            "budget_ratio",
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
        ],
    )
    export_rows(
        phase2_rows,
        output_root / "phase2_summary.csv",
        [
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
        ],
    )
    run_summary_path = output_root / "run_summary.json"
    run_summary_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "phase1_json": str(phase1_json),
                "phase2_json": str(phase2_json),
                "best_phase1_config": best_phase1_config,
                "run_summary": str(run_summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
