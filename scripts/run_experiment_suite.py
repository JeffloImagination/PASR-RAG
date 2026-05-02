from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PASR-RAG and FlashRAG experiment suite")
    parser.add_argument("--dataset", default="data/raw/hotpotqa_validation_small.jsonl")
    parser.add_argument("--metadata-path", default="data/agents_hotpot_small/agent_metadata.json")
    parser.add_argument("--flashrag-asset-dir", default="data/flashrag_assets/hotpot_small")
    parser.add_argument("--output-root", default="outputs/results/experiment_suite")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--run-pasr", action="store_true")
    parser.add_argument("--run-bm25", action="store_true")
    parser.add_argument("--run-dense", action="store_true")
    parser.add_argument("--run-hybrid", action="store_true")
    parser.add_argument("--retrieval-model-path", default="models/bge-base-en-v1.5")
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


def find_latest_flashrag_run(parent_dir: Path) -> Path | None:
    if not parent_dir.exists():
        return None
    candidates = [path for path in parent_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> int:
    args = build_parser().parse_args()
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not any([args.run_pasr, args.run_bm25, args.run_dense, args.run_hybrid]):
        args.run_pasr = True
        args.run_bm25 = True

    results = {}

    if args.run_pasr:
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
            str(output_root / "pasr"),
            "--limit",
            str(args.limit),
            "--strategy",
            "pasr",
        ]
        proc = run_command(cmd)
        results["pasr"] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    if args.run_bm25 or args.run_dense or args.run_hybrid:
        prepare_cmd = [
            sys.executable,
            "scripts/prepare_flashrag_assets.py",
            "--agent-root",
            str(resolve_repo_path(args.metadata_path).parent),
            "--dataset",
            args.dataset,
            "--output-dir",
            args.flashrag_asset_dir,
            "--limit",
            str(args.limit),
        ]
        prep = run_command(prepare_cmd)
        results["prepare_flashrag_assets"] = {
            "returncode": prep.returncode,
            "stdout": prep.stdout,
            "stderr": prep.stderr,
        }

    baseline_modes = []
    if args.run_bm25:
        baseline_modes.append("bm25")
    if args.run_dense:
        baseline_modes.append("bge")
    if args.run_hybrid:
        baseline_modes.append("hybrid")

    for mode in baseline_modes:
        cmd = [
            sys.executable,
            "scripts/run_flashrag_baseline.py",
            "--asset-dir",
            args.flashrag_asset_dir,
            "--save-dir",
            str(output_root / f"flashrag_{mode}"),
            "--retrieval-method",
            mode,
            "--generator-framework",
            "openai",
            "--generator-model",
            "qwen-plus",
            "--generator-base-url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ]
        if mode in {"bge", "hybrid"}:
            if not args.retrieval_model_path:
                results[f"flashrag_{mode}"] = {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": "Missing --retrieval-model-path for dense/hybrid baseline.",
                }
                continue
            cmd.extend(["--retrieval-model-path", args.retrieval_model_path])

        proc = run_command(cmd)
        entry = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        if proc.returncode == 0:
            run_dir = find_latest_flashrag_run(output_root / f"flashrag_{mode}")
            if run_dir is not None:
                export_cmd = [
                    sys.executable,
                    "scripts/export_flashrag_results.py",
                    "--flashrag-run-dir",
                    str(run_dir),
                    "--output-dir",
                    str(output_root / f"flashrag_{mode}_normalized"),
                    "--method-name",
                    f"flashrag_{mode}",
                ]
                export_proc = run_command(export_cmd)
                entry["run_dir"] = str(run_dir)
                entry["normalized_export"] = {
                    "returncode": export_proc.returncode,
                    "stdout": export_proc.stdout,
                    "stderr": export_proc.stderr,
                }
        results[f"flashrag_{mode}"] = entry

    summary_path = output_root / "suite_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "runs": list(results.keys())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
