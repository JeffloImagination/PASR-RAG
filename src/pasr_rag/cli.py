from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .config import load_app_config
from .pipeline import PASRExperimentPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PASR-RAG experiment runner")
    parser.add_argument(
        "--config",
        default="configs/pasr_rag/base.yaml",
        help="Path to the PASR-RAG YAML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Bootstrap the runtime and print the resolved configuration summary.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = load_app_config(args.config)
    pipeline = PASRExperimentPipeline(config)
    report = pipeline.bootstrap()
    report_path = pipeline.save_bootstrap_report(report)

    if args.dry_run:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
        print(f"bootstrap_report={report_path}")
        return 0

    print("PASR-RAG skeleton bootstrapped successfully.")
    print(f"bootstrap_report={report_path}")
    return 0
