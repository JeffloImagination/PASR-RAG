from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pasr_rag.config import load_app_config
from pasr_rag.pipeline import PASRExperimentPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run minimal end-to-end PASR-RAG QA step")
    parser.add_argument("--config", default="configs/pasr_rag/base.yaml")
    parser.add_argument("--metadata-path", default="data/agents_hotpot_small/agent_metadata.json")
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["pasr", "rel_only", "ma_rag_lite", "random", "threshold"],
    )
    return parser


def resolve_repo_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def main() -> int:
    args = build_parser().parse_args()
    config = load_app_config(resolve_repo_path(args.config))
    if args.strategy is not None:
        config.router.strategy = args.strategy
    pipeline = PASRExperimentPipeline(config)
    result = pipeline.answer_query(args.query, resolve_repo_path(args.metadata_path))
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
