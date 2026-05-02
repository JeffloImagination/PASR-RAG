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
from pasr_rag.preprocessing import HotpotPreprocessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess HotpotQA-like data for PASR-RAG")
    parser.add_argument("--config", default="configs/pasr_rag/base.yaml")
    parser.add_argument(
        "--input",
        default="data/raw/hotpotqa_smoke.jsonl",
        help="Path to a HotpotQA-like JSONL file",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for the agent output directory",
    )
    return parser


def resolve_repo_path(path_str: str | None) -> str | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def main() -> int:
    args = build_parser().parse_args()
    config = load_app_config(resolve_repo_path(args.config))
    preprocessor = HotpotPreprocessor(config)
    report = preprocessor.run(resolve_repo_path(args.input), resolve_repo_path(args.output_root))
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
