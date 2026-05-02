from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pasr_rag.dataio import write_jsonl
from pasr_rag.preprocessing.loaders import load_hotpotqa_from_hf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and normalize HotpotQA from Hugging Face")
    parser.add_argument("--name", default="distractor")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", default="data/raw/hotpotqa_validation_small.jsonl")
    return parser


def resolve_repo_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def main() -> int:
    args = build_parser().parse_args()
    rows = load_hotpotqa_from_hf(name=args.name, split=args.split, limit=args.limit)
    output_path = write_jsonl(resolve_repo_path(args.output), rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "output": str(output_path),
                "dataset": "hotpotqa/hotpot_qa",
                "name": args.name,
                "split": args.split,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
