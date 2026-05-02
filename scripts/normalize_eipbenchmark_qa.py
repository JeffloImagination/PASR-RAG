from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pasr_rag.preprocessing.loaders import normalize_eipbenchmark_test_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize eipBenchmark QA into HotpotQA-like JSONL")
    parser.add_argument("--input", default="new-database/QA/test.jsonl")
    parser.add_argument("--output", default="data/datasets/eipBenchmark/test_normalized.jsonl")
    parser.add_argument("--copy-raw", action="store_true")
    parser.add_argument("--raw-qa-dir", default="data/raw/eipBenchmark/qa")
    return parser


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def main() -> int:
    args = build_parser().parse_args()
    input_path = resolve_repo_path(args.input)
    output_path = resolve_repo_path(args.output)
    normalized_path = normalize_eipbenchmark_test_jsonl(input_path, output_path)

    payload: dict[str, str] = {
        "input": str(input_path),
        "normalized_output": str(normalized_path),
    }

    if args.copy_raw:
        raw_qa_dir = resolve_repo_path(args.raw_qa_dir)
        raw_qa_dir.mkdir(parents=True, exist_ok=True)
        copied_path = raw_qa_dir / input_path.name
        shutil.copy2(input_path, copied_path)
        payload["raw_copy"] = str(copied_path)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
