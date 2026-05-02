from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pasr_rag.config import load_app_config
from pasr_rag.dataio import write_json
from pasr_rag.preprocessing import EIPBenchmarkPreprocessor
from pasr_rag.preprocessing.loaders import normalize_eipbenchmark_test_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare eipBenchmark corpus, QA, and agent assets")
    parser.add_argument("--config", default="configs/pasr_rag/eipBenchmark.yaml")
    parser.add_argument("--source-root", default="new-database")
    parser.add_argument("--raw-root", default="data/raw/eipBenchmark")
    parser.add_argument("--dataset-root", default="data/datasets/eipBenchmark")
    parser.add_argument("--agent-root", default="data/agents_eipBenchmark")
    return parser


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def sync_raw_tree(source_root: Path, raw_root: Path) -> dict[str, str]:
    raw_root.mkdir(parents=True, exist_ok=True)
    source_corpus = source_root / "corpus"
    source_qa = source_root / "QA"
    target_corpus = raw_root / "corpus"
    target_qa = raw_root / "qa"
    target_corpus.mkdir(parents=True, exist_ok=True)
    target_qa.mkdir(parents=True, exist_ok=True)

    for item in source_corpus.glob("*.jsonl"):
        shutil.copy2(item, target_corpus / item.name)
    for item in source_qa.iterdir():
        if item.is_file():
            shutil.copy2(item, target_qa / item.name)

    return {
        "raw_root": str(raw_root),
        "corpus_root": str(target_corpus),
        "qa_root": str(target_qa),
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_app_config(resolve_repo_path(args.config))
    source_root = resolve_repo_path(args.source_root)
    raw_root = resolve_repo_path(args.raw_root)
    dataset_root = resolve_repo_path(args.dataset_root)
    agent_root = resolve_repo_path(args.agent_root)

    sync_info = sync_raw_tree(source_root, raw_root)

    dataset_root.mkdir(parents=True, exist_ok=True)
    normalized_qa_path = normalize_eipbenchmark_test_jsonl(
        raw_root / "qa" / "test.jsonl",
        dataset_root / "test_normalized.jsonl",
    )

    preprocessor = EIPBenchmarkPreprocessor(config)
    report = preprocessor.run(raw_root / "corpus", agent_root)
    source_registry_path = agent_root / "source_registry.json"
    dataset_registry_path = dataset_root / "source_registry.json"
    if source_registry_path.exists():
        write_json(dataset_registry_path, json.loads(source_registry_path.read_text(encoding="utf-8")))

    payload = {
        "sync": sync_info,
        "normalized_qa": str(normalized_qa_path),
        "preprocess_report": asdict(report),
        "source_registry": str(source_registry_path),
        "dataset_registry": str(dataset_registry_path),
        "metadata_path": report.metadata_path,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
