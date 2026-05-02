from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pasr_rag.preprocessing.loaders import load_hotpotqa_like_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare FlashRAG-compatible dataset and merged corpus assets")
    parser.add_argument("--agent-root", default="data/agents_hotpot_small")
    parser.add_argument("--dataset", default="data/raw/hotpotqa_validation_small.jsonl")
    parser.add_argument("--output-dir", default="data/flashrag_assets/hotpot_small")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def main() -> int:
    args = build_parser().parse_args()
    agent_root = resolve_repo_path(args.agent_root)
    dataset_path = resolve_repo_path(args.dataset)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_hotpotqa_like_jsonl(str(dataset_path))
    if args.limit is not None:
        examples = examples[: args.limit]

    dataset_rows = []
    for example in examples:
        dataset_rows.append(
            {
                "id": example.question_id,
                "question": example.question,
                "golden_answers": [example.answer],
                "metadata": {
                    "supporting_titles": example.supporting_titles,
                },
            }
        )

    corpus_rows = []
    seen_ids: set[str] = set()
    for chunk_file in sorted(agent_root.glob("A_*/*_chunks.jsonl")):
        with chunk_file.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                chunk_id = item["chunk_id"]
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                corpus_rows.append(
                    {
                        "id": chunk_id,
                        "contents": f"{item['title']}\n{item['content']}",
                    }
                )

    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / "test.jsonl").open("w", encoding="utf-8") as file:
        for row in dataset_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    corpus_path = output_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as file:
        for row in corpus_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "dataset_path": str(dataset_dir / "test.jsonl"),
                "corpus_path": str(corpus_path),
                "examples": len(dataset_rows),
                "corpus_rows": len(corpus_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
