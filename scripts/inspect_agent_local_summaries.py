from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pasr_rag.config import load_app_config
from pasr_rag.retrieval.executor import RetrievalAgentExecutor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect cached local summaries returned by retrieval agents."
    )
    parser.add_argument(
        "--config",
        default="configs/pasr_rag/eipBenchmark.yaml",
        help="Config yaml path.",
    )
    parser.add_argument(
        "--query-results",
        required=True,
        help="Path to query_results.jsonl produced by run_eval_batch.py.",
    )
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output json path to save samples (utf-8).",
    )
    return parser


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def main() -> int:
    args = build_parser().parse_args()
    config = load_app_config(resolve_repo_path(args.config))
    executor = RetrievalAgentExecutor(config)

    query_results_path = resolve_repo_path(args.query_results)
    rows: list[dict] = []
    with query_results_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    emitted = 0
    samples: list[dict] = []
    for row in rows:
        query = row.get("query", "")
        for agent_id in row.get("selected_agents", []):
            cache_path = executor._cache_path(query, agent_id)
            if not cache_path.exists():
                continue
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            sample = {
                "query": query,
                "agent_id": agent_id,
                "cache_path": str(cache_path),
                "local_summary": payload.get("local_summary", ""),
            }
            samples.append(sample)

            if args.out is None:
                print(f"=== SAMPLE {emitted + 1} ===")
                print(f"QUERY: {query}")
                print(f"AGENT: {agent_id}")
                print(f"CACHE: {cache_path}")
                print("LOCAL_SUMMARY_START")
                print(sample["local_summary"])
                print("LOCAL_SUMMARY_END")
                print()
            emitted += 1
            if emitted >= args.max_samples:
                if args.out is not None:
                    out_path = resolve_repo_path(args.out)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(
                        json.dumps(samples, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(f"Wrote {len(samples)} samples to: {out_path}")
                return 0

    print(f"Only emitted {emitted} samples (cache misses may have occurred).")
    if args.out is not None:
        out_path = resolve_repo_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(samples, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {len(samples)} samples to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
