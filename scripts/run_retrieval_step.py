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
from pasr_rag.privacy import AgentMetadataStore
from pasr_rag.retrieval import execute_agents_for_query
from pasr_rag.router.router import build_router


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run routing + local retrieval step for a single query")
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

    metadata_store = AgentMetadataStore(resolve_repo_path(args.metadata_path))
    router = build_router(config, metadata_store)
    decision = router.select_agents(args.query)
    responses = execute_agents_for_query(
        query=args.query,
        selected_agent_ids=decision.selected_agents,
        metadata_store=metadata_store,
        config=config,
    )
    print(
        json.dumps(
            {
                "router_decision": asdict(decision),
                "agent_responses": [asdict(item) for item in responses],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
