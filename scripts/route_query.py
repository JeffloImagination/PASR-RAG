from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pasr_rag.config import load_app_config
from pasr_rag.privacy import AgentMetadataStore
from pasr_rag.router.router import build_router


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PASR-style routing for a single query")
    parser.add_argument("--config", default="configs/pasr_rag/base.yaml")
    parser.add_argument("--metadata-path", default="data/agents_hotpot_small/agent_metadata.json")
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["pasr", "rel_only", "ma_rag_lite", "random", "threshold"],
    )
    parser.add_argument("--save-log", action="store_true")
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
    if args.save_log:
        logs_root = config.paths.logs_root
        logs_root.mkdir(parents=True, exist_ok=True)
        log_path = logs_root / "router_decisions.jsonl"
        payload = asdict(decision)
        payload["timestamp"] = datetime.utcnow().isoformat(timespec="seconds")
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps(asdict(decision), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
