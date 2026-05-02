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
from pasr_rag.privacy import AgentMetadataStore, PrivacyCostEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate source-level privacy costs for PASR-RAG agents")
    parser.add_argument("--config", default=None)
    parser.add_argument("--agent-root", default="data/agents_hotpot_small")
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--mode", default="llm_eval", choices=["llm_eval", "fixed_label", "random_label"])
    parser.add_argument("--fixed-level", default="L1", choices=["L0", "L1", "L2", "L3", "L4"])
    parser.add_argument("--prompt-path", default="docs/隐私成本评估提示词.md")
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
    config = load_app_config(resolve_repo_path(args.config)) if args.config else None
    agent_root = (
        config.paths.agent_root
        if config is not None
        else Path(resolve_repo_path(args.agent_root))
    )
    metadata_path = resolve_repo_path(args.metadata_path) or str(agent_root / "agent_metadata.json")
    store = AgentMetadataStore(metadata_path)
    prompt_path = (
        resolve_repo_path(args.prompt_path)
        if args.prompt_path
        else str((REPO_ROOT / "docs" / "隐私成本评估提示词.md").resolve())
    )
    evaluator = PrivacyCostEvaluator(
        prompt_path=prompt_path if not config else resolve_repo_path(config.privacy.prompt_path),
        random_seed=42 if not config else config.project.random_seed,
        llm_model="qwen-plus" if not config else config.models.llm_model,
        api_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not config
        else config.generation.api_base_url,
        api_key_env="DASHSCOPE_API_KEY" if not config else config.generation.api_key_env,
        sample_ratio=0.1 if not config else config.privacy.sample_ratio,
        max_sample_chunks=48 if not config else config.privacy.max_sample_chunks,
        max_chunk_chars=1200 if not config else config.privacy.max_chunk_chars,
    )

    assessments: list[dict] = []
    for agent in store.get_all_agents():
        chunk_path = Path(agent.chunk_path) if agent.chunk_path else (agent_root / agent.agent_id / f"{agent.agent_id}_chunks.jsonl")
        assessment = evaluator.evaluate_source(
            chunk_path=chunk_path,
            mode=args.mode,
            fixed_level=args.fixed_level,
        )
        store.update_agent_privacy(
            agent_id=agent.agent_id,
            privacy_level=assessment.privacy_level,
            privacy_cost=assessment.privacy_cost,
            reason=assessment.reason,
            confidence=assessment.confidence,
        )
        assessments.append(
            {
                "agent_id": agent.agent_id,
                **asdict(assessment),
            }
        )

    output_path = store.save()
    print(json.dumps({"metadata_path": str(output_path), "assessments": assessments}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
