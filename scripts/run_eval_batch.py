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
from pasr_rag.evaluation import BatchEvaluator
from pasr_rag.pipeline import PASRExperimentPipeline
from pasr_rag.privacy import AgentMetadataStore
from pasr_rag.preprocessing.loaders import load_hotpotqa_like_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PASR-RAG batch evaluation on a HotpotQA-like dataset")
    parser.add_argument("--config", default="configs/pasr_rag/base.yaml")
    parser.add_argument("--metadata-path", default="data/agents_hotpot_small/agent_metadata.json")
    parser.add_argument("--dataset", default="data/raw/hotpotqa_validation_small.jsonl")
    parser.add_argument("--output-dir", default="outputs/results/pasr_batch_eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["pasr", "rel_only", "ma_rag_lite", "random", "threshold"],
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--privacy-budget-ratio", type=float, default=None)
    parser.add_argument("--privacy-budget-B", type=float, default=None)
    parser.add_argument("--max-active-sources-k", type=int, default=None)
    parser.add_argument("--top-m-per-source", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--enable-privacy-term", dest="enable_privacy_term", action="store_true")
    parser.add_argument("--disable-privacy-term", dest="enable_privacy_term", action="store_false")
    parser.add_argument("--enable-summarization", dest="enable_summarization", action="store_true")
    parser.add_argument("--disable-summarization", dest="enable_summarization", action="store_false")
    parser.add_argument(
        "--enable-parallel-retrieval",
        dest="enable_parallel_retrieval",
        action="store_true",
    )
    parser.add_argument(
        "--disable-parallel-retrieval",
        dest="enable_parallel_retrieval",
        action="store_false",
    )
    parser.set_defaults(
        enable_privacy_term=None,
        enable_summarization=None,
        enable_parallel_retrieval=None,
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
    if args.alpha is not None:
        config.router.alpha = args.alpha
    if args.beta is not None:
        config.router.beta = args.beta
    if args.privacy_budget_ratio is not None:
        config.router.privacy_budget_mode = "ratio"
        config.router.privacy_budget_ratio = args.privacy_budget_ratio
    if args.privacy_budget_B is not None:
        config.router.privacy_budget_mode = "absolute"
        config.router.privacy_budget_B = args.privacy_budget_B
    if args.max_active_sources_k is not None:
        config.router.max_active_sources_k = args.max_active_sources_k
    if args.top_m_per_source is not None:
        config.retrieval.top_m_per_source = args.top_m_per_source
    if args.threshold is not None:
        config.router.threshold = args.threshold
    if args.enable_privacy_term is not None:
        config.privacy.enable_privacy_term = args.enable_privacy_term
    if args.enable_summarization is not None:
        config.generation.enable_summarization = args.enable_summarization
    if args.enable_parallel_retrieval is not None:
        config.retrieval.enable_parallel_retrieval = args.enable_parallel_retrieval

    metadata_path = resolve_repo_path(args.metadata_path)
    dataset_path = resolve_repo_path(args.dataset)
    output_dir = resolve_repo_path(args.output_dir)

    pipeline = PASRExperimentPipeline(config)
    metadata_store = AgentMetadataStore(metadata_path)
    evaluator = BatchEvaluator(metadata_store)

    examples = load_hotpotqa_like_jsonl(dataset_path)
    if args.limit is not None:
        examples = examples[: args.limit]

    rows: list[dict] = []
    for example in examples:
        result = pipeline.answer_query(example.question, metadata_path)
        rows.append(
            evaluator.evaluate_item(
                result=result,
                gold_answer=example.answer,
                supporting_titles=example.supporting_titles,
            )
        )

    summary = evaluator.summarize(rows)
    jsonl_path, csv_path, summary_path = evaluator.export(rows, summary, output_dir)
    print(
        json.dumps(
            {
                "count": len(rows),
                "strategy": config.router.strategy,
                "alpha": config.router.alpha,
                "beta": config.router.beta,
                "privacy_budget_mode": config.router.privacy_budget_mode,
                "privacy_budget_ratio": config.router.privacy_budget_ratio,
                "privacy_budget_B": config.router.privacy_budget_B,
                "max_active_sources_k": config.router.max_active_sources_k,
                "top_m_per_source": config.retrieval.top_m_per_source,
                "enable_privacy_term": config.privacy.enable_privacy_term,
                "enable_summarization": config.generation.enable_summarization,
                "enable_parallel_retrieval": config.retrieval.enable_parallel_retrieval,
                "jsonl": str(jsonl_path),
                "csv": str(csv_path),
                "summary": str(summary_path),
                "summary_metrics": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
