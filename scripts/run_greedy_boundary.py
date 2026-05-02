from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pasr_rag.config import load_app_config
from pasr_rag.preprocessing.loaders import load_hotpotqa_like_jsonl
from pasr_rag.privacy import AgentMetadataStore
from pasr_rag.router.router import build_router


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run H3 greedy-vs-exact boundary validation")
    parser.add_argument("--config", default="configs/pasr_rag/base.yaml")
    parser.add_argument("--metadata-path", default="data/agents_hotpot_50/agent_metadata.json")
    parser.add_argument("--dataset", default="data/raw/hotpotqa_validation_50.jsonl")
    parser.add_argument("--output-root", default="outputs/results/h3_greedy_boundary")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--strategy", default="pasr", choices=["pasr", "rel_only", "threshold"])
    return parser


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def exact_best_subset(scored_agents: list[dict], budget: float, max_k: int) -> tuple[list[str], float, float]:
    best_agents: list[str] = []
    best_utility = float("-inf")
    best_cost = 0.0
    for size in range(0, min(max_k, len(scored_agents)) + 1):
        for combo in itertools.combinations(scored_agents, size):
            cost = sum(item["privacy_cost"] for item in combo)
            if cost > budget + 1e-9:
                continue
            utility = sum(item["utility_score"] for item in combo)
            selected_ids = [item["agent_id"] for item in combo]
            candidate = (utility, -cost, selected_ids)
            current = (best_utility, -best_cost, best_agents)
            if candidate > current:
                best_agents = selected_ids
                best_utility = utility
                best_cost = cost
    if best_utility == float("-inf"):
        return [], 0.0, 0.0
    return best_agents, best_utility, best_cost


def main() -> int:
    args = build_parser().parse_args()
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = load_app_config(resolve_repo_path(args.config))
    config.router.strategy = args.strategy
    metadata_store = AgentMetadataStore(resolve_repo_path(args.metadata_path))
    router = build_router(config, metadata_store)
    examples = load_hotpotqa_like_jsonl(resolve_repo_path(args.dataset))[: args.limit]

    rows: list[dict] = []
    for example in examples:
        greedy_start = time.perf_counter()
        decision = router.select_agents(example.question)
        greedy_runtime_ms = (time.perf_counter() - greedy_start) * 1000
        exact_start = time.perf_counter()
        exact_agents, exact_utility, exact_cost = exact_best_subset(
            decision.scored_agents,
            decision.privacy_budget,
            config.router.max_active_sources_k,
        )
        exact_runtime_ms = (time.perf_counter() - exact_start) * 1000
        greedy_utility = 0.0
        for item in decision.scored_agents:
            if item["agent_id"] in decision.selected_agents:
                greedy_utility += float(item["utility_score"])
        utility_gap = exact_utility - greedy_utility
        approximation_ratio = 1.0 if exact_utility == 0 else greedy_utility / exact_utility
        rows.append(
            {
                "question_id": example.question_id,
                "query": example.question,
                "greedy_selected_agents": decision.selected_agents,
                "exact_selected_agents": exact_agents,
                "greedy_utility": greedy_utility,
                "exact_utility": exact_utility,
                "utility_gap": utility_gap,
                "approximation_ratio": approximation_ratio,
                "greedy_cost": decision.total_privacy_cost,
                "exact_cost": exact_cost,
                "budget": decision.privacy_budget,
                "same_selection": decision.selected_agents == exact_agents,
                "greedy_runtime_ms": greedy_runtime_ms,
                "exact_runtime_ms": exact_runtime_ms,
            }
        )

    summary = {
        "count": len(rows),
        "strategy": args.strategy,
        "same_selection_rate": sum(float(row["same_selection"]) for row in rows) / len(rows) if rows else 0.0,
        "avg_greedy_utility": sum(float(row["greedy_utility"]) for row in rows) / len(rows) if rows else 0.0,
        "avg_exact_utility": sum(float(row["exact_utility"]) for row in rows) / len(rows) if rows else 0.0,
        "avg_utility_gap": sum(float(row["utility_gap"]) for row in rows) / len(rows) if rows else 0.0,
        "avg_approximation_ratio": (
            sum(float(row["approximation_ratio"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "avg_greedy_runtime_ms": (
            sum(float(row["greedy_runtime_ms"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "avg_exact_runtime_ms": (
            sum(float(row["exact_runtime_ms"]) for row in rows) / len(rows) if rows else 0.0
        ),
    }

    jsonl_path = output_root / "query_results.jsonl"
    csv_path = output_root / "query_results.csv"
    summary_path = output_root / "summary.json"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            flat_rows = []
            for row in rows:
                row_copy = dict(row)
                row_copy["greedy_selected_agents"] = ",".join(row_copy["greedy_selected_agents"])
                row_copy["exact_selected_agents"] = ",".join(row_copy["exact_selected_agents"])
                flat_rows.append(row_copy)
            writer.writerows(flat_rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_root": str(output_root),
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
