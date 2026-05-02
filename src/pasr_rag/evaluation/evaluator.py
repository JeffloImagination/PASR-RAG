from __future__ import annotations

import csv
import json
import re
import string
import unicodedata
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ..privacy import AgentMetadataStore
from ..schemas import QueryResult


CHINESE_PUNCTUATION = "，。！？；：、（）【】《》“”‘’—…￥·"
ANSWER_PREFIX_PATTERNS = [
    r"^答案[:：]\s*",
    r"^最终答案[:：]\s*",
    r"^结论[:：]\s*",
    r"^答[:：]\s*",
]


def normalize_business_qa_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.strip()
    for pattern in ANSWER_PREFIX_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    # Drop obvious retrieval/context markers that should not participate in answer scoring.
    text = re.sub(r"\[Agent\s+[^\]]+\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]+:[^\]]+\]", " ", text)
    text = re.sub(r"Query\s*:\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Context\s*:\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Fused\s*Context\s*:\s*", " ", text, flags=re.IGNORECASE)
    # Keep the head of the answer if the model emitted a long paragraph.
    text = re.split(r"[\r\n]+", text, maxsplit=1)[0]
    text = re.split(r"[。！？;；]", text, maxsplit=1)[0]
    lowered = text.lower()
    lowered = lowered.translate(str.maketrans("", "", string.punctuation + CHINESE_PUNCTUATION))
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    lowered = re.sub(r"\s+", "", lowered)
    return lowered


def mixed_tokenize(text: str) -> list[str]:
    normalized = normalize_business_qa_answer(text)
    if not normalized:
        return []
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+(?:\.[0-9]+)?%?", normalized)


def normalize_answer(text: str) -> str:
    return normalize_business_qa_answer(text)


def exact_match_score(prediction: str, gold: str) -> float:
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)
    if not pred and not target:
        return 1.0
    if not pred or not target:
        return 0.0
    if pred == target:
        return 1.0
    # Chinese/business QA often returns a short answer span inside a longer explanation.
    short, long_ = (pred, target) if len(pred) <= len(target) else (target, pred)
    if len(short) <= 24 and short in long_:
        return 1.0
    return 0.0


def soft_exact_match_score(prediction: str, gold: str) -> float:
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)
    if not pred and not target:
        return 1.0
    if not pred or not target:
        return 0.0
    return float(pred == target or pred in target or target in pred)


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = mixed_tokenize(prediction)
    gold_tokens = mixed_tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def relaxed_f1_score(prediction: str, gold: str) -> float:
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)
    if not pred and not target:
        return 1.0
    if not pred or not target:
        return 0.0
    if pred == target or pred in target or target in pred:
        return 1.0
    return f1_score(prediction, gold)


class BatchEvaluator:
    def __init__(self, metadata_store: AgentMetadataStore) -> None:
        self.metadata_store = metadata_store
        self.agent_title_map = self._build_agent_title_map()

    def evaluate_item(
        self,
        result: QueryResult,
        gold_answer: str,
        supporting_titles: list[str],
    ) -> dict:
        em = exact_match_score(result.answer, gold_answer)
        f1 = f1_score(result.answer, gold_answer)
        soft_em = soft_exact_match_score(result.answer, gold_answer)
        relaxed_f1 = relaxed_f1_score(result.answer, gold_answer)
        acc = em
        sfr = self.support_fact_recall(result.selected_agents, supporting_titles)
        return {
            "query": result.query,
            "prediction": result.answer,
            "answer_source": result.answer_source,
            "gold_answer": gold_answer,
            "selected_agents": result.selected_agents,
            "router_strategy": result.router_strategy,
            "em": em,
            "f1": f1,
            "soft_em": soft_em,
            "relaxed_f1": relaxed_f1,
            "accuracy": acc,
            "support_fact_recall": sfr,
            "avg_selected_privacy_cost": result.total_privacy_cost,
            "hrhr": float(self.high_risk_hit(result.selected_agents)),
            "bbr": float(result.total_privacy_cost > result.privacy_budget + 1e-9),
            "avg_activated_sources": len(result.selected_agents),
            "latency_routing": result.router_latency_ms,
            "latency_e2e": result.e2e_latency_ms,
            "privacy_budget": result.privacy_budget,
            "fused_context": result.fused_context,
            "fusion_state": result.fusion_state,
            "agent_responses": result.agent_responses,
            "router_scored_agents": result.router_scored_agents,
        }

    def summarize(self, rows: list[dict]) -> dict:
        if not rows:
            return {}
        keys = [
            "em",
            "f1",
            "soft_em",
            "relaxed_f1",
            "accuracy",
            "support_fact_recall",
            "avg_selected_privacy_cost",
            "hrhr",
            "bbr",
            "avg_activated_sources",
            "latency_routing",
            "latency_e2e",
        ]
        summary = {"count": len(rows)}
        for key in keys:
            summary[key] = sum(float(row[key]) for row in rows) / len(rows)
        return summary

    def export(self, rows: list[dict], summary: dict, output_dir: str | Path) -> tuple[Path, Path, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "query_results.jsonl"
        csv_path = output_dir / "query_results.csv"
        summary_path = output_dir / "summary.json"

        with jsonl_path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        flat_rows = []
        for row in rows:
            row_copy = dict(row)
            row_copy["selected_agents"] = ",".join(row_copy["selected_agents"])
            for key in ["fusion_state", "agent_responses", "router_scored_agents"]:
                row_copy[key] = json.dumps(row_copy[key], ensure_ascii=False)
            flat_rows.append(row_copy)
        if flat_rows:
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(flat_rows[0].keys()))
                writer.writeheader()
                writer.writerows(flat_rows)
        else:
            csv_path.write_text("", encoding="utf-8")

        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        return jsonl_path, csv_path, summary_path

    def support_fact_recall(self, selected_agents: list[str], supporting_titles: list[str]) -> float:
        if not supporting_titles:
            return 0.0
        covered = 0
        target_titles = {title.lower() for title in supporting_titles}
        selected_titles = set()
        for agent_id in selected_agents:
            selected_titles.update(self.agent_title_map.get(agent_id, set()))
        for title in target_titles:
            if title in selected_titles:
                covered += 1
        return covered / len(target_titles)

    def high_risk_hit(self, selected_agents: list[str]) -> bool:
        for agent_id in selected_agents:
            agent = self.metadata_store.get_agent(agent_id)
            if agent.privacy_cost >= 0.75:
                return True
        return False

    def _build_agent_title_map(self) -> dict[str, set[str]]:
        title_map: dict[str, set[str]] = {}
        for agent in self.metadata_store.get_all_agents():
            titles: set[str] = set()
            chunk_path = Path(agent.chunk_path)
            if chunk_path.exists():
                with chunk_path.open("r", encoding="utf-8") as file:
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        titles.add(item.get("title", "").lower())
            title_map[agent.agent_id] = titles
        return title_map
