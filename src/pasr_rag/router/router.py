from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import AppConfig
from ..privacy import AgentMetadataStore
from ..preprocessing.embedding import build_embedding_encoder
from ..schemas import AgentMeta


@dataclass
class RouterDecision:
    query: str
    strategy: str
    selected_agents: list[str]
    total_privacy_cost: float
    privacy_budget: float
    routing_latency_ms: float
    scored_agents: list[dict]


class BaseRouter:
    strategy_name = "base"

    def __init__(self, config: AppConfig, metadata_store: AgentMetadataStore) -> None:
        self.config = config
        self.metadata_store = metadata_store
        self.encoder = build_embedding_encoder(config)
        self.random = random.Random(config.project.random_seed)

    def select_agents(self, query: str) -> RouterDecision:
        start = time.perf_counter()
        agent_metas = [agent for agent in self.metadata_store.get_all_agents() if agent.status == "active"]
        scored = self.score_agents(query, agent_metas)
        budget = self.resolve_budget(scored)
        selected = self.apply_selection(scored, budget)
        latency_ms = (time.perf_counter() - start) * 1000
        return RouterDecision(
            query=query,
            strategy=self.strategy_name,
            selected_agents=[item["agent_id"] for item in selected],
            total_privacy_cost=sum(item["privacy_cost"] for item in selected),
            privacy_budget=budget,
            routing_latency_ms=latency_ms,
            scored_agents=scored,
        )

    def score_agents(self, query: str, agent_metas: list[AgentMeta]) -> list[dict]:
        query_vector = self.encoder.encode([query], is_query=True)[0]
        scored: list[dict] = []
        for agent in agent_metas:
            relevance = self.compute_relevance(query_vector, agent)
            utility = self.compute_utility(relevance, agent.privacy_cost)
            scored.append(
                {
                    "agent_id": agent.agent_id,
                    "relevance_score": relevance,
                    "privacy_cost": agent.privacy_cost,
                    "utility_score": utility,
                    "privacy_level": agent.privacy_level,
                }
            )
        return sorted(scored, key=lambda item: item["utility_score"], reverse=True)

    def compute_relevance(self, query_vector: np.ndarray, agent: AgentMeta) -> float:
        relevance_method = self.config.router.relevance_method.lower()
        if relevance_method in {"centroid_cosine", "centroid"}:
            return self._compute_centroid_relevance(query_vector, agent)
        return self._compute_max_cosine_relevance(query_vector, agent)

    def _compute_centroid_relevance(self, query_vector: np.ndarray, agent: AgentMeta) -> float:
        centroid_path = getattr(agent, "centroid_path", "")
        if centroid_path and Path(centroid_path).exists():
            centroid = np.load(centroid_path)
            if centroid.size == 0:
                return 0.0
            centroid = centroid.astype(np.float32)
            norm = float(np.linalg.norm(centroid))
            if norm > 0:
                centroid = centroid / norm
            return float(np.dot(query_vector, centroid))
        return self._compute_max_cosine_relevance(query_vector, agent)

    def _compute_max_cosine_relevance(self, query_vector: np.ndarray, agent: AgentMeta) -> float:
        if not agent.vector_path or not Path(agent.vector_path).exists():
            return 0.0
        vectors = np.load(agent.vector_path)
        if vectors.size == 0:
            return 0.0
        scores = vectors @ query_vector
        return float(np.max(scores))

    def compute_utility(self, relevance: float, privacy_cost: float) -> float:
        alpha = self.config.router.alpha if self.config.privacy.enable_relevance_term else 0.0
        beta = self.config.router.beta if self.config.privacy.enable_privacy_term else 0.0
        return alpha * relevance - beta * privacy_cost

    def resolve_budget(self, scored_agents: list[dict]) -> float:
        if self.config.router.privacy_budget_mode == "ratio":
            return sum(item["privacy_cost"] for item in scored_agents) * self.config.router.privacy_budget_ratio
        return self.config.router.privacy_budget_B

    def apply_selection(self, scored_agents: list[dict], budget: float) -> list[dict]:
        raise NotImplementedError


class PASRRouter(BaseRouter):
    strategy_name = "pasr"

    def apply_selection(self, scored_agents: list[dict], budget: float) -> list[dict]:
        selected: list[dict] = []
        current_cost = 0.0
        max_k = self.config.router.max_active_sources_k
        for item in scored_agents:
            if len(selected) >= max_k:
                break
            if current_cost + item["privacy_cost"] <= budget:
                selected.append(item)
                current_cost += item["privacy_cost"]
        return selected


class RelOnlyRouter(PASRRouter):
    strategy_name = "rel_only"

    def compute_utility(self, relevance: float, privacy_cost: float) -> float:
        return relevance


class MARAGLiteRouter(RelOnlyRouter):
    strategy_name = "ma_rag_lite"


class RandomRouter(PASRRouter):
    strategy_name = "random"

    def score_agents(self, query: str, agent_metas: list[AgentMeta]) -> list[dict]:
        scored = super().score_agents(query, agent_metas)
        self.random.shuffle(scored)
        return scored


class ThresholdRouter(PASRRouter):
    strategy_name = "threshold"

    def apply_selection(self, scored_agents: list[dict], budget: float) -> list[dict]:
        threshold = self.config.router.threshold
        filtered = [item for item in scored_agents if item["relevance_score"] >= threshold]
        return super().apply_selection(filtered, budget)


def build_router(config: AppConfig, metadata_store: AgentMetadataStore) -> BaseRouter:
    strategy = config.router.strategy.lower()
    if strategy == "pasr":
        return PASRRouter(config, metadata_store)
    if strategy == "rel_only":
        return RelOnlyRouter(config, metadata_store)
    if strategy == "ma_rag_lite":
        return MARAGLiteRouter(config, metadata_store)
    if strategy == "random":
        return RandomRouter(config, metadata_store)
    if strategy == "threshold":
        return ThresholdRouter(config, metadata_store)
    raise ValueError(f"Unknown router strategy: {config.router.strategy}")
