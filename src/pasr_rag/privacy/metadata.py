from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ..schemas import AgentMeta


class AgentMetadataStore:
    def __init__(self, metadata_path: str | Path) -> None:
        self.metadata_path = Path(metadata_path)
        self._cache: dict[str, AgentMeta] = {}
        self.reload()

    def reload(self) -> None:
        if not self.metadata_path.exists():
            self._cache = {}
            return
        with self.metadata_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        self._cache = {
            item["agent_id"]: AgentMeta(**self._normalize_item(item))
            for item in payload
        }

    def _normalize_item(self, item: dict) -> dict:
        normalized = dict(item)
        agent_id = normalized["agent_id"]
        if "chunk_path" not in normalized or not normalized["chunk_path"]:
            normalized["chunk_path"] = str(
                self.metadata_path.parent / agent_id / f"{agent_id}_chunks.jsonl"
            )
        if "vector_path" not in normalized or not normalized["vector_path"]:
            normalized["vector_path"] = str(
                self.metadata_path.parent / agent_id / f"{agent_id}_vectors.npy"
            )
        if "centroid_path" not in normalized or not normalized["centroid_path"]:
            normalized["centroid_path"] = str(
                self.metadata_path.parent / agent_id / f"{agent_id}_centroid.npy"
            )
        if "bm25_index_path" not in normalized or not normalized["bm25_index_path"]:
            normalized["bm25_index_path"] = str(
                self.metadata_path.parent / agent_id / f"{agent_id}_bm25_index.json"
            )
        normalized.setdefault("privacy_level", "")
        normalized.setdefault("privacy_reason", "")
        normalized.setdefault("privacy_confidence", 0.0)
        normalized.setdefault("status", "active")
        normalized.setdefault("update_time", "")
        return normalized

    def get_all_agents(self) -> list[AgentMeta]:
        return list(self._cache.values())

    def get_agent(self, agent_id: str) -> AgentMeta:
        return self._cache[agent_id]

    def update_agent_privacy(
        self,
        agent_id: str,
        privacy_level: str,
        privacy_cost: float,
        reason: str,
        confidence: float,
    ) -> None:
        agent = self._cache[agent_id]
        agent.privacy_level = privacy_level
        agent.privacy_cost = privacy_cost
        agent.privacy_reason = reason
        agent.privacy_confidence = confidence
        agent.update_time = datetime.utcnow().isoformat(timespec="seconds")

    def save(self) -> Path:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metadata_path.open("w", encoding="utf-8") as file:
            json.dump(
                [asdict(item) for item in self.get_all_agents()],
                file,
                ensure_ascii=False,
                indent=2,
            )
        return self.metadata_path
