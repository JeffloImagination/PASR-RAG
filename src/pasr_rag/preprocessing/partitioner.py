from __future__ import annotations

import hashlib
from collections import defaultdict

from ..schemas import QAExample, SourceDocument


class EntityBalancedPartitioner:
    """Partition documents into agent-local sources.

    Supporting titles are intentionally spread across 2-3 nearby agents to
    approximate the thesis requirement that supporting facts are distributed
    across multiple sources.
    """

    def __init__(self, target_sources: int) -> None:
        if target_sources < 2:
            raise ValueError("target_sources must be at least 2")
        self.target_sources = target_sources

    def partition(self, examples: list[QAExample]) -> dict[str, list[SourceDocument]]:
        sources: dict[str, list[SourceDocument]] = defaultdict(list)
        seen_doc_ids: set[str] = set()

        for example in examples:
            support_map = {
                title: self._support_agents(example.question_id, title)
                for title in example.supporting_titles
            }
            for document in example.documents:
                if document.doc_id in seen_doc_ids:
                    continue
                target_agent = self._pick_agent(document, support_map)
                sources[target_agent].append(document)
                seen_doc_ids.add(document.doc_id)

        for idx in range(self.target_sources):
            agent_id = f"A_{idx:02d}"
            sources.setdefault(agent_id, [])
        return dict(sorted(sources.items()))

    def _support_agents(self, question_id: str, title: str) -> list[str]:
        seed = self._hash_to_int(f"{question_id}:{title}")
        primary = seed % self.target_sources
        spread = 2 + (seed % 2)
        return [f"A_{(primary + offset) % self.target_sources:02d}" for offset in range(spread)]

    def _pick_agent(
        self,
        document: SourceDocument,
        support_map: dict[str, list[str]],
    ) -> str:
        if document.title in support_map:
            support_agents = support_map[document.title]
            selected = support_agents[self._hash_to_int(document.doc_id) % len(support_agents)]
            return selected
        return f"A_{self._hash_to_int(document.title) % self.target_sources:02d}"

    @staticmethod
    def _hash_to_int(text: str) -> int:
        return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
