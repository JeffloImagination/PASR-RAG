from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AppConfig
from ..preprocessing.embedding import build_embedding_encoder
from ..privacy import AgentMetadataStore
from ..schemas import AgentMeta, AgentResponse
from .bm25 import BM25Index
from .query_analysis import compute_slot_hits, extract_query_analysis


def _is_connection_error(exc: Exception | str) -> bool:
    detail = str(exc).lower()
    return "connection error" in detail or "apiconnectionerror" in detail


class RetrievalAgentExecutor:
    _bm25_cache: dict[tuple[str, float], BM25Index] = {}

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.encoder = (
            build_embedding_encoder(config)
            if self.config.retrieval.backend.lower() != "bm25"
            else None
        )
        self._summary_prompt_template = self._load_summary_prompt_template()
        self._prompt_digest = hashlib.md5(
            self._summary_prompt_template.encode("utf-8")
        ).hexdigest()[:8]
        self._summary_chunk_char_limit = 900
        self._summary_prompt_char_budget = 9000

    def run(self, query: str, agent: AgentMeta) -> AgentResponse:
        start = time.perf_counter()
        analysis = extract_query_analysis(query)
        try:
            cache_path = self._cache_path(query, agent.agent_id)
            if self.config.retrieval.cache_enabled and cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                payload["retrieval_latency_ms"] = (time.perf_counter() - start) * 1000
                return AgentResponse(**payload)

            chunks = self._load_chunks(agent.chunk_path)
            if not chunks:
                response = AgentResponse(
                    agent_id=agent.agent_id,
                    local_summary="",
                    source_chunks_count=0,
                    retrieval_latency_ms=(time.perf_counter() - start) * 1000,
                    summary_mode="empty",
                    structured_summary={},
                    retrieval_debug={"question_type": analysis.question_type},
                    error=None,
                )
                self._save_cache(cache_path, response)
                return response

            scores, top_indices, retrieval_debug = self._retrieve_top_indices(
                query,
                agent,
                chunks,
                analysis,
            )

            retrieved_chunks: list[dict] = []
            for idx in top_indices:
                item = dict(chunks[int(idx)])
                item["score"] = float(scores[int(idx)]) if len(scores) > int(idx) else 0.0
                item["slot_hits"] = compute_slot_hits(item.get("content", ""), analysis)
                retrieved_chunks.append(item)

            local_summary, summary_mode, structured_summary = self._summarize(
                query,
                agent.agent_id,
                retrieved_chunks,
                analysis,
            )
            response = AgentResponse(
                agent_id=agent.agent_id,
                local_summary=local_summary,
                source_chunks_count=len(retrieved_chunks),
                retrieval_latency_ms=(time.perf_counter() - start) * 1000,
                summary_mode=summary_mode,
                structured_summary=structured_summary,
                retrieval_debug={
                    **retrieval_debug,
                    "top_chunks": [
                        {
                            "chunk_id": item.get("chunk_id"),
                            "title": item.get("title"),
                            "score": item.get("score"),
                            "slot_hits": item.get("slot_hits", {}),
                        }
                        for item in retrieved_chunks
                    ],
                },
                error=None,
            )
            self._save_cache(cache_path, response)
            return response
        except Exception as exc:
            if _is_connection_error(exc):
                raise
            return AgentResponse(
                agent_id=agent.agent_id,
                local_summary="",
                source_chunks_count=0,
                retrieval_latency_ms=(time.perf_counter() - start) * 1000,
                summary_mode="error",
                structured_summary={},
                retrieval_debug={"question_type": analysis.question_type},
                error=str(exc),
            )

    def _load_chunks(self, chunk_path: str) -> list[dict]:
        path = Path(chunk_path)
        if not path.exists():
            return []
        items: list[dict] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def _retrieve_top_indices(
        self,
        query: str,
        agent: AgentMeta,
        chunks: list[dict],
        analysis: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        backend = self.config.retrieval.backend.lower()
        top_k = min(self.config.retrieval.top_m_per_source, len(chunks))
        if top_k == 0:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=int),
                {"backend": backend, "question_type": analysis.question_type},
            )
        if backend == "bm25":
            scores, top_indices, debug = self._retrieve_bm25(query, agent, chunks, top_k, analysis)
            return scores, top_indices, debug
        scores, top_indices = self._retrieve_vector(query, agent, chunks, top_k)
        return (
            scores,
            top_indices,
            {
                "backend": backend,
                "question_type": analysis.question_type,
                "expanded_queries": [query],
                "extracted_slots": self._analysis_to_slots(analysis),
            },
        )

    def _retrieve_vector(
        self,
        query: str,
        agent: AgentMeta,
        chunks: list[dict],
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.encoder is None:
            raise RuntimeError("Vector retrieval requested but embedding encoder is unavailable.")
        vectors = np.load(agent.vector_path)
        query_vector = self.encoder.encode([query], is_query=True)[0]
        scores = vectors @ query_vector
        top_indices = np.argsort(scores)[::-1][:top_k]
        return scores, top_indices

    def _retrieve_bm25(
        self,
        query: str,
        agent: AgentMeta,
        chunks: list[dict],
        top_k: int,
        analysis: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        index = self._load_bm25_index(agent, chunks)
        expanded_queries = analysis.expanded_queries or [query]
        weights = self._expanded_query_weights(expanded_queries, analysis)
        combined = np.zeros(len(chunks), dtype=np.float32)

        for expanded_query, weight in zip(expanded_queries, weights):
            query_scores = np.asarray(index.score(expanded_query), dtype=np.float32)
            combined = np.maximum(combined, query_scores * weight)

        top_indices = np.argsort(combined)[::-1][:top_k]
        debug = {
            "backend": "bm25",
            "question_type": analysis.question_type,
            "expanded_queries": expanded_queries,
            "expanded_query_weights": weights,
            "extracted_slots": self._analysis_to_slots(analysis),
        }
        return combined, top_indices, debug

    def _load_bm25_index(self, agent: AgentMeta, chunks: list[dict]) -> BM25Index:
        bm25_path = Path(getattr(agent, "bm25_index_path", ""))
        if bm25_path.exists():
            mtime = bm25_path.stat().st_mtime
            cache_key = (str(bm25_path.resolve()), mtime)
            index = self._bm25_cache.get(cache_key)
            if index is None:
                index = BM25Index.load(bm25_path)
                self._bm25_cache[cache_key] = index
            return index

        chunk_path = Path(agent.chunk_path)
        mtime = chunk_path.stat().st_mtime if chunk_path.exists() else 0.0
        cache_key = (str(chunk_path.resolve()) if chunk_path.exists() else str(chunk_path), mtime)
        index = self._bm25_cache.get(cache_key)
        if index is None:
            texts = [f"{item.get('title', '')}\n{item.get('content', '')}".strip() for item in chunks]
            index = BM25Index.from_texts(
                texts,
                k1=self.config.retrieval.bm25_k1,
                b=self.config.retrieval.bm25_b,
            )
            self._bm25_cache[cache_key] = index
        return index

    def _expanded_query_weights(self, expanded_queries: list[str], analysis: Any) -> list[float]:
        weights: list[float] = []
        for idx, expanded_query in enumerate(expanded_queries):
            weight = 1.0
            if idx == 0:
                weight = 1.2
            if analysis.metric and analysis.metric in expanded_query:
                weight += 0.2
            if analysis.comparison_target and analysis.comparison_target in expanded_query:
                weight += 0.1
            if any(company in expanded_query for company in analysis.companies):
                weight += 0.1
            weights.append(weight)
        return weights

    def _summarize(
        self,
        query: str,
        agent_id: str,
        retrieved_chunks: list[dict],
        analysis: Any,
    ) -> tuple[str, str, dict[str, Any]]:
        if not retrieved_chunks:
            empty_payload = {
                "direct_support": [],
                "inferential_bridges": [],
                "answer_candidates": [],
                "reasoning_notes": "该知识源未检索到可用证据。",
                "missing_slots": [slot for slot, value in self._analysis_to_slots(analysis).items() if value],
                "confidence": 0.0,
            }
            return "", "empty", empty_payload

        if not self.config.generation.enable_summarization:
            fallback_text = "\n".join(item["content"] for item in retrieved_chunks)
            return fallback_text, "disabled", {
                "direct_support": [],
                "inferential_bridges": [],
                "answer_candidates": [],
                "reasoning_notes": "局部总结被禁用，直接返回原始文本。",
                "missing_slots": [],
                "confidence": 0.0,
            }

        payload = self._summarize_with_llm(query, agent_id, retrieved_chunks, analysis)
        if payload:
            summary_text = self._render_structured_summary(payload)
            return summary_text, "llm_structured", payload

        fallback_payload = self._extractive_fallback_payload(retrieved_chunks, analysis)
        fallback_text = self._render_structured_summary(fallback_payload)
        return fallback_text, "extractive_fallback", fallback_payload

    def _summarize_with_llm(
        self,
        query: str,
        agent_id: str,
        retrieved_chunks: list[dict],
        analysis: Any,
    ) -> dict[str, Any] | None:
        api_key = os.environ.get(self.config.generation.api_key_env)
        if not api_key:
            return None

        prompt = self._build_summary_prompt(query, agent_id, retrieved_chunks, analysis)
        try:
            from openai import OpenAI
        except Exception as exc:
            self._log_llm_debug("openai_import_failed", agent_id, query, str(exc))
            return None

        try:
            client = OpenAI(
                api_key=api_key,
                base_url=self.config.generation.api_base_url,
                timeout=90.0,
            )
        except Exception as exc:
            self._log_llm_debug("client_init_failed", agent_id, query, str(exc))
            return None

        request_kwargs = dict(
            model=self.config.models.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a local evidence extractor. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.generation.temperature,
            max_tokens=self.config.generation.max_summary_tokens,
        )
        response = None
        try:
            response = client.chat.completions.create(
                response_format={"type": "json_object"},
                timeout=90.0,
                **request_kwargs,
            )
        except Exception as exc:
            self._log_llm_debug("json_mode_failed", agent_id, query, str(exc))
            if _is_connection_error(exc):
                raise RuntimeError(
                    f"Local summary LLM connection error for {agent_id}: {exc}"
                ) from exc
            try:
                response = client.chat.completions.create(**request_kwargs)
            except Exception as retry_exc:
                self._log_llm_debug("plain_mode_failed", agent_id, query, str(retry_exc))
                if _is_connection_error(retry_exc):
                    raise RuntimeError(
                        f"Local summary LLM connection error for {agent_id}: {retry_exc}"
                    ) from retry_exc
                return None

        try:
            content = response.choices[0].message.content or ""
        except Exception as exc:
            self._log_llm_debug("empty_choice", agent_id, query, str(exc))
            return None

        payload = self._parse_summary_json(content)
        if not payload:
            self._log_llm_debug("json_parse_failed", agent_id, query, content[:1200])
            return None
        normalized = self._normalize_summary_payload(payload, analysis)
        if not normalized:
            self._log_llm_debug(
                "payload_normalize_failed",
                agent_id,
                query,
                json.dumps(payload, ensure_ascii=False)[:1200],
            )
            return None
        return normalized

    def _build_summary_prompt(
        self,
        query: str,
        agent_id: str,
        retrieved_chunks: list[dict],
        analysis: Any,
    ) -> str:
        max_tokens = self.config.generation.max_summary_tokens
        prompt = self._summary_prompt_template.replace("{{MAX_TOKENS}}", str(max_tokens))
        prompt = prompt.replace("{{AGENT_ID}}", agent_id)

        evidence_blocks = []
        for idx, item in enumerate(retrieved_chunks, start=1):
            evidence_blocks.append(
                "\n".join(
                    [
                        f"[片段{idx}]",
                        f"标题: {item.get('title', '')}",
                        f"相关性分数: {item.get('score', 0.0):.4f}",
                        f"关键槽位命中: {json.dumps(item.get('slot_hits', {}), ensure_ascii=False)}",
                        "内容:",
                        self._clip_for_prompt(str(item.get("content", "")).strip()),
                    ]
                )
            )

        schema_block = (
            "你必须返回 JSON，包含以下字段：\n"
            "{\n"
            '  "direct_support": ["短句，保留公司名/年份/指标/数值/单位"],\n'
            '  "inferential_bridges": ["必要的桥接信息"],\n'
            '  "answer_candidates": ["是/否/数值/实体名/高于/低于/持平"],\n'
            '  "reasoning_notes": "给中心智能体的简短建议",\n'
            '  "missing_slots": ["缺失字段"],\n'
            '  "confidence": 0.0\n'
            "}\n"
            "如果没有证据，也必须返回上述字段，且 answer_candidates 为空。"
        )

        full_prompt = (
            f"{prompt}\n\n"
            f"用户查询:\n{query}\n\n"
            f"题型与槽位:\n{json.dumps(self._analysis_to_slots(analysis) | {'question_type': analysis.question_type}, ensure_ascii=False)}\n\n"
            f"知识源ID:\n{agent_id}\n\n"
            "本地检索片段:\n"
            + "\n\n".join(evidence_blocks)
            + f"\n\n{schema_block}"
        )
        return self._clip_prompt_budget(full_prompt)

    def _parse_summary_json(self, content: str) -> Optional[Dict[str, Any]]:
        text = content.strip()
        if not text:
            return None

        # 第一步：直接尝试解析原始字符串
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 第二步：首次解析失败，移除 Markdown JSON 代码块标记（```json ... ``` / ``` ... ```）
        # 匹配并删除开头的 ```json 、```  和结尾的 ``` 标记
        cleaned_text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'\s*```$', '', cleaned_text, flags=re.MULTILINE)
        cleaned_text = cleaned_text.strip()

        # 尝试解析去除代码块标记后的内容
        if cleaned_text != text:
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                pass

        # 第三步：去除标记后仍失败，执行原有逻辑：提取 {} 包裹的内容再解析
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _normalize_summary_payload(
        self,
        payload: dict[str, Any],
        analysis: Any,
    ) -> dict[str, Any] | None:
        evidence_payload = payload.get("evidence_payload")
        base = evidence_payload if isinstance(evidence_payload, dict) else payload

        def as_clean_list(value: Any) -> list[str]:
            if not isinstance(value, list):
                return []
            cleaned_items: list[str] = []
            for item in value:
                cleaned = self._clean_support_text(str(item))
                if cleaned:
                    cleaned_items.append(cleaned)
            return cleaned_items

        direct_support = as_clean_list(base.get("direct_support"))
        inferential_bridges = as_clean_list(base.get("inferential_bridges"))
        answer_candidates = as_clean_list(base.get("answer_candidates"))
        missing_slots = as_clean_list(base.get("missing_slots"))
        reasoning_notes = self._clean_support_text(str(base.get("reasoning_notes", "")).strip())
        confidence = base.get("confidence", payload.get("confidence", 0.0))

        if not answer_candidates:
            answer_candidates = self._derive_answer_candidates(direct_support, analysis)
        if not missing_slots:
            missing_slots = self._infer_missing_slots(direct_support, analysis)

        normalized = {
            "direct_support": direct_support,
            "inferential_bridges": inferential_bridges,
            "answer_candidates": answer_candidates,
            "reasoning_notes": reasoning_notes,
            "missing_slots": missing_slots,
            "confidence": float(confidence or 0.0),
        }
        if not any(
            [
                normalized["direct_support"],
                normalized["inferential_bridges"],
                normalized["answer_candidates"],
                normalized["reasoning_notes"],
                normalized["missing_slots"],
            ]
        ):
            return None
        return normalized

    def _extractive_fallback_payload(self, retrieved_chunks: list[dict], analysis: Any) -> dict[str, Any]:
        direct_support = []
        for item in retrieved_chunks[:3]:
            text = self._clean_support_text(str(item.get("content", "")).strip())
            if not text:
                continue
            source_label = self._clean_support_text(
                str(item.get("source_hint") or item.get("title", "")).strip()
            )
            direct_support.append(f"{source_label}: {text}" if source_label else text)
        return {
            "direct_support": direct_support,
            "inferential_bridges": [],
            "answer_candidates": self._derive_answer_candidates(direct_support, analysis),
            "reasoning_notes": "局部结构化摘要失败，当前为提取式回退结果，中心端应降低其权重。",
            "missing_slots": self._infer_missing_slots(direct_support, analysis),
            "confidence": 0.2,
        }

    def _render_structured_summary(self, payload: dict[str, Any]) -> str:
        sections: list[str] = []
        if payload.get("direct_support"):
            sections.append("Direct Support:")
            sections.extend(f"- {item}" for item in payload["direct_support"])
        if payload.get("inferential_bridges"):
            sections.append("Inferential Bridges:")
            sections.extend(f"- {item}" for item in payload["inferential_bridges"])
        if payload.get("answer_candidates"):
            sections.append("Answer Candidates:")
            sections.extend(f"- {item}" for item in payload["answer_candidates"])
        if payload.get("missing_slots"):
            sections.append("Missing Slots:")
            sections.extend(f"- {item}" for item in payload["missing_slots"])
        if payload.get("reasoning_notes"):
            sections.append("Reasoning Notes:")
            sections.append(str(payload["reasoning_notes"]))
        sections.append(f"Confidence: {payload.get('confidence', 0.0):.2f}")
        return "\n".join(sections).strip()

    def _derive_answer_candidates(self, support_lines: list[str], analysis: Any) -> list[str]:
        candidates: list[str] = []
        joined = "\n".join(support_lines)
        if analysis.question_type == "yes_no_compare":
            if "高于" in joined:
                candidates.append("是")
            if "低于" in joined and analysis.direction == "higher":
                candidates.append("否")
            if "低于" in joined and analysis.direction == "lower":
                candidates.append("是")
            if "高于" in joined and analysis.direction == "lower":
                candidates.append("否")
            if "是" in joined:
                candidates.append("是")
            if "否" in joined:
                candidates.append("否")
        elif analysis.question_type == "difference_numeric":
            candidates.extend(self._extract_numeric_candidates(joined, analysis, limit=3))
        elif analysis.question_type == "argmax_or_choice":
            for company in analysis.companies:
                if company in joined:
                    candidates.append(company)
            if analysis.comparison_target and analysis.comparison_target in joined:
                candidates.append(analysis.comparison_target)
        else:
            candidates.extend(self._extract_numeric_candidates(joined, analysis, limit=2))
        return _unique_keep_order(candidates)

    def _infer_missing_slots(self, support_lines: list[str], analysis: Any) -> list[str]:
        joined = "\n".join(support_lines)
        missing: list[str] = []
        if analysis.metric and analysis.metric not in joined:
            missing.append("metric")
        if analysis.years and not any(year in joined for year in analysis.years):
            missing.append("year")
        if analysis.companies and not any(company in joined for company in analysis.companies):
            missing.append("company")
        if analysis.comparison_target and analysis.comparison_target not in joined:
            missing.append("comparison_target")
        if (
            analysis.question_type in {"difference_numeric", "attribute_lookup"}
            and not self._extract_numeric_candidates(joined, analysis, limit=1)
        ):
            missing.append("numeric_value")
        if analysis.question_type == "yes_no_compare" and not any(token in joined for token in ["是", "否", "高于", "低于"]):
            missing.append("comparison_result")
        return missing

    def _clean_support_text(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"\b[a-z_]+:[0-9a-f]{8}-[0-9a-f-]{27,}\b", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" :,-")
        return self._clip_for_prompt(cleaned)

    def _extract_numeric_candidates(self, text: str, analysis: Any, limit: int) -> list[str]:
        candidates: list[str] = []
        for match in re.finditer(r"-?\d[\d,]*(?:\.\d+)?%?", text):
            token = match.group(0).replace(",", "")
            window_start = max(0, match.start() - 18)
            window_end = min(len(text), match.end() + 18)
            window = text[window_start:window_end]
            if analysis.years and token in analysis.years:
                continue
            if re.fullmatch(r"\d{6,}", token) and "." not in token and "%" not in token:
                continue
            if (
                "%" not in token
                and "." not in token
                and len(token) >= 5
                and not any(
                    keyword in window
                    for keyword in [
                        analysis.metric,
                        "人数",
                        "总数",
                        "均值",
                        "平均",
                        "同比",
                        "差值",
                        "高于",
                        "低于",
                        "金额",
                        "营收",
                        "利润",
                    ]
                    if keyword
                )
            ):
                continue
            candidates.append(token)
            if len(candidates) >= limit:
                break
        return _unique_keep_order(candidates)

    def _analysis_to_slots(self, analysis: Any) -> dict[str, Any]:
        return {
            "years": analysis.years,
            "companies": analysis.companies,
            "metric": analysis.metric,
            "comparison_target": analysis.comparison_target,
            "direction": analysis.direction,
        }

    def _load_summary_prompt_template(self) -> str:
        prompt_path = Path(self.config.generation.local_summary_prompt_path)
        if not prompt_path.is_absolute():
            prompt_path = prompt_path.resolve()
        if not prompt_path.exists():
            return (
                "你是隐私感知多智能体RAG系统中的一个局部知识摘要智能体。"
                "请基于本知识源内检索到的文档片段，返回紧凑摘要。"
            )
        return prompt_path.read_text(encoding="utf-8")

    def _cache_path(self, query: str, agent_id: str) -> Path:
        backend = self.config.retrieval.backend.lower()
        version = f"llm_local_summary_v4_{backend}_{self._prompt_digest}"
        digest = hashlib.md5(f"{version}:{agent_id}:{query}".encode("utf-8")).hexdigest()
        return self.config.paths.cache_root / f"{agent_id}_{digest}.json"

    def _save_cache(self, cache_path: Path, response: AgentResponse) -> None:
        if not self.config.retrieval.cache_enabled:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "agent_id": response.agent_id,
                    "local_summary": response.local_summary,
                    "source_chunks_count": response.source_chunks_count,
                    "retrieval_latency_ms": response.retrieval_latency_ms,
                    "summary_mode": response.summary_mode,
                    "structured_summary": response.structured_summary,
                    "retrieval_debug": response.retrieval_debug,
                    "error": response.error,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _clip_for_prompt(self, text: str) -> str:
        if len(text) <= self._summary_chunk_char_limit:
            return text
        return text[: self._summary_chunk_char_limit].rstrip() + " ..."

    def _clip_prompt_budget(self, prompt: str) -> str:
        if len(prompt) <= self._summary_prompt_char_budget:
            return prompt
        return prompt[: self._summary_prompt_char_budget].rstrip() + "\n\n[Truncated for model budget]"

    def _log_llm_debug(self, tag: str, agent_id: str, query: str, detail: str) -> None:
        try:
            log_path = self.config.paths.logs_root / "local_summary_llm_debug.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "tag": tag,
                            "agent_id": agent_id,
                            "query": query,
                            "detail": detail,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass


def execute_agents_for_query(
    query: str,
    selected_agent_ids: list[str],
    metadata_store: AgentMetadataStore,
    config: AppConfig,
) -> list[AgentResponse]:
    executor = RetrievalAgentExecutor(config)
    agents = [metadata_store.get_agent(agent_id) for agent_id in selected_agent_ids]
    if not config.retrieval.enable_parallel_retrieval or len(agents) <= 1:
        return [executor.run(query, agent) for agent in agents]

    responses: list[AgentResponse] = []
    with ThreadPoolExecutor(max_workers=config.retrieval.max_workers) as pool:
        future_map = {pool.submit(executor.run, query, agent): agent.agent_id for agent in agents}
        for future in as_completed(future_map):
            responses.append(future.result())
    responses.sort(key=lambda item: selected_agent_ids.index(item.agent_id))
    return responses


def _unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        stripped = str(item).strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        result.append(stripped)
    return result
