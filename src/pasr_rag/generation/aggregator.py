from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..schemas import AgentResponse
from ..retrieval.query_analysis import extract_query_analysis


def _is_connection_error(exc: Exception | str) -> bool:
    detail = str(exc).lower()
    return "connection error" in detail or "apiconnectionerror" in detail


def _load_text_template(path_text: str, fallback: str) -> str:
    prompt_path = Path(path_text)
    if not prompt_path.is_absolute():
        prompt_path = prompt_path.resolve()
    if not prompt_path.exists():
        return fallback
    return prompt_path.read_text(encoding="utf-8")


class _OpenAICompatibleMixin:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client: Any | None = None

    def _get_client(self, api_key: str) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package is not installed") from exc

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.config.generation.api_base_url,
            timeout=60.0,
        )
        return self._client

    def _chat_once(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        api_key = os.environ.get(self.config.generation.api_key_env)
        if not api_key:
            self._log_chat_debug("missing_api_key", "", "")
            return ""

        try:
            client = self._get_client(api_key)
            response = client.chat.completions.create(
                model=self.config.models.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.generation.temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            self._log_chat_debug("chat_failed", user_prompt[:300], str(exc))
            if _is_connection_error(exc):
                raise RuntimeError(f"Central LLM connection error: {exc}") from exc
            return ""

        try:
            content = response.choices[0].message.content
        except Exception as exc:
            self._log_chat_debug("empty_choice", user_prompt[:300], str(exc))
            return ""
        if content is None:
            self._log_chat_debug("none_content", user_prompt[:300], "")
            return ""
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        text_parts.append(str(text))
                else:
                    text_parts.append(str(item))
            return "\n".join(text_parts).strip()
        return str(content).strip()

    def _log_chat_debug(self, tag: str, prompt_preview: str, detail: str) -> None:
        try:
            log_path = self.config.paths.logs_root / "central_llm_debug.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "tag": tag,
                            "prompt_preview": prompt_preview,
                            "detail": detail,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass


class InformationFusion(_OpenAICompatibleMixin):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._fusion_prompt_template = _load_text_template(
            self.config.generation.central_fusion_prompt_path,
            (
                "你是中心证据聚合器。请整合多个检索智能体返回的结构化证据，"
                "输出可供最终回答器或规则比较器直接使用的 JSON。"
            ),
        )
        self._fusion_prompt_budget = 12000
        self._last_state: dict[str, Any] = {}

    @property
    def last_state(self) -> dict[str, Any]:
        return self._last_state

    def fuse(self, query: str, responses: list[AgentResponse]) -> str:
        fusion_state = self.fuse_state(query, responses)
        self._last_state = fusion_state
        return self._render_fusion_state(fusion_state)

    def fuse_state(self, query: str, responses: list[AgentResponse]) -> dict[str, Any]:
        analysis = extract_query_analysis(query)
        base_state = self._build_base_state(query, responses, analysis)
        if self.config.generation.enable_central_fusion:
            llm_state = self._fuse_with_llm(query, responses, analysis)
            if llm_state:
                merged = self._merge_fusion_states(base_state, llm_state)
                merged["fusion_mode"] = "llm_structured"
                return merged
        base_state["fusion_mode"] = "rule_based"
        return base_state

    def _build_base_state(self, query: str, responses: list[AgentResponse], analysis: Any) -> dict[str, Any]:
        direct_support: list[dict[str, Any]] = []
        inferential_bridges: list[dict[str, Any]] = []
        candidate_answers: list[str] = []
        missing_information: list[str] = []
        conflicts: list[str] = []
        reasoning_basis: list[str] = []

        for response in responses:
            structured = response.structured_summary or {}
            direct_support.extend(
                {"agent_id": response.agent_id, "text": item}
                for item in structured.get("direct_support", [])
            )
            inferential_bridges.extend(
                {"agent_id": response.agent_id, "text": item}
                for item in structured.get("inferential_bridges", [])
            )
            candidate_answers.extend(structured.get("answer_candidates", []))
            missing_information.extend(structured.get("missing_slots", []))
            if response.summary_mode == "extractive_fallback":
                conflicts.append(f"{response.agent_id}: summary degraded to extractive fallback")
            if structured.get("reasoning_notes"):
                reasoning_basis.append(f"{response.agent_id}: {structured['reasoning_notes']}")

        state = {
            "question_type": analysis.question_type,
            "resolved_entities": {
                "companies": analysis.companies,
                "years": analysis.years,
                "metric": analysis.metric,
                "comparison_target": analysis.comparison_target,
                "direction": analysis.direction,
            },
            "key_values": self._extract_key_values(direct_support),
            "comparison_axis": analysis.metric,
            "candidate_answers": _unique(candidate_answers),
            "conflicts": _unique(conflicts),
            "missing_information": _unique(missing_information),
            "final_reasoning_basis": _unique(reasoning_basis),
            "direct_support": direct_support,
            "inferential_bridges": inferential_bridges,
        }
        return state

    def _extract_key_values(self, direct_support: list[dict[str, Any]]) -> list[dict[str, str]]:
        values: list[dict[str, str]] = []
        for item in direct_support:
            text = item.get("text", "")
            for match in re.finditer(r"-?\d+(?:\.\d+)?%?", text):
                values.append(
                    {
                        "agent_id": str(item.get("agent_id", "")),
                        "value": match.group(0),
                        "evidence": text[:200],
                    }
                )
        return values

    def _fuse_with_llm(self, query: str, responses: list[AgentResponse], analysis: Any) -> dict[str, Any] | None:
        agent_payloads = []
        for response in responses:
            if response.error:
                continue
            agent_payloads.append(
                {
                    "agent_id": response.agent_id,
                    "summary_mode": response.summary_mode,
                    "structured_summary": response.structured_summary,
                    "retrieval_debug": response.retrieval_debug,
                }
            )
        if not agent_payloads:
            return None

        schema = (
            '{'
            '"question_type":"...",'
            '"resolved_entities":{"companies":[],"years":[],"metric":"","comparison_target":"","direction":""},'
            '"key_values":[{"agent_id":"","value":"","evidence":""}],'
            '"comparison_axis":"...",'
            '"candidate_answers":["..."],'
            '"conflicts":["..."],'
            '"missing_information":["..."],'
            '"final_reasoning_basis":["..."]'
            '}'
        )
        prompt = (
            f"{self._fusion_prompt_template}\n\n"
            f"用户问题:\n{query}\n\n"
            f"问题槽位:\n{json.dumps({'question_type': analysis.question_type, 'companies': analysis.companies, 'years': analysis.years, 'metric': analysis.metric, 'comparison_target': analysis.comparison_target, 'direction': analysis.direction}, ensure_ascii=False)}\n\n"
            f"检索智能体结果:\n{json.dumps(agent_payloads, ensure_ascii=False)}\n\n"
            f"请严格输出 JSON，结构如下：\n{schema}"
        )
        prompt = prompt[: self._fusion_prompt_budget]
        text = self._chat_once(
            system_prompt="You are a structured evidence fuser. Return valid JSON only.",
            user_prompt=prompt,
            max_tokens=self.config.generation.max_fusion_tokens,
        )
        if not text:
            return None
        return self._parse_json(text)

    def _parse_json(self, text: str) -> dict[str, Any] | None:
        text = text.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    def _merge_fusion_states(self, base: dict[str, Any], llm_state: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key in [
            "question_type",
            "resolved_entities",
            "key_values",
            "comparison_axis",
            "candidate_answers",
            "conflicts",
            "missing_information",
            "final_reasoning_basis",
        ]:
            if key in llm_state and llm_state[key]:
                merged[key] = llm_state[key]
        return merged

    def _render_fusion_state(self, state: dict[str, Any]) -> str:
        lines = [
            f"问题类型: {state.get('question_type', '')}",
            f"关键对象: {json.dumps(state.get('resolved_entities', {}), ensure_ascii=False)}",
        ]
        key_values = state.get("key_values", [])
        if key_values:
            lines.append("关键数值:")
            lines.extend(
                f"- {item.get('value')} ({item.get('agent_id')}) <- {item.get('evidence')}"
                for item in key_values[:8]
            )
        candidates = state.get("candidate_answers", [])
        if candidates:
            lines.append("候选答案:")
            lines.extend(f"- {item}" for item in candidates[:8])
        if state.get("missing_information"):
            lines.append("缺失信息:")
            lines.extend(f"- {item}" for item in state["missing_information"][:8])
        if state.get("conflicts"):
            lines.append("冲突信息:")
            lines.extend(f"- {item}" for item in state["conflicts"][:8])
        if state.get("final_reasoning_basis"):
            lines.append("推理依据:")
            lines.extend(f"- {item}" for item in state["final_reasoning_basis"][:8])
        return "\n".join(lines).strip()


class ContextAssembler:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._answer_prompt_template = _load_text_template(
            self.config.generation.final_answer_prompt_path,
            (
                "你是 PASR-RAG 的中心回答器。"
                "请基于聚合后的结构化证据直接给出最终短答案。"
                "如果是是非题，只输出“是”或“否”；"
                "如果是比较题，只输出比较结果或实体名；"
                "如果是数值题，只输出数值及必要单位。"
            ),
        )

    def assemble(self, query: str, fused_context: str) -> str:
        prompt = (
            f"{self._answer_prompt_template}\n\n"
            f"用户问题:\n{query}\n\n"
            f"聚合证据:\n{fused_context}\n\n"
            "最终答案:"
        )
        if len(prompt) <= self.config.generation.max_context_tokens:
            return prompt
        return prompt[: self.config.generation.max_context_tokens].rstrip()


class FinalAnswerResolver:
    def resolve(self, query: str, fusion_state: dict[str, Any]) -> tuple[str, str]:
        analysis = extract_query_analysis(query)
        candidates = [str(item).strip() for item in fusion_state.get("candidate_answers", []) if str(item).strip()]
        key_values = fusion_state.get("key_values", [])
        missing = fusion_state.get("missing_information", [])

        if analysis.question_type == "yes_no_compare":
            answer = self._resolve_yes_no(candidates)
            if answer:
                return answer, "rule"
        elif analysis.question_type == "difference_numeric":
            answer = self._resolve_numeric_difference(candidates, key_values)
            if answer:
                return answer, "rule"
        elif analysis.question_type == "argmax_or_choice":
            answer = self._resolve_choice(candidates, analysis)
            if answer:
                return answer, "rule"
        elif analysis.question_type == "attribute_lookup":
            answer = self._resolve_attribute(candidates, key_values, analysis)
            if answer:
                return answer, "rule"

        if missing:
            return "证据不足", "rule"
        return "", "llm"

    def _resolve_yes_no(self, candidates: list[str]) -> str:
        joined = "\n".join(candidates)
        if "是" in candidates or "高于" in joined or "低于" in joined:
            if "否" in candidates and "是" not in candidates:
                return "否"
            if "是" in candidates:
                return "是"
        if "否" in candidates:
            return "否"
        return ""

    def _resolve_numeric_difference(self, candidates: list[str], key_values: list[dict[str, Any]]) -> str:
        for candidate in candidates:
            if re.fullmatch(r"-?\d+(?:\.\d+)?%?", candidate):
                return candidate
        if key_values:
            return str(key_values[0].get("value", ""))
        return ""

    def _resolve_choice(self, candidates: list[str], analysis: Any) -> str:
        for candidate in candidates:
            if candidate in analysis.companies or candidate == analysis.comparison_target:
                return candidate
        return candidates[0] if candidates else ""

    def _resolve_attribute(self, candidates: list[str], key_values: list[dict[str, Any]], analysis: Any) -> str:
        for candidate in candidates:
            if candidate in {"是", "否"}:
                return candidate
            if candidate in analysis.companies or re.fullmatch(r"-?\d+(?:\.\d+)?%?", candidate):
                return candidate
        if key_values:
            return str(key_values[0].get("value", ""))
        return ""


class CentralGenerator(_OpenAICompatibleMixin):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self.resolver = FinalAnswerResolver()

    def generate(
        self,
        query: str,
        assembled_prompt: str,
        fused_context: str,
        fusion_state: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        if not fused_context.strip():
            return "Insufficient information to answer the question.", "fallback"

        fusion_state = fusion_state or {}
        rule_answer, rule_source = self.resolver.resolve(query, fusion_state)
        if rule_answer:
            return rule_answer, rule_source

        backend = self.config.generation.backend.lower()
        if backend in {"openai_compatible", "openai-compatible", "dashscope"}:
            answer = self._generate_via_openai_compatible(assembled_prompt)
            if answer:
                return self._compress_answer(query, answer), "llm"
            if not self.config.generation.fallback_to_extract:
                return "Generation failed: no answer returned from configured backend.", "fallback"
        return self._generate_by_extraction(query, fused_context), "fallback"

    def _generate_via_openai_compatible(self, assembled_prompt: str) -> str:
        return self._chat_once(
            system_prompt="You are the final answer generator. Return only the final short answer.",
            user_prompt=assembled_prompt,
            max_tokens=self.config.generation.max_answer_tokens,
        )

    def _generate_by_extraction(self, query: str, fused_context: str) -> str:
        cleaned = self._clean_context(fused_context)
        if not cleaned:
            return "证据不足"
        if any(marker in query for marker in ["是否", "是不是"]):
            if "否" in cleaned[:40]:
                return "否"
            if "是" in cleaned[:40]:
                return "是"
        match = re.search(r"-?\d+(?:\.\d+)?%?", cleaned)
        if match:
            return match.group(0)
        return cleaned.splitlines()[0][:80]

    def _compress_answer(self, query: str, answer: str) -> str:
        text = answer.strip()
        if len(text) <= 60:
            return text
        if any(marker in query for marker in ["是否", "是不是"]):
            if "否" in text[:40]:
                return "否"
            if "是" in text[:40]:
                return "是"
        match = re.search(r"-?\d+(?:\.\d+)?%?", text)
        if match:
            return match.group(0)
        company_match = re.search(r"([\u4e00-\u9fffA-Za-z0-9（）()·\-]+?公司)", text)
        if company_match:
            return company_match.group(1)
        return text[:60].strip()

    def _clean_context(self, fused_context: str) -> str:
        lines = []
        for line in fused_context.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("[Agent "):
                continue
            if stripped.startswith("Query:"):
                continue
            lines.append(stripped)
        return "\n".join(lines)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        stripped = str(item).strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        result.append(stripped)
    return result
