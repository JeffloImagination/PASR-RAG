from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..dataio import read_jsonl


@dataclass
class PrivacyAssessment:
    privacy_level: str
    privacy_cost: float
    reason: str
    confidence: float
    method: str


class PrivacyCostEvaluator:
    """Offline knowledge-source privacy evaluator.

    Preferred path:
    - sample chunk texts via `sample_chunks_for_privacy_eval`
    - ask qwen-plus to estimate continuous privacy cost in [0, 1]

    Fallback path:
    - heuristic keyword-based scorer when remote LLM is unavailable
    """

    def __init__(
        self,
        prompt_path: str | Path,
        *,
        random_seed: int = 42,
        llm_model: str = "qwen-plus",
        api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env: str = "DASHSCOPE_API_KEY",
        sample_ratio: float = 0.1,
        max_sample_chunks: int = 48,
        max_chunk_chars: int = 1200,
    ) -> None:
        self.prompt_path = Path(prompt_path)
        self.random = random.Random(random_seed)
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8")
        self.llm_model = llm_model
        self.api_base_url = api_base_url
        self.api_key_env = api_key_env
        self.sample_ratio = sample_ratio
        self.max_sample_chunks = max_sample_chunks
        self.max_chunk_chars = max_chunk_chars
        self._client: Any | None = None

    def evaluate_source(
        self,
        chunk_path: str | Path,
        mode: str = "llm_eval",
        fixed_level: str = "L1",
    ) -> PrivacyAssessment:
        chunks = read_jsonl(chunk_path)
        texts = [str(item.get("content", "")).strip() for item in chunks if str(item.get("content", "")).strip()]
        if mode == "fixed_label":
            return self._make_assessment(
                level=fixed_level,
                cost=self._cost_from_level(fixed_level),
                reason=f"Fixed privacy label configured as {fixed_level}.",
                confidence=1.0,
                method=mode,
            )
        if mode == "random_label":
            level = self.random.choice(["L0", "L1", "L2", "L3", "L4"])
            return self._make_assessment(
                level=level,
                cost=self._cost_from_level(level),
                reason="Random privacy label assigned for ablation baseline.",
                confidence=0.5,
                method=mode,
            )
        return self._llm_eval_with_sampling(texts)

    def sample_chunks_for_privacy_eval(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        sample_size = max(1, math.ceil(len(texts) * self.sample_ratio))
        sample_size = min(sample_size, len(texts), self.max_sample_chunks)
        indices = list(range(len(texts)))
        self.random.shuffle(indices)
        selected = sorted(indices[:sample_size])
        return [self._truncate_chunk(texts[idx]) for idx in selected]

    def _llm_eval_with_sampling(self, texts: list[str]) -> PrivacyAssessment:
        sampled_chunks = self.sample_chunks_for_privacy_eval(texts)
        if not sampled_chunks:
            return self._make_assessment(
                level="L0",
                cost=0.0,
                reason="No chunk content available; defaulted to the lowest privacy-cost setting.",
                confidence=0.3,
                method="llm_eval_empty",
            )

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            return self._heuristic_llm_eval(texts)

        try:
            payload = self._call_llm_with_sampling_tool(api_key, sampled_chunks)
        except Exception:
            try:
                payload = self._call_llm_direct_sample(api_key, sampled_chunks)
            except Exception:
                return self._heuristic_llm_eval(texts)

        if not payload:
            return self._heuristic_llm_eval(texts)

        privacy_cost = self._clamp_float(payload.get("privacy_cost", 0.0))
        confidence = self._clamp_float(payload.get("confidence", 0.0))
        reason = str(payload.get("reason", "")).strip() or "LLM privacy evaluation returned no reason."
        level = self._level_from_cost(privacy_cost)
        return self._make_assessment(
            level=level,
            cost=privacy_cost,
            reason=reason,
            confidence=confidence,
            method="llm_eval",
        )

    def _call_llm_direct_sample(self, api_key: str, sampled_chunks: list[str]) -> dict[str, Any]:
        client = self._get_client(api_key)
        sampled_text = "\n\n".join(
            f"[chunk_{idx + 1}]\n{text}" for idx, text in enumerate(sampled_chunks)
        )
        direct_prompt = (
            self.prompt_template
            + "\n\n你已经通过函数获得了以下采样 chunk 内容，请基于它们完成评估，并严格输出 JSON：\n\n"
            + sampled_text
        )
        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "Return valid JSON only, with privacy_cost, confidence, and reason.",
                },
                {"role": "user", "content": direct_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        payload = self._extract_json_payload(response.choices[0].message.content)
        if payload is None:
            raise RuntimeError("Direct sampled-chunk privacy evaluation did not return valid JSON.")
        return payload

    def _call_llm_with_sampling_tool(self, api_key: str, sampled_chunks: list[str]) -> dict[str, Any]:
        client = self._get_client(api_key)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "sample_chunks_for_privacy_eval",
                    "description": (
                        "Return a random sample of chunk texts from the current knowledge source "
                        "for privacy-cost estimation."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
        ]
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a careful privacy-cost evaluator. "
                    "You must call the provided function before giving the final result, "
                    "and your final answer must be valid JSON only."
                ),
            },
            {"role": "user", "content": self.prompt_template},
        ]
        first_response = client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "sample_chunks_for_privacy_eval"}},
            temperature=0.0,
            max_tokens=256,
        )
        first_message = first_response.choices[0].message
        tool_calls = getattr(first_message, "tool_calls", None) or []
        if not tool_calls:
            raise RuntimeError("LLM did not call sample_chunks_for_privacy_eval as required.")

        assistant_message = {
            "role": "assistant",
            "content": first_message.content or "",
            "tool_calls": [],
        }
        for tool_call in tool_calls:
            assistant_message["tool_calls"].append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments or "{}",
                    },
                }
            )
        messages.append(assistant_message)

        for tool_call in tool_calls:
            if tool_call.function.name != "sample_chunks_for_privacy_eval":
                continue
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "sample_chunks_for_privacy_eval",
                    "content": json.dumps(sampled_chunks, ensure_ascii=False),
                }
            )

        second_response = client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        parsed = self._extract_json_payload(second_response.choices[0].message.content)
        if parsed is None:
            raise RuntimeError("LLM privacy evaluation did not return a valid JSON payload.")
        return parsed

    def _heuristic_llm_eval(self, texts: list[str]) -> PrivacyAssessment:
        joined = " ".join(texts).lower()
        high_risk_keywords = {
            "L4": ["private key", "secret key", "password", "credential", "ssn", "密钥", "密码"],
            "L3": ["contract", "invoice", "salary", "financial statement", "medical", "patient", "合同", "财务", "病人", "医疗"],
            "L2": ["internal", "restricted", "business strategy", "customer", "workflow", "内部", "受限", "客户", "流程"],
            "L1": ["draft", "team", "roadmap", "meeting", "草稿", "团队", "会议"],
        }
        for level, keywords in high_risk_keywords.items():
            hits = [keyword for keyword in keywords if keyword in joined]
            if hits:
                return self._make_assessment(
                    level=level,
                    cost=self._cost_from_level(level),
                    reason=(
                        "Heuristic llm_eval matched sensitive indicators: "
                        + ", ".join(hits[:3])
                        + "."
                    ),
                    confidence=0.75 if level in {"L2", "L3", "L4"} else 0.65,
                    method="heuristic_fallback",
                )

        return self._make_assessment(
            level="L0",
            cost=0.1,
            reason="Heuristic fallback found no strong privacy-sensitive indicators.",
            confidence=0.6,
            method="heuristic_fallback",
        )

    def _get_client(self, api_key: str) -> Any:
        if self._client is not None:
            return self._client
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.api_base_url,
            timeout=60.0,
        )
        return self._client

    def _extract_json_payload(self, content: Any) -> dict[str, Any] | None:
        if content is None:
            return None
        if isinstance(content, list):
            text = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item) for item in content
            ).strip()
        else:
            text = str(content).strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None

    def _truncate_chunk(self, text: str) -> str:
        stripped = text.strip()
        if len(stripped) <= self.max_chunk_chars:
            return stripped
        return stripped[: self.max_chunk_chars].rstrip() + "\n[TRUNCATED]"

    def _make_assessment(
        self,
        *,
        level: str,
        cost: float,
        reason: str,
        confidence: float,
        method: str,
    ) -> PrivacyAssessment:
        return PrivacyAssessment(
            privacy_level=level,
            privacy_cost=self._clamp_float(cost),
            reason=reason,
            confidence=self._clamp_float(confidence),
            method=method,
        )

    def _clamp_float(self, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, parsed))

    def _cost_from_level(self, level: str) -> float:
        mapping = {
            "L0": 0.1,
            "L1": 0.3,
            "L2": 0.5,
            "L3": 0.75,
            "L4": 0.95,
        }
        return mapping.get(level, 0.1)

    def _level_from_cost(self, cost: float) -> str:
        if cost < 0.2:
            return "L0"
        if cost < 0.4:
            return "L1"
        if cost < 0.6:
            return "L2"
        if cost < 0.8:
            return "L3"
        return "L4"

    def render_prompt_preview(self, sampled_text: str) -> str:
        return f"{self.prompt_template}\n\n[知识源样本]\n{sampled_text}"
