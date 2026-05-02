from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProjectConfig:
    name: str = "PASR-RAG"
    version: str = "0.1.0"
    random_seed: int = 42

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectConfig":
        return cls(
            name=payload.get("name", cls.name),
            version=payload.get("version", cls.version),
            random_seed=payload.get("random_seed", cls.random_seed),
        )


@dataclass
class ModelConfig:
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_model_path: str = "models/bge-base-en-v1.5"
    embedding_batch_size: int = 32
    llm_model: str = "qwen-plus"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        return cls(
            embedding_model=payload.get("embedding_model", cls.embedding_model),
            embedding_model_path=payload.get("embedding_model_path", cls.embedding_model_path),
            embedding_batch_size=payload.get("embedding_batch_size", cls.embedding_batch_size),
            llm_model=payload.get("llm_model", cls.llm_model),
        )


@dataclass
class SourcePartitionConfig:
    target_sources: int = 8
    strategy: str = "entity_balanced"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourcePartitionConfig":
        return cls(
            target_sources=payload.get("target_sources", cls.target_sources),
            strategy=payload.get("strategy", cls.strategy),
        )


@dataclass
class PreprocessingConfig:
    chunk_size: int = 384
    chunk_overlap: int = 80
    source_partition: SourcePartitionConfig = field(default_factory=SourcePartitionConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreprocessingConfig":
        return cls(
            chunk_size=payload.get("chunk_size", cls.chunk_size),
            chunk_overlap=payload.get("chunk_overlap", cls.chunk_overlap),
            source_partition=SourcePartitionConfig.from_dict(payload.get("source_partition", {})),
        )


@dataclass
class RouterConfig:
    strategy: str = "pasr"
    alpha: float = 1.0
    beta: float = 0.5
    privacy_budget_mode: str = "ratio"
    privacy_budget_B: float = 3.0
    privacy_budget_ratio: float = 0.3
    max_active_sources_k: int = 3
    relevance_method: str = "centroid_cosine"
    threshold: float = 0.2

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouterConfig":
        return cls(
            strategy=payload.get("strategy", cls.strategy),
            alpha=payload.get("alpha", cls.alpha),
            beta=payload.get("beta", cls.beta),
            privacy_budget_mode=payload.get("privacy_budget_mode", cls.privacy_budget_mode),
            privacy_budget_B=payload.get("privacy_budget_B", cls.privacy_budget_B),
            privacy_budget_ratio=payload.get("privacy_budget_ratio", cls.privacy_budget_ratio),
            max_active_sources_k=payload.get("max_active_sources_k", cls.max_active_sources_k),
            relevance_method=payload.get("relevance_method", cls.relevance_method),
            threshold=payload.get("threshold", cls.threshold),
        )


@dataclass
class RetrievalConfig:
    backend: str = "vector"
    top_m_per_source: int = 5
    enable_parallel_retrieval: bool = True
    max_workers: int = 4
    cache_enabled: bool = True
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievalConfig":
        return cls(
            backend=payload.get("backend", cls.backend),
            top_m_per_source=payload.get("top_m_per_source", cls.top_m_per_source),
            enable_parallel_retrieval=payload.get(
                "enable_parallel_retrieval", cls.enable_parallel_retrieval
            ),
            max_workers=payload.get("max_workers", cls.max_workers),
            cache_enabled=payload.get("cache_enabled", cls.cache_enabled),
            bm25_k1=payload.get("bm25_k1", cls.bm25_k1),
            bm25_b=payload.get("bm25_b", cls.bm25_b),
        )


@dataclass
class GenerationConfig:
    backend: str = "openai_compatible"
    enable_summarization: bool = True
    enable_central_fusion: bool = True
    max_summary_tokens: int = 200
    max_fusion_tokens: int = 512
    max_context_tokens: int = 4096
    temperature: float = 0.0
    max_answer_tokens: int = 256
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    fallback_to_extract: bool = True
    local_summary_prompt_path: str = "docs/检索智能体内部摘要提示词.md"
    central_fusion_prompt_path: str = "docs/中心智能体证据聚合提示词.md"
    final_answer_prompt_path: str = "docs/中心智能体最终答案提示词.md"
    local_summary_fallback_to_extract: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerationConfig":
        return cls(
            backend=payload.get("backend", cls.backend),
            enable_summarization=payload.get("enable_summarization", cls.enable_summarization),
            enable_central_fusion=payload.get("enable_central_fusion", cls.enable_central_fusion),
            max_summary_tokens=payload.get("max_summary_tokens", cls.max_summary_tokens),
            max_fusion_tokens=payload.get("max_fusion_tokens", cls.max_fusion_tokens),
            max_context_tokens=payload.get("max_context_tokens", cls.max_context_tokens),
            temperature=payload.get("temperature", cls.temperature),
            max_answer_tokens=payload.get("max_answer_tokens", cls.max_answer_tokens),
            api_base_url=payload.get("api_base_url", cls.api_base_url),
            api_key_env=payload.get("api_key_env", cls.api_key_env),
            fallback_to_extract=payload.get("fallback_to_extract", cls.fallback_to_extract),
            local_summary_prompt_path=payload.get(
                "local_summary_prompt_path", cls.local_summary_prompt_path
            ),
            central_fusion_prompt_path=payload.get(
                "central_fusion_prompt_path", cls.central_fusion_prompt_path
            ),
            final_answer_prompt_path=payload.get(
                "final_answer_prompt_path", cls.final_answer_prompt_path
            ),
            local_summary_fallback_to_extract=payload.get(
                "local_summary_fallback_to_extract", cls.local_summary_fallback_to_extract
            ),
        )


@dataclass
class PrivacyConfig:
    enable_privacy_term: bool = True
    enable_relevance_term: bool = True
    privacy_eval_method: str = "llm_eval"
    prompt_path: str = "docs/隐私成本评估提示词.md"
    sample_ratio: float = 0.1
    max_sample_chunks: int = 48
    max_chunk_chars: int = 1200

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PrivacyConfig":
        return cls(
            enable_privacy_term=payload.get("enable_privacy_term", cls.enable_privacy_term),
            enable_relevance_term=payload.get("enable_relevance_term", cls.enable_relevance_term),
            privacy_eval_method=payload.get("privacy_eval_method", cls.privacy_eval_method),
            prompt_path=payload.get("prompt_path", cls.prompt_path),
            sample_ratio=payload.get("sample_ratio", cls.sample_ratio),
            max_sample_chunks=payload.get("max_sample_chunks", cls.max_sample_chunks),
            max_chunk_chars=payload.get("max_chunk_chars", cls.max_chunk_chars),
        )


@dataclass
class EvaluationConfig:
    metrics: list[str] = field(default_factory=lambda: ["em", "f1", "acc"])

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationConfig":
        return cls(metrics=payload.get("metrics", cls().metrics))


@dataclass
class PathConfig:
    data_root: Path
    agent_root: Path
    raw_data_root: Path
    cache_root: Path
    logs_root: Path
    results_root: Path
    flashrag_root: Path

    @classmethod
    def from_dict(cls, payload: dict[str, Any], base_dir: Path) -> "PathConfig":
        def resolve_path(key: str, default: str) -> Path:
            return (base_dir / payload.get(key, default)).resolve()

        return cls(
            data_root=resolve_path("data_root", "data"),
            agent_root=resolve_path("agent_root", "data/agents"),
            raw_data_root=resolve_path("raw_data_root", "data/raw"),
            cache_root=resolve_path("cache_root", "outputs/cache"),
            logs_root=resolve_path("logs_root", "outputs/logs"),
            results_root=resolve_path("results_root", "outputs/results"),
            flashrag_root=resolve_path("flashrag_root", "FlashRAG"),
        )


@dataclass
class AgentMeta:
    agent_id: str
    privacy_cost: float
    index_path: str
    chunk_path: str = ""
    vector_path: str = ""
    centroid_path: str = ""
    bm25_index_path: str = ""
    privacy_level: str = ""
    privacy_reason: str = ""
    privacy_confidence: float = 0.0
    status: str = "active"
    update_time: str = ""


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    content: str
    agent_id: str
    score: float | None = None


@dataclass
class SourceDocument:
    doc_id: str
    title: str
    content: str
    source_hint: str = ""


@dataclass
class QAExample:
    question_id: str
    question: str
    answer: str
    supporting_titles: list[str]
    documents: list[SourceDocument]


@dataclass
class PreprocessReport:
    dataset_name: str
    input_path: str
    output_root: str
    total_examples: int
    total_agents: int
    total_documents: int
    total_chunks: int
    index_backend: str
    metadata_path: str


@dataclass
class AgentSelection:
    agent_id: str
    relevance_score: float
    privacy_cost: float
    utility_score: float


@dataclass
class AgentResponse:
    agent_id: str
    local_summary: str
    source_chunks_count: int
    retrieval_latency_ms: float
    summary_mode: str = ""
    structured_summary: dict[str, Any] = field(default_factory=dict)
    retrieval_debug: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class QueryResult:
    query: str
    selected_agents: list[str]
    answer: str
    fused_context: str
    total_privacy_cost: float
    privacy_budget: float
    router_strategy: str
    router_latency_ms: float
    e2e_latency_ms: float
    fusion_state: dict[str, Any] = field(default_factory=dict)
    answer_source: str = ""
    agent_responses: list[dict[str, Any]] = field(default_factory=list)
    router_scored_agents: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineBootstrapReport:
    project_name: str
    project_version: str
    config_path: str
    router_strategy: str
    summarization_enabled: bool
    parallel_retrieval_enabled: bool
    ensured_paths: list[str]
