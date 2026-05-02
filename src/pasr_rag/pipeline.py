from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .config import AppConfig
from .generation import CentralGenerator, ContextAssembler, InformationFusion
from .privacy import AgentMetadataStore
from .retrieval import execute_agents_for_query
from .router.router import build_router
from .schemas import QueryResult
from .schemas import PipelineBootstrapReport


class PASRExperimentPipeline:
    """Stage-B pipeline skeleton.

    This class only validates configuration, prepares runtime directories,
    and emits a bootstrap report so later modules can plug into a stable entrypoint.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.fusion = InformationFusion(config)
        self.assembler = ContextAssembler(config)
        self.generator = CentralGenerator(config)

    def bootstrap(self) -> PipelineBootstrapReport:
        ensured_paths = self._ensure_runtime_directories()
        return PipelineBootstrapReport(
            project_name=self.config.project.name,
            project_version=self.config.project.version,
            config_path=str(self.config.config_path),
            router_strategy=self.config.router.strategy,
            summarization_enabled=self.config.generation.enable_summarization,
            parallel_retrieval_enabled=self.config.retrieval.enable_parallel_retrieval,
            ensured_paths=[str(path) for path in ensured_paths],
        )

    def save_bootstrap_report(self, report: PipelineBootstrapReport) -> Path:
        report_dir = self.config.paths.logs_root
        report_dir.mkdir(parents=True, exist_ok=True)
        output_path = report_dir / "bootstrap_report.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(report), file, ensure_ascii=False, indent=2)
        return output_path

    def _ensure_runtime_directories(self) -> list[Path]:
        paths = [
            self.config.paths.data_root,
            self.config.paths.agent_root,
            self.config.paths.cache_root,
            self.config.paths.logs_root,
            self.config.paths.results_root,
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def answer_query(self, query: str, metadata_path: str | Path) -> QueryResult:
        start = time.perf_counter()
        metadata_store = AgentMetadataStore(metadata_path)
        router = build_router(self.config, metadata_store)
        decision = router.select_agents(query)
        responses = execute_agents_for_query(
            query=query,
            selected_agent_ids=decision.selected_agents,
            metadata_store=metadata_store,
            config=self.config,
        )
        fused_context = self.fusion.fuse(query, responses)
        fusion_state = self.fusion.last_state
        assembled_prompt = self.assembler.assemble(query, fused_context)
        answer, answer_source = self.generator.generate(
            query,
            assembled_prompt,
            fused_context,
            fusion_state=fusion_state,
        )
        e2e_latency_ms = (time.perf_counter() - start) * 1000
        return QueryResult(
            query=query,
            selected_agents=decision.selected_agents,
            answer=answer,
            fused_context=fused_context,
            total_privacy_cost=decision.total_privacy_cost,
            privacy_budget=decision.privacy_budget,
            router_strategy=decision.strategy,
            router_latency_ms=decision.routing_latency_ms,
            e2e_latency_ms=e2e_latency_ms,
            fusion_state=fusion_state,
            answer_source=answer_source,
            agent_responses=[asdict(item) for item in responses],
            router_scored_agents=decision.scored_agents,
        )
