from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schemas import (
    EvaluationConfig,
    GenerationConfig,
    ModelConfig,
    PathConfig,
    PreprocessingConfig,
    PrivacyConfig,
    ProjectConfig,
    RetrievalConfig,
    RouterConfig,
)


@dataclass
class AppConfig:
    project: ProjectConfig
    models: ModelConfig
    preprocessing: PreprocessingConfig
    router: RouterConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    privacy: PrivacyConfig
    evaluation: EvaluationConfig
    paths: PathConfig
    config_path: Path

    @classmethod
    def from_dict(cls, payload: dict[str, Any], config_path: Path) -> "AppConfig":
        return cls(
            project=ProjectConfig.from_dict(payload.get("project", {})),
            models=ModelConfig.from_dict(payload.get("models", {})),
            preprocessing=PreprocessingConfig.from_dict(payload.get("preprocessing", {})),
            router=RouterConfig.from_dict(payload.get("router", {})),
            retrieval=RetrievalConfig.from_dict(payload.get("retrieval", {})),
            generation=GenerationConfig.from_dict(payload.get("generation", {})),
            privacy=PrivacyConfig.from_dict(payload.get("privacy", {})),
            evaluation=EvaluationConfig.from_dict(payload.get("evaluation", {})),
            paths=PathConfig.from_dict(payload.get("paths", {}), base_dir=config_path.parent.parent.parent),
            config_path=config_path.resolve(),
        )


def load_app_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    return AppConfig.from_dict(payload, path)
