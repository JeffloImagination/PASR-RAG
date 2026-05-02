from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json

import numpy as np

from ..config import AppConfig
from ..dataio import write_json, write_jsonl
from ..schemas import AgentMeta, Chunk, PreprocessReport, SourceDocument
from .chunker import TokenWindowChunker
from .embedding import build_embedding_encoder
from .indexing import FlatInnerProductIndexBuilder
from .loaders import load_hotpotqa_like_jsonl
from .partitioner import EntityBalancedPartitioner


class HotpotPreprocessor:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.chunker = TokenWindowChunker(
            chunk_size=config.preprocessing.chunk_size,
            chunk_overlap=config.preprocessing.chunk_overlap,
        )
        self.partitioner = EntityBalancedPartitioner(
            target_sources=config.preprocessing.source_partition.target_sources
        )
        self.encoder = build_embedding_encoder(config)
        self.index_builder = FlatInnerProductIndexBuilder()

    def run(self, input_path: str | Path, output_root: str | Path | None = None) -> PreprocessReport:
        input_path = Path(input_path).resolve()
        output_root = Path(output_root or self.config.paths.agent_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        existing_metadata = self._load_existing_metadata(output_root / "agent_metadata.json")

        examples = load_hotpotqa_like_jsonl(str(input_path))
        sources = self.partitioner.partition(examples)

        total_chunks = 0
        total_documents = sum(len(documents) for documents in sources.values())
        metadata_items: list[AgentMeta] = []
        backend_name = "unknown"

        for agent_id, documents in sources.items():
            agent_dir = output_root / agent_id
            agent_dir.mkdir(parents=True, exist_ok=True)

            chunks = self._build_chunks(agent_id, documents)
            total_chunks += len(chunks)
            chunk_path = write_jsonl(agent_dir / f"{agent_id}_chunks.jsonl", [asdict(chunk) for chunk in chunks])

            embeddings = (
                self.encoder.encode([chunk.content for chunk in chunks], is_query=False)
                if chunks
                else self.encoder.encode([""], is_query=False)
            )
            if not chunks:
                embeddings = embeddings[:0]
            vector_path = agent_dir / f"{agent_id}_vectors.npy"
            np.save(vector_path, embeddings.astype(np.float32))
            centroid = self._build_centroid_vector(embeddings)
            centroid_path = agent_dir / f"{agent_id}_centroid.npy"
            np.save(centroid_path, centroid.astype(np.float32))
            backend_name, index_path = self.index_builder.build(
                embeddings if len(chunks) else self.encoder.encode(["empty"], is_query=False)[:0],
                output_dir=agent_dir,
                stem=f"{agent_id}_index",
            )
            metadata_items.append(
                AgentMeta(
                    agent_id=agent_id,
                    privacy_cost=existing_metadata.get(agent_id, {}).get("privacy_cost", 0.0),
                    index_path=index_path,
                    chunk_path=str(chunk_path),
                    vector_path=str(vector_path),
                    centroid_path=str(centroid_path),
                    privacy_level=existing_metadata.get(agent_id, {}).get("privacy_level", ""),
                    privacy_reason=existing_metadata.get(agent_id, {}).get("privacy_reason", ""),
                    privacy_confidence=existing_metadata.get(agent_id, {}).get("privacy_confidence", 0.0),
                    status="active",
                    update_time=datetime.utcnow().isoformat(timespec="seconds"),
                )
            )

            write_json(
                agent_dir / f"{agent_id}_manifest.json",
                {
                    "agent_id": agent_id,
                    "documents": len(documents),
                    "chunks": len(chunks),
                    "chunk_path": str(chunk_path),
                    "vector_path": str(vector_path),
                    "centroid_path": str(centroid_path),
                    "index_path": index_path,
                    "index_backend": backend_name,
                },
            )

        metadata_path = write_json(
            output_root / "agent_metadata.json",
            [asdict(item) for item in metadata_items],
        )
        report = PreprocessReport(
            dataset_name="hotpotqa_like",
            input_path=str(input_path),
            output_root=str(output_root),
            total_examples=len(examples),
            total_agents=len(sources),
            total_documents=total_documents,
            total_chunks=total_chunks,
            index_backend=backend_name,
            metadata_path=str(metadata_path),
        )
        write_json(output_root / "preprocess_report.json", asdict(report))
        return report

    def _load_existing_metadata(self, metadata_path: Path) -> dict[str, dict]:
        if not metadata_path.exists():
            return {}
        with metadata_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return {item["agent_id"]: item for item in payload}

    def _build_chunks(self, agent_id: str, documents: list[SourceDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            source_text = f"{document.title}\n{document.content}".strip()
            spans = self.chunker.chunk_text(source_text)
            if not spans and source_text:
                spans = self.chunker.chunk_text(document.content)
            for idx, span in enumerate(spans):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}_chunk_{idx}",
                        doc_id=document.doc_id,
                        title=document.title,
                        content=span.text,
                        agent_id=agent_id,
                    )
                )
        return chunks

    def _build_centroid_vector(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return np.zeros((0,), dtype=np.float32)
        centroid = embeddings.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm
        return centroid.astype(np.float32)
