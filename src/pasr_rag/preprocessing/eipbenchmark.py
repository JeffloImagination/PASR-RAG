from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import AppConfig
from ..dataio import write_json, write_jsonl
from ..retrieval.bm25 import BM25Index
from ..schemas import AgentMeta, Chunk, PreprocessReport, SourceDocument
from .chunker import TokenWindowChunker
from .embedding import build_embedding_encoder
from .indexing import FlatInnerProductIndexBuilder
from .loaders import load_eipbenchmark_corpus_jsonl

EIPBENCHMARK_SOURCE_TO_AGENT = {
    "company_core": "A_00",
    "company_operation_status": "A_01",
    "company_profile": "A_02",
    "national_industry_status": "A_03",
    "policy_release_status": "A_04",
    "policy_resource": "A_05",
    "regional_industry_status": "A_06",
}


class EIPBenchmarkPreprocessor:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.chunker = TokenWindowChunker(
            chunk_size=config.preprocessing.chunk_size,
            chunk_overlap=config.preprocessing.chunk_overlap,
        )
        self.encoder = build_embedding_encoder(config)
        self.index_builder = FlatInnerProductIndexBuilder()

    def run(self, corpus_root: str | Path, output_root: str | Path | None = None) -> PreprocessReport:
        corpus_root = Path(corpus_root).resolve()
        output_root = Path(output_root or self.config.paths.agent_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        total_documents = 0
        total_chunks = 0
        metadata_items: list[AgentMeta] = []
        registry_items: list[dict] = []
        backend_name = "unknown"

        for source_name, agent_id in EIPBENCHMARK_SOURCE_TO_AGENT.items():
            source_path = corpus_root / f"{source_name}.jsonl"
            documents = load_eipbenchmark_corpus_jsonl(source_path, source_name)
            total_documents += len(documents)
            agent_dir = output_root / agent_id
            agent_dir.mkdir(parents=True, exist_ok=True)

            chunks = self._build_chunks(agent_id, documents)
            total_chunks += len(chunks)
            chunk_path = write_jsonl(agent_dir / f"{agent_id}_chunks.jsonl", [asdict(chunk) for chunk in chunks])

            embeddings = (
                self.encoder.encode([chunk.content for chunk in chunks], is_query=False)
                if chunks
                else self.encoder.encode(["empty"], is_query=False)[:0]
            )
            vector_path = agent_dir / f"{agent_id}_vectors.npy"
            np.save(vector_path, embeddings.astype(np.float32))
            centroid = self._build_centroid_vector(embeddings)
            centroid_path = agent_dir / f"{agent_id}_centroid.npy"
            np.save(centroid_path, centroid.astype(np.float32))
            bm25_index = self._build_bm25_index(chunks)
            bm25_index_path = agent_dir / f"{agent_id}_bm25_index.json"
            bm25_index.save(bm25_index_path)
            backend_name, index_path = self.index_builder.build(
                embeddings if embeddings.size else self.encoder.encode(["empty"], is_query=False)[:0],
                output_dir=agent_dir,
                stem=f"{agent_id}_index",
            )

            manifest = {
                "agent_id": agent_id,
                "source_name": source_name,
                "source_path": str(source_path),
                "documents": len(documents),
                "chunks": len(chunks),
                "chunk_path": str(chunk_path),
                "vector_path": str(vector_path),
                "centroid_path": str(centroid_path),
                "bm25_index_path": str(bm25_index_path),
                "index_path": index_path,
                "index_backend": backend_name,
            }
            write_json(agent_dir / f"{agent_id}_manifest.json", manifest)
            registry_items.append(manifest)
            metadata_items.append(
                AgentMeta(
                    agent_id=agent_id,
                    privacy_cost=0.1,
                    index_path=index_path,
                    chunk_path=str(chunk_path),
                    vector_path=str(vector_path),
                    centroid_path=str(centroid_path),
                    bm25_index_path=str(bm25_index_path),
                    privacy_level="PENDING",
                    privacy_reason="Pending formal privacy evaluation for eipBenchmark.",
                    privacy_confidence=0.0,
                    status="active",
                    update_time=datetime.utcnow().isoformat(timespec="seconds"),
                )
            )

        metadata_path = write_json(output_root / "agent_metadata.json", [asdict(item) for item in metadata_items])
        write_json(output_root / "source_registry.json", registry_items)
        report = PreprocessReport(
            dataset_name="eipBenchmark",
            input_path=str(corpus_root),
            output_root=str(output_root),
            total_examples=0,
            total_agents=len(EIPBENCHMARK_SOURCE_TO_AGENT),
            total_documents=total_documents,
            total_chunks=total_chunks,
            index_backend=backend_name,
            metadata_path=str(metadata_path),
        )
        write_json(output_root / "preprocess_report.json", asdict(report))
        return report

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

    def _build_bm25_index(self, chunks: list[Chunk]) -> BM25Index:
        texts = [f"{chunk.title}\n{chunk.content}".strip() for chunk in chunks]
        return BM25Index.from_texts(
            texts,
            k1=self.config.retrieval.bm25_k1,
            b=self.config.retrieval.bm25_b,
        )
