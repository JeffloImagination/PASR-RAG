from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


class FlatInnerProductIndexBuilder:
    """Build FAISS index when available, otherwise persist a numpy fallback."""

    def __init__(self) -> None:
        try:
            import faiss  # type: ignore
        except ImportError:
            faiss = None
        self.faiss = faiss

    def build(self, embeddings: np.ndarray, output_dir: str | Path, stem: str) -> tuple[str, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.faiss is not None:
            index_path = output_dir / f"{stem}.faiss"
            index = self.faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings.astype(np.float32))
            temp_dir = Path(tempfile.gettempdir()) / "pasr_rag_faiss"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"{stem}.faiss"
            self.faiss.write_index(index, str(temp_path))
            shutil.move(str(temp_path), str(index_path))
            return "faiss", str(index_path)

        vector_path = output_dir / f"{stem}.npy"
        np.save(vector_path, embeddings.astype(np.float32))
        manifest_path = output_dir / f"{stem}.manifest.json"
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "backend": "numpy_flat_ip",
                    "vector_path": str(vector_path),
                    "dimension": int(embeddings.shape[1]),
                    "count": int(embeddings.shape[0]),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        return "numpy_flat_ip", str(manifest_path)
