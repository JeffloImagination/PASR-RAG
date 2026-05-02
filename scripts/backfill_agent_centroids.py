from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pasr_rag.privacy import AgentMetadataStore


def build_centroid(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros((0,), dtype=np.float32)
    centroid = vectors.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 0:
        centroid = centroid / norm
    return centroid.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill centroid vectors for existing agent assets.")
    parser.add_argument("--metadata-path", required=True, help="Path to agent_metadata.json")
    args = parser.parse_args()

    store = AgentMetadataStore(args.metadata_path)
    updated = 0

    for agent in store.get_all_agents():
        vector_path = Path(agent.vector_path)
        centroid_path = Path(agent.centroid_path)
        if not vector_path.exists():
            print(f"[skip] {agent.agent_id}: missing vector file {vector_path}")
            continue
        vectors = np.load(vector_path)
        centroid = build_centroid(vectors)
        centroid_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(centroid_path, centroid)
        updated += 1
        print(f"[ok] {agent.agent_id}: wrote {centroid_path}")

    store.save()
    print(f"completed: {updated} agents updated")


if __name__ == "__main__":
    main()
