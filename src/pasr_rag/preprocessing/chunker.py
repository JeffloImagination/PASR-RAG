from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkSpan:
    text: str
    start_token: int
    end_token: int


class TokenWindowChunker:
    """Whitespace token chunker with overlap.

    This keeps the Stage-C pipeline dependency-light while preserving the
    fixed-size/overlap behavior required by the thesis docs.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[ChunkSpan]:
        tokens = text.split()
        if not tokens:
            return []

        spans: list[ChunkSpan] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for start in range(0, len(tokens), step):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                continue
            spans.append(
                ChunkSpan(
                    text=" ".join(chunk_tokens),
                    start_token=start,
                    end_token=end,
                )
            )
            if end >= len(tokens):
                break
        return spans
