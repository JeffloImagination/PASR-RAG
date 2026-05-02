from __future__ import annotations

import math
import re
import string
import unicodedata
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


CHINESE_PUNCTUATION = "，。！？；：（）【】《》“”‘’—…￥·、"


def bm25_tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation + CHINESE_PUNCTUATION))
    normalized = re.sub(r"\s+", "", normalized)
    if not normalized:
        return []
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+(?:\.[0-9]+)?%?", normalized)


@dataclass
class BM25Index:
    doc_freqs: list[Counter[str]]
    idf: dict[str, float]
    doc_lengths: list[int]
    avgdl: float
    k1: float
    b: float

    @classmethod
    def from_texts(cls, texts: list[str], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        tokenized = [bm25_tokenize(text) for text in texts]
        doc_freqs = [Counter(tokens) for tokens in tokenized]
        doc_lengths = [len(tokens) for tokens in tokenized]
        avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0

        df: Counter[str] = Counter()
        for freq in doc_freqs:
            for term in freq:
                df[term] += 1

        total_docs = len(doc_freqs)
        idf = {
            term: math.log(1.0 + (total_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        return cls(
            doc_freqs=doc_freqs,
            idf=idf,
            doc_lengths=doc_lengths,
            avgdl=avgdl,
            k1=k1,
            b=b,
        )

    def score(self, query: str) -> list[float]:
        query_terms = bm25_tokenize(query)
        if not query_terms:
            return [0.0 for _ in self.doc_freqs]

        scores: list[float] = []
        for idx, freqs in enumerate(self.doc_freqs):
            dl = self.doc_lengths[idx]
            score = 0.0
            for term in query_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl > 0 else 0.0)
                )
                score += idf * (tf * (self.k1 + 1.0) / denom)
            scores.append(score)
        return scores

    def to_dict(self) -> dict:
        return {
            "doc_freqs": [dict(freq) for freq in self.doc_freqs],
            "idf": self.idf,
            "doc_lengths": self.doc_lengths,
            "avgdl": self.avgdl,
            "k1": self.k1,
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BM25Index":
        return cls(
            doc_freqs=[Counter(item) for item in payload.get("doc_freqs", [])],
            idf={str(k): float(v) for k, v in payload.get("idf", {}).items()},
            doc_lengths=[int(v) for v in payload.get("doc_lengths", [])],
            avgdl=float(payload.get("avgdl", 0.0)),
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
        )

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
