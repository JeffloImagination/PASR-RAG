from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def read_jsonl(path: str | Path) -> list[dict]:
    items: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str | Path, payload: dict | list) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return output_path


def write_jsonl(path: str | Path, items: Iterable[T]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as file:
        for item in items:
            payload = asdict(item) if is_dataclass(item) else item
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return output_path
