from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from ..dataio import iter_jsonl
from ..schemas import QAExample, SourceDocument


def load_hotpotqa_like_jsonl(path: str) -> list[QAExample]:
    """Load a HotpotQA-like JSONL file.

    Expected fields per line:
    - `_id` or `id`
    - `question`
    - `answer`
    - `supporting_facts`: [[title, sent_id], ...]
    - `context`: [[title, [sentence1, sentence2, ...]], ...]
    """

    examples: list[QAExample] = []
    for item in iter_jsonl(path):
        supporting_titles = list(
            OrderedDict.fromkeys(fact[0] for fact in item.get("supporting_facts", []))
        )
        documents: list[SourceDocument] = []
        for doc_idx, context_item in enumerate(item.get("context", [])):
            title, sentences = context_item
            content = " ".join(sentences).strip()
            documents.append(
                SourceDocument(
                    doc_id=f"{item.get('_id', item.get('id', 'q'))}_doc_{doc_idx}",
                    title=title,
                    content=content,
                    source_hint=title,
                )
            )

        examples.append(
            QAExample(
                question_id=str(item.get("_id", item.get("id", f"example_{len(examples)}"))),
                question=item["question"],
                answer=item.get("answer", ""),
                supporting_titles=supporting_titles,
                documents=documents,
            )
        )
    return examples


def normalize_hotpotqa_record(item: dict) -> dict:
    supporting_facts = item.get("supporting_facts", [])
    if isinstance(supporting_facts, dict):
        supporting_facts = [
            [title, sent_id]
            for title, sent_id in zip(
                supporting_facts.get("title", []),
                supporting_facts.get("sent_id", []),
            )
        ]
    return {
        "_id": item.get("_id", item.get("id")),
        "question": item["question"],
        "answer": item.get("answer", ""),
        "supporting_facts": supporting_facts,
        "context": _normalize_context(item.get("context", {})),
    }


def load_hotpotqa_from_hf(
    name: str = "distractor",
    split: str = "validation",
    limit: int | None = None,
) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("hotpotqa/hotpot_qa", name=name, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [normalize_hotpotqa_record(item) for item in dataset]


def _normalize_context(context: dict | list) -> list[list]:
    if isinstance(context, list):
        return context

    titles = context.get("title", [])
    sentences = context.get("sentences", [])
    return [[title, sentence_list] for title, sentence_list in zip(titles, sentences)]


EIPBENCHMARK_SOURCE_ALIASES = {
    "company_core": "company_core",
    "company_operation_status": "company_operation_status",
    "company_profile": "company_profile",
    "national_industry_status": "national_industry_status",
    "policy_release_status": "policy_release_status",
    "policy_resource": "policy_resource",
    "regional_industry_status": "regional_industry_status",
    "ragCompanyData": "company_operation_status",
    "ragCompany": "company_profile",
    "ragCountryIndustryData": "national_industry_status",
    "ragPolicyRelease": "policy_release_status",
    "ragPolicyResource": "policy_resource",
    "ragRegionalIndustryData": "regional_industry_status",
    "ragCompanyCore": "company_core",
}


def load_eipbenchmark_corpus_jsonl(path: str | Path, source_name: str) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for item in iter_jsonl(path):
        uuid = str(item.get("uuid", item.get("id", f"doc_{len(documents)}")))
        documents.append(
            SourceDocument(
                doc_id=uuid,
                title=f"{source_name}:{uuid}",
                content=str(item.get("contents", "")).strip(),
                source_hint=source_name,
            )
        )
    return documents


def normalize_eipbenchmark_test_record(item: dict) -> dict:
    answers = [str(answer) for answer in item.get("golden_answers", []) if str(answer).strip()]
    supporting_titles: list[list] = []
    for reference_item in item.get("reference", []):
        for source_key, source_uuid in reference_item.items():
            normalized_source = EIPBENCHMARK_SOURCE_ALIASES.get(source_key, source_key)
            supporting_titles.append([f"{normalized_source}:{source_uuid}", 0])

    return {
        "_id": str(item.get("id", item.get("_id"))),
        "question": str(item.get("question", "")).strip(),
        "answer": answers[0] if answers else "",
        "answers": answers,
        "supporting_facts": supporting_titles,
        "context": [],
        "metadata": item.get("metadata", {}),
        "evidence": item.get("evidence", []),
        "reference": item.get("reference", []),
    }


def normalize_eipbenchmark_test_jsonl(input_path: str | Path, output_path: str | Path) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            normalized = normalize_eipbenchmark_test_record(json.loads(line))
            dst.write(json.dumps(normalized, ensure_ascii=False) + "\n")
    return output_path
