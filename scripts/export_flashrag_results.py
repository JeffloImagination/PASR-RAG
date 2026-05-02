from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pasr_rag.evaluation import normalize_answer
from pasr_rag.evaluation.evaluator import relaxed_f1_score, soft_exact_match_score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize FlashRAG output directory into PASR-style exports")
    parser.add_argument("--flashrag-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="flashrag_baseline")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.flashrag_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    intermediate_path = run_dir / "intermediate_data.json"
    metric_path = run_dir / "metric_score.txt"
    if not intermediate_path.exists():
        raise SystemExit(f"Missing intermediate_data.json in {run_dir}")

    rows = json.loads(intermediate_path.read_text(encoding="utf-8"))
    normalized_rows = []
    for row in rows:
        output = row.get("output", {})
        metric_score = output.get("metric_score", {})
        normalized_rows.append(
            {
                "id": row.get("id"),
                "query": row.get("question"),
                "prediction": output.get("pred", ""),
                "gold_answer": (row.get("golden_answers") or [""])[0],
                "method": args.method_name,
                "em": metric_score.get("em", 0.0),
                "f1": metric_score.get("f1", 0.0),
                "soft_em": soft_exact_match_score(output.get("pred", ""), (row.get("golden_answers") or [""])[0]),
                "relaxed_f1": relaxed_f1_score(output.get("pred", ""), (row.get("golden_answers") or [""])[0]),
                "accuracy": metric_score.get("acc", 0.0),
                "retrieval_result_count": len(output.get("retrieval_result", [])),
                "prompt": json.dumps(output.get("prompt", []), ensure_ascii=False),
            }
        )

    jsonl_path = output_dir / "query_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in normalized_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = output_dir / "query_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(normalized_rows[0].keys()) if normalized_rows else ["id"])
        writer.writeheader()
        writer.writerows(normalized_rows)

    summary = {}
    if metric_path.exists():
        for line in metric_path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                summary[key.strip()] = float(value.strip())
            except ValueError:
                summary[key.strip()] = value.strip()
    summary["count"] = len(normalized_rows)
    summary["method"] = args.method_name
    if normalized_rows:
        summary["soft_em"] = sum(float(row["soft_em"]) for row in normalized_rows) / len(normalized_rows)
        summary["relaxed_f1"] = sum(float(row["relaxed_f1"]) for row in normalized_rows) / len(normalized_rows)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "jsonl": str(jsonl_path),
                "csv": str(csv_path),
                "summary": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
