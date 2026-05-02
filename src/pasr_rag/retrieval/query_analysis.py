from __future__ import annotations

import re
from dataclasses import dataclass, field


COMMON_METRICS = [
    "雇员总数",
    "雇员同比增减幅",
    "净利润同比增减幅",
    "营业收入金额",
    "资产负债率",
    "公司市值",
    "研发投入占比",
    "累计所有专利被引用数",
    "累计PCT发明专利申请数",
    "专利被引用数",
    "PCT发明专利申请数",
    "净利润",
    "营业收入",
    "研发投入",
    "雇员",
]

COMPARISON_TARGET_PATTERNS = [
    r"与(.+?)相比",
    r"是否高于(.+?)(?:\?|？|$)",
    r"是否低于(.+?)(?:\?|？|$)",
    r"低于(.+?)(?:\?|？|$)",
    r"高于(.+?)(?:\?|？|$)",
]


@dataclass
class QueryAnalysis:
    question_type: str
    years: list[str] = field(default_factory=list)
    companies: list[str] = field(default_factory=list)
    metric: str = ""
    comparison_target: str = ""
    direction: str = ""
    expanded_queries: list[str] = field(default_factory=list)


def classify_query_type(query: str) -> str:
    if "差值" in query or "多少人" in query or "多少" in query and "差" in query:
        return "difference_numeric"
    if "是否" in query or "是不是" in query:
        return "yes_no_compare"
    if "哪个更高" in query or "哪个更低" in query or "哪一个更高" in query:
        return "argmax_or_choice"
    return "attribute_lookup"


def extract_query_analysis(query: str) -> QueryAnalysis:
    years = re.findall(r"(20\d{2})年", query)
    companies = _unique(re.findall(r"([\u4e00-\u9fffA-Za-z0-9（）()·\-]+?公司)", query))
    metric = next((metric for metric in COMMON_METRICS if metric in query), "")
    comparison_target = ""
    for pattern in COMPARISON_TARGET_PATTERNS:
        match = re.search(pattern, query)
        if match:
            comparison_target = match.group(1).strip()
            break

    direction = ""
    if "高于" in query:
        direction = "higher"
    elif "低于" in query:
        direction = "lower"
    elif "差值" in query:
        direction = "difference"

    analysis = QueryAnalysis(
        question_type=classify_query_type(query),
        years=_unique(years),
        companies=companies,
        metric=metric,
        comparison_target=comparison_target,
        direction=direction,
    )
    analysis.expanded_queries = build_expanded_queries(query, analysis)
    return analysis


def build_expanded_queries(query: str, analysis: QueryAnalysis) -> list[str]:
    variants = [query]
    year_text = " ".join(f"{year}年" for year in analysis.years)
    company_text = " ".join(analysis.companies[:2])
    metric_text = analysis.metric

    if company_text and year_text and metric_text:
        variants.append(f"{company_text} {year_text} {metric_text}")
    if company_text and metric_text:
        variants.append(f"{company_text} {metric_text}")
    if analysis.comparison_target and metric_text and year_text:
        variants.append(f"{analysis.comparison_target} {year_text} {metric_text}")
    if metric_text and any(token in query for token in ["行业", "省份", "区域"]):
        variants.append(f"{year_text} 行业 {metric_text}".strip())
        variants.append(f"{year_text} 省份 {metric_text}".strip())

    return _unique([variant.strip() for variant in variants if variant.strip()])


def compute_slot_hits(text: str, analysis: QueryAnalysis) -> dict[str, bool]:
    return {
        "year": any(year in text for year in analysis.years),
        "company": any(company in text for company in analysis.companies),
        "metric": bool(analysis.metric and analysis.metric in text),
        "comparison_target": bool(
            analysis.comparison_target and analysis.comparison_target in text
        ),
        "direction": bool(
            (analysis.direction == "higher" and "高于" in text)
            or (analysis.direction == "lower" and "低于" in text)
            or (analysis.direction == "difference" and "差值" in text)
        ),
    }


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        stripped = item.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        result.append(stripped)
    return result
