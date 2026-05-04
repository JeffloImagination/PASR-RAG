# PASR-RAG

[中文说明](README.zh-CN.md)

PASR-RAG is a research-oriented retrieval-augmented generation project for **privacy-aware source routing over multi-source knowledge bases**. The repository combines:

- a custom PASR-RAG pipeline in `src/pasr_rag`
- preprocessing and evaluation scripts for HotpotQA-like and eipBenchmark-style data
- experiment runners for ablations, hyper-parameter sweeps, Pareto studies, and baseline comparison
- a vendored `FlashRAG/` subproject for centralized RAG baselines

The core idea is to score candidate source agents with both **relevance** and **privacy cost**, then activate only a limited subset of sources before local retrieval, evidence fusion, and final answer generation.

## Highlights

- Privacy-aware router with multiple strategies: `pasr`, `rel_only`, `threshold`, `random`, and `ma_rag_lite`
- Agent-level retrieval over partitioned knowledge sources
- Optional local summarization and central evidence fusion
- Batch evaluation with answer quality, privacy, and efficiency metrics
- Built-in experiment scripts for main matrix, ablation, alpha sweep, and parameter sweep
- Support for both English-style HotpotQA experiments and Chinese enterprise-policy style eipBenchmark experiments

## Repository Structure

```text
.
|-- src/pasr_rag/                 # Core PASR-RAG implementation
|-- scripts/                      # Preprocessing, evaluation, and experiment runners
|-- configs/pasr_rag/             # YAML configs for HotpotQA and eipBenchmark
|-- data/                         # Prepared datasets, agent indexes, FlashRAG assets
|-- new-database/                 # Source corpus and QA files for eipBenchmark-style experiments
|-- outputs/                      # Logs and experiment results
|-- docs/                         # Prompts, design notes, and thesis materials
|-- models/                       # Local embedding models
`-- FlashRAG/                     # Baseline framework used for centralized RAG comparison
```

## Method Overview

The default PASR-RAG workflow is:

1. Split a corpus into multiple source agents.
2. Build per-agent embeddings and indexes.
3. Estimate each agent's privacy cost.
4. Route a query with utility `alpha * relevance - beta * privacy_cost`.
5. Retrieve top-`m` evidence from selected agents.
6. Optionally summarize local evidence and fuse it centrally.
7. Generate the final answer and evaluate quality, privacy, and latency.

## Environment

This repository does not currently include a pinned root `requirements.txt` or `pyproject.toml`, so the environment is assembled from the codebase itself.

Recommended baseline environment:

- Python 3.10 or 3.11
- `numpy`
- `pyyaml`
- `openai`
- `sentence-transformers`
- `torch`
- `faiss-cpu` or `faiss-gpu` for FAISS indexing

Optional dependencies:

- `datasets` for downloading HotpotQA from Hugging Face
- FlashRAG runtime dependencies inside `FlashRAG/` for centralized baselines

Example installation:

```bash
pip install numpy pyyaml openai sentence-transformers torch faiss-cpu datasets
```

## Model and API Setup

Default configs expect:

- embedding model at `models/bge-base-en-v1.5` or `models/bge-base-zh-v1.5`
- generation backend `openai_compatible`
- API key from environment variable `DASHSCOPE_API_KEY`
- default API base URL `https://dashscope.aliyuncs.com/compatible-mode/v1`

If you use a different provider, update:

- `configs/pasr_rag/base.yaml`
- `configs/pasr_rag/eipBenchmark.yaml`

## Quick Start

### 1. Bootstrap the project

```bash
python scripts/run_pasr.py --dry-run
```

This validates the config and creates runtime folders such as `outputs/logs` and `outputs/results`.

### 2. Preprocess a HotpotQA-like dataset

```bash
python scripts/preprocess_hotpotqa.py \
  --config configs/pasr_rag/base.yaml \
  --input data/raw/hotpotqa_validation_small.jsonl \
  --output-root data/agents_hotpot_small
```

### 3. Run a single PASR-RAG query

```bash
python scripts/run_qa_step.py \
  --config configs/pasr_rag/base.yaml \
  --metadata-path data/agents_hotpot_small/agent_metadata.json \
  --query "Were Scott Derrickson and Ed Wood of the same nationality?"
```

### 4. Run batch evaluation

```bash
python scripts/run_eval_batch.py \
  --config configs/pasr_rag/base.yaml \
  --metadata-path data/agents_hotpot_small/agent_metadata.json \
  --dataset data/raw/hotpotqa_validation_small.jsonl \
  --output-dir outputs/results/pasr_batch_eval_smoke \
  --strategy pasr
```

### 5. Run the main comparison matrix

```bash
python scripts/run_main_experiment_matrix.py \
  --dataset data/raw/hotpotqa_validation_small.jsonl \
  --metadata-path data/agents_hotpot_small/agent_metadata.json \
  --flashrag-asset-dir data/flashrag_assets/hotpot_small \
  --output-root outputs/results/main_experiment_matrix_smoke
```

## eipBenchmark Workflow

To prepare the enterprise/policy benchmark assets:

```bash
python scripts/prepare_eipbenchmark.py \
  --config configs/pasr_rag/eipBenchmark.yaml \
  --source-root new-database \
  --raw-root data/raw/eipBenchmark \
  --dataset-root data/datasets/eipBenchmark \
  --agent-root data/agents_eipBenchmark
```

Then you can evaluate with the generated metadata and normalized QA set.

## Main Scripts

- `scripts/run_pasr.py`: bootstrap entrypoint
- `scripts/preprocess_hotpotqa.py`: build agent assets from HotpotQA-like JSONL
- `scripts/prepare_eipbenchmark.py`: normalize and build eipBenchmark assets
- `scripts/run_qa_step.py`: run one end-to-end QA example
- `scripts/run_retrieval_step.py`: inspect routing and retrieval behavior
- `scripts/route_query.py`: inspect router decisions only
- `scripts/run_eval_batch.py`: batch evaluation for PASR-RAG variants
- `scripts/run_main_experiment_matrix.py`: compare PASR-RAG variants with FlashRAG baselines
- `scripts/run_ablation_suite.py`: ablation study runner
- `scripts/run_alpha_sweep.py`: alpha sweep
- `scripts/run_hparam_sweep.py`: hierarchical hyper-parameter sweep
- `scripts/run_pareto_experiment.py`: privacy-performance trade-off study
- `scripts/run_flashrag_baseline.py`: run centralized FlashRAG baselines

## Configurations

The main settings live in:

- `configs/pasr_rag/base.yaml`: default English/HotpotQA-style setup
- `configs/pasr_rag/eipBenchmark.yaml`: Chinese enterprise benchmark setup

Important knobs:

- router: `strategy`, `alpha`, `beta`, `privacy_budget_ratio`, `max_active_sources_k`, `threshold`
- retrieval: `backend`, `top_m_per_source`, `enable_parallel_retrieval`
- generation: `enable_summarization`, `enable_central_fusion`, `llm_model`, `api_base_url`
- privacy: `enable_privacy_term`, `privacy_eval_method`

## Metrics

The repository includes both answer quality metrics and privacy/efficiency metrics:

- `em`, `f1`, `accuracy`
- `support_fact_recall`
- `avg_selected_privacy_cost`
- `hrhr`
- `bbr`
- `avg_activated_sources`
- `latency_routing`
- `latency_e2e`
