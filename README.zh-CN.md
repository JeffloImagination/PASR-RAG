# PASR-RAG

[English README](README.md)

PASR-RAG 是一个面向**多源知识库隐私感知路由**的研究型 RAG 项目。仓库主要包含：

- `src/pasr_rag` 中的自研 PASR-RAG 核心流程
- 面向 HotpotQA 风格数据与 eipBenchmark 风格数据的预处理、路由、检索与评测脚本
- 消融实验、超参数搜索、Pareto 权衡实验和主实验矩阵脚本
- 用于集中式基线对比的 `FlashRAG/` 子项目

项目的核心思想是：在回答问题前，先对候选知识源同时计算**相关性**与**隐私成本**，只激活满足预算约束的部分 source agents，再执行局部检索、证据融合和最终生成。

## 项目亮点

- 提供多种路由策略：`pasr`、`rel_only`、`threshold`、`random`、`ma_rag_lite`
- 基于 source agent 的分布式检索与路由
- 支持局部摘要与中心证据融合
- 同时评估答案质量、隐私成本与效率指标
- 内置主实验、消融、`alpha` 扫描和分层超参数搜索脚本
- 同时支持英文 HotpotQA 类实验与中文企业政策类实验

## 目录结构

```text
.
|-- src/pasr_rag/                 # PASR-RAG 核心实现
|-- scripts/                      # 预处理、评测、实验运行脚本
|-- configs/pasr_rag/             # HotpotQA 与 eipBenchmark 配置
|-- data/                         # 数据集、agent 索引、FlashRAG 资产
|-- new-database/                 # eipBenchmark 风格原始语料与 QA
|-- outputs/                      # 日志与实验结果
|-- docs/                         # 提示词、设计文档、论文材料
|-- models/                       # 本地 embedding 模型
`-- FlashRAG/                     # 集中式 RAG 基线框架
```

## 方法流程

默认 PASR-RAG 流程如下：

1. 将语料切分为多个 source agents。
2. 为每个 agent 构建 embedding 与索引。
3. 估计各 agent 的隐私成本。
4. 使用 `alpha * relevance - beta * privacy_cost` 进行路由打分。
5. 从被选中的 agents 内部分别检索 top-`m` 证据。
6. 可选地执行局部摘要与中心融合。
7. 生成最终答案，并统计质量、隐私与时延指标。

## 运行环境

仓库根目录目前没有完整的 `requirements.txt` 或 `pyproject.toml`，因此环境需要根据代码依赖自行组装。

建议的基础环境：

- Python 3.10 或 3.11
- `numpy`
- `pyyaml`
- `openai`
- `sentence-transformers`
- `torch`
- `faiss-cpu` 或 `faiss-gpu`

可选依赖：

- `datasets`：用于从 Hugging Face 下载 HotpotQA
- `FlashRAG/` 自身运行所需依赖：用于集中式基线实验

示例安装命令：

```bash
pip install numpy pyyaml openai sentence-transformers torch faiss-cpu datasets
```

## 模型与 API 配置

默认配置假定：

- embedding 模型位于 `models/bge-base-en-v1.5` 或 `models/bge-base-zh-v1.5`
- 生成后端为 `openai_compatible`
- API Key 从环境变量 `DASHSCOPE_API_KEY` 读取
- 默认 Base URL 为 `https://dashscope.aliyuncs.com/compatible-mode/v1`

如果你使用其他模型服务，请修改：

- `configs/pasr_rag/base.yaml`
- `configs/pasr_rag/eipBenchmark.yaml`

## 快速开始

### 1. 初始化运行目录

```bash
python scripts/run_pasr.py --dry-run
```

该命令会校验配置，并创建 `outputs/logs`、`outputs/results` 等运行目录。

### 2. 预处理 HotpotQA 风格数据

```bash
python scripts/preprocess_hotpotqa.py \
  --config configs/pasr_rag/base.yaml \
  --input data/raw/hotpotqa_validation_small.jsonl \
  --output-root data/agents_hotpot_small
```

### 3. 运行单条问答

```bash
python scripts/run_qa_step.py \
  --config configs/pasr_rag/base.yaml \
  --metadata-path data/agents_hotpot_small/agent_metadata.json \
  --query "Were Scott Derrickson and Ed Wood of the same nationality?"
```

### 4. 运行批量评测

```bash
python scripts/run_eval_batch.py \
  --config configs/pasr_rag/base.yaml \
  --metadata-path data/agents_hotpot_small/agent_metadata.json \
  --dataset data/raw/hotpotqa_validation_small.jsonl \
  --output-dir outputs/results/pasr_batch_eval_smoke \
  --strategy pasr
```

### 5. 运行主实验矩阵

```bash
python scripts/run_main_experiment_matrix.py \
  --dataset data/raw/hotpotqa_validation_small.jsonl \
  --metadata-path data/agents_hotpot_small/agent_metadata.json \
  --flashrag-asset-dir data/flashrag_assets/hotpot_small \
  --output-root outputs/results/main_experiment_matrix_smoke
```

## eipBenchmark 数据流程

如果你要运行中文企业政策类实验，可先执行：

```bash
python scripts/prepare_eipbenchmark.py \
  --config configs/pasr_rag/eipBenchmark.yaml \
  --source-root new-database \
  --raw-root data/raw/eipBenchmark \
  --dataset-root data/datasets/eipBenchmark \
  --agent-root data/agents_eipBenchmark
```

该脚本会同步原始语料、规范化 QA，并构建 agent 级索引与元数据。

## 主要脚本

- `scripts/run_pasr.py`：项目 bootstrap 入口
- `scripts/preprocess_hotpotqa.py`：HotpotQA 类数据预处理
- `scripts/prepare_eipbenchmark.py`：eipBenchmark 数据准备
- `scripts/run_qa_step.py`：单条端到端问答
- `scripts/run_retrieval_step.py`：单条检索链路检查
- `scripts/route_query.py`：仅查看路由决策
- `scripts/run_eval_batch.py`：PASR-RAG 变体批量评测
- `scripts/run_main_experiment_matrix.py`：与 FlashRAG 基线的主实验矩阵
- `scripts/run_ablation_suite.py`：消融实验
- `scripts/run_alpha_sweep.py`：`alpha` 扫描
- `scripts/run_hparam_sweep.py`：分层超参数搜索
- `scripts/run_pareto_experiment.py`：隐私-性能权衡实验
- `scripts/run_flashrag_baseline.py`：集中式 FlashRAG 基线

## 配置文件

主要配置位于：

- `configs/pasr_rag/base.yaml`：英文/HotpotQA 风格实验配置
- `configs/pasr_rag/eipBenchmark.yaml`：中文企业数据实验配置

常用参数包括：

- router：`strategy`、`alpha`、`beta`、`privacy_budget_ratio`、`max_active_sources_k`、`threshold`
- retrieval：`backend`、`top_m_per_source`、`enable_parallel_retrieval`
- generation：`enable_summarization`、`enable_central_fusion`、`llm_model`、`api_base_url`
- privacy：`enable_privacy_term`、`privacy_eval_method`

## 评测指标

仓库同时支持答案质量指标与隐私/效率指标：

- `em`、`f1`、`accuracy`
- `support_fact_recall`
- `avg_selected_privacy_cost`
- `hrhr`
- `bbr`
- `avg_activated_sources`
- `latency_routing`
- `latency_e2e`

## 已有实验结果

仓库中已经包含较多实验输出，主要位于 `outputs/results/`。

例如 `outputs/results/main_experiment_matrix_full50_llm_local_summary_prompt_v2/matrix_summary.json` 中记录了 50 条样本上的 PASR-RAG 与 FlashRAG 基线对比结果，因此该仓库既适合复现实验，也适合继续扩展。

## 发布到 GitHub 前的建议

- 当前仓库包含较多本地大文件，主要在 `models/`、`data/` 和 `outputs/`。
- 如果准备公开发布，建议清理生成产物，或将大文件交给 Git LFS 管理。
- `FlashRAG/` 看起来是随仓携带的外部框架，公开发布时建议在 README 或 LICENSE 中说明其来源与许可证。

## 引用

如果你在论文、毕业设计或其他学术工作中使用了本项目，建议补充对应论文、项目主页或 BibTeX 引用信息。

## 许可证

仓库根目录目前没有顶层许可证文件。若你准备将项目发布到 GitHub，建议补充明确的开源许可证。
