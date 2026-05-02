from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def patch_bm25s_utf8_io() -> None:
    import json as std_json
    from pathlib import Path as StdPath
    from typing import Iterable

    import bm25s
    import bm25s.tokenization as tokenization
    from bm25s import __version__ as bm25s_version
    from bm25s import json_functions, logging, utils
    import numpy as np

    def save_utf8(
        self,
        save_dir,
        corpus=None,
        data_name="data.csc.index.npy",
        indices_name="indices.csc.index.npy",
        indptr_name="indptr.csc.index.npy",
        vocab_name="vocab.index.json",
        params_name="params.index.json",
        nnoc_name="nonoccurrence_array.index.npy",
        corpus_name="corpus.jsonl",
        allow_pickle=False,
    ):
        save_dir = StdPath(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(save_dir / data_name, self.scores["data"], allow_pickle=allow_pickle)
        np.save(save_dir / indices_name, self.scores["indices"], allow_pickle=allow_pickle)
        np.save(save_dir / indptr_name, self.scores["indptr"], allow_pickle=allow_pickle)

        if self.nonoccurrence_array is not None:
            np.save(save_dir / nnoc_name, self.nonoccurrence_array, allow_pickle=allow_pickle)

        with open(save_dir / vocab_name, "w", encoding="utf-8") as f:
            f.write(json_functions.dumps(self.vocab_dict))

        params = dict(
            k1=self.k1,
            b=self.b,
            delta=self.delta,
            method=self.method,
            idf_method=self.idf_method,
            dtype=self.dtype,
            int_dtype=self.int_dtype,
            num_docs=self.scores["num_docs"],
            version=bm25s_version,
            backend=self.backend,
        )
        with open(save_dir / params_name, "w", encoding="utf-8") as f:
            std_json.dump(params, f, indent=4, ensure_ascii=False)

        corpus = corpus if corpus is not None else self.corpus
        if corpus is not None:
            with open(save_dir / corpus_name, "w", encoding="utf-8") as f:
                if not isinstance(corpus, Iterable):
                    logging.warning("The corpus is not an iterable. Skipping saving the corpus.")
                else:
                    for i, doc in enumerate(corpus):
                        if isinstance(doc, str):
                            doc = {"id": i, "text": doc}
                        elif not isinstance(doc, (dict, list, tuple)):
                            logging.warning(
                                f"Document at index {i} is not a string, dictionary, list or tuple. Skipping."
                            )
                            continue

                        try:
                            doc_str = json_functions.dumps(doc)
                        except Exception as e:
                            logging.warning(f"Error saving document at index {i}: {e}")
                        else:
                            f.write(doc_str + "\n")

            mmidx = utils.corpus.find_newline_positions(save_dir / corpus_name)
            utils.corpus.save_mmindex(mmidx, path=save_dir / corpus_name)

    def save_vocab_utf8(self, save_dir: str, vocab_name: str = "vocab.tokenizer.json"):
        save_dir = StdPath(save_dir)
        path = save_dir / vocab_name
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            payload = {
                "word_to_stem": self.word_to_stem,
                "stem_to_sid": self.stem_to_sid,
                "word_to_id": self.word_to_id,
            }
            f.write(json_functions.dumps(payload))

    def save_stopwords_utf8(self, save_dir: str, stopwords_name: str = "stopwords.tokenizer.json"):
        save_dir = StdPath(save_dir)
        path = save_dir / stopwords_name
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_functions.dumps(self.stopwords))

    bm25s.BM25.save = save_utf8
    tokenization.Tokenizer.save_vocab = save_vocab_utf8
    tokenization.Tokenizer.save_stopwords = save_stopwords_utf8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a FlashRAG baseline on prepared assets")
    parser.add_argument("--asset-dir", default="data/flashrag_assets/hotpot_small")
    parser.add_argument("--save-dir", default="outputs/results/flashrag_baseline")
    parser.add_argument("--retrieval-method", default="bm25", choices=["bm25", "bge", "hybrid"])
    parser.add_argument("--generator-model", default="qwen-plus", help="FlashRAG generator model key or path")
    parser.add_argument("--generator-framework", default="openai", choices=["openai", "hf", "vllm", "fschat"])
    parser.add_argument("--generator-api-key", default=None)
    parser.add_argument("--generator-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--generator-max-tokens", type=int, default=32)
    parser.add_argument(
        "--retrieval-model-path",
        default="models/bge-base-en-v1.5",
        help="Required for dense retrieval methods",
    )
    parser.add_argument("--index-path", default=None, help="Provide a built FlashRAG index path")
    parser.add_argument("--bm25-backend", default="bm25s", choices=["bm25s", "pyserini"])
    parser.add_argument("--dense-faiss-type", default="Flat")
    parser.add_argument("--dense-batch-size", type=int, default=16)
    parser.add_argument("--dense-max-length", type=int, default=512)
    parser.add_argument("--hybrid-merge-method", default="rrf", choices=["concat", "rrf", "rerank"])
    return parser


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def main() -> int:
    args = build_parser().parse_args()
    asset_dir = resolve_repo_path(args.asset_dir)
    save_dir = resolve_repo_path(args.save_dir)
    flashrag_root = resolve_repo_path("FlashRAG")
    sys.path.insert(0, str(flashrag_root))

    try:
        from flashrag.config import Config
        from flashrag.dataset.dataset import Dataset
        from flashrag.pipeline import SequentialPipeline
        from flashrag.prompt import PromptTemplate
        from flashrag.retriever.index_builder import Index_Builder
    except Exception as exc:
        raise SystemExit(
            f"FlashRAG import failed. Please install its runtime dependencies before running baselines.\n{exc}"
        )

    if args.retrieval_method in {"bm25", "hybrid"} and args.bm25_backend == "bm25s":
        patch_bm25s_utf8_io()

    dataset_rows = []
    dataset_path = asset_dir / "dataset" / "test.jsonl"
    with dataset_path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            dataset_rows.append(row)

    save_dir.mkdir(parents=True, exist_ok=True)
    api_key = args.generator_api_key or os.environ.get("DASHSCOPE_API_KEY")
    if args.generator_framework == "openai" and not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is not set. Please export it before running FlashRAG baselines.")

    built_indexes = prepare_indexes(args, asset_dir, Index_Builder)

    config_dict = {
        "dataset_name": "pasr_hotpot_small",
        "data_dir": str(asset_dir),
        "dataset_path": str(asset_dir / "dataset"),
        "split": ["test"],
        "save_dir": str(save_dir),
        "save_note": "flashrag_baseline",
        "save_intermediate_data": True,
        "save_metric_score": True,
        "metrics": ["em", "f1", "acc"],
        "framework": args.generator_framework,
        "generator_model": args.generator_model,
        "generation_params": {"max_tokens": args.generator_max_tokens},
        "retrieval_method": "bm25" if args.retrieval_method == "hybrid" else args.retrieval_method,
        "retrieval_topk": 5,
        "index_path": built_indexes.get("primary_index_path"),
        "corpus_path": str(asset_dir / "corpus.jsonl"),
        "generator_model_path": args.generator_model,
        "retrieval_model_path": args.retrieval_model_path,
        "bm25_backend": args.bm25_backend,
        "use_reranker": False,
        "save_retrieval_cache": False,
        "use_retrieval_cache": False,
        "test_sample_num": None,
        "random_sample": False,
        "instruction": None,
        "openai_setting": {
            "api_key": api_key,
            "base_url": args.generator_base_url,
        },
    }

    retrieval_model_path = str(resolve_repo_path(args.retrieval_model_path)) if args.retrieval_model_path else None
    if args.retrieval_method in {"bge", "hybrid"} and not retrieval_model_path:
        raise SystemExit("Dense/Hybrid FlashRAG baselines require --retrieval-model-path.")
    if args.retrieval_method in {"bge", "hybrid"} and not Path(retrieval_model_path).exists():
        raise SystemExit(f"Dense retrieval model path does not exist: {retrieval_model_path}")
    config_dict["retrieval_model_path"] = retrieval_model_path
    if args.generator_framework != "openai" and not args.generator_model:
        raise SystemExit("Non-openai FlashRAG baselines require --generator-model.")

    if args.retrieval_method == "hybrid":
        config_dict["use_multi_retriever"] = True
        config_dict["multi_retriever_setting"] = {
            "merge_method": args.hybrid_merge_method,
            "topk": 5,
            "retriever_list": [
                {
                    "retrieval_method": "bm25",
                    "corpus_path": str(asset_dir / "corpus.jsonl"),
                    "index_path": built_indexes["bm25_index_path"],
                    "retrieval_topk": 5,
                    "bm25_backend": args.bm25_backend,
                },
                {
                    "retrieval_method": "bge",
                    "corpus_path": str(asset_dir / "corpus.jsonl"),
                    "index_path": built_indexes["dense_index_path"],
                    "retrieval_model_path": retrieval_model_path,
                    "retrieval_topk": 5,
                },
            ],
        }

    config = Config(config_dict=config_dict)
    prompt_template = PromptTemplate(
        config,
        system_prompt=(
            "Answer the question based on the given document. "
            "Only give me the answer and do not output any other words.\n"
            "The following are given documents.\n\n{reference}"
        ),
        user_prompt="Question: {question}\nAnswer:",
    )

    dataset = Dataset(config=config, data=dataset_rows)
    pipeline = SequentialPipeline(config, prompt_template=prompt_template)
    output_dataset = pipeline.run(dataset, do_eval=True)
    print(
        json.dumps(
            {
                "save_dir": str(save_dir),
                "predictions": output_dataset.pred,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def prepare_indexes(args, asset_dir: Path, Index_Builder) -> dict[str, str]:
    corpus_path = str(asset_dir / "corpus.jsonl")
    result: dict[str, str] = {}

    if args.retrieval_method in {"bm25", "hybrid"}:
        bm25_dir = asset_dir / "bm25_index"
        bm25_dir.mkdir(parents=True, exist_ok=True)
        builder = Index_Builder(
            retrieval_method="bm25",
            model_path=None,
            corpus_path=corpus_path,
            save_dir=str(bm25_dir),
            max_length=180,
            batch_size=16,
            use_fp16=False,
            bm25_backend=args.bm25_backend,
        )
        builder.build_index()
        result["bm25_index_path"] = str(bm25_dir / "bm25")

    retrieval_model_path = str(resolve_repo_path(args.retrieval_model_path)) if args.retrieval_model_path else None

    if args.retrieval_method in {"bge", "hybrid"}:
        dense_dir = asset_dir / "dense_index"
        dense_dir.mkdir(parents=True, exist_ok=True)
        builder = Index_Builder(
            retrieval_method="bge",
            model_path=retrieval_model_path,
            corpus_path=corpus_path,
            save_dir=str(dense_dir),
            max_length=args.dense_max_length,
            batch_size=args.dense_batch_size,
            use_fp16=False,
            faiss_type=args.dense_faiss_type,
        )
        builder.build_index()
        result["dense_index_path"] = str(dense_dir / f"bge_{args.dense_faiss_type}.index")

    if args.retrieval_method == "bm25":
        result["primary_index_path"] = result["bm25_index_path"]
    elif args.retrieval_method == "bge":
        result["primary_index_path"] = result["dense_index_path"]
    else:
        result["primary_index_path"] = result["bm25_index_path"]
    return result


if __name__ == "__main__":
    raise SystemExit(main())
