#!/usr/bin/env python3
"""
Unified topic modelling pipeline for simulation runs and MADOC Voat data.

Example usage:

    python scripts/run_topic_similarity_pipeline.py \
        --simulation-run simulation \
        --simulation-run simulation2 \
        --simulation-run simulation3 \
        --madoc-path MADOC/voat-technology/voat_technology_madoc.parquet \
        --output-dir topic_comparison_results/unified_topics \
        --min-topic-size 20 \
        --similarity-top-n 5 \
        --similarity-min 0.3

The script aggregates thread-level documents, preprocesses text, fits BERTopic
models, and scores similarity between each simulation run and the MADOC
reference. Artefacts include CSV tables, summary JSON, topic size plots, and
heatmaps under the specified output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from topic_pipeline import data_io, modeling, plots, preprocessing, similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified topic similarity pipeline.")
    parser.add_argument(
        "--simulation-run",
        dest="simulation_runs",
        type=Path,
        action="append",
        required=True,
        help="Path to a simulation run directory containing posts.csv (can be repeated).",
    )
    parser.add_argument(
        "--madoc-path",
        type=Path,
        required=True,
        help="Path to MADOC Voat parquet export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for pipeline artefacts (default: topic_comparison_results/unified_<timestamp>).",
    )
    parser.add_argument(
        "--min-thread-chars",
        type=int,
        default=80,
        help="Minimum characters required for a thread document to be retained.",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase documents during preprocessing.",
    )
    parser.add_argument(
        "--max-simulation-threads",
        type=int,
        default=None,
        help="Optional cap on number of simulation threads (after filtering).",
    )
    parser.add_argument(
        "--max-madoc-threads",
        type=int,
        default=None,
        help="Optional cap on number of MADOC threads (after filtering).",
    )
    parser.add_argument(
        "--madoc-language",
        type=str,
        default="English",
        help="Language filter for MADOC rows (set to '' to skip).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for embeddings (e.g., 'cpu', 'cuda'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=20,
        help="Minimum number of documents per topic for BERTopic.",
    )
    parser.add_argument(
        "--madoc-min-topic-size",
        type=int,
        default=None,
        help="Override BERTopic min_topic_size for MADOC reference (defaults to --min-topic-size).",
    )
    parser.add_argument(
        "--vectorizer-min-df",
        type=float,
        default=0.01,
        help="Minimum document frequency (fraction or absolute >=1) for the BERTopic vectorizer.",
    )
    parser.add_argument(
        "--vectorizer-max-df",
        type=float,
        default=0.45,
        help="Maximum document frequency (fraction) for the BERTopic vectorizer.",
    )
    parser.add_argument(
        "--extra-stopword",
        dest="extra_stopwords",
        action="append",
        default=None,
        help="Additional stopwords for the BERTopic vectorizer (repeatable).",
    )
    parser.add_argument(
        "--top-n-terms",
        type=int,
        default=10,
        help="Number of terms to include in topic summaries.",
    )
    parser.add_argument(
        "--similarity-top-n",
        type=int,
        default=5,
        help="Top N matches to keep per simulation topic.",
    )
    parser.add_argument(
        "--similarity-min",
        type=float,
        default=0.2,
        help="Minimum cosine similarity required to keep a match.",
    )
    parser.add_argument(
        "--use-umap",
        dest="use_umap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable UMAP dimensionality reduction inside BERTopic.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def ensure_output_dir(path: Path | None) -> Path:
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("topic_comparison_results") / f"unified_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_config(output_dir: Path, args: argparse.Namespace, resolved_stopwords: List[str]) -> None:
    config_path = output_dir / "config.json"
    serialisable = {}
    for key, value in vars(args).items():
        if isinstance(value, list):
            serialisable[key] = [str(item) for item in value]
        elif isinstance(value, Path):
            serialisable[key] = str(value)
        else:
            serialisable[key] = value
    serialisable["resolved_extra_stopwords"] = resolved_stopwords
    config_path.write_text(json.dumps(serialisable, indent=2))


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    default_stopwords = [
        "justsaying",
        "kiddie",
        "nerd",
        "nerds",
        "updates",
        "update",
        "upgrade",
        "upgrades",
        "ubuntu",
        "driver",
        "drivers",
        "games",
        "gaming",
        "steam",
        "desktop",
    ]
    resolved_stopwords = default_stopwords + (args.extra_stopwords or [])

    output_dir = ensure_output_dir(args.output_dir)
    save_config(output_dir, args, resolved_stopwords)

    embedding_model = modeling.load_embedding_model(
        model_name=args.embedding_model,
        device=args.device,
    )

    simulation_results: List[modeling.TopicModelResult] = []
    for sim_path in args.simulation_runs:
        corpus_name = sim_path.name
        logger = logging.getLogger(f"simulation.{corpus_name}")

        documents = data_io.load_simulation_threads(
            sim_path,
            min_chars=args.min_thread_chars,
            max_threads=args.max_simulation_threads,
        )
        documents = preprocessing.preprocess_documents(
            documents,
            min_chars=args.min_thread_chars,
            lowercase=args.lowercase,
        )

        corpus_dir = output_dir / corpus_name
        corpus_dir.mkdir(parents=True, exist_ok=True)
        documents.to_csv(corpus_dir / "thread_documents.csv", index=False)

        result = modeling.train_topic_model(
            corpus_name=corpus_name,
            documents=documents,
            embedding_model=embedding_model,
            min_topic_size=args.min_topic_size,
            vectorizer_min_df=args.vectorizer_min_df,
            vectorizer_max_df=args.vectorizer_max_df,
            top_n_terms=args.top_n_terms,
            use_umap=args.use_umap,
            extra_stopwords=resolved_stopwords,
        )

        result.topics_info.to_csv(corpus_dir / "topics.csv", index=False)
        result.document_topics.to_csv(corpus_dir / "document_topics.csv", index=False)
        plots.plot_topic_sizes(
            result.topics_info,
            corpus_dir / "topic_sizes.png",
            title=f"{corpus_name} topic sizes",
        )

        simulation_results.append(result)
        logger.info("Saved artefacts to %s", corpus_dir)

    madoc_documents = data_io.load_madoc_threads(
        args.madoc_path,
        min_chars=args.min_thread_chars,
        language_filter=args.madoc_language or None,
        max_threads=args.max_madoc_threads,
    )
    madoc_documents = preprocessing.preprocess_documents(
        madoc_documents,
        min_chars=args.min_thread_chars,
        lowercase=args.lowercase,
    )

    madoc_dir = output_dir / "madoc"
    madoc_dir.mkdir(parents=True, exist_ok=True)
    madoc_documents.to_csv(madoc_dir / "thread_documents.csv", index=False)

    madoc_min_topic_size = args.madoc_min_topic_size or args.min_topic_size

    madoc_result = modeling.train_topic_model(
        corpus_name="madoc",
        documents=madoc_documents,
        embedding_model=embedding_model,
        min_topic_size=madoc_min_topic_size,
        vectorizer_min_df=args.vectorizer_min_df,
        vectorizer_max_df=args.vectorizer_max_df,
        top_n_terms=args.top_n_terms,
        use_umap=args.use_umap,
        extra_stopwords=resolved_stopwords,
    )
    madoc_result.topics_info.to_csv(madoc_dir / "topics.csv", index=False)
    madoc_result.document_topics.to_csv(madoc_dir / "document_topics.csv", index=False)
    plots.plot_topic_sizes(
        madoc_result.topics_info,
        madoc_dir / "topic_sizes.png",
        title="MADOC topic sizes",
    )

    combined_matches: List[pd.DataFrame] = []
    for sim_result in simulation_results:
        pair_dir = output_dir / f"{sim_result.corpus_name}_vs_madoc"
        sim_matrix = similarity.topic_similarity_matrix(
            sim_result,
            madoc_result,
            embedding_model=embedding_model,
            embedding_model_name=args.embedding_model,
        )
        matches = similarity.extract_top_matches(
            sim_matrix,
            sim_result,
            madoc_result,
            top_n=args.similarity_top_n,
            min_similarity=args.similarity_min,
        )

        metadata = {
            "simulation_run": sim_result.corpus_name,
            "madoc_reference": "madoc",
            "min_topic_size": args.min_topic_size,
            "similarity_min": args.similarity_min,
            "top_n": args.similarity_top_n,
        }
        similarity.write_similarity_outputs(pair_dir, sim_matrix, matches, metadata=metadata)
        plots.plot_similarity_heatmap(
            sim_matrix,
            pair_dir / "similarity_heatmap.png",
            title=f"{sim_result.corpus_name} vs MADOC",
        )

        if not matches.empty:
            temp = matches.copy()
            temp["simulation_run"] = sim_result.corpus_name
            combined_matches.append(temp)

    if combined_matches:
        all_matches = pd.concat(combined_matches, ignore_index=True)
        all_matches.to_csv(output_dir / "all_simulation_matches.csv", index=False)

    logging.getLogger(__name__).info("Pipeline complete. Artefacts stored in %s", output_dir)


if __name__ == "__main__":
    main()
