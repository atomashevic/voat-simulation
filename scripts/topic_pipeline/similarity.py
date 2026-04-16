"""
Topic similarity utilities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .modeling import TopicModelResult, load_embedding_model

logger = logging.getLogger(__name__)


def _ensure_topic_embeddings(
    result: TopicModelResult,
    embedding_model=None,
    *,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    embeddings = result.topic_embeddings
    if embeddings.size and embeddings.shape[0] == len(result.topic_ids):
        return embeddings, embedding_model

    logger.warning(
        "Topic embeddings missing or misaligned for %s; encoding top terms instead.",
        result.corpus_name,
    )

    if embedding_model is None:
        embedding_model = load_embedding_model(embedding_model_name)

    texts = result.topics_info["top_terms"].tolist()
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    return embeddings, embedding_model


def topic_similarity_matrix(
    source: TopicModelResult,
    reference: TopicModelResult,
    *,
    embedding_model=None,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """
    Compute cosine similarity between topics of two corpora.

    Returns a DataFrame where rows correspond to `source` topics and columns
    correspond to `reference` topics.
    """
    emb_source, embedding_model = _ensure_topic_embeddings(
        source,
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
    )
    emb_reference, _ = _ensure_topic_embeddings(
        reference,
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
    )

    if not len(emb_source) or not len(emb_reference):
        raise ValueError("Cannot compute similarity matrix with empty embeddings.")

    matrix = cosine_similarity(emb_source, emb_reference)
    df = pd.DataFrame(matrix, index=source.topic_ids, columns=reference.topic_ids)
    return df


def extract_top_matches(
    sim_matrix: pd.DataFrame,
    source: TopicModelResult,
    reference: TopicModelResult,
    *,
    top_n: int = 5,
    min_similarity: float = 0.2,
) -> pd.DataFrame:
    """
    Build a tidy table of the top matches for each source topic.
    """
    info_source = source.topics_info.set_index("topic_id")
    info_reference = reference.topics_info.set_index("topic_id")

    rows = []
    for topic_id in sim_matrix.index:
        scores = sim_matrix.loc[topic_id].sort_values(ascending=False)
        for rank, (ref_topic_id, score) in enumerate(scores.items(), start=1):
            if rank > top_n or score < min_similarity:
                continue

            s_info = info_source.loc[topic_id]
            r_info = info_reference.loc[ref_topic_id]
            rows.append(
                {
                    "source_topic_id": topic_id,
                    "source_label": s_info["label"],
                    "source_size": int(s_info["size"]),
                    "source_top_terms": s_info["top_terms"],
                    "reference_topic_id": ref_topic_id,
                    "reference_label": r_info["label"],
                    "reference_size": int(r_info["size"]),
                    "reference_top_terms": r_info["top_terms"],
                    "similarity": float(score),
                    "rank": rank,
                }
            )

    matches = pd.DataFrame(rows)
    matches = matches.sort_values(["similarity", "source_topic_id", "rank"], ascending=[False, True, True])
    return matches.reset_index(drop=True)


def similarity_summary(matches: pd.DataFrame) -> dict:
    if matches.empty:
        return {
            "matches": 0,
            "average_similarity": 0.0,
            "median_similarity": 0.0,
            "max_similarity": 0.0,
        }
    return {
        "matches": int(len(matches)),
        "average_similarity": float(matches["similarity"].mean()),
        "median_similarity": float(matches["similarity"].median()),
        "max_similarity": float(matches["similarity"].max()),
    }


def write_similarity_outputs(
    output_dir: Path,
    sim_matrix: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    metadata: Optional[dict] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_matrix.to_csv(output_dir / "similarity_matrix.csv", index=True)
    matches.to_csv(output_dir / "top_matches.csv", index=False)

    summary = similarity_summary(matches)
    if metadata:
        summary["metadata"] = metadata
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
