"""
Data loading helpers for the topic modelling pipeline.

Simulation corpora are stored as CSV files under `simulation*/posts.csv`,
while MADOC Voat data is provided as a parquet export. The functions below
aggregate raw rows into thread-level documents that the modeller can consume.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _aggregate_threads(
    df: pd.DataFrame,
    thread_column: str,
    text_column: str,
    sort_column: Optional[str] = None,
    joiner: str = "\n\n",
) -> pd.DataFrame:
    """Collapse rows into thread-level documents."""
    if thread_column not in df.columns:
        raise ValueError(f"Missing thread identifier column '{thread_column}'")
    if text_column not in df.columns:
        raise ValueError(f"Missing text column '{text_column}'")

    working = df.copy()
    working[text_column] = (
        working[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    working = working[working[text_column].str.len() > 0]

    if sort_column and sort_column in working.columns:
        working = working.sort_values(sort_column)

    grouped = working.groupby(thread_column, dropna=False)

    docs = grouped[text_column].apply(lambda parts: joiner.join(p for p in parts if p))
    counts = grouped.size()

    result = pd.DataFrame(
        {
            "document_id": docs.index.astype(str),
            "text": docs.values,
            "num_messages": counts.reindex(docs.index).astype(int).values,
        }
    )

    if sort_column and sort_column in working.columns:
        result["first_order_value"] = grouped[sort_column].min().reindex(docs.index).values
        result["last_order_value"] = grouped[sort_column].max().reindex(docs.index).values

    return result.reset_index(drop=True)


def load_simulation_threads(
    sim_dir: Path,
    *,
    text_column: str = "tweet",
    thread_column: str = "thread_id",
    sort_column: Optional[str] = None,
    min_chars: int = 80,
    max_threads: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load and aggregate simulation posts into thread-level documents.

    Parameters
    ----------
    sim_dir: Path
        Directory that contains `posts.csv`.
    text_column: str
        Column containing message bodies (default: `tweet`).
    thread_column: str
        Column denoting thread membership (default: `thread_id`).
    sort_column: Optional[str]
        Column used for ordering messages before concatenation.
    min_chars: int
        Minimum character count required for a document to be kept.
    max_threads: Optional[int]
        Optional cap on the number of documents (uniform random sample).
    seed: int
        Random seed for sampling.
    """
    posts_path = sim_dir / "posts.csv"
    if not posts_path.exists():
        raise FileNotFoundError(f"Simulation posts file not found: {posts_path}")

    logger.info("Loading simulation posts from %s", posts_path)
    df = pd.read_csv(posts_path)
    logger.info("Loaded %d rows from %s", len(df), posts_path.name)

    documents = _aggregate_threads(df, thread_column, text_column, sort_column)
    documents["source"] = sim_dir.name

    documents = documents[documents["text"].str.len() >= int(min_chars)].reset_index(drop=True)

    if max_threads is not None and len(documents) > max_threads:
        before = len(documents)
        documents = (
            documents.sample(n=min(max_threads, len(documents)), random_state=seed)
            .reset_index(drop=True)
        )
        logger.info("Sampled %d threads out of %d", len(documents), before)

    return documents


def _resolve_threads(post_ids: pd.Series, parent_ids: pd.Series) -> pd.Series:
    """Resolve each row to a thread id by walking parent pointers."""
    mapping_series = parent_ids.fillna("").astype(str).replace({"nan": ""})
    post_series = post_ids.astype(str)

    parent_lookup = dict(zip(post_series.tolist(), mapping_series.tolist()))
    cache: dict[str, str] = {}

    def find_root(pid: str) -> str:
        if not pid:
            return ""
        if pid in cache:
            return cache[pid]
        seen = set()
        current = pid
        while True:
            parent = parent_lookup.get(current, "")
            if not parent or parent == current:
                cache[pid] = current
                return current
            if parent in seen:
                cache[pid] = current
                return current
            seen.add(parent)
            current = parent

    roots = []
    for pid, parent in zip(post_series.tolist(), mapping_series.tolist()):
        candidate = pid if not parent else find_root(parent)
        roots.append((candidate or pid) or "")
    return pd.Series(roots, index=post_ids.index, dtype="string")


def load_madoc_threads(
    parquet_path: Path,
    *,
    text_column: str = "content",
    interaction_column: str = "interaction_type",
    parent_column: str = "parent_id",
    post_id_column: str = "post_id",
    timestamp_column: Optional[str] = "publish_date",
    min_chars: int = 80,
    language_column: Optional[str] = "language",
    language_filter: Optional[str] = "English",
    max_threads: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load Voat MADOC data and collapse to thread-level documents.

    Thread identifiers are derived by following each row's `parent_id` to the
    root post. Root posts (where `parent_id` is null) own their thread.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"MADOC parquet not found: {parquet_path}")

    logger.info("Loading MADOC Voat data from %s", parquet_path)
    try:
        df = pd.read_parquet(parquet_path)
    except ImportError as exc:
        raise ImportError(
            "Reading parquet requires 'pyarrow' or 'fastparquet'. Install one of them."
        ) from exc

    logger.info("Loaded %d rows from %s", len(df), parquet_path.name)

    if language_column and language_column in df.columns and language_filter:
        before = len(df)
        df = df[df[language_column] == language_filter]
        logger.info(
            "Filtered to %d %s rows by language (%s)",
            len(df),
            parquet_path.name,
            language_filter,
        )
        if not len(df):
            raise ValueError("Language filter removed all rows; adjust filters.")

    if post_id_column not in df.columns:
        raise ValueError(f"Column '{post_id_column}' not found in MADOC data")

    df = df.copy()

    df["thread_id"] = _resolve_threads(df[post_id_column], df[parent_column] if parent_column in df.columns else pd.Series(index=df.index, dtype=object))

    documents = _aggregate_threads(
        df,
        thread_column="thread_id",
        text_column=text_column,
        sort_column=timestamp_column if timestamp_column in df.columns else None,
    )
    documents["source"] = parquet_path.stem

    documents = documents[documents["text"].str.len() >= int(min_chars)].reset_index(drop=True)

    if max_threads is not None and len(documents) > max_threads:
        before = len(documents)
        documents = (
            documents.sample(n=min(max_threads, len(documents)), random_state=seed)
            .reset_index(drop=True)
        )
        logger.info("Sampled %d MADOC threads out of %d", len(documents), before)

    return documents
