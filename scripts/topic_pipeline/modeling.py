"""
Topic modelling utilities built around BERTopic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TopicModelResult:
    """Container for a fitted BERTopic model and its artefacts."""

    corpus_name: str
    documents: pd.DataFrame
    model: "BERTopic"
    topics_info: pd.DataFrame
    document_topics: pd.DataFrame

    @property
    def topic_ids(self) -> List[int]:
        return self.topics_info["topic_id"].tolist()

    @property
    def topic_embeddings(self) -> np.ndarray:
        emb = getattr(self.model, "topic_embeddings_", None)
        if emb is None or not len(emb):
            return np.empty((0, 0))
        # topic_embeddings_ is aligned to topic index; filter by valid ids
        max_index = emb.shape[0]
        indices = [tid for tid in self.topic_ids if 0 <= tid < max_index]
        return emb[indices]


def load_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
):
    """Load a sentence-transformers embedding model."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Install 'torch' and 'sentence-transformers' to run the topic pipeline."
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading embedding model %s on %s", model_name, device)
    return SentenceTransformer(model_name, device=device)


def _build_topic_model(
    embedding_model,
    *,
    min_topic_size: int = 20,
    vectorizer_min_df: float = 0.01,
    vectorizer_max_df: float = 0.45,
    extra_stopwords: Optional[Sequence[str]] = None,
    use_umap: bool = True,
    random_state: int = 42,
    verbose: bool = False,
):
    """Create a BERTopic instance with defensive defaults."""
    try:
        from bertopic import BERTopic
    except ImportError as exc:
        raise ImportError("Install 'bertopic' to run the topic pipeline.") from exc

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stopwords = set(ENGLISH_STOP_WORDS)
    if extra_stopwords:
        stopwords.update(word.strip().lower() for word in extra_stopwords if word)

    min_df_value = vectorizer_min_df
    if isinstance(min_df_value, (int, float)) and min_df_value > 1:
        # Guard against invalid configurations when c-TFIDF aggregates
        # documents per topic (often < 50). Convert large absolute thresholds
        # to a conservative proportion.
        min_df_value = max(0.01, min_df_value / max(min_topic_size, 1))

    vectorizer_model = CountVectorizer(
        stop_words=list(stopwords),
        ngram_range=(1, 2),
        min_df=min_df_value,
        max_df=vectorizer_max_df,
        token_pattern=r"(?u)\b[^\W\d][\w\-]{2,}\b",
    )

    try:
        from hdbscan import HDBSCAN
    except ImportError as exc:
        raise ImportError("Install 'hdbscan' to run the topic pipeline.") from exc

    min_samples = max(2, min_topic_size // 2 or 1)

    clusterer = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    reducer = None
    if use_umap:
        try:
            from umap import UMAP  # type: ignore

            reducer = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=random_state,
            )
        except Exception as exc:  # pragma: no cover - dependency specific
            logger.warning("UMAP unavailable (%s); falling back to SVD.", exc)
    if reducer is None:
        try:
            from sklearn.decomposition import TruncatedSVD

            class SVDReducer:
                def __init__(self, n_components=5, random_state: int = 42):
                    self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)

                def fit(self, X, y=None):
                    self._svd.fit(X)
                    return self

                def fit_transform(self, X, y=None):
                    return self._svd.fit_transform(X)

                def transform(self, X):
                    return self._svd.transform(X)

            reducer = SVDReducer(n_components=5, random_state=random_state)
        except Exception:
            reducer = None

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=reducer,
        hdbscan_model=clusterer,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=verbose,
    )
    return topic_model


def _summarise_topics(model: "BERTopic", top_n: int = 10) -> pd.DataFrame:
    info = model.get_topic_info()
    info = info[info["Topic"] != -1].copy()
    info = info.rename(columns={"Topic": "topic_id", "Count": "size"})

    top_terms: List[str] = []
    labels: List[str] = []
    for topic_id in info["topic_id"]:
        words = model.get_topic(topic_id) or []
        ordered = [word for word, _ in words[:top_n]]
        summary = ", ".join(ordered)
        top_terms.append(summary)
        labels.append(f"Topic {topic_id}: {' '.join(ordered[:3])}".strip())

    info["top_terms"] = top_terms
    info["label"] = labels
    info = info.reset_index(drop=True)
    return info


def train_topic_model(
    corpus_name: str,
    documents: pd.DataFrame,
    *,
    text_column: str = "text",
    embedding_model=None,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    min_topic_size: int = 20,
    vectorizer_min_df: float = 0.01,
    vectorizer_max_df: float = 0.45,
    extra_stopwords: Optional[Sequence[str]] = None,
    use_umap: bool = True,
    top_n_terms: int = 10,
    random_state: int = 42,
    verbose: bool = False,
) -> TopicModelResult:
    """
    Fit a BERTopic model and return structured artefacts.

    Parameters mirror `_build_topic_model`; the function handles embedding model
    instantiation if one is not supplied.
    """
    if text_column not in documents.columns:
        raise ValueError(f"Column '{text_column}' not found in documents for {corpus_name}")

    texts = documents[text_column].tolist()
    if not texts:
        raise ValueError(f"No documents found for corpus '{corpus_name}'")

    if embedding_model is None:
        embedding_model = load_embedding_model(embedding_model_name, device=device)

    logger.info(
        "Training BERTopic model for %s (%d documents, min_topic_size=%d)",
        corpus_name,
        len(texts),
        min_topic_size,
    )

    topic_model = _build_topic_model(
        embedding_model,
        min_topic_size=min_topic_size,
        vectorizer_min_df=vectorizer_min_df,
        vectorizer_max_df=vectorizer_max_df,
        extra_stopwords=extra_stopwords,
        use_umap=use_umap,
        random_state=random_state,
        verbose=verbose,
    )

    topics, _ = topic_model.fit_transform(texts)
    doc_topics = pd.DataFrame(
        {
            "document_id": documents["document_id"].astype(str),
            "topic_id": topics,
        }
    )

    topic_info = _summarise_topics(topic_model, top_n=top_n_terms)

    logger.info(
        "Model for %s discovered %d topics (avg size %.2f)",
        corpus_name,
        len(topic_info),
        topic_info["size"].mean(),
    )

    return TopicModelResult(
        corpus_name=corpus_name,
        documents=documents.copy(),
        model=topic_model,
        topics_info=topic_info,
        document_topics=doc_topics,
    )
