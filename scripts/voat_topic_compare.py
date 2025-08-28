#!/usr/bin/env python3
"""
Compare topics between simulation2 (Voat) and MADOC Voat samples using BERTopic.

Pipeline:
- Load and preprocess two corpora (simulation2 posts and MADOC Voat).
- Train BERTopic models separately with MiniLM, UMAP, HDBSCAN, c-TFIDF.
- Extract topic info and compute topic embeddings.
- Compute cosine similarity matrix and perform Hungarian matching.
- Filter by similarity threshold and compute Jaccard overlap of top words.
- Export CSVs/JSON and optional heatmap plot.

Run:
  python scripts/voat_topic_compare.py       --sim2-posts-csv simulation2/posts.csv       --madoc-input MADOC/voat-technology/voat_technology_madoc.parquet       --outdir simulation2/topic_compare --save-heatmap

Dependencies:
  pip install bertopic[visualization] sentence-transformers umap-learn hdbscan               scikit-learn pandas numpy scipy matplotlib
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_text_columns(df: pd.DataFrame, preferred: Optional[List[str]] = None) -> List[str]:
    if preferred:
        cols = [c for c in preferred if c in df.columns]
        if cols:
            return cols
    # Prefer reasonable combinations and singles in order
    candidates = [
        ["title", "body"],
        ["title", "selftext"],
        ["title", "text"],
        ["title", "content"],
        ["content"],
        ["text"],
        ["body"],
        ["title"],
    ]
    for group in candidates:
        if all(c in df.columns for c in group):
            return group
    # Heuristic fallback: choose the most text-like object column, avoid ids/urls/meta
    meta_pat = re.compile(
        r"(^|_)id$|user|date|time|url|lang|platform|community|interaction|sentiment|subjectiv|toxicity|score|vote|strict|parent",
        re.IGNORECASE,
    )
    obj_cols = [c for c in df.columns if df[c].dtype == object and not meta_pat.search(c)]
    if obj_cols:
        def mean_len(col: str) -> float:
            s = df[col].dropna()
            if not len(s):
                return 0.0
            s = s.astype(str)
            return float(s.str.len().mean())
        obj_cols.sort(key=mean_len, reverse=True)
        return [obj_cols[0]]
    any_obj = [c for c in df.columns if df[c].dtype == object]
    return any_obj[:1] if any_obj else []


URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = s or ""
    # Remove URLs and normalize whitespace/newlines
    s = URL_RE.sub(" ", s)
    s = s.replace("\n", " ")
    # Fix missing space after periods before capital letters (e.g., "end.Sentence")
    s = re.sub(r"\.([A-Z])", r". \1", s)
    # Strip username/mention prefix at start of text (e.g., ".user", "@user")
    s = re.sub(r"^[.@]\S+\s+", "", s)
    s = WS_RE.sub(" ", s)
    return s.strip()


def build_text_series(df: pd.DataFrame, text_cols: List[str]) -> pd.Series:
    def join_row(row: pd.Series) -> str:
        parts = []
        for c in text_cols:
            val = row.get(c)
            if isinstance(val, str) and val.strip():
                # Remove leading column-name prefixes like 'title:' 'Title:' etc.
                cleaned_val = re.sub(rf'^\s*{c}\s*:\s*', '', val, flags=re.IGNORECASE)
                # Also generic TITLE:, BODY:, CONTENT: if mismatched column naming
                cleaned_val = re.sub(r'^(title|body|content|text|selftext)\s*:\s*', '', cleaned_val, flags=re.IGNORECASE)
                parts.append(cleaned_val)
        return clean_text(". ".join(parts))

    return df[text_cols].apply(lambda r: join_row(r), axis=1)


def _looks_like_header(s: str) -> bool:
    if not s:
        return False
    low = s.lower().strip()
    # Heuristics: very short and contains only column-name like tokens
    return low in {"title", "body", "selftext", "text", "content"}


def filter_docs(texts: pd.Series, min_chars: int, drop_headers: bool) -> pd.Series:
    mask = texts.str.len() >= min_chars
    if drop_headers:
        mask &= ~texts.apply(_looks_like_header)
    return texts[mask]


def load_corpus(
    path: Path,
    text_cols: Optional[List[str]] = None,
    max_docs: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], pd.DataFrame, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    used_cols = detect_text_columns(df, preferred=text_cols)
    if not used_cols:
        raise ValueError("No suitable text columns found. Use --text-cols to specify.")

    texts = build_text_series(df, used_cols)
    texts = texts[texts.str.len() > 0]
    if max_docs is not None and len(texts) > max_docs:
        texts = texts.sample(n=max_docs, random_state=seed)

    return texts.tolist(), df.loc[texts.index].reset_index(drop=True), used_cols


def load_madoc_samples(root: Path, max_docs: Optional[int], seed: int, text_cols: Optional[List[str]] = None) -> Tuple[List[str], pd.DataFrame, List[str]]:
    sample_dirs = sorted([p for p in root.glob("sample_*") if p.is_dir()])
    frames = []
    for d in sample_dirs:
        for f in d.glob("**/*.csv"):
            try:
                frames.append(pd.read_csv(f))
            except Exception:
                continue
        for f in d.glob("**/*.parquet"):
            try:
                frames.append(pd.read_parquet(f))
            except Exception:
                continue
    if not frames:
        raise FileNotFoundError(f"No sample files found under {root}")
    df = pd.concat(frames, ignore_index=True)
    used_cols = detect_text_columns(df, preferred=text_cols)
    texts = build_text_series(df, used_cols)
    texts = texts[texts.str.len() > 0]
    if max_docs is not None and len(texts) > max_docs:
        texts = texts.sample(n=max_docs, random_state=seed)
    return texts.tolist(), df.loc[texts.index].reset_index(drop=True), used_cols


def make_embedding_model(device_preference: Optional[str] = None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", local_path: Optional[str] = None, hf_offline: bool = False):
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        device = device_preference
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if hf_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        load_target = local_path if local_path else model_name
        model = SentenceTransformer(load_target, device=device)
        return model
    except Exception as e:
        raise RuntimeError(
            "Failed to load SentenceTransformer model. Ensure 'sentence-transformers' and 'torch' are installed and the model is available."
        ) from e


def build_bertopic(
    embedding_model,
    min_topic_size: int,
    seed: int,
    vectorizer_min_df: int = 5,
):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    import sys
    
    # Try UMAP; if unavailable (e.g., numba/NumPy version conflict), fall back to a lightweight reducer
    umap_model = None
    try:
        from umap import UMAP  # type: ignore
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
        )
    except Exception as e:
        print(f"Warning: UMAP unavailable ({e}). Falling back to TruncatedSVD/identity reducer.", file=sys.stderr)

        # Minimal reducer with sklearn TruncatedSVD if available; otherwise identity (no reduction)
        try:
            from sklearn.decomposition import TruncatedSVD  # type: ignore

            class SVDReducer:
                def __init__(self, n_components=5, random_state=42):
                    self.n_components = n_components
                    self.random_state = random_state
                    self._svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)

                def fit(self, X, y=None):
                    self._svd.fit(X)
                    return self

                def fit_transform(self, X, y=None):
                    return self._svd.fit_transform(X)

                def transform(self, X):
                    return self._svd.transform(X)

            umap_model = SVDReducer(n_components=5, random_state=seed)
        except Exception:
            class IdentityReducer:
                def fit(self, X, y=None):
                    return self

                def fit_transform(self, X, y=None):
                    return X

                def transform(self, X):
                    return X

            umap_model = IdentityReducer()
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    # Vectorizer with stricter token pattern:
    # - drop pure numbers and very short tokens
    # - keep words starting with a letter, allow digits/underscore/hyphen after
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=vectorizer_min_df,
        token_pattern=r"(?u)\\b[^\W\d][\w\-]{2,}\\b",
    )
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=True,
    )
    return topic_model


@dataclass
class TopicInfo:
    topic_id: int
    size: int
    label: str
    top_words: List[str]


def _clean_tokens(words: List[str]) -> List[str]:
    cleaned = []
    for w in words:
        if not w:
            continue
        if len(w) >= 4 and w.isupper():
            continue
        if re.fullmatch(r"[0-9a-fA-F]{4,}", w):
            continue
        if re.fullmatch(r"\d+", w):
            continue
        if len(w) > 30:
            continue
        cleaned.append(w)
    return cleaned


def extract_topic_info(model, docs: List[str], top_n_words: int = 10) -> List[TopicInfo]:
    _topics, _ = model.transform(docs)
    info = model.get_topic_info()
    try:
        label_map = model.get_topic_labels()
    except Exception:
        label_map = {int(row["Topic"]): row.get("Name", f"Topic {int(row['Topic'])}") for _, row in info.iterrows()}

    topic_infos: List[TopicInfo] = []
    for _, row in info.iterrows():
        tid = int(row["Topic"])  # -1 are outliers
        if tid == -1:
            continue
        name = label_map.get(tid, str(row.get("Name", f"Topic {tid}")))
        words_scores = model.get_topic(tid) or []
        top_words = [w for w, _ in words_scores[: (top_n_words * 2)]]
        top_words = _clean_tokens(top_words)[:top_n_words]
        topic_infos.append(TopicInfo(topic_id=tid, size=int(row["Count"]), label=name, top_words=top_words))
    return topic_infos


def compute_topic_embeddings(model, topic_infos: List[TopicInfo], embedding_model) -> np.ndarray:
    if getattr(model, "topic_embeddings_", None) is not None:
        topic_info_df = model.get_topic_info()
        topic_info_df = topic_info_df[topic_info_df["Topic"] != -1].reset_index(drop=True)
        id_to_idx = {int(row.Topic): i for i, row in topic_info_df.iterrows()}
        embs = []
        for t in topic_infos:
            idx = id_to_idx.get(t.topic_id)
            if idx is None:
                text = " ".join(t.top_words)
                embs.append(embedding_model.encode([text], convert_to_numpy=True)[0])
            else:
                embs.append(model.topic_embeddings_[idx])
        return np.vstack(embs) if embs else np.zeros((0, 384))
    texts = [" ".join(t.top_words) for t in topic_infos]
    if not texts:
        return np.zeros((0, 384))
    return embedding_model.encode(texts, convert_to_numpy=True)


def compute_topic_centroids(
    texts: List[str], topics: List[int], topic_infos: List[TopicInfo], embedding_model, max_docs_per_topic: int = 200
) -> np.ndarray:
    topic_to_indices = {ti.topic_id: [] for ti in topic_infos}
    for idx, t in enumerate(topics):
        if t in topic_to_indices and len(topic_to_indices[t]) < max_docs_per_topic:
            topic_to_indices[t].append(idx)
    embs = []
    for ti in topic_infos:
        idxs = topic_to_indices.get(ti.topic_id, [])
        if not idxs:
            embs.append(np.zeros((384,), dtype=np.float32))
            continue
        batch_texts = [texts[i] for i in idxs]
        vecs = embedding_model.encode(batch_texts, convert_to_numpy=True)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        embs.append(vecs.mean(axis=0))
    return np.vstack(embs) if embs else np.zeros((0, 384))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def soft_jaccard(words_a: List[str], words_b: List[str], embedding_model, sim_thresh: float = 0.6) -> float:
    """Soft Jaccard using word embeddings: intersection is similarity-weighted matches.

    - Encode words with the same embedding model.
    - Compute cosine similarity matrix.
    - Intersection = average(sum of best matches from A->B and B->A) with thresholding.
    - Union = |A| + |B| - Intersection.
    """
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    try:
        a_vecs = embedding_model.encode(words_a, convert_to_numpy=True)
        b_vecs = embedding_model.encode(words_b, convert_to_numpy=True)
    except Exception:
        return 0.0
    if a_vecs.size == 0 or b_vecs.size == 0:
        return 0.0
    sim = cosine_similarity(a_vecs, b_vecs)
    best_a = np.maximum(sim.max(axis=1), 0)
    best_b = np.maximum(sim.max(axis=0), 0)
    best_a = best_a[best_a >= sim_thresh]
    best_b = best_b[best_b >= sim_thresh]
    inter = (best_a.sum() + best_b.sum()) / 2.0
    union = len(words_a) + len(words_b) - inter
    return float(inter / union) if union > 0 else 0.0


def match_topics_topk(
    sim_matrix: np.ndarray, threshold: float, topk: int
) -> Tuple[List[Tuple[int, int, float, int]], List[int], List[int], np.ndarray, np.ndarray]:
    """Many-to-one matching: for each sim2 topic (row), keep top-k MADOC topics above threshold.

    Returns:
      - pairs: list of (row_idx, col_idx, sim, rank)
      - unmatched_rows: sim2 rows without any match
      - unmatched_cols: MADOC cols never used in any pair
      - topk_indices: (n_rows, topk) indices of top matches (-1 if none)
      - topk_scores: (n_rows, topk) similarity scores (0 if none)
    """
    n_rows, n_cols = sim_matrix.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols)), np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=float)
    pairs: List[Tuple[int, int, float, int]] = []
    topk_indices = -np.ones((n_rows, topk), dtype=int)
    topk_scores = np.zeros((n_rows, topk), dtype=float)
    used_cols = set()
    matched_rows = set()
    for i in range(n_rows):
        sims = sim_matrix[i]
        order = np.argsort(-sims)
        rank = 0
        for j in order:
            if sims[j] < threshold:
                break
            if rank < topk:
                pairs.append((i, int(j), float(sims[j]), rank))
                topk_indices[i, rank] = int(j)
                topk_scores[i, rank] = float(sims[j])
                used_cols.add(int(j))
                matched_rows.add(i)
                rank += 1
            else:
                break
    unmatched_rows = [i for i in range(n_rows) if i not in matched_rows]
    unmatched_cols = [j for j in range(n_cols) if j not in used_cols]
    return pairs, unmatched_rows, unmatched_cols, topk_indices, topk_scores


def match_topics(
    sim2_embs: np.ndarray, madoc_embs: np.ndarray, threshold: float
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int], np.ndarray]:
    if sim2_embs.size == 0 or madoc_embs.size == 0:
        return [], list(range(sim2_embs.shape[0])), list(range(madoc_embs.shape[0])), np.zeros((sim2_embs.shape[0], madoc_embs.shape[0]))

    sim_matrix = cosine_similarity(sim2_embs, madoc_embs)
    cost = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = []
    used_rows = set()
    used_cols = set()
    for r, c in zip(row_ind, col_ind):
        sim = float(sim_matrix[r, c])
        if sim >= threshold:
            pairs.append((int(r), int(c), sim))
            used_rows.add(int(r))
            used_cols.add(int(c))
    unmatched_rows = [i for i in range(sim2_embs.shape[0]) if i not in used_rows]
    unmatched_cols = [j for j in range(madoc_embs.shape[0]) if j not in used_cols]
    return pairs, unmatched_rows, unmatched_cols, sim_matrix


def plot_heatmap_topk(sim2_labels: List[str], topk_scores: np.ndarray, outpath: Path) -> None:
    import matplotlib.pyplot as plt

    if topk_scores.size == 0:
        return
    order = np.argsort(-topk_scores[:, 0])
    scores = topk_scores[order]
    labels = [sim2_labels[i] for i in order]
    labels = [l if len(l) <= 40 else l[:37] + "..." for l in labels]

    plt.figure(figsize=(6, max(4, len(labels) * 0.25)))
    im = plt.imshow(scores, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Cosine similarity")
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.xticks(range(scores.shape[1]), [f"top{j+1}" for j in range(scores.shape[1])], fontsize=8)
    plt.xlabel("Matched MADOC topics (ranks)")
    plt.ylabel("Simulation topics")
    plt.title("Similarity to top-K MADOC topics")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare topics between simulation2 Voat and MADOC Voat samples using BERTopic.")
    parser.add_argument("--sim2-posts-csv", type=Path, default=Path("simulation3/posts.csv"))
    parser.add_argument("--madoc-input", type=Path, default=Path("MADOC/voat-technology/voat_technology_madoc.parquet"))
    parser.add_argument("--madoc-merge-samples", action="store_true", help="Merge MADOC/voat-technology/sample_* files instead of using --madoc-input")
    parser.add_argument("--madoc-root", type=Path, default=Path("MADOC/voat-technology"), help="Root directory for MADOC voat samples when --madoc-merge-samples is used")
    parser.add_argument("--text-cols", type=str, default=None, help="Comma-separated list of text columns to use for both corpora (e.g., 'title,body'). Use per-source overrides below if needed.")
    parser.add_argument("--sim2-text-cols", type=str, default=None, help="Comma-separated list of text columns specifically for simulation2")
    parser.add_argument("--madoc-text-cols", type=str, default=None, help="Comma-separated list of text columns specifically for MADOC (default prefers 'content')")
    parser.add_argument("--min-docs", type=int, default=200)
    parser.add_argument("--min-topic-size", type=int, default=10)
    parser.add_argument("--top-n-words", type=int, default=10)
    parser.add_argument("--similarity-threshold", type=float, default=0.65)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Force device for embeddings: 'cuda' or 'cpu'")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="HuggingFace model id for sentence embeddings")
    parser.add_argument("--embedding-local-path", type=str, default=None, help="Path to a local SentenceTransformer model directory (offline)")
    parser.add_argument("--hf-offline", action="store_true", help="Set HF_HUB_OFFLINE=1 to avoid network calls when loading the model")
    parser.add_argument("--outdir", type=Path, default=Path("simulation3/topic_compare"))
    parser.add_argument("--save-heatmap", action="store_true")
    parser.add_argument("--vectorizer-min-df", type=int, default=1)
    parser.add_argument("--topk-per-sim2", type=int, default=3, help="Top-K MADOC matches to keep per simulation topic (many-to-one)")
    parser.add_argument("--word-sim-threshold", type=float, default=0.6, help="Threshold for word embedding similarity in soft Jaccard")
    parser.add_argument("--topic-repr", type=str, default="centroid", choices=["centroid", "bertopic", "hybrid"], help="Topic vector representation")
    parser.add_argument("--centroid-max-docs", type=int, default=200, help="Max docs per topic to use for centroid computation")
    parser.add_argument("--repr-alpha", type=float, default=0.5, help="Blend weight for hybrid representation (alpha*centroid + (1-alpha)*bertopic)")
    parser.add_argument("--composite-alpha", type=float, default=0.8, help="Weight for cosine vs soft-Jaccard when re-ranking matches")
    parser.add_argument("--remove-contraction-fragments", action="store_true", help="Remove common detached contraction fragments (don, ve, ll, re, isn, t) before vectorization")
    parser.add_argument("--df-threshold", type=float, default=0.4, help="If in (0,1): remove tokens appearing in >= fraction of docs across both corpora (document frequency)")
    parser.add_argument("--min-doc-chars", type=int, default=15, help="Minimum cleaned character length to keep a document")
    parser.add_argument("--drop-header-rows", action="store_true", help="Attempt to drop rows whose text is just a column header like 'title'")
    parser.add_argument("--extra-stopwords", type=str, default=None, help="Comma-separated extra stopwords to remove (added to english + column names)")
    parser.add_argument(
        "--min-token-freq",
        type=float,
        default=0.0,
        help=(
            "Dynamic stopwords: if 0<val<1 treat as fraction of docs; "
            "if >=1 treat as absolute doc count. Set 0 to disable."
        ),
    )

    args = parser.parse_args(argv)

    set_seeds(args.seed)
    ensure_outdir(args.outdir)

    text_cols_global = [c.strip() for c in args.text_cols.split(",")] if args.text_cols else None
    sim2_text_cols = [c.strip() for c in args.sim2_text_cols.split(",")] if args.sim2_text_cols else text_cols_global
    # For MADOC, default preference is ['content'] when not specified
    if args.madoc_text_cols:
        madoc_text_cols = [c.strip() for c in args.madoc_text_cols.split(",")]
    elif text_cols_global:
        madoc_text_cols = text_cols_global
    else:
        madoc_text_cols = ["content"]

    # Load corpora
    sim2_texts, sim2_df, sim2_used_cols = load_corpus(args.sim2_posts_csv, text_cols=sim2_text_cols, max_docs=args.max_docs, seed=args.seed)
    sim2_series = pd.Series(sim2_texts)
    sim2_series = filter_docs(sim2_series, min_chars=args.min_doc_chars, drop_headers=args.drop_header_rows)
    sim2_texts = sim2_series.tolist()
    if len(sim2_texts) < args.min_docs:
        print(f"Simulation2 corpus too small: {len(sim2_texts)} < min_docs={args.min_docs}", file=sys.stderr)
    if args.madoc_merge_samples:
        madoc_texts, madoc_df, madoc_used_cols = load_madoc_samples(args.madoc_root, max_docs=args.max_docs, seed=args.seed, text_cols=madoc_text_cols)
    else:
        madoc_texts, madoc_df, madoc_used_cols = load_corpus(args.madoc_input, text_cols=madoc_text_cols, max_docs=args.max_docs, seed=args.seed)
    madoc_series = pd.Series(madoc_texts)
    madoc_series = filter_docs(madoc_series, min_chars=args.min_doc_chars, drop_headers=args.drop_header_rows)
    madoc_texts = madoc_series.tolist()
    if len(madoc_texts) < args.min_docs:
        print(f"MADOC corpus too small: {len(madoc_texts)} < min_docs={args.min_docs}", file=sys.stderr)

    # Build embedding model (GPU if available)
    embedding_model = make_embedding_model(
        device_preference=args.device,
        model_name=args.embedding_model,
        local_path=args.embedding_local_path,
        hf_offline=args.hf_offline,
    )

    # Train BERTopic models
    # Dynamic stopword extension
    extra_stop = set()
    if args.extra_stopwords:
        extra_stop.update([w.strip().lower() for w in args.extra_stopwords.split(",") if w.strip()])
    # Add column names (often leak like 'title')
    for col in (sim2_used_cols + madoc_used_cols):
        extra_stop.add(col.lower())
    if args.remove_contraction_fragments:
        extra_stop.update({"don", "ve", "ll", "re", "isn", "ain", "cant", "won", "shouldn", "wouldn", "couldn"})
    # Optionally find ultra-frequent tokens (appearing in too many docs)
    # Dynamic frequency-based stopwords
    from collections import Counter
    token_counts = Counter()
    total_docs = len(sim2_texts) + len(madoc_texts)
    def doc_tokens(doc: str) -> set:
        return set(re.findall(r"[A-Za-z][A-Za-z0-9_'-]+", doc.lower()))
    for d in sim2_texts + madoc_texts:
        for t in doc_tokens(d):
            token_counts[t] += 1
    # Dynamic stopwords by doc frequency threshold
    mtf = float(args.min_token_freq or 0.0)
    if 0 < mtf < 1:
        df_cut = mtf * total_docs
        for tok, cnt in token_counts.items():
            if cnt >= df_cut:
                extra_stop.add(tok)
    elif mtf >= 1:
        for tok, cnt in token_counts.items():
            if cnt >= mtf:
                extra_stop.add(tok)
    # Fractional DF threshold
    if 0 < args.df_threshold < 1:
        df_cut = args.df_threshold * total_docs
        for tok, cnt in token_counts.items():
            if cnt >= df_cut:
                extra_stop.add(tok)

    topic_model_sim2 = build_bertopic(embedding_model, min_topic_size=args.min_topic_size, seed=args.seed, vectorizer_min_df=args.vectorizer_min_df)
    if extra_stop and hasattr(topic_model_sim2, 'vectorizer_model'):
        base_stop = set(topic_model_sim2.vectorizer_model.get_stop_words() or [])
        topic_model_sim2.vectorizer_model.stop_words = list(base_stop.union(extra_stop))
    # Fit with fallback if vocabulary collapses
    try:
        _topics_sim2, _ = topic_model_sim2.fit_transform(sim2_texts)
    except ValueError as e:
        if "empty vocabulary" in str(e):
            # Relax vectorizer and retry once
            print("Warning: empty vocabulary for sim2 topics; relaxing vectorizer (min_df=1, default token pattern)", file=sys.stderr)
            topic_model_sim2.vectorizer_model = CountVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                token_pattern=r"(?u)\\b\\w\\w+\\b",
            )
            if extra_stop:
                base_stop = set(topic_model_sim2.vectorizer_model.get_stop_words() or [])
                topic_model_sim2.vectorizer_model.stop_words = list(base_stop.union(extra_stop))
            _topics_sim2, _ = topic_model_sim2.fit_transform(sim2_texts)
        else:
            raise
    topic_model_madoc = build_bertopic(embedding_model, min_topic_size=args.min_topic_size, seed=args.seed, vectorizer_min_df=args.vectorizer_min_df)
    if extra_stop and hasattr(topic_model_madoc, 'vectorizer_model'):
        base_stop = set(topic_model_madoc.vectorizer_model.get_stop_words() or [])
        topic_model_madoc.vectorizer_model.stop_words = list(base_stop.union(extra_stop))
    try:
        _topics_madoc, _ = topic_model_madoc.fit_transform(madoc_texts)
    except ValueError as e:
        if "empty vocabulary" in str(e):
            print("Warning: empty vocabulary for MADOC topics; relaxing vectorizer (min_df=1, default token pattern)", file=sys.stderr)
            topic_model_madoc.vectorizer_model = CountVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                token_pattern=r"(?u)\\b\\w\\w+\\b",
            )
            if extra_stop:
                base_stop = set(topic_model_madoc.vectorizer_model.get_stop_words() or [])
                topic_model_madoc.vectorizer_model.stop_words = list(base_stop.union(extra_stop))
            _topics_madoc, _ = topic_model_madoc.fit_transform(madoc_texts)
        else:
            raise

    # Extract topic infos
    sim2_infos = extract_topic_info(topic_model_sim2, sim2_texts, top_n_words=args.top_n_words)
    madoc_infos = extract_topic_info(topic_model_madoc, madoc_texts, top_n_words=args.top_n_words)

    # Compute topic vectors per requested representation
    topics_sim2, _ = topic_model_sim2.transform(sim2_texts)
    topics_madoc, _ = topic_model_madoc.transform(madoc_texts)
    if args.topic_repr == "centroid":
        sim2_embs = compute_topic_centroids(sim2_texts, topics_sim2, sim2_infos, embedding_model, max_docs_per_topic=args.centroid_max_docs)
        madoc_embs = compute_topic_centroids(madoc_texts, topics_madoc, madoc_infos, embedding_model, max_docs_per_topic=args.centroid_max_docs)
    elif args.topic_repr == "bertopic":
        sim2_embs = compute_topic_embeddings(topic_model_sim2, sim2_infos, embedding_model)
        madoc_embs = compute_topic_embeddings(topic_model_madoc, madoc_infos, embedding_model)
    else:  # hybrid
        sim2_c = compute_topic_centroids(sim2_texts, topics_sim2, sim2_infos, embedding_model, max_docs_per_topic=args.centroid_max_docs)
        madoc_c = compute_topic_centroids(madoc_texts, topics_madoc, madoc_infos, embedding_model, max_docs_per_topic=args.centroid_max_docs)
        sim2_b = compute_topic_embeddings(topic_model_sim2, sim2_infos, embedding_model)
        madoc_b = compute_topic_embeddings(topic_model_madoc, madoc_infos, embedding_model)
        a = float(args.repr_alpha)
        sim2_embs = a * sim2_c + (1 - a) * sim2_b
        madoc_embs = a * madoc_c + (1 - a) * madoc_b

    # Similarity matrix and many-to-one matching (top-K per sim2 topic)
    sim_matrix = cosine_similarity(sim2_embs, madoc_embs) if (sim2_embs.size and madoc_embs.size) else np.zeros((len(sim2_embs), len(madoc_embs)))
    pairs, unmatched_sim2_idx, unmatched_madoc_idx, topk_indices, topk_scores = match_topics_topk(
        sim_matrix, threshold=args.similarity_threshold, topk=args.topk_per_sim2
    )

    # Rerank each row's top-K by composite score: alpha*cosine + (1-alpha)*soft_jaccard(top words)
    composite_alpha = float(args.composite_alpha)
    reranked_pairs = []
    heatmap_scores = np.zeros_like(topk_scores)
    for i in range(len(sim2_infos)):
        row_pairs = [(r, c, s, rk) for (r, c, s, rk) in pairs if r == i]
        if not row_pairs:
            continue
        scored = []
        for (r, c, s, rk) in row_pairs:
            sj = soft_jaccard(sim2_infos[r].top_words, madoc_infos[c].top_words, embedding_model, sim_thresh=args.word_sim_threshold)
            comp = composite_alpha * s + (1 - composite_alpha) * sj
            scored.append((r, c, s, rk, sj, comp))
        scored.sort(key=lambda x: -x[5])
        for new_rank, (r, c, s, rk, sj, comp) in enumerate(scored[: args.topk_per_sim2]):
            reranked_pairs.append((r, c, s, new_rank))
            heatmap_scores[r, new_rank] = s
    pairs = reranked_pairs
    topk_scores = heatmap_scores

    # Build label arrays aligned to embeddings order
    sim2_labels = [ti.label for ti in sim2_infos]
    madoc_labels = [ti.label for ti in madoc_infos]

    # Save topic info CSVs
    sim2_info_df = pd.DataFrame(
        {
            "topic_id": [t.topic_id for t in sim2_infos],
            "size": [t.size for t in sim2_infos],
            "label": sim2_labels,
            "top_words": [" ".join(t.top_words) for t in sim2_infos],
        }
    )
    madoc_info_df = pd.DataFrame(
        {
            "topic_id": [t.topic_id for t in madoc_infos],
            "size": [t.size for t in madoc_infos],
            "label": madoc_labels,
            "top_words": [" ".join(t.top_words) for t in madoc_infos],
        }
    )
    write_csv(sim2_info_df, args.outdir / "topic_info_sim2.csv")
    write_csv(madoc_info_df, args.outdir / "topic_info_madoc.csv")

    # Matches table (may include multiple MADOC matches per sim2 topic)
    matches_rows = []
    for r_idx, c_idx, sim, rank in pairs:
        s_info = sim2_infos[r_idx]
        m_info = madoc_infos[c_idx]
        jac = jaccard(s_info.top_words, m_info.top_words)
        sjac = soft_jaccard(s_info.top_words, m_info.top_words, embedding_model, sim_thresh=args.word_sim_threshold)
        matches_rows.append(
            {
                "sim2_topic_id": s_info.topic_id,
                "sim2_label": s_info.label,
                "sim2_size": s_info.size,
                "madoc_topic_id": m_info.topic_id,
                "madoc_label": m_info.label,
                "madoc_size": m_info.size,
                "cosine_sim": sim,
                "jaccard_topn": jac,
                "soft_jaccard_topn": sjac,
                "rank": rank + 1,
            }
        )
    matches_df = pd.DataFrame(matches_rows)
    write_csv(matches_df, args.outdir / "topic_matches.csv")

    # Full similarity matrix
    sim_df = pd.DataFrame(sim_matrix, columns=[f"madoc_{i}" for i in range(sim_matrix.shape[1])])
    sim_df.insert(0, "sim2_idx", range(sim_matrix.shape[0]))
    write_csv(sim_df, args.outdir / "topic_similarity.csv")

    # Summary
    matched_sim2_set = {p[0] for p in pairs}
    coverage = float(len(matched_sim2_set) / max(1, len(sim2_infos)))
    best_matches = [row for row in matches_rows if row.get("rank", 1) == 1]
    mean_cos = float(np.mean([m["cosine_sim"] for m in best_matches])) if best_matches else 0.0
    med_cos = float(np.median([m["cosine_sim"] for m in best_matches])) if best_matches else 0.0
    mean_jac = float(np.mean([m["jaccard_topn"] for m in best_matches])) if best_matches else 0.0
    med_jac = float(np.median([m["jaccard_topn"] for m in best_matches])) if best_matches else 0.0
    mean_sjac = float(np.mean([m["soft_jaccard_topn"] for m in best_matches])) if best_matches else 0.0
    med_sjac = float(np.median([m["soft_jaccard_topn"] for m in best_matches])) if best_matches else 0.0
    summary = {
        "sim2_docs": len(sim2_texts),
        "madoc_docs": len(madoc_texts),
        "sim2_topics": len(sim2_infos),
        "madoc_topics": len(madoc_infos),
        "matches": len(matches_rows),
        "coverage_sim2": coverage,
        "similarity_threshold": args.similarity_threshold,
        "mean_cosine": mean_cos,
        "median_cosine": med_cos,
        "mean_jaccard": mean_jac,
        "median_jaccard": med_jac,
        "mean_soft_jaccard": mean_sjac,
        "median_soft_jaccard": med_sjac,
        "unmatched_sim2_count": len(unmatched_sim2_idx),
        "unmatched_madoc_count": len(unmatched_madoc_idx),
        "seed": args.seed,
    }
    with open(args.outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Heatmap
    if args.save_heatmap and sim_matrix.size > 0:
        heatmap_path = args.outdir / "similarity_heatmap.png"
        plot_heatmap_topk(sim2_labels, topk_scores, heatmap_path)

    # Run info
    # Ensure args are JSON-serializable (e.g., convert Paths)
    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run_info = {
        "args": args_dict,
        "sim2_used_text_cols": sim2_used_cols,
        "madoc_used_text_cols": madoc_used_cols,
        "timestamp": int(time.time()),
    }
    with open(args.outdir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2, default=str)

    print(f"Wrote outputs to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
