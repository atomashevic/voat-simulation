#!/usr/bin/env python3
"""
Semantic diversity analysis for simulation data.

The script ingests posts/comments plus simulation demographics, redacts identifiers,
builds sentence embeddings, and reports mean pairwise cosine distances per
demographic grouping (and optionally per topic).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - import guard for optional dependency
    SentenceTransformer = None


URL_RE = re.compile(r"https?://\S+|www\.\S+")
USERNAME_RE = re.compile(r"(@|u/)[A-Za-z0-9_\-]+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
QUOTE_RE = re.compile(r"^>.*$", re.MULTILINE)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class ClusterMetrics:
    mean_pairwise_distance: float
    sampled_pairs: int
    status: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute semantic diversity grouped by simulation demographics."
    )
    parser.add_argument(
        "--simulation-dir",
        type=Path,
        default=Path("simulation"),
        help="Directory containing posts.csv and users.csv (default: simulation).",
    )
    parser.add_argument(
        "--posts-file",
        type=str,
        default="posts.csv",
        help="Posts CSV file name relative to --simulation-dir (default: posts.csv).",
    )
    parser.add_argument(
        "--users-file",
        type=str,
        default="users.csv",
        help="Users CSV file name relative to --simulation-dir (default: users.csv).",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="tweet",
        help="Column containing free text to embed (default: tweet).",
    )
    parser.add_argument(
        "--posts-user-id-column",
        type=str,
        default="user_id",
        help="User id column in the posts file (default: user_id).",
    )
    parser.add_argument(
        "--users-id-column",
        type=str,
        default="id",
        help="User id column in the users file (default: id).",
    )
    parser.add_argument(
        "--demographic-fields",
        nargs="+",
        default=["leaning", "education_level", "gender"],
        help="Demographic fields to include (default: leaning education_level gender).",
    )
    parser.add_argument(
        "--group-spec",
        nargs="+",
        action="append",
        help=(
            "Field(s) defining a grouping. Provide multiple times for multiple "
            "views. Defaults to each demographic individually and the full combination."
        ),
    )
    parser.add_argument(
        "--topic-column",
        type=str,
        default=None,
        help="Optional column (e.g., thread_id) to append to every grouping.",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=50,
        help="Minimum character length after preprocessing (default: 50).",
    )
    parser.add_argument(
        "--min-word-count",
        type=int,
        default=8,
        help="Minimum token count after preprocessing (default: 8).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=100,
        help="Minimum number of responses required per cluster (default: 100).",
    )
    parser.add_argument(
        "--max-responses-per-user",
        type=int,
        default=25,
        help="Cap per-user contributions to reduce dominance (default: 25).",
    )
    parser.add_argument(
        "--max-responses-per-cluster",
        type=int,
        default=2000,
        help="Optional cap on responses sampled per cluster before measuring distances.",
    )
    parser.add_argument(
        "--pairwise-sample-size",
        type=int,
        default=10000,
        help="Random pair count for distance estimation when clusters are large.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: sentence-transformers/all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device override passed to SentenceTransformer.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional hard limit on number of responses processed (debug only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation/semantic_diversity"),
        help="Directory to store csv/json outputs (default: simulation/semantic_diversity).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for reproducibility (default: 17).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating summary plots (default: generate boxplot + point cloud).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def load_csv(path: Path, **read_kwargs) -> pd.DataFrame:
    logging.info("Loading %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, **read_kwargs)


def compile_label_patterns(values: Iterable[str]) -> List[re.Pattern]:
    patterns = []
    for value in values:
        if not value or not isinstance(value, str):
            continue
        escaped = re.escape(value.strip())
        if not escaped:
            continue
        pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
        patterns.append(pattern)
    return patterns


def redact_text(text: str, demographic_terms: Sequence[re.Pattern]) -> str:
    redacted = text
    for pattern in demographic_terms:
        redacted = pattern.sub(" ", redacted)
    return redacted


def preprocess_text(text: str, demographic_patterns: Sequence[re.Pattern]) -> str:
    if not isinstance(text, str):
        return ""
    working = text
    working = URL_RE.sub(" ", working)
    working = USERNAME_RE.sub(" ", working)
    working = EMAIL_RE.sub(" ", working)
    working = QUOTE_RE.sub(" ", working)
    working = CONTROL_CHAR_RE.sub(" ", working)
    working = redact_text(working, demographic_patterns)
    working = working.replace("TITLE:", " ")
    working = working.replace("QUOTE:", " ")
    working = WHITESPACE_RE.sub(" ", working)
    return working.strip()


def enforce_user_cap(df: pd.DataFrame, user_col: str, max_responses: Optional[int]) -> pd.DataFrame:
    if not max_responses:
        return df
    sort_cols = [c for c in ("round", "id", "created_at") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    capped = (
        df.groupby(user_col, group_keys=False)
        .head(max_responses)
        .reset_index(drop=True)
    )
    logging.info(
        "Capped responses per user at %s (from %s to %s rows).",
        max_responses,
        len(df),
        len(capped),
    )
    return capped


def embed_texts(
    texts: Sequence[str],
    model_name: str,
    batch_size: int,
    device: Optional[str],
) -> np.ndarray:
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is required. Install via `pip install sentence-transformers`."
        )
    logging.info("Loading embedding model %s", model_name)
    model = SentenceTransformer(model_name, device=device)
    logging.info("Encoding %s texts.", len(texts))
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def mean_pairwise_cosine(
    vectors: np.ndarray,
    rng: np.random.Generator,
    sample_size: Optional[int],
) -> Tuple[float, int]:
    n = len(vectors)
    if n < 2:
        return math.nan, 0
    total_pairs = n * (n - 1) // 2
    if not sample_size or sample_size >= total_pairs:
        sims = vectors @ vectors.T
        tri_upper = np.triu_indices(n, k=1)
        distances = 1.0 - sims[tri_upper]
        return float(distances.mean()), len(distances)
    distances = []
    attempts = 0
    while len(distances) < sample_size and attempts < sample_size * 5:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            attempts += 1
            continue
        cosine_sim = float(np.dot(vectors[i], vectors[j]))
        distances.append(1.0 - cosine_sim)
        attempts += 1
    if not distances:
        return math.nan, 0
    return float(np.mean(distances)), len(distances)


def compute_cluster_metrics(
    cluster_df: pd.DataFrame,
    embeddings: np.ndarray,
    indexer: np.ndarray,
    rng: np.random.Generator,
    min_cluster_size: int,
    max_responses_per_cluster: Optional[int],
    pairwise_sample_size: Optional[int],
) -> ClusterMetrics:
    n = len(cluster_df)
    if n < min_cluster_size:
        return ClusterMetrics(
            mean_pairwise_distance=math.nan,
            sampled_pairs=0,
            status="too_small",
            note=f"size<{min_cluster_size}",
        )
    if max_responses_per_cluster and n > max_responses_per_cluster:
        sample_idx = rng.choice(
            n,
            size=max_responses_per_cluster,
            replace=False,
        )
        selected = indexer[cluster_df.index.values[sample_idx]]
        note = f"downsampled_to_{max_responses_per_cluster}"
    else:
        selected = indexer[cluster_df.index.values]
        note = ""

    mean_distance, sampled_pairs = mean_pairwise_cosine(
        embeddings[selected], rng, pairwise_sample_size
    )
    status = "ok" if not math.isnan(mean_distance) else "insufficient_pairs"
    return ClusterMetrics(
        mean_pairwise_distance=mean_distance,
        sampled_pairs=sampled_pairs,
        status=status,
        note=note,
    )


def render_summary_plots(summary_df: pd.DataFrame, output_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        logging.warning("Skipping plots because %s", exc)
        return []

    # Limit to the single-field demographics the team cares about for visualization.
    focus_fields = ["gender", "education_level", "leaning"]
    ready_df = summary_df[
        (summary_df["status"] == "ok") & (summary_df["group_fields"].isin(focus_fields))
    ].copy()
    if ready_df.empty:
        logging.info("No successful single-field clusters to plot; skipping.")
        return []

    def _pretty_field(field: str) -> str:
        return field.replace("_", " ").title()

    ready_df["field_label"] = ready_df["group_fields"].map(_pretty_field)

    def _format_value(raw: str) -> str:
        txt = str(raw or "").strip().replace("_", " ")
        if not txt:
            return "Unknown"
        normalized = " ".join(txt.split())
        return normalized.title() if normalized == normalized.lower() else normalized

    ready_df["display_label"] = ready_df["group_values"].map(_format_value)

    palette = {
        "gender": "#1f77b4",
        "education_level": "#f3722c",
        "leaning": "#43aa8b",
    }
    markers = {
        "gender": "o",
        "education_level": "s",
        "leaning": "^",
    }

    # Build plotting order grouped by field while keeping values sorted for readability.
    plot_rows: List[Dict[str, object]] = []
    y_position = 0
    gap = 0.6
    for field in focus_fields:
        field_df = ready_df[ready_df["group_fields"] == field].copy()
        if field_df.empty:
            continue
        field_df = field_df.sort_values("display_label")
        for _, row in field_df.iterrows():
            plot_rows.append(
                {
                    "y": y_position,
                    "value": row["mean_pairwise_cosine_distance"],
                    "label": row["display_label"],
                    "field": field,
                }
            )
            y_position += 1
        y_position += gap  # spacer before next block

    if not plot_rows:
        logging.info("No rows available for combined plot; skipping.")
        return []

    max_value = max(row["value"] for row in plot_rows)
    fig_height = max(6.0, 0.4 * len(plot_rows) + 2.0)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    used_fields = set()
    for row in plot_rows:
        field = row["field"]
        y = row["y"]
        val = row["value"]
        ax.hlines(
            y,
            xmin=0,
            xmax=val,
            color=palette.get(field, "#888888"),
            alpha=0.35,
            linewidth=3,
        )
        legend_label = None
        if field not in used_fields:
            legend_label = _pretty_field(field)
            used_fields.add(field)
        ax.scatter(
            val,
            y,
            color=palette.get(field, "#888888"),
            marker=markers.get(field, "o"),
            edgecolor="black",
            linewidths=0.4,
            s=80,
            label=legend_label,
        )

    ax.set_yticks([row["y"] for row in plot_rows])
    ax.set_yticklabels([row["label"] for row in plot_rows])
    ax.set_xlabel("Mean pairwise cosine distance")
    ax.set_title("Semantic diversity overview")
    ax.set_xlim(0, max_value * 1.05)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()  # top-to-bottom reading order
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Demographic", loc="upper left")
    fig.tight_layout()

    out_path = output_dir / "semantic_diversity_overview.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    logging.info("Wrote plot %s", out_path)
    return [str(out_path)]


def prepare_group_specs(args: argparse.Namespace) -> List[List[str]]:
    if args.group_spec:
        return [spec for spec in args.group_spec if spec]
    specs = [[field] for field in args.demographic_fields]
    if len(args.demographic_fields) > 1:
        specs.append(list(args.demographic_fields))
    return specs


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    rng = np.random.default_rng(args.seed)

    posts_path = args.simulation_dir / args.posts_file
    users_path = args.simulation_dir / args.users_file

    posts_df = load_csv(posts_path)
    users_df = load_csv(users_path)

    required_user_cols = set(args.demographic_fields + [args.users_id_column])
    missing_user_cols = required_user_cols - set(users_df.columns)
    if missing_user_cols:
        raise KeyError(f"Missing columns in users file: {missing_user_cols}")

    merged_df = posts_df.merge(
        users_df[args.demographic_fields + [args.users_id_column]],
        left_on=args.posts_user_id_column,
        right_on=args.users_id_column,
        how="inner",
        suffixes=("_post", ""),
    )
    if args.limit:
        merged_df = merged_df.head(args.limit).copy()
    logging.info("Merged dataset shape: %s", merged_df.shape)

    # Drop rows lacking text or demographics
    subset_cols = [args.text_column, args.posts_user_id_column] + args.demographic_fields
    if args.topic_column:
        subset_cols.append(args.topic_column)
    merged_df = merged_df.dropna(subset=subset_cols)

    # Build demographic-specific redaction patterns
    demographic_patterns: Dict[int, List[re.Pattern]] = {}
    pattern_cache: Dict[Tuple[str, ...], List[re.Pattern]] = {}
    for idx, row in merged_df.iterrows():
        label_values = tuple(
            str(row[field]) for field in args.demographic_fields if pd.notna(row[field])
        )
        if label_values not in pattern_cache:
            pattern_cache[label_values] = compile_label_patterns(label_values)
        demographic_patterns[idx] = pattern_cache[label_values]

    cleaned_texts = []
    word_counts = []
    char_counts = []
    for idx, text in enumerate(merged_df[args.text_column].tolist()):
        patterns = demographic_patterns.get(merged_df.index[idx], [])
        cleaned = preprocess_text(text, patterns)
        cleaned_texts.append(cleaned)
        words = cleaned.split()
        word_counts.append(len(words))
        char_counts.append(len(cleaned))

    merged_df = merged_df.assign(
        cleaned_text=cleaned_texts,
        word_count=word_counts,
        char_count=char_counts,
    )
    merged_df = merged_df[
        (merged_df["char_count"] >= args.min_text_length)
        & (merged_df["word_count"] >= args.min_word_count)
    ].copy()

    if merged_df.empty:
        raise ValueError("No responses remain after preprocessing filters.")

    merged_df = enforce_user_cap(
        merged_df,
        args.posts_user_id_column,
        args.max_responses_per_user,
    )

    embeddings = embed_texts(
        merged_df["cleaned_text"].tolist(),
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )

    merged_df = merged_df.reset_index(drop=True)
    embedding_indexer = np.arange(len(merged_df))

    group_specs = prepare_group_specs(args)
    output_records = []

    for fields in group_specs:
        group_columns = list(fields)
        if args.topic_column:
            group_columns.append(args.topic_column)
        missing_cols = [col for col in group_columns if col not in merged_df.columns]
        if missing_cols:
            logging.warning("Skipping grouping %s; missing columns %s", fields, missing_cols)
            continue

        logging.info("Evaluating grouping on columns: %s", group_columns)
        for group_values, cluster_df in merged_df.groupby(group_columns, dropna=True):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            metadata = dict(zip(group_columns, group_values))
            metrics = compute_cluster_metrics(
                cluster_df=cluster_df,
                embeddings=embeddings,
                indexer=embedding_indexer,
                rng=rng,
                min_cluster_size=args.min_cluster_size,
                max_responses_per_cluster=args.max_responses_per_cluster,
                pairwise_sample_size=args.pairwise_sample_size,
            )
            record = {
                "group_fields": "|".join(fields),
                "group_values": "|".join(str(metadata[col]) for col in fields),
                "topic_column": args.topic_column or "",
                "topic_value": metadata.get(args.topic_column, "") if args.topic_column else "",
                "n_responses": len(cluster_df),
                "n_users": cluster_df[args.posts_user_id_column].nunique(),
                "mean_word_count": cluster_df["word_count"].mean(),
                "mean_char_count": cluster_df["char_count"].mean(),
                "mean_pairwise_cosine_distance": metrics.mean_pairwise_distance,
                "distance_sampled_pairs": metrics.sampled_pairs,
                "status": metrics.status,
                "note": metrics.note,
            }
            output_records.append(record)

    if not output_records:
        raise RuntimeError("No cluster metrics were computed. Check group specifications.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(output_records)
    summary_path = output_dir / "semantic_diversity_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info("Wrote summary metrics to %s", summary_path)

    plot_paths = []
    if not args.no_plot:
        plot_paths = render_summary_plots(summary_df, output_dir)

    metadata = {
        "simulation_dir": str(args.simulation_dir),
        "posts_file": args.posts_file,
        "users_file": args.users_file,
        "text_column": args.text_column,
        "demographic_fields": args.demographic_fields,
        "group_specs": group_specs,
        "topic_column": args.topic_column,
        "min_text_length": args.min_text_length,
        "min_word_count": args.min_word_count,
        "min_cluster_size": args.min_cluster_size,
        "max_responses_per_user": args.max_responses_per_user,
        "max_responses_per_cluster": args.max_responses_per_cluster,
        "pairwise_sample_size": args.pairwise_sample_size,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "n_rows_after_filters": len(merged_df),
        "plot_paths": plot_paths,
    }
    metadata_path = output_dir / "semantic_diversity_run.json"
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
    logging.info("Wrote run metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
