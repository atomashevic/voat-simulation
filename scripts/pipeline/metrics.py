from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .io import MadocSampleData, SimulationRun
from .validation import require_columns

logger = logging.getLogger(__name__)


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return mean(vals) if vals else None


@dataclass
class MadocSampleMetrics:
    name: str
    source: str
    activity: Dict[str, Optional[float]] = field(default_factory=dict)
    toxicity: Dict[str, Optional[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationMetrics:
    name: str
    source: str
    activity: Dict[str, Optional[float]] = field(default_factory=dict)
    toxicity: Dict[str, Optional[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_madoc_sample(sample: MadocSampleData) -> MadocSampleMetrics:
    fr = sample.full_results or {}
    tox = sample.toxicity or {}
    tl = sample.thread_length or {}

    activity = {
        "posts_total": _nested_get(tox, ["interaction_toxicity", "posts", "count"]),
        "comments_total": _nested_get(tox, ["interaction_toxicity", "comments", "count"]),
        "unique_users": _nested_get(fr, ["active_users", "unique_users"]),
        "avg_thread_length": tl.get("avg_thread_length"),
        "mean_posts_per_user": _nested_get(fr, ["active_users", "avg_interactions_per_user"]),
        "mean_daily_active_users": _mean_daily_value(fr),
    }

    toxicity = {
        "overall_mean": _nested_get(tox, ["toxicity_stats", "mean"]),
        "posts_mean": _nested_get(tox, ["interaction_toxicity", "posts", "mean"]),
        "comments_mean": _nested_get(tox, ["interaction_toxicity", "comments", "mean"]),
        "coverage_pct": tox.get("toxicity_coverage_pct"),
    }

    metadata = {
        "sample_id": _nested_get(fr, ["sample_info", "sample_id"]),
        "start_date": _nested_get(fr, ["sample_info", "start_date"]),
        "end_date": _nested_get(fr, ["sample_info", "end_date"]),
        "record_count": _nested_get(fr, ["sample_info", "record_count"]),
        "path": str(sample.path),
    }

    return MadocSampleMetrics(
        name=sample.name,
        source=str(sample.path),
        activity=activity,
        toxicity=toxicity,
        metadata=metadata,
    )


def summarize_simulation_run(run: SimulationRun) -> SimulationMetrics:
    posts = run.posts
    activity: Dict[str, Optional[float]] = {}
    toxicity: Dict[str, Optional[float]] = {}

    if posts is not None and not posts.empty:
        comments_mask = _detect_comment_mask(posts)
        total_rows = len(posts)
        comments_total = int(comments_mask.sum())
        posts_total = int(total_rows - comments_total)
        activity["posts_total"] = posts_total
        activity["comments_total"] = comments_total
        activity["all_interactions"] = total_rows
        if "user_id" in posts.columns:
            activity["unique_users"] = int(posts["user_id"].nunique())
            posts_per_user = posts.groupby("user_id").size()
            activity["mean_posts_per_user"] = float(posts_per_user.mean())
            activity["median_posts_per_user"] = float(posts_per_user.median())
        if "thread_id" in posts.columns:
            activity["avg_thread_length"] = float(posts.groupby("thread_id").size().mean())
        if "round" in posts.columns and "user_id" in posts.columns:
            # Convert rounds to days: 1 round = 1 hour, 24 rounds = 1 day
            posts_with_day = posts.copy()
            posts_with_day["day"] = posts_with_day["round"] // 24
            daily_active = posts_with_day.groupby("day")["user_id"].nunique()
            activity["mean_daily_active_users"] = float(daily_active.mean())
        if "round" in posts.columns and "user_id" in posts.columns:
            span_per_user = posts.groupby("user_id")["round"].agg(lambda s: s.max() - s.min() + 1)
            activity["mean_activity_span_rounds"] = float(span_per_user.mean())
    else:
        logger.warning("Simulation run '%s' missing posts dataframe.", run.name)

    toxigen = run.toxigen
    if toxigen is not None and not toxigen.empty:
        require_columns(toxigen, ["toxicity"], context=f"{run.name} toxigen.csv")
        toxicity["overall_mean"] = float(toxigen["toxicity"].mean())
        normalized_types = _normalize_post_types(toxigen)
        type_means = toxigen.groupby(normalized_types)["toxicity"].mean().to_dict()
        toxicity["posts_mean"] = float(type_means.get("posts")) if "posts" in type_means else None
        toxicity["comments_mean"] = float(type_means.get("comments")) if "comments" in type_means else None
        toxicity["by_post_type"] = {
            str(k): float(v) for k, v in toxigen.groupby(_raw_type_label(toxigen)).toxicity.mean().to_dict().items()
        }
        type_counts = toxigen.groupby(normalized_types).size().to_dict()
        toxicity["type_counts"] = {str(k): int(v) for k, v in type_counts.items()}
    else:
        logger.warning("Simulation run '%s' missing toxigen.csv", run.name)

    metadata = {
        "path": str(run.path),
        "has_users_csv": run.users is not None,
        "has_news_csv": run.news is not None,
        "has_toxigen": toxigen is not None,
    }

    return SimulationMetrics(
        name=run.name,
        source=str(run.path),
        activity=activity,
        toxicity=toxicity,
        metadata=metadata,
    )


def _nested_get(obj: Dict[str, Any], path: List[str]) -> Optional[Any]:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _mean_daily_value(full_results: Dict[str, Any]) -> Optional[float]:
    dynamics = _nested_get(full_results, ["user_dynamics", "daily_dynamics"])
    if not isinstance(dynamics, list):
        return None
    values = [entry.get("active_users") for entry in dynamics if entry.get("active_users") is not None]
    return _safe_mean(values)


def _detect_comment_mask(df: pd.DataFrame) -> pd.Series:
    if "comment_to" in df.columns:
        mask = df["comment_to"].fillna(-1).astype("int64") != -1
        return mask
    if "is_comment" in df.columns:
        return df["is_comment"].astype(bool)
    return pd.Series(False, index=df.index)


def _normalize_post_types(df: pd.DataFrame) -> pd.Series:
    if "post_type" in df.columns:
        normalized = (
            df["post_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("posts", "post", regex=False)
            .str.replace("comments", "comment", regex=False)
        )
        return normalized.apply(lambda val: "comments" if "comment" in val else "posts")
    if "is_comment" in df.columns:
        return df["is_comment"].astype(bool).map({True: "comments", False: "posts"})
    return pd.Series("posts", index=df.index)


def _raw_type_label(df: pd.DataFrame) -> pd.Series:
    if "post_type" in df.columns:
        return df["post_type"].astype(str)
    if "is_comment" in df.columns:
        return df["is_comment"].astype(bool).map({True: "comment", False: "post"})
    return pd.Series("post", index=df.index)
