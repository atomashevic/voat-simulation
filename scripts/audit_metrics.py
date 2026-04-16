from __future__ import annotations

from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx
import numpy as np
import pandas as pd


def bootstrap_summary(
    values: Iterable[float],
    confidence: float = 0.99,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    vals = [float(v) for v in values if _is_finite(v)]
    if not vals:
        return {
            "mean": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "std": np.nan,
            "n_samples": 0,
            "ci_method": "bootstrap",
        }

    arr = np.asarray(vals, dtype=float)
    if arr.size == 1:
        value = float(arr[0])
        return {
            "mean": value,
            "ci_lower": value,
            "ci_upper": value,
            "std": 0.0,
            "n_samples": 1,
            "ci_method": "bootstrap",
        }

    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(n_bootstrap, arr.size), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "mean": float(arr.mean()),
        "ci_lower": float(np.percentile(boot, 100.0 * alpha / 2.0)),
        "ci_upper": float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0))),
        "std": float(np.std(arr, ddof=1)),
        "n_samples": int(arr.size),
        "ci_method": "bootstrap",
    }


def t_summary(
    values: Iterable[float],
    confidence: float = 0.99,
) -> Dict[str, Any]:
    vals = [float(v) for v in values if _is_finite(v)]
    if not vals:
        return {
            "mean": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "std": np.nan,
            "n_samples": 0,
            "ci_method": "t",
        }

    arr = np.asarray(vals, dtype=float)
    if arr.size == 1:
        value = float(arr[0])
        return {
            "mean": value,
            "ci_lower": value,
            "ci_upper": value,
            "std": 0.0,
            "n_samples": 1,
            "ci_method": "t",
        }

    mean = float(arr.mean())
    std = float(np.std(arr, ddof=1))
    sem = std / np.sqrt(arr.size)
    t_critical = _student_t_critical(confidence, arr.size - 1)
    margin = t_critical * sem
    return {
        "mean": mean,
        "ci_lower": mean - margin,
        "ci_upper": mean + margin,
        "std": std,
        "n_samples": int(arr.size),
        "ci_method": "t",
    }


def summarize_values(
    values: Iterable[float],
    confidence: float = 0.99,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    if ci_method == "bootstrap":
        return bootstrap_summary(
            values,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    if ci_method == "t":
        return t_summary(values, confidence=confidence)
    raise ValueError(f"Unsupported ci_method: {ci_method}")


def ci_overlap(
    ci_a: tuple[Optional[float], Optional[float]],
    ci_b: tuple[Optional[float], Optional[float]],
) -> Optional[bool]:
    a_lo, a_hi = ci_a
    b_lo, b_hi = ci_b
    if not all(_is_finite(v) for v in [a_lo, a_hi, b_lo, b_hi]):
        return None
    return not (float(a_hi) < float(b_lo) or float(a_lo) > float(b_hi))


def build_sim_interaction_graph(posts_csv: Path) -> nx.Graph:
    df = pd.read_csv(posts_csv)
    graph = nx.Graph()

    if "user_id" not in df.columns:
        return graph

    graph.add_nodes_from(df["user_id"].dropna().unique())
    if "comment_to" not in df.columns or "id" not in df.columns:
        return graph

    post_users = df.set_index("id")["user_id"].to_dict()
    comment_df = df[df["comment_to"].fillna(-1) != -1]

    for _, row in comment_df.iterrows():
        commenter = row["user_id"]
        parent_user = post_users.get(row["comment_to"])
        if parent_user is None or parent_user == commenter:
            continue
        _increment_edge(graph, commenter, parent_user)

    return graph


def build_voat_interaction_graph(parquet_path: Path) -> nx.Graph:
    df = pd.read_parquet(parquet_path)
    graph = nx.Graph()

    if "user_id" not in df.columns:
        return graph

    graph.add_nodes_from(df["user_id"].dropna().unique())
    if "parent_id" not in df.columns or "post_id" not in df.columns:
        return graph

    post_users = df.set_index("post_id")["user_id"].to_dict()
    for _, row in df.iterrows():
        parent_id = row.get("parent_id")
        if pd.isna(parent_id) or parent_id == row["post_id"]:
            continue
        child_user = row["user_id"]
        parent_user = post_users.get(parent_id)
        if parent_user is None or child_user is None or parent_user == child_user:
            continue
        _increment_edge(graph, child_user, parent_user)

    return graph


def compute_repeated_interaction_pct(graph: nx.Graph) -> float:
    total_edges = graph.number_of_edges()
    if total_edges == 0:
        return 0.0
    repeated_edges = sum(
        1
        for _, _, attrs in graph.edges(data=True)
        if attrs.get("weight", 1) > 1
    )
    return 100.0 * repeated_edges / total_edges


def compute_repeated_interaction_pct_from_posts(posts_csv: Path) -> float:
    return compute_repeated_interaction_pct(build_sim_interaction_graph(posts_csv))


def compute_repeated_interaction_pct_from_parquet(parquet_path: Path) -> float:
    return compute_repeated_interaction_pct(build_voat_interaction_graph(parquet_path))


def metric_rows_from_baseline(baseline: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric, entry in sorted(baseline.items()):
        row = {"metric": metric}
        row.update(entry)
        rows.append(row)
    return rows


def _increment_edge(graph: nx.Graph, user_a: Any, user_b: Any) -> None:
    if graph.has_edge(user_a, user_b):
        graph[user_a][user_b]["weight"] += 1
    else:
        graph.add_edge(user_a, user_b, weight=1)


def _is_finite(value: Any) -> bool:
    try:
        return value is not None and np.isfinite(float(value))
    except Exception:
        return False


def _student_t_critical(confidence: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between 0 and 1")

    alpha = 1.0 - confidence
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    nu = float(degrees_of_freedom)

    # Asymptotic expansion for the Student-t quantile around the normal quantile.
    z2 = z * z
    z3 = z2 * z
    z5 = z3 * z2
    z7 = z5 * z2

    term1 = (z3 + z) / (4.0 * nu)
    term2 = (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * nu * nu)
    term3 = (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / (384.0 * nu * nu * nu)
    return float(z + term1 + term2 + term3)
