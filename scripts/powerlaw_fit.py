#!/usr/bin/env python3
"""
Formal tail-shape analysis for the main EPJ benchmark.

This script evaluates the two heavy-tail claims that appear in the paper:

1. Posts per user (root posts only)
2. Network degree (positive degrees in the interaction graph)

For each of the 30 simulation runs and 30 matched Voat windows, it:
    - fits a discrete power law with estimated x_min using the powerlaw package
    - computes a Clauset-style bootstrap goodness-of-fit p-value
    - compares power law to lognormal, truncated power law, and exponential tails
    - records distribution-free concentration metrics

Outputs:
    - per-sample fit CSV
    - corpus-level summary CSV / JSON
    - distribution-free comparison CSV
    - compact Markdown report for manuscript drafting
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
from scipy import stats

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
OBSERVABLES = ("posts_per_user", "degree")
COMPARISON_DISTS = ("lognormal", "truncated_power_law", "exponential")
SUMMARY_METRICS = ("gini", "top_1pct_share", "top_5pct_share", "top_10pct_share", "max_to_mean_ratio", "skewness")


def build_network_from_posts(posts_csv: Path) -> nx.Graph:
    """Build an undirected weighted interaction graph from simulation posts.csv."""
    df = pd.read_csv(posts_csv)
    graph = nx.Graph()
    graph.add_nodes_from(df["user_id"].dropna().unique())

    comment_df = df[df["comment_to"] != -1]
    parent_users = df.set_index("id")["user_id"].to_dict()

    for _, row in comment_df.iterrows():
        commenter = row["user_id"]
        parent_user = parent_users.get(row["comment_to"])
        if pd.notna(commenter) and pd.notna(parent_user) and commenter != parent_user:
            if graph.has_edge(commenter, parent_user):
                graph[commenter][parent_user]["weight"] += 1
            else:
                graph.add_edge(commenter, parent_user, weight=1)

    return graph


def build_network_from_parquet(parquet_path: Path) -> nx.Graph:
    """Build an undirected weighted interaction graph from a Voat parquet window."""
    df = pd.read_parquet(parquet_path)
    graph = nx.Graph()

    if "user_id" not in df.columns:
        return graph

    graph.add_nodes_from(df["user_id"].dropna().unique())
    if "parent_id" not in df.columns:
        return graph

    parent_users = df.set_index("post_id")["user_id"].to_dict()
    for _, row in df.iterrows():
        parent_id = row.get("parent_id")
        child_user = row.get("user_id")
        if pd.isna(parent_id) or parent_id == row.get("post_id"):
            continue

        parent_user = parent_users.get(parent_id)
        if pd.notna(child_user) and pd.notna(parent_user) and child_user != parent_user:
            if graph.has_edge(child_user, parent_user):
                graph[child_user][parent_user]["weight"] += 1
            else:
                graph.add_edge(child_user, parent_user, weight=1)

    return graph


def load_sim_posts_per_user(run_dir: Path) -> np.ndarray:
    posts = pd.read_csv(run_dir / "posts.csv", engine="python")
    comment_to = pd.to_numeric(posts["comment_to"], errors="coerce")
    is_root = (comment_to <= 0) | comment_to.isna()
    counts = posts.loc[is_root, "user_id"].value_counts().astype(int).to_numpy()
    return np.asarray(counts, dtype=int)


def load_voat_posts_per_user(sample_dir: Path) -> np.ndarray:
    parquet_path = next(sample_dir.glob("*.parquet"))
    df = pd.read_parquet(parquet_path)
    mask = df["interaction_type"].astype(str).str.lower().eq("posts")
    counts = df.loc[mask, "user_id"].value_counts().astype(int).to_numpy()
    return np.asarray(counts, dtype=int)


def load_sim_degree(run_dir: Path) -> np.ndarray:
    graph = build_network_from_posts(run_dir / "posts.csv")
    degrees = np.asarray([degree for _, degree in graph.degree() if degree > 0], dtype=int)
    return degrees


def load_voat_degree(sample_dir: Path) -> np.ndarray:
    parquet_path = next(sample_dir.glob("*.parquet"))
    graph = build_network_from_parquet(parquet_path)
    degrees = np.asarray([degree for _, degree in graph.degree() if degree > 0], dtype=int)
    return degrees


def gini_coefficient(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if values.size < 2:
        return float("nan")
    sorted_values = np.sort(values)
    n = sorted_values.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_values)) / (n * sorted_values.sum()) - (n + 1) / n)


def top_share(values: np.ndarray, pct: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if values.size == 0:
        return float("nan")
    sorted_values = np.sort(values)[::-1]
    top_n = max(1, int(math.ceil(values.size * pct / 100.0)))
    return float(100.0 * sorted_values[:top_n].sum() / sorted_values.sum())


def max_to_mean_ratio(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if values.size == 0 or values.mean() == 0:
        return float("nan")
    return float(values.max() / values.mean())


def cliffs_delta(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    greater = 0
    less = 0
    for x_val in x_arr:
        greater += np.sum(x_val > y_arr)
        less += np.sum(x_val < y_arr)
    return float((greater - less) / (len(x_arr) * len(y_arr)))


def empirical_fit(data: np.ndarray) -> powerlaw.Fit:
    return powerlaw.Fit(
        data,
        discrete=True,
        estimate_discrete=True,
        xmin=None,
        verbose=False,
    )


def bootstrap_powerlaw_gof(data: np.ndarray, fit: powerlaw.Fit, reps: int, seed: int) -> tuple[float, int]:
    """Clauset-style bootstrap GOF p-value preserving the empirical body below x_min."""
    observed_d = float(fit.D)
    xmin = int(round(float(fit.xmin)))
    body = data[data < xmin]
    n_tail = int(np.sum(data >= xmin))
    if reps <= 0 or n_tail < 10:
        return float("nan"), 0

    rng = np.random.default_rng(seed)
    simulated_ds: list[float] = []

    for _ in range(reps):
        if body.size:
            sampled_body = rng.choice(body, size=body.size, replace=True)
        else:
            sampled_body = np.empty(0, dtype=int)

        sampled_tail = np.asarray(
            fit.power_law.generate_random(size=n_tail, estimate_discrete=True),
            dtype=float,
        )
        sampled_tail = np.rint(sampled_tail).astype(int)
        sampled_tail[sampled_tail < xmin] = xmin

        synthetic = np.concatenate([sampled_body, sampled_tail]).astype(int)
        try:
            synthetic_fit = empirical_fit(synthetic)
            simulated_ds.append(float(synthetic_fit.D))
        except Exception:
            continue

    if not simulated_ds:
        return float("nan"), 0

    simulated_arr = np.asarray(simulated_ds, dtype=float)
    p_value = float(np.mean(simulated_arr >= observed_d))
    return p_value, int(simulated_arr.size)


def compare_distribution(fit: powerlaw.Fit, alt_name: str) -> tuple[float, float, str]:
    try:
        ratio, p_value = fit.distribution_compare("power_law", alt_name, normalized_ratio=True)
    except Exception:
        return float("nan"), float("nan"), "error"

    if not np.isfinite(ratio) or not np.isfinite(p_value):
        return float("nan"), float("nan"), "error"
    if p_value < 0.05 and ratio > 0:
        winner = "power_law"
    elif p_value < 0.05 and ratio < 0:
        winner = alt_name
    else:
        winner = "indeterminate"
    return float(ratio), float(p_value), winner


def analyze_sample(task: tuple[str, str, str, int, int]) -> dict:
    observable, source, path_str, bootstrap_reps, seed = task
    sample_path = Path(path_str)

    if observable == "posts_per_user" and source == "simulation":
        data = load_sim_posts_per_user(sample_path)
    elif observable == "posts_per_user" and source == "voat":
        data = load_voat_posts_per_user(sample_path)
    elif observable == "degree" and source == "simulation":
        data = load_sim_degree(sample_path)
    elif observable == "degree" and source == "voat":
        data = load_voat_degree(sample_path)
    else:
        raise ValueError(f"Unknown task combination: {observable}, {source}")

    data = np.asarray(data, dtype=int)
    data = data[np.isfinite(data) & (data > 0)]

    result = {
        "observable": observable,
        "source": source,
        "label": sample_path.name,
        "sample_path": str(sample_path),
        "n": int(data.size),
        "mean": float(np.mean(data)) if data.size else float("nan"),
        "median": float(np.median(data)) if data.size else float("nan"),
        "max": int(np.max(data)) if data.size else float("nan"),
        "gini": gini_coefficient(data),
        "top_1pct_share": top_share(data, 1),
        "top_5pct_share": top_share(data, 5),
        "top_10pct_share": top_share(data, 10),
        "max_to_mean_ratio": max_to_mean_ratio(data),
        "skewness": float(stats.skew(data)) if data.size >= 3 else float("nan"),
    }

    if data.size < 25:
        result.update(
            {
                "xmin": float("nan"),
                "alpha": float("nan"),
                "ks_d": float("nan"),
                "gof_p": float("nan"),
                "bootstrap_success": 0,
                "n_tail": 0,
                "accepted_powerlaw": False,
                "fit_note": "too_few_points",
            }
        )
        for alt_name in COMPARISON_DISTS:
            result[f"R_powerlaw_vs_{alt_name}"] = float("nan")
            result[f"p_powerlaw_vs_{alt_name}"] = float("nan")
            result[f"winner_powerlaw_vs_{alt_name}"] = "error"
        return result

    try:
        fit = empirical_fit(data)
        xmin = int(round(float(fit.xmin)))
        alpha = float(fit.alpha)
        ks_d = float(fit.D)
        n_tail = int(np.sum(data >= xmin))
        gof_p, bootstrap_success = bootstrap_powerlaw_gof(data, fit, bootstrap_reps, seed)
        accepted = bool(n_tail >= 50 and np.isfinite(gof_p) and gof_p >= 0.10)

        result.update(
            {
                "xmin": xmin,
                "alpha": alpha,
                "ks_d": ks_d,
                "gof_p": gof_p,
                "bootstrap_success": bootstrap_success,
                "n_tail": n_tail,
                "accepted_powerlaw": accepted,
                "fit_note": "ok",
            }
        )

        for alt_name in COMPARISON_DISTS:
            ratio, p_value, winner = compare_distribution(fit, alt_name)
            result[f"R_powerlaw_vs_{alt_name}"] = ratio
            result[f"p_powerlaw_vs_{alt_name}"] = p_value
            result[f"winner_powerlaw_vs_{alt_name}"] = winner
    except Exception as exc:
        result.update(
            {
                "xmin": float("nan"),
                "alpha": float("nan"),
                "ks_d": float("nan"),
                "gof_p": float("nan"),
                "bootstrap_success": 0,
                "n_tail": 0,
                "accepted_powerlaw": False,
                "fit_note": f"fit_error:{type(exc).__name__}",
            }
        )
        for alt_name in COMPARISON_DISTS:
            result[f"R_powerlaw_vs_{alt_name}"] = float("nan")
            result[f"p_powerlaw_vs_{alt_name}"] = float("nan")
            result[f"winner_powerlaw_vs_{alt_name}"] = "error"

    return result


def select_comparison_test(x: np.ndarray, y: np.ndarray) -> tuple[str, float, float, str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    shapiro_x = stats.shapiro(x).pvalue if 3 <= len(x) <= 5000 else float("nan")
    shapiro_y = stats.shapiro(y).pvalue if 3 <= len(y) <= 5000 else float("nan")
    levene_p = stats.levene(x, y).pvalue if len(x) >= 2 and len(y) >= 2 else float("nan")

    both_normal = np.isfinite(shapiro_x) and np.isfinite(shapiro_y) and shapiro_x > 0.05 and shapiro_y > 0.05
    if both_normal:
        test_name = "welch_t"
        statistic, p_value = stats.ttest_ind(x, y, equal_var=False)
        pooled = math.sqrt(((x.std(ddof=1) ** 2) + (y.std(ddof=1) ** 2)) / 2)
        effect = float((np.mean(x) - np.mean(y)) / pooled) if pooled > 0 else float("nan")
        effect_name = "cohens_d"
    else:
        test_name = "mannwhitney"
        statistic, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
        effect = cliffs_delta(x, y)
        effect_name = "cliffs_delta"

    return test_name, float(statistic), float(p_value), effect_name, float(effect)


def summarize_results(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    comparison_rows = []

    for observable in OBSERVABLES:
        subset = results_df[results_df["observable"] == observable].copy()
        sim_subset = subset[subset["source"] == "simulation"]
        voat_subset = subset[subset["source"] == "voat"]

        for source_name, source_df in (("simulation", sim_subset), ("voat", voat_subset)):
            accepted = source_df[source_df["accepted_powerlaw"]].copy()
            row = {
                "observable": observable,
                "source": source_name,
                "n_samples": int(len(source_df)),
                "n_powerlaw_accepted": int(len(accepted)),
                "powerlaw_accept_frac": float(len(accepted) / len(source_df)) if len(source_df) else float("nan"),
                "median_n_tail": float(accepted["n_tail"].median()) if len(accepted) else float("nan"),
                "median_xmin": float(accepted["xmin"].median()) if len(accepted) else float("nan"),
                "median_alpha": float(accepted["alpha"].median()) if len(accepted) else float("nan"),
                "median_gof_p": float(accepted["gof_p"].median()) if len(accepted) else float("nan"),
                "median_gini": float(source_df["gini"].median()) if len(source_df) else float("nan"),
                "median_top_1pct_share": float(source_df["top_1pct_share"].median()) if len(source_df) else float("nan"),
                "median_top_5pct_share": float(source_df["top_5pct_share"].median()) if len(source_df) else float("nan"),
                "median_top_10pct_share": float(source_df["top_10pct_share"].median()) if len(source_df) else float("nan"),
                "median_max_to_mean_ratio": float(source_df["max_to_mean_ratio"].median()) if len(source_df) else float("nan"),
                "median_skewness": float(source_df["skewness"].median()) if len(source_df) else float("nan"),
            }
            for alt_name in COMPARISON_DISTS:
                winner_col = f"winner_powerlaw_vs_{alt_name}"
                row[f"powerlaw_beats_{alt_name}_frac"] = float((source_df[winner_col] == "power_law").mean())
                row[f"{alt_name}_beats_powerlaw_frac"] = float((source_df[winner_col] == alt_name).mean())
            summary_rows.append(row)

        for metric in SUMMARY_METRICS:
            sim_vals = sim_subset[metric].dropna().to_numpy(dtype=float)
            voat_vals = voat_subset[metric].dropna().to_numpy(dtype=float)
            if len(sim_vals) < 3 or len(voat_vals) < 3:
                continue
            test_name, statistic, p_value, effect_name, effect = select_comparison_test(sim_vals, voat_vals)
            comparison_rows.append(
                {
                    "observable": observable,
                    "metric": metric,
                    "sim_mean": float(np.mean(sim_vals)),
                    "voat_mean": float(np.mean(voat_vals)),
                    "sim_median": float(np.median(sim_vals)),
                    "voat_median": float(np.median(voat_vals)),
                    "sim_shapiro_p": float(stats.shapiro(sim_vals).pvalue),
                    "voat_shapiro_p": float(stats.shapiro(voat_vals).pvalue),
                    "levene_p": float(stats.levene(sim_vals, voat_vals).pvalue),
                    "test": test_name,
                    "statistic": statistic,
                    "p_value": p_value,
                    "effect_name": effect_name,
                    "effect_value": effect,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(comparison_rows)


def build_tasks(results_dir: Path, voat_dir: Path, bootstrap_reps: int, base_seed: int) -> list[tuple[str, str, str, int, int]]:
    tasks: list[tuple[str, str, str, int, int]] = []
    run_dirs = sorted(path for path in results_dir.glob("run*") if path.is_dir())
    sample_dirs = sorted(path for path in voat_dir.glob("sample_*") if path.is_dir())
    observable_offsets = {"posts_per_user": 1_000_000, "degree": 2_000_000}

    for observable in OBSERVABLES:
        for idx, run_dir in enumerate(run_dirs):
            tasks.append((observable, "simulation", str(run_dir), bootstrap_reps, base_seed + observable_offsets[observable] + idx))
        for idx, sample_dir in enumerate(sample_dirs):
            tasks.append((observable, "voat", str(sample_dir), bootstrap_reps, base_seed + observable_offsets[observable] + 10_000 + idx))

    return tasks


def render_markdown_report(summary_df: pd.DataFrame, comparison_df: pd.DataFrame) -> str:
    lines = ["# Tail-shape validation summary", ""]
    for observable in OBSERVABLES:
        obs_summary = summary_df[summary_df["observable"] == observable].copy()
        lines.append(f"## {observable}")
        for _, row in obs_summary.iterrows():
            lines.append(
                (
                    f"- {row['source']}: accepted power-law fits in "
                    f"{int(row['n_powerlaw_accepted'])}/{int(row['n_samples'])} samples "
                    f"({100 * row['powerlaw_accept_frac']:.1f}%), median "
                    f"x_min = {row['median_xmin']:.1f}, median alpha = {row['median_alpha']:.3f}, "
                    f"median top-1% share = {row['median_top_1pct_share']:.1f}%."
                )
            )
        obs_comparisons = comparison_df[comparison_df["observable"] == observable].copy()
        if not obs_comparisons.empty:
            lines.append("")
            lines.append("Distribution-free comparisons:")
            for _, row in obs_comparisons.iterrows():
                effect_label = row["effect_name"]
                lines.append(
                    f"- {row['metric']}: {row['test']} p = {row['p_value']:.4f}, "
                    f"{effect_label} = {row['effect_value']:.3f}."
                )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal tail-shape validation for EPJ benchmark observables.")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    parser.add_argument("--voat-dir", type=Path, default=REPO / "MADOC" / "voat-technology-midlife30")
    parser.add_argument("--output-dir", type=Path, default=REPO / "results" / "tail_analysis_midlife30")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(8, (os.cpu_count() or 1))))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args.results_dir, args.voat_dir, args.bootstrap_reps, args.seed)
    logger.info("Running %d tail-analysis tasks with %d workers", len(tasks), args.n_jobs)

    results = []
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {executor.submit(analyze_sample, task): task for task in tasks}
        for idx, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            result = future.result()
            results.append(result)
            logger.info(
                "[%d/%d] %s %s %s: n=%s, accepted=%s, note=%s",
                idx,
                len(tasks),
                result["observable"],
                result["source"],
                result["label"],
                result["n"],
                result.get("accepted_powerlaw"),
                result.get("fit_note"),
            )

    results_df = pd.DataFrame(results).sort_values(["observable", "source", "label"]).reset_index(drop=True)
    summary_df, comparison_df = summarize_results(results_df)

    results_csv = args.output_dir / "tail_fits_per_sample.csv"
    summary_csv = args.output_dir / "tail_summary.csv"
    comparisons_csv = args.output_dir / "tail_metric_tests.csv"
    summary_json = args.output_dir / "tail_summary.json"
    report_md = args.output_dir / "tail_report.md"

    results_df.to_csv(results_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    comparison_df.to_csv(comparisons_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": summary_df.to_dict(orient="records"),
                "distribution_free_tests": comparison_df.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )

    report_text = render_markdown_report(summary_df, comparison_df)
    report_md.write_text(report_text, encoding="utf-8")

    logger.info("Saved per-sample fits to %s", results_csv)
    logger.info("Saved corpus summary to %s", summary_csv)
    logger.info("Saved comparison tests to %s", comparisons_csv)
    logger.info("Saved Markdown report to %s", report_md)


if __name__ == "__main__":
    main()
