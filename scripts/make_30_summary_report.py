#!/usr/bin/env python3
"""
make_30_summary_report.py

Generates `30-summary-cdx.md` by aggregating `results/runXX/metrics.json` (n=30)
and comparing against MADOC Voat sample summaries (n=10) and the precomputed
Voat network metric confidence intervals CSV.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from scripts.pipeline import load_madoc_samples, summarize_madoc_sample
except Exception:
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.pipeline import load_madoc_samples, summarize_madoc_sample  # type: ignore


def _is_finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def bootstrap_ci(
    values: Iterable[float],
    confidence: float = 0.95,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    vals = [float(v) for v in values if _is_finite(v)]
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(vals, dtype=float)
    if arr.size < 2:
        mean_val = float(np.mean(arr))
        return (mean_val, float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    n = arr.size
    boot = rng.choice(arr, size=(n_bootstrap, n), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    lo = float(np.percentile(boot, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return (float(arr.mean()), lo, hi)


def ci_overlap(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Optional[bool]:
    _, a_lo, a_hi = a
    _, b_lo, b_hi = b
    if not (_is_finite(a_lo) and _is_finite(a_hi) and _is_finite(b_lo) and _is_finite(b_hi)):
        return None
    return not (a_hi < b_lo or a_lo > b_hi)


def _format_ci(mean: float, lo: float, hi: float, fmt: str) -> str:
    if not _is_finite(mean):
        return "N/A"

    def f(x: float) -> str:
        if not _is_finite(x):
            return "N/A"
        if fmt == "count":
            return f"{x:,.1f}"
        if fmt == "count_int":
            return f"{int(round(x)):,}"
        if fmt == "pct":
            return f"{x:.2f}"
        if fmt == "tox":
            return f"{x:.4f}"
        if fmt == "dens":
            return f"{x:.6f}"
        if fmt == "float4":
            return f"{x:.4f}"
        if fmt == "float3":
            return f"{x:.3f}"
        return f"{x:.4f}"

    if _is_finite(lo) and _is_finite(hi):
        return f"{f(mean)} [{f(lo)}, {f(hi)}]"
    return f"{f(mean)} [N/A]"


def _get_path(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


@dataclass(frozen=True)
class MetricSpec:
    section: str
    label: str
    sim_path: Optional[str] = None
    voat_kind: Optional[str] = None  # "samples" | "network_ci" | None
    voat_key: Optional[str] = None
    fmt: str = "float4"
    sim_transform: Optional[Callable[[float], float]] = None
    voat_transform: Optional[Callable[[float], float]] = None
    notes: str = ""


def load_sim_metrics(results_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for run_dir in sorted(results_dir.glob("run*")):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        out.append(json.loads(metrics_path.read_text(encoding="utf-8", errors="ignore")))
    return out


def load_voat_network_ci(path: Path) -> Dict[str, Tuple[float, float, float]]:
    if not path.exists():
        return {}
    rows = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not rows:
        return {}
    header = rows[0].split(",")
    idx_metric = header.index("metric")
    idx_mean = header.index("mean")
    idx_lo = header.index("ci_lower")
    idx_hi = header.index("ci_upper")
    out: Dict[str, Tuple[float, float, float]] = {}
    for line in rows[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) <= max(idx_metric, idx_mean, idx_lo, idx_hi):
            continue
        metric = parts[idx_metric].strip()
        try:
            mean = float(parts[idx_mean])
            lo = float(parts[idx_lo])
            hi = float(parts[idx_hi])
        except ValueError:
            continue
        out[metric] = (mean, lo, hi)
    return out


def compute_voat_sample_cis(madoc_root: Path) -> Dict[str, Tuple[float, float, float]]:
    samples = load_madoc_samples(madoc_root)
    sample_metrics = [summarize_madoc_sample(s) for s in samples]

    def vals(getter: Callable[[Any], Any]) -> List[float]:
        out: List[float] = []
        for sm in sample_metrics:
            v = getter(sm)
            if _is_finite(v):
                out.append(float(v))
        return out

    posts = vals(lambda sm: sm.activity.get("posts_total"))
    comments = vals(lambda sm: sm.activity.get("comments_total"))

    return {
        "activity.total_interactions": bootstrap_ci([p + c for p, c in zip(posts, comments)]),
        "activity.root_posts": bootstrap_ci(posts),
        "activity.comments": bootstrap_ci(comments),
        "activity.unique_users": bootstrap_ci(vals(lambda sm: sm.activity.get("unique_users"))),
        "activity.avg_thread_length": bootstrap_ci(vals(lambda sm: sm.activity.get("avg_thread_length"))),
        "activity.mean_posts_per_user": bootstrap_ci(vals(lambda sm: sm.activity.get("mean_posts_per_user"))),
        "activity.mean_daily_active_users": bootstrap_ci(vals(lambda sm: sm.activity.get("mean_daily_active_users"))),
        "toxicity.overall_mean": bootstrap_ci(vals(lambda sm: sm.toxicity.get("overall_mean"))),
        "toxicity.posts_mean": bootstrap_ci(vals(lambda sm: sm.toxicity.get("posts_mean"))),
        "toxicity.comments_mean": bootstrap_ci(vals(lambda sm: sm.toxicity.get("comments_mean"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 30-summary-cdx.md from run metrics + Voat baselines")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--madoc-root", type=Path, default=Path("MADOC/voat-technology"))
    parser.add_argument(
        "--voat-network-ci",
        type=Path,
        default=Path("MADOC/voat-technology/voat_network_metrics_confidence_intervals.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("30-summary-cdx.md"))
    args = parser.parse_args()

    sim_runs = load_sim_metrics(args.results_dir)
    if len(sim_runs) < 2:
        raise SystemExit(f"Expected >=2 runs with metrics.json under {args.results_dir}, found {len(sim_runs)}")

    voat_sample_ci = compute_voat_sample_cis(args.madoc_root)
    voat_net_ci = load_voat_network_ci(args.voat_network_ci)

    specs: List[MetricSpec] = [
        MetricSpec(
            section="Basic metrics",
            label="Total interactions (posts+comments)",
            sim_path="activity.total_posts",
            voat_kind="samples",
            voat_key="activity.total_interactions",
            fmt="count",
        ),
        MetricSpec(
            section="Basic metrics",
            label="Root posts",
            sim_path="activity.root_posts",
            voat_kind="samples",
            voat_key="activity.root_posts",
            fmt="count",
        ),
        MetricSpec(
            section="Basic metrics",
            label="Comments",
            sim_path="activity.comments",
            voat_kind="samples",
            voat_key="activity.comments",
            fmt="count",
        ),
        MetricSpec(
            section="Basic metrics",
            label="Unique users",
            sim_path="activity.unique_users",
            voat_kind="samples",
            voat_key="activity.unique_users",
            fmt="count",
        ),
        MetricSpec(
            section="Basic metrics",
            label="Mean interactions per user",
            sim_path="activity.mean_posts_per_user",
            voat_kind="samples",
            voat_key="activity.mean_posts_per_user",
            fmt="float3",
        ),
        MetricSpec(
            section="Basic metrics",
            label="Avg thread length",
            sim_path="activity.avg_thread_length",
            voat_kind="samples",
            voat_key="activity.avg_thread_length",
            fmt="float3",
        ),
        MetricSpec(
            section="Basic metrics",
            label="Mean daily active users",
            sim_path="activity.mean_daily_active_users",
            voat_kind="samples",
            voat_key="activity.mean_daily_active_users",
            fmt="float3",
        ),
        MetricSpec(
            section="Network metrics",
            label="Density (whole network)",
            sim_path="network.density_total",
            voat_kind="network_ci",
            voat_key="density",
            fmt="dens",
        ),
        MetricSpec(
            section="Network metrics",
            label="Avg degree",
            sim_path="network.avg_degree",
            voat_kind="network_ci",
            voat_key="avg_degree",
            fmt="float3",
        ),
        MetricSpec(
            section="Network metrics",
            label="Avg weighted degree",
            sim_path="network.avg_weighted_degree",
            voat_kind="network_ci",
            voat_key="avg_weighted_degree",
            fmt="float3",
        ),
        MetricSpec(
            section="Network metrics",
            label="Avg clustering",
            sim_path="network.avg_clustering",
            voat_kind="network_ci",
            voat_key="avg_clustering",
            fmt="float4",
        ),
        MetricSpec(
            section="Network metrics",
            label="Largest connected component (%)",
            sim_path="network.lcc_ratio",
            sim_transform=lambda x: 100.0 * x,
            voat_kind="network_ci",
            voat_key="lcc_percentage",
            fmt="pct",
        ),
        MetricSpec(
            section="Core-periphery metrics",
            label="Core size (% of LCC)",
            sim_path="network.core_pct",
            voat_kind="network_ci",
            voat_key="core_percentage_of_lcc",
            fmt="pct",
        ),
        MetricSpec(
            section="Core-periphery metrics",
            label="Core density",
            sim_path="network.core_density",
            voat_kind="network_ci",
            voat_key="core_density",
            fmt="float3",
        ),
        MetricSpec(
            section="Core-periphery metrics",
            label="Core–periphery density",
            sim_path="network.core_periphery_density",
            voat_kind="network_ci",
            voat_key="core_periphery_density",
            fmt="float3",
        ),
        MetricSpec(
            section="Core-periphery metrics",
            label="Core avg degree",
            sim_path="network.core_avg_degree",
            voat_kind=None,
            fmt="float3",
        ),
        MetricSpec(
            section="Toxicity",
            label="Mean toxicity (all)",
            sim_path="toxicity.mean",
            voat_kind="samples",
            voat_key="toxicity.overall_mean",
            fmt="tox",
        ),
        MetricSpec(
            section="Toxicity",
            label="Mean toxicity (posts)",
            sim_path="toxicity.regular_post_mean",
            voat_kind="samples",
            voat_key="toxicity.posts_mean",
            fmt="tox",
            notes="Voat uses interaction type 'posts'; sim uses 'regular_post'.",
        ),
        MetricSpec(
            section="Toxicity",
            label="Mean toxicity (comments)",
            sim_path="toxicity.comment_mean",
            voat_kind="samples",
            voat_key="toxicity.comments_mean",
            fmt="tox",
        ),
        MetricSpec(
            section="Similarity (sim→Voat)",
            label="Embedding similarity (posts, mean cosine)",
            sim_path="embedding.posts_mean_cosine",
            voat_kind=None,
            fmt="float3",
            notes="Nearest-neighbor cosine similarity against MADOC Voat corpus.",
        ),
        MetricSpec(
            section="Similarity (sim→Voat)",
            label="Embedding similarity (comments, mean cosine)",
            sim_path="embedding.comments_mean_cosine",
            voat_kind=None,
            fmt="float3",
            notes="Nearest-neighbor cosine similarity against MADOC Voat corpus.",
        ),
        MetricSpec(
            section="Topic matching (sim↔Voat)",
            label="Topic coverage (sim topics matched)",
            sim_path="topic.coverage_sim2",
            voat_kind=None,
            sim_transform=lambda x: 100.0 * x,
            fmt="pct",
        ),
        MetricSpec(
            section="Topic matching (sim↔Voat)",
            label="Topic match mean cosine",
            sim_path="topic.mean_cosine",
            voat_kind=None,
            fmt="float3",
        ),
        MetricSpec(
            section="Topic matching (sim↔Voat)",
            label="Topic match mean soft-Jaccard",
            sim_path="topic.mean_soft_jaccard",
            voat_kind=None,
            fmt="float3",
        ),
        MetricSpec(
            section="Thread-level topics (sim↔Voat)",
            label="Thread-topic coverage (sim topics matched)",
            sim_path="topic_threads.coverage_sim2",
            voat_kind=None,
            sim_transform=lambda x: 100.0 * x,
            fmt="pct",
        ),
        MetricSpec(
            section="Thread-level topics (sim↔Voat)",
            label="Thread-topic match mean cosine",
            sim_path="topic_threads.mean_cosine",
            voat_kind=None,
            fmt="float3",
        ),
        MetricSpec(
            section="Thread-level topics (sim↔Voat)",
            label="Thread-topic match mean soft-Jaccard",
            sim_path="topic_threads.mean_soft_jaccard",
            voat_kind=None,
            fmt="float3",
        ),
    ]

    # Build markdown
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = []
    lines.append("# 30-run summary (CDX)\n")
    lines.append(f"- Generated: {now}\n")
    lines.append(f"- Simulation runs: n={len(sim_runs)} (`results/runXX/metrics.json`)\n")
    lines.append("- Voat baseline: n=10 MADOC temporal samples (`MADOC/voat-technology/sample_*`)\n")
    lines.append("- CIs: bootstrap 95% CI for the mean (5,000 resamples)\n")
    lines.append("- CI overlap: intervals intersect (descriptive, not a hypothesis test)\n\n")

    lines.append("## Main comparison table\n\n")
    lines.append("| Section | Metric | Sim mean [CI] | Voat mean [CI] | CI overlap? | Notes |\n")
    lines.append("|---|---|---:|---:|:---:|---|\n")

    for spec in specs:
        sim_vals: List[float] = []
        if spec.sim_path:
            for run in sim_runs:
                v = _get_path(run, spec.sim_path)
                if _is_finite(v):
                    fv = float(v)
                    if spec.sim_transform:
                        fv = float(spec.sim_transform(fv))
                    sim_vals.append(fv)
        sim_ci = bootstrap_ci(sim_vals) if sim_vals else (float("nan"), float("nan"), float("nan"))

        voat_ci: Optional[Tuple[float, float, float]] = None
        if spec.voat_kind == "samples" and spec.voat_key:
            voat_ci = voat_sample_ci.get(spec.voat_key)
        elif spec.voat_kind == "network_ci" and spec.voat_key:
            voat_ci = voat_net_ci.get(spec.voat_key)

        if voat_ci and spec.voat_transform:
            mean, lo, hi = voat_ci
            voat_ci = (
                float(spec.voat_transform(mean)),
                float(spec.voat_transform(lo)),
                float(spec.voat_transform(hi)),
            )

        overlap = ci_overlap(sim_ci, voat_ci) if voat_ci else None

        sim_cell = _format_ci(*sim_ci, spec.fmt)
        voat_cell = _format_ci(*voat_ci, spec.fmt) if voat_ci else "N/A"
        overlap_cell = "YES" if overlap is True else ("NO" if overlap is False else "N/A")
        notes = spec.notes or ""

        lines.append(
            f"| {spec.section} | {spec.label} | {sim_cell} | {voat_cell} | {overlap_cell} | {notes} |\n"
        )

    lines.append("\n## Diversity (not directly comparable)\n\n")
    lines.append("| Metric | Sim mean [CI] |\n")
    lines.append("|---|---:|\n")
    diversity_specs = [
        MetricSpec(
            section="Diversity",
            label="Semantic diversity (mean pairwise cosine distance)",
            sim_path="semantic_diversity.mean_pairwise_cosine_distance_mean",
            fmt="float4",
        )
    ]
    for spec in diversity_specs:
        vals: List[float] = []
        for run in sim_runs:
            v = _get_path(run, spec.sim_path or "")
            if _is_finite(v):
                vals.append(float(v))
        ci = bootstrap_ci(vals)
        lines.append(f"| {spec.label} | {_format_ci(*ci, spec.fmt)} |\n")

    lines.append("\n## Entropy (not directly comparable)\n\n")
    lines.append("| Metric | Sim mean [CI] |\n")
    lines.append("|---|---:|\n")
    entropy_specs = [
        MetricSpec(
            section="Entropy",
            label="H per token (overall mean)",
            sim_path="entropy.H_per_token_mean",
            fmt="float4",
        ),
        MetricSpec(
            section="Entropy",
            label="H per token (interpersonal mean)",
            sim_path="entropy.inter_hpt_mean",
            fmt="float4",
        ),
        MetricSpec(
            section="Entropy",
            label="H per token (intrapersonal mean)",
            sim_path="entropy.intra_hpt_mean",
            fmt="float4",
        ),
    ]
    for spec in entropy_specs:
        vals = []
        for run in sim_runs:
            v = _get_path(run, spec.sim_path or "")
            if _is_finite(v):
                vals.append(float(v))
        ci = bootstrap_ci(vals)
        lines.append(f"| {spec.label} | {_format_ci(*ci, spec.fmt)} |\n")

    args.out.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
