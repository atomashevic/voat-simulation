#!/usr/bin/env python3
"""Generate a summary of MADOC Reddit/Voat samples.

This script scans the 10 time-sample folders for Reddit and Voat under
`MADOC/reddit-technology` and `MADOC/voat-technology`, reads the JSON
artifacts produced by the sampling/analysis scripts, and computes
per-sample metrics which are then aggregated across the 10 samples for
each platform. It writes a human-readable `.txt` summary into `reports/`.

Outputs: reports/madoc_samples_summary.txt

Run: python scripts/madoc_samples_summary.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MADOC_DIR = REPO_ROOT / "MADOC"
REPORTS_DIR = REPO_ROOT / "reports"


@dataclass
class SampleMetrics:
    """Metrics computed for a single 30-day sample window."""

    sample_id: int
    unique_users: int  # total unique users in the sample
    mean_daily_active_users: float
    mean_daily_new_users_pct: float
    mean_daily_churned_users_pct: float
    total_posts: int
    total_comments: int
    comments_per_post: float
    first_day_active_users: int


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_sample_dirs(platform_dir: Path) -> List[Path]:
    """Return 10 sample_* directories for a platform, if present.

    Prefers `sample_1..10`. If those are missing for Reddit (older runs),
    falls back to `early_sample_1..10`.
    """
    sample_dirs = [platform_dir / f"sample_{i}" for i in range(1, 11)]
    if all(d.exists() for d in sample_dirs):
        return sample_dirs

    # fallback for early_sample_* pattern (seen in some reddit runs)
    early_dirs = [platform_dir / f"early_sample_{i}" for i in range(1, 11)]
    if all(d.exists() for d in early_dirs):
        return early_dirs

    # try to pick whichever exists sample-by-sample (robustness)
    mixed: List[Path] = []
    for i in range(1, 11):
        d1, d2 = platform_dir / f"sample_{i}", platform_dir / f"early_sample_{i}"
        if d1.exists():
            mixed.append(d1)
        elif d2.exists():
            mixed.append(d2)
    return mixed


def compute_sample_metrics(sample_dir: Path) -> SampleMetrics:
    """Compute metrics for a single sample directory.

    Expects the following files (created by the sampling/analysis scripts):
    - full_results.json (contains user_dynamics.daily_dynamics with daily active/new/churned)
    - active_users.json (contains overall unique_users)
    - post_distribution.json (contains daily_counts with posts/comments per day)
    """

    # Detect sample_id from the directory name
    try:
        sid = int(sample_dir.name.split("_")[-1])
    except Exception:
        sid = -1

    # Load JSON artifacts
    full_results_path = sample_dir / "full_results.json"
    active_users_path = sample_dir / "active_users.json"
    post_dist_path = sample_dir / "post_distribution.json"

    if not full_results_path.exists():
        raise FileNotFoundError(f"Missing {full_results_path}")
    if not active_users_path.exists():
        raise FileNotFoundError(f"Missing {active_users_path}")
    if not post_dist_path.exists():
        raise FileNotFoundError(f"Missing {post_dist_path}")

    full_results = read_json(full_results_path)
    active_users = read_json(active_users_path)
    post_dist = read_json(post_dist_path)

    # 1) Total unique users in sample
    unique_users = int(active_users.get("unique_users", 0))

    # 2/3/4) Daily metrics from full_results.user_dynamics.daily_dynamics
    daily = full_results.get("user_dynamics", {}).get("daily_dynamics", [])
    daily_active = [int(d.get("active_users", 0)) for d in daily]
    daily_new_pct = [float(d.get("new_users_percentage", 0.0)) for d in daily]
    daily_churned_pct = [float(d.get("churned_percentage", 0.0)) for d in daily]

    mean_daily_active = float(np.mean(daily_active)) if daily_active else 0.0
    mean_daily_new_pct = float(np.mean(daily_new_pct)) if daily_new_pct else 0.0
    mean_daily_churned_pct = (
        float(np.mean(daily_churned_pct)) if daily_churned_pct else 0.0
    )

    # 6/7/5) Posts and comments per day from post_distribution.json
    # Aggregate over the whole sample window
    daily_counts: Dict[str, Dict[str, float]] = post_dist.get("daily_counts", {})
    total_posts = 0
    total_comments = 0
    for _, rec in daily_counts.items():
        total_posts += int(rec.get("posts", 0))
        total_comments += int(rec.get("comments", 0))
    comments_per_post = (total_comments / total_posts) if total_posts > 0 else 0.0

    # 8) Active users on the first day
    first_day_active = daily_active[0] if daily_active else 0

    return SampleMetrics(
        sample_id=sid,
        unique_users=unique_users,
        mean_daily_active_users=mean_daily_active,
        mean_daily_new_users_pct=mean_daily_new_pct,
        mean_daily_churned_users_pct=mean_daily_churned_pct,
        total_posts=total_posts,
        total_comments=total_comments,
        comments_per_post=comments_per_post,
        first_day_active_users=first_day_active,
    )


def summarize(values: List[float]) -> Tuple[float, float, float, float]:
    """Return (mean, sd, min, max) for a list of numbers.

    Uses sample standard deviation (ddof=1) when there are 2+ values,
    otherwise returns 0.0 for sd.
    """
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, sd, float(np.min(arr)), float(np.max(arr))


def collect_platform_metrics(platform: str, platform_dir: Path) -> Tuple[List[SampleMetrics], Dict[str, Tuple[float, float, float, float]]]:
    """Compute metrics for all 10 samples of a platform and summarize them.

    Returns both per-sample metrics and aggregate summaries per requested item.
    """
    samples_dirs = list_sample_dirs(platform_dir)
    if len(samples_dirs) != 10:
        raise RuntimeError(
            f"Expected 10 samples for {platform} in {platform_dir}, found {len(samples_dirs)}"
        )

    per_sample: List[SampleMetrics] = []
    for d in sorted(samples_dirs, key=lambda p: int(p.name.split("_")[-1])):
        per_sample.append(compute_sample_metrics(d))

    # Aggregate across samples for each requested metric
    users_per_sample = [s.unique_users for s in per_sample]
    active_users_per_day = [s.mean_daily_active_users for s in per_sample]
    new_users_pct_per_day = [s.mean_daily_new_users_pct for s in per_sample]
    churned_pct_per_day = [s.mean_daily_churned_users_pct for s in per_sample]
    comments_per_post = [s.comments_per_post for s in per_sample]
    posts_counts = [s.total_posts for s in per_sample]
    comments_counts = [s.total_comments for s in per_sample]
    first_day_active = [s.first_day_active_users for s in per_sample]

    summary = {
        "number_of_users_per_sample": summarize(users_per_sample),
        "active_users_per_day": summarize(active_users_per_day),
        "new_users_percentage_per_day": summarize(new_users_pct_per_day),
        "churned_users_percentage_per_day": summarize(churned_pct_per_day),
        "comments_per_post": summarize(comments_per_post),
        "number_of_posts": summarize(posts_counts),
        "number_of_comments": summarize(comments_counts),
        "first_day_active_users": summarize(first_day_active),
    }

    return per_sample, summary


def load_sampling_description() -> str:
    """Build a short description of how samples were created.

    Reads the header comments and constants from the sampling scripts to
    reflect the methodology used.
    """
    reddit_script = REPO_ROOT / "scripts" / "reddit-samples.py"
    voat_script = REPO_ROOT / "scripts" / "voat-samples.py"

    def extract_header_lines(path: Path) -> List[str]:
        lines: List[str] = []
        if not path.exists():
            return lines
        with path.open("r", encoding="utf-8") as f:
            for _ in range(20):  # read top portion where the description lives
                line = f.readline()
                if not line:
                    break
                if line.strip().startswith("#"):
                    # strip leading shebang and comment marker
                    content = line.strip().lstrip("#").strip()
                    if content:
                        lines.append(content)
        return lines

    reddit_lines = extract_header_lines(reddit_script)
    voat_lines = extract_header_lines(voat_script)

    # Try to capture the numbered steps if present
    def collapse_steps(lines: List[str]) -> str:
        steps = [
            ln
            for ln in lines
            if ln[:2].strip().startswith(tuple(str(i) for i in range(1, 6)))
        ]
        if steps:
            return " ".join(steps)
        return ""

    steps_text = collapse_steps(reddit_lines) or collapse_steps(voat_lines)

    # Constants (we know both scripts define these)
    sample_length = 30
    number_of_samples = 10

    base = (
        "Samples were generated by project scripts (reddit-samples.py / voat-samples.py) "
        f"as {number_of_samples} non-overlapping windows of {sample_length} consecutive days "
        "selected from the middle portion of each platform's technology dataset. "
        "For each window, per-day activity, user dynamics (new/churn), and thread metrics were computed."
    )

    if steps_text:
        return base + " Source steps: " + steps_text
    return base


def format_summary_block(platform: str, summary: Dict[str, Tuple[float, float, float, float]]) -> str:
    def line(label: str, stats: Tuple[float, float, float, float], pct: bool = False) -> str:
        mean, sd, mn, mx = stats
        if pct:
            return (
                f"- {label}: mean={mean:.2f}%, sd={sd:.2f}%, min={mn:.2f}%, max={mx:.2f}%\n"
            )
        else:
            return f"- {label}: mean={mean:.2f}, sd={sd:.2f}, min={mn:.2f}, max={mx:.2f}\n"

    out = []
    out.append(f"Platform: {platform}\n")
    out.append(line("1) Number of users per sample", summary["number_of_users_per_sample"]))
    out.append(line("2) Active users per day", summary["active_users_per_day"]))
    out.append(
        line(
            "3) New users percentage per day",
            summary["new_users_percentage_per_day"],
            pct=True,
        )
    )
    out.append(
        line(
            "4) Churned users percentage per day",
            summary["churned_users_percentage_per_day"],
            pct=True,
        )
    )
    out.append(line("5) Comments per post (sample-level)", summary["comments_per_post"]))
    out.append(line("6) Number of posts", summary["number_of_posts"]))
    out.append(line("7) Number of comments", summary["number_of_comments"]))
    out.append(line("8) Active users on day 1", summary["first_day_active_users"]))
    return "".join(out)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect per-platform summaries
    platforms = {
        "Reddit (technology)": MADOC_DIR / "reddit-technology",
        "Voat (technology)": MADOC_DIR / "voat-technology",
    }

    blocks: List[str] = []
    description = load_sampling_description()
    header = [
        "MADOC Sample Summary",
        "",
        "Description of sampling procedure:",
        description,
        "",
    ]
    blocks.append("\n".join(header))

    for label, path in platforms.items():
        per_sample, summary = collect_platform_metrics(label, path)
        blocks.append(format_summary_block(label, summary))
        blocks.append("")

    output_path = REPORTS_DIR / "madoc_samples_summary.txt"
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(blocks).strip() + "\n")

    print(f"Wrote summary to {output_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

