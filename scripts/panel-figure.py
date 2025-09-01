"""
Panel figure of daily metrics for a simulation directory.

Generates a 2x2 panel with:
  - Posts per day
  - Comments per day
  - Unique active users per day (by activity)
  - Interactions per active user per day (with median line)

Usage:
  python scripts/panel-figure.py --sim-dir simulation \
    [--out-file daily_metrics_panel.png]

Notes:
  - Prefers existing CSV time series exported by additional_plots.py:
      timeseries_posts_comments.csv
      timeseries_unique_active_users_activity.csv
    If absent, attempts to compute daily aggregates from posts.csv.
  - "Interactions" = posts + comments per day.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_timeseries_from_csv(sim_dir: Path):
    posts_comments_path = sim_dir / "timeseries_posts_comments.csv"
    active_path = sim_dir / "timeseries_unique_active_users_activity.csv"
    if posts_comments_path.exists() and active_path.exists():
        ts_pc = pd.read_csv(posts_comments_path)
        ts_active = pd.read_csv(active_path)
        # Normalize columns
        ts_pc.columns = [c.strip().lower() for c in ts_pc.columns]
        ts_active.columns = [c.strip().lower() for c in ts_active.columns]
        if "day" not in ts_pc.columns:
            raise ValueError(f"Missing 'day' column in {posts_comments_path}")
        if "day" not in ts_active.columns:
            raise ValueError(f"Missing 'day' column in {active_path}")
        # Expected columns with defaults
        for col in ("posts", "comments", "interactions"):
            if col not in ts_pc.columns:
                ts_pc[col] = 0
        if "active_users_activity" not in ts_active.columns:
            # allow alternative header name
            alt = [c for c in ts_active.columns if "active" in c]
            if alt:
                ts_active = ts_active.rename(columns={alt[0]: "active_users_activity"})
            else:
                raise ValueError(f"Missing 'active_users_activity' column in {active_path}")

        # Merge and sort by day
        merged = pd.merge(ts_pc[["day", "posts", "comments", "interactions"]],
                          ts_active[["day", "active_users_activity"]], on="day", how="outer")
        merged = merged.sort_values("day").reset_index(drop=True)
        # Fill NaNs with zeros for counts
        for c in ("posts", "comments", "interactions", "active_users_activity"):
            if c in merged.columns:
                merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype(int)
        return merged
    return None


def _compute_from_posts(sim_dir: Path, time_col: Optional[str] = None, hours_per_day: int = 24) -> pd.DataFrame:
    """Fallback: compute daily aggregates from posts.csv if the time series CSVs are absent."""
    posts_path = sim_dir / "posts.csv"
    if not posts_path.exists():
        raise FileNotFoundError("Could not find time series CSVs nor posts.csv to compute them.")

    # Read posts with tolerant CSV parsing (multi-line text)
    df = pd.read_csv(posts_path, engine="python")
    cols_lower = {c.lower(): c for c in df.columns}

    # Required columns
    uid_col = cols_lower.get("user_id")
    if uid_col is None:
        raise ValueError("posts.csv must contain a 'user_id' column")

    # Time column inference
    tcol = None
    if time_col and time_col in df.columns:
        tcol = time_col
    else:
        for cand in ("round", "created_at", "created_utc", "timestamp", "date"):
            if cand in cols_lower:
                tcol = cols_lower[cand]
                break
    if tcol is None:
        raise ValueError("Could not infer a time column in posts.csv; pass --time-col")

    # Determine posts vs comments using comment_to if available
    comment_to_col = cols_lower.get("comment_to")
    is_comment = None
    if comment_to_col is not None and comment_to_col in df.columns:
        cto_num = pd.to_numeric(df[comment_to_col], errors="coerce")
        # In these simulations, top-level posts can be marked with -1.
        # Treat comments as strictly > 0; posts as <= 0 or NaN.
        is_comment = (cto_num > 0)
    else:
        # If not available, treat all as posts (counts will go to posts)
        is_comment = pd.Series([False] * len(df))

    # Normalize to numeric hour index and then to day index starting at min
    hour = pd.to_numeric(df[tcol], errors="coerce")
    base = pd.to_numeric(hour.dropna()).min()
    day = ((hour - base) // hours_per_day).astype("Int64")

    # Build per-day counts
    tmp = pd.DataFrame({
        "day": day,
        "is_comment": is_comment.astype(bool),
        "user_id": df[uid_col],
    }).dropna(subset=["day"])  # drop rows without day

    ts_pc = (
        tmp.groupby(["day", "is_comment"]).size().unstack(fill_value=0).rename(columns={False: "posts", True: "comments"})
    )
    # Ensure both columns exist
    for c in ("posts", "comments"):
        if c not in ts_pc.columns:
            ts_pc[c] = 0
    ts_pc["interactions"] = ts_pc["posts"] + ts_pc["comments"]

    ts_active = tmp.groupby("day")["user_id"].nunique().rename("active_users_activity")

    merged = (
        ts_pc.join(ts_active, how="outer").reset_index().sort_values("day").fillna(0)
    )
    # Cast to ints for counts
    for c in ("posts", "comments", "interactions", "active_users_activity"):
        merged[c] = merged[c].astype(int)
    return merged


def _autocorr_at(series: pd.Series, lag: int) -> float:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    x = x - x.mean()
    if lag <= 0 or lag >= len(x):
        return np.nan
    num = np.dot(x[:-lag], x[lag:])
    den = np.dot(x, x)
    return float(num / den) if den != 0 else np.nan


def make_panel(df: pd.DataFrame, out_file: Path) -> dict:
    # Compute interactions per active user per day; guard against division by zero
    active = df["active_users_activity"].replace(0, np.nan)
    inter_per_active = df["interactions"] / active
    median_val = float(np.nanmedian(inter_per_active.values))

    # Style: clean scientific style with thicker lines
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.5,
        "figure.dpi": 150,
    })

    fig, axs = plt.subplots(2, 2, figsize=(7.5, 5.5), constrained_layout=True)
    days = df["day"].values

    # Posts/day
    ax = axs[0, 0]
    ax.plot(days, df["posts"].values, color="#1f77b4")
    ax.set_title("Posts/day")
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")

    # Comments/day
    ax = axs[0, 1]
    ax.plot(days, df["comments"].values, color="#ff7f0e")
    ax.set_title("Comments/day")
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")

    # Unique active users/day (by activity)
    ax = axs[1, 0]
    ax.plot(days, df["active_users_activity"].values, color="#2ca02c")
    ax.set_title("Active users/day")
    ax.set_xlabel("Day")
    ax.set_ylabel("Users")

    # Interactions per active user/day
    ax = axs[1, 1]
    ax.plot(days, inter_per_active.values, color="#9467bd")
    ax.axhline(median_val, color="#9467bd", linestyle="--", linewidth=1.8, alpha=0.8, label=f"Median = {median_val:.2f}")
    ax.set_title("Interactions per active user/day")
    ax.set_xlabel("Day")
    ax.set_ylabel("Mean per-active")
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Daily Metrics Panel", fontsize=12)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    # Simple weekly pattern note via lag-7 autocorrelation on interactions
    acf7 = _autocorr_at(df["interactions"], lag=7)
    acf1 = _autocorr_at(df["interactions"], lag=1)
    return {
        "median_interactions_per_active": median_val,
        "acf_lag1": acf1,
        "acf_lag7": acf7,
    }


def main():
    parser = argparse.ArgumentParser(description="Create a small daily metrics panel figure for a simulation directory.")
    parser.add_argument("--sim-dir", type=Path, default=Path("simulation"), help="Path to simulation directory")
    parser.add_argument("--out-file", type=Path, default=None, help="Output image file (default: <sim-dir>/daily_metrics_panel.png)")
    parser.add_argument("--time-col", type=str, default=None, help="Optional posts time column if computing from posts.csv")
    parser.add_argument("--hours-per-day", type=int, default=24, help="Rounds per day if computing from posts.csv")
    args = parser.parse_args()

    sim_dir: Path = args.sim_dir
    out_file: Path = args.out_file or (sim_dir / "daily_metrics_panel.png")

    if not sim_dir.exists():
        raise SystemExit(f"Simulation directory not found: {sim_dir}")

    # Prefer computing from posts.csv to ensure correct posts/comments split
    # (some exported CSVs have posts=0 due to comment_to encoding).
    try:
        df = _compute_from_posts(sim_dir, time_col=args.time_col, hours_per_day=args.hours_per_day)
    except Exception as e:
        print(f"Falling back to existing time series CSVs due to: {e}")
        df = _load_timeseries_from_csv(sim_dir)
        if df is None:
            raise

    # Remove padded last day if present (e.g., day 30)
    if 30 in set(df["day"].tolist()):
        df = df[df["day"] != 30].reset_index(drop=True)

    stats = make_panel(df, out_file)

    # Console note about diurnal/weekly patterns
    print(f"Saved panel: {out_file}")
    print(f"Median interactions per active user/day: {stats['median_interactions_per_active']:.3f}")
    if not np.isnan(stats["acf_lag7"]):
        weekly_hint = "likely" if stats["acf_lag7"] >= 0.3 else "unclear"
        print(f"Weekly pattern (lag-7 ACF = {stats['acf_lag7']:.2f}) is {weekly_hint}.")
    else:
        print("Weekly pattern: insufficient data to assess (lag-7).")


if __name__ == "__main__":
    main()
