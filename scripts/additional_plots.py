"""
Generate additional per-day activity plots and onboarding/churn summaries
for simulation outputs, aligning with real-data diagnostics.

Usage:
  python scripts/additional_plots.py \
    --sim-dir simulation2 \
    --out-dir simulation2 \
    [--generate-network-sim] \
    [--generate-network-voat --voat-dir MADOC/voat-technology/sample_1]

Inputs (defaults to --sim-dir):
  - users.csv: must contain at least columns: user_id (or id), joined_on, left_on
  - posts.csv: must contain at least columns: user_id, round, comment_to

Outputs (to --out-dir):
  - posts_per_day.png, comments_per_day.png, unique_active_users_timeseries.png
  - onboarding_churn_per_day.png, unique_active_users_distribution.png
  - active_days_distribution.png
  - CSV exports: *_timeseries.csv with the underlying data

Optional network visuals (saved next to their respective data directories):
  - For --generate-network-sim: four additional images in --sim-dir
  - For --generate-network-voat: same four images in --voat-dir
  The network visuals color nodes as Core or Periphery only (no Unknown class).

Notes:
  - "round" is treated as a day index for activity. If a true timestamp column
    exists (e.g., created_at), prefer that by providing --time-col.
  - Unique active users are reported in two ways:
      1) By activity: users who posted/commented that day (from posts.csv)
      2) By presence: users whose [joined_on, left_on] interval covers that day
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_users(users_path: Path) -> pd.DataFrame:
    df = pd.read_csv(users_path)
    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    # Identify id column
    uid_col = None
    for cand in ("user_id", "id"):
        if cand in cols:
            uid_col = cols[cand]
            break
    if uid_col is None:
        raise ValueError("users.csv must have a 'user_id' or 'id' column")

    # Identify joined/left columns
    joined_col = None
    left_col = None
    for c in df.columns:
        lc = c.lower()
        if lc == "joined_on":
            joined_col = c
        elif lc == "left_on":
            left_col = c
    if joined_col is None:
        raise ValueError("users.csv must have a 'joined_on' column")

    # Coerce to numeric day indices (allow NaN in left_on)
    df[joined_col] = pd.to_numeric(df[joined_col], errors="coerce").astype("Int64")
    if left_col is not None:
        df[left_col] = pd.to_numeric(df[left_col], errors="coerce").astype("Int64")
    else:
        # Create placeholder left_on as NA
        left_col = "left_on"
        df[left_col] = pd.Series([pd.NA] * len(df), dtype="Int64")

    # Standardize names: user_id, joined_on, left_on
    if uid_col != "user_id":
        df = df.rename(columns={uid_col: "user_id"})
    if joined_col != "joined_on":
        df = df.rename(columns={joined_col: "joined_on"})
    if left_col != "left_on":
        df = df.rename(columns={left_col: "left_on"})

    return df[["user_id", "joined_on", "left_on"]]


def read_posts(posts_path: Path, time_col: Optional[str] = None) -> pd.DataFrame:
    # Use python engine to tolerate multi-line text fields
    df = pd.read_csv(posts_path, engine="python")
    cols_lower = {c.lower(): c for c in df.columns}

    # Identify essentials
    uid_col = cols_lower.get("user_id")
    if uid_col is None:
        raise ValueError("posts.csv must have a 'user_id' column")

    # Determine time column (hour-level index, e.g., 'round')
    if time_col is not None and time_col in df.columns:
        tcol = time_col
    else:
        # default to 'round' if present
        tcol = cols_lower.get("round")
        if tcol is None:
            # try a few common timestamp columns
            for cand in ("created_at", "created_utc", "timestamp", "date"):
                if cand in cols_lower:
                    tcol = cols_lower[cand]
                    break
    if tcol is None:
        raise ValueError(
            "Could not infer time column. Specify with --time-col or include 'round'."
        )

    # Identify comment/post marker
    comment_to_col = cols_lower.get("comment_to")

    # Standardize
    out = df[[uid_col, tcol]].copy()
    out = out.rename(columns={uid_col: "user_id", tcol: "hour"})

    # Normalize hour index to numeric
    if not np.issubdtype(out["hour"].dtype, np.number):
        out["hour"] = pd.to_numeric(out["hour"], errors="coerce")
    out["hour"] = out["hour"].astype("Int64")

    # Determine post vs comment if possible
    if comment_to_col is not None and comment_to_col in df.columns:
        cto_raw = df[comment_to_col]
        # Prefer numeric interpretation: 0 or NaN => post; >0 => comment
        cto_num = pd.to_numeric(cto_raw, errors="coerce")
        is_comment = (cto_num.notna()) & (cto_num != 0)
        # If all NaN after coercion, fallback to string heuristic
        if is_comment.isna().all():
            is_comment = ~(cto_raw.isna() | (cto_raw.astype(str).str.len() == 0))
        out["is_comment"] = is_comment.astype("boolean")
    else:
        out["is_comment"] = pd.Series([pd.NA] * len(out), dtype="boolean")

    return out


def normalize_hours_to_days(series: pd.Series, base_hour: int, hours_per_day: int) -> pd.Series:
    """Convert hour index to day index starting at 0 using base_hour."""
    out = pd.to_numeric(series, errors="coerce") - base_hour
    out = (out // hours_per_day).astype("Int64")
    return out


def build_day_range(max_day: int) -> pd.Index:
    if pd.isna(max_day):
        raise ValueError("Unable to determine day range: max_day is NA")
    return pd.RangeIndex(start=0, stop=int(max_day) + 1)


def compute_timeseries(users: pd.DataFrame, posts: pd.DataFrame, days: pd.Index):
    # Posts/comments per day from posts
    ts_posts = (
        posts.assign(is_comment=posts["is_comment"].fillna(False))
        .groupby(["day", "is_comment"], dropna=True)
        .size()
        .unstack(fill_value=0)
        .reindex(days, fill_value=0)
    )
    ts_posts = ts_posts.rename(columns={False: "posts", True: "comments"})
    if "posts" not in ts_posts.columns:
        ts_posts["posts"] = 0
    if "comments" not in ts_posts.columns:
        ts_posts["comments"] = 0
    ts_posts["interactions"] = ts_posts["posts"] + ts_posts["comments"]

    # Unique active users by activity (posted/commented that day)
    ts_active_by_activity = (
        posts.groupby("day")["user_id"].nunique().reindex(days, fill_value=0)
    )

    # Onboarding/churn from users
    onboard = users.groupby("joined_day").size().reindex(days, fill_value=0)
    churn = (
        users["left_day"].dropna().groupby(users["left_day"]).size().reindex(days, fill_value=0)
    )
    net = onboard - churn
    present = net.cumsum()

    # Unique active users by presence interval equals present users
    ts_active_by_presence = present

    return ts_posts, ts_active_by_activity, onboard, churn, ts_active_by_presence


def save_csvs(out_dir: Path, ts_posts: pd.DataFrame, ts_active_by_activity: pd.Series,
              onboard: pd.Series, churn: pd.Series, ts_active_by_presence: pd.Series) -> None:
    ts_posts.to_csv(out_dir / "timeseries_posts_comments.csv", index_label="day")
    ts_active_by_activity.to_csv(out_dir / "timeseries_unique_active_users_activity.csv", index_label="day", header=["active_users_activity"])
    pd.DataFrame({
        "new_users": onboard,
        "churned_users": churn,
        "net_change": onboard - churn,
        "cumulative_present": ts_active_by_presence,
    }).to_csv(out_dir / "timeseries_onboarding_churn.csv", index_label="day")


def plot_timeseries(out_dir: Path, ts_posts: pd.DataFrame, ts_active_by_activity: pd.Series,
                    onboard: pd.Series, churn: pd.Series, ts_active_by_presence: pd.Series) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Posts and comments per day
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts_posts.index, ts_posts["posts"], label="Posts/day")
    ax.plot(ts_posts.index, ts_posts["comments"], label="Comments/day")
    ax.set_title("Posts and Comments per Day")
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "posts_per_day.png", dpi=150)
    plt.close(fig)

    # Unique active users (by activity) and presence
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts_active_by_activity.index, ts_active_by_activity.values, label="Active users (activity)")
    ax.plot(ts_active_by_presence.index, ts_active_by_presence.values, label="Present users (presence)", alpha=0.7)
    ax.set_title("Unique Active Users per Day")
    ax.set_xlabel("Day")
    ax.set_ylabel("Users")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "unique_active_users_timeseries.png", dpi=150)
    plt.close(fig)

    # Onboarding vs churn per day (stacked bars)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(onboard.index, onboard.values, label="New users/day", color="#2ca02c", alpha=0.8)
    ax.bar(churn.index, -churn.values, label="Churned users/day", color="#d62728", alpha=0.8)
    ax.set_title("Onboarding and Churn per Day")
    ax.set_xlabel("Day")
    ax.set_ylabel("Users (+/-)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "onboarding_churn_per_day.png", dpi=150)
    plt.close(fig)

    # Distribution of unique active users (by activity)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ts_active_by_activity.values, bins=20, color="#1f77b4", alpha=0.8)
    ax.set_title("Distribution: Unique Active Users per Day (activity)")
    ax.set_xlabel("Active users")
    ax.set_ylabel("Days")
    fig.tight_layout()
    fig.savefig(out_dir / "unique_active_users_distribution.png", dpi=150)
    plt.close(fig)


def plot_user_active_days(posts: pd.DataFrame, out_dir: Path) -> None:
    # Count distinct active days per user (activity-based)
    active_days = posts.dropna(subset=["day"]).groupby("user_id")["day"].nunique()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(active_days.values, bins=30, color="#9467bd", alpha=0.85)
    ax.set_title("Distribution: Active Days per User (activity)")
    ax.set_xlabel("Days active (posted/commented)")
    ax.set_ylabel("Users")
    fig.tight_layout()
    fig.savefig(out_dir / "active_days_distribution.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate per-day activity and onboarding/churn plots; optionally add network visuals.")
    parser.add_argument("--sim-dir", type=Path, default=Path("simulation3"), help="Directory containing users.csv and posts.csv")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for plots/CSVs (default: sim-dir)")
    parser.add_argument("--time-col", type=str, default=None, help="Optional posts time column to use instead of 'round'")
    parser.add_argument("--hours-per-day", type=int, default=24, help="Number of rounds per day (default: 24)")
    # Optional network plots
    parser.add_argument("--generate-network-sim", action="store_true", help="Also generate additional network plots for --sim-dir")
    parser.add_argument("--generate-network-voat", action="store_true", help="Also generate the same network plots for a Voat sample directory")
    parser.add_argument("--voat-dir", type=Path, default=Path("MADOC/voat-technology/sample_1"), help="Voat sample directory (default: MADOC/voat-technology/sample_1)")
    args = parser.parse_args()

    sim_dir: Path = args.sim_dir
    out_dir: Path = args.out_dir or sim_dir

    users_path = sim_dir / "users.csv"
    posts_path = sim_dir / "posts.csv"

    if not users_path.exists() or not posts_path.exists():
        raise SystemExit(f"Expected users.csv and posts.csv in {sim_dir}")

    print(f"Reading users from {users_path}")
    users = read_users(users_path)
    print(f"Reading posts from {posts_path}")
    posts = read_posts(posts_path, time_col=args.time_col)

    # Establish base hour and normalize to day indices
    base_hour = pd.concat([users["joined_on"].dropna(), posts["hour"].dropna()]).min()
    if pd.isna(base_hour):
        raise SystemExit("Unable to determine base hour from users.joined_on and posts time column")

    users["joined_day"] = normalize_hours_to_days(users["joined_on"], base_hour, args.hours_per_day)
    users["left_day"] = normalize_hours_to_days(users["left_on"], base_hour, args.hours_per_day)
    posts["day"] = normalize_hours_to_days(posts["hour"], base_hour, args.hours_per_day)

    max_day = int(
        pd.concat([
            posts["day"].dropna(),
            users["joined_day"].dropna(),
            users["left_day"].dropna(),
        ]).max()
    )
    days = build_day_range(max_day)
    print(f"Computed day range (relative): 0..{days.stop - 1} ({len(days)} days); base hour={int(base_hour)}")

    ts_posts, ts_active_by_activity, onboard, churn, ts_active_by_presence = compute_timeseries(users, posts, days)

    print("Saving CSV time series...")
    save_csvs(out_dir, ts_posts, ts_active_by_activity, onboard, churn, ts_active_by_presence)

    print("Creating plots...")
    plot_timeseries(out_dir, ts_posts, ts_active_by_activity, onboard, churn, ts_active_by_presence)
    plot_user_active_days(posts, out_dir)

    # Optional: network visuals
    if args.generate_network_sim or args.generate_network_voat:
        try:
            # Local import to avoid adding a hard dependency when unused
            import scripts.visualize_simulation2_additional as vis  # type: ignore
        except Exception:
            try:
                import visualize_simulation2_additional as vis  # fallback if running from scripts/
            except Exception as e:
                raise SystemExit(f"Unable to import network visual module: {e}")

        if args.generate_network_sim:
            print("Creating additional network visuals for simulation directory...")
            vis.generate_simulation_plots(sim_dir)

        if args.generate_network_voat:
            print(f"Creating additional network visuals for Voat sample at {args.voat_dir}...")
            vis.generate_voat_plots(args.voat_dir)

    print(f"Done. Plots and CSVs saved under {out_dir}")


if __name__ == "__main__":
    main()
