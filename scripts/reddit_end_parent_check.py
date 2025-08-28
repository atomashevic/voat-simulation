#!/usr/bin/env python3
"""
reddit_end_parent_check.py

Purpose:
- Load MADOC/reddit-technology/reddit_technology_madoc.parquet
- Take 10 random samples from the last 10% of the community lifetime
- For each sample window, compute the percentage of COMMENT rows with missing parent_user_id
- Print results only (no files saved)

Notes:
- Default sample length is 30 days. If the last 10% window is shorter than 30 days,
  the sample length is reduced to fit fully inside the window (min 1 day).
- Samples can overlap; we only need 10 independent draws within the end window.
"""

import os
import sys
import random
import math
from typing import Tuple, List

import pandas as pd
 # numpy not required for this script


# Constants
DATA_PATH = os.path.join('MADOC', 'reddit-technology', 'reddit_technology_madoc.parquet')
NUMBER_OF_SAMPLES = 10
DEFAULT_SAMPLE_LENGTH_DAYS = 30
SECONDS_PER_DAY = 24 * 60 * 60


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Data file not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(path)

    # Basic column sanity checks
    required_cols = {'publish_date', 'interaction_type'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Missing required columns: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    # parent_user_id is required for the check; if completely missing, exit early
    if 'parent_user_id' not in df.columns:
        print("Column 'parent_user_id' not found in dataset.", file=sys.stderr)
        sys.exit(1)

    # Ensure publish_date is numeric seconds
    df = df.dropna(subset=['publish_date']).copy()
    df['publish_date'] = pd.to_numeric(df['publish_date'], errors='coerce')
    df = df.dropna(subset=['publish_date']).copy()

    return df


def last_10_percent_window(df: pd.DataFrame) -> Tuple[float, float, float]:
    min_ts = float(df['publish_date'].min())
    max_ts = float(df['publish_date'].max())
    total_span = max_ts - min_ts
    if total_span <= 0:
        print("Invalid time span in data.", file=sys.stderr)
        sys.exit(1)

    start_10 = max_ts - 0.10 * total_span
    span_10 = max_ts - start_10
    return start_10, max_ts, span_10


def pick_sample_ranges(start_10: float, end_10: float, span_10: float) -> Tuple[int, List[Tuple[float, float]]]:
    # Determine sample duration in seconds; ensure it fits entirely inside the last-10% window
    max_days = max(1, int(math.floor(span_10 / SECONDS_PER_DAY)))
    sample_days = min(DEFAULT_SAMPLE_LENGTH_DAYS, max_days)
    sample_seconds = sample_days * SECONDS_PER_DAY

    if end_10 - start_10 < 1:
        print("End window too small to sample.", file=sys.stderr)
        sys.exit(1)

    # If the 10% window is shorter than the desired window, we already reduced sample_days
    # Now choose random starts so that the window [start, start+sample_seconds) lies within [start_10, end_10]
    max_start = end_10 - sample_seconds
    if max_start < start_10:
        # Degenerate: set start equal to start_10 to at least produce a single valid window
        max_start = start_10

    random.seed(42)
    ranges = []
    for _ in range(NUMBER_OF_SAMPLES):
        s = random.uniform(start_10, max_start) if max_start > start_10 else start_10
        e = s + sample_seconds
        # Clamp just in case of float precision
        if e > end_10:
            s = end_10 - sample_seconds
            e = end_10
        ranges.append((s, e))

    return sample_days, ranges


def compute_missing_parent_pct(df: pd.DataFrame, start_ts: float, end_ts: float) -> Tuple[int, int, float]:
    mask = (df['publish_date'] >= start_ts) & (df['publish_date'] < end_ts)
    window = df.loc[mask]

    if window.empty:
        return 0, 0, float('nan')

    # Robust comparison for interaction_type == 'COMMENT'
    itype = window['interaction_type']
    itype = itype.astype(str).str.upper()
    comments = window.loc[itype == 'COMMENT']

    total_comments = len(comments)
    if total_comments == 0:
        return 0, 0, float('nan')

    # Missing parent_user_id: NaN or empty/whitespace
    pcol = comments['parent_user_id']
    missing_mask = pcol.isna() | (pcol.astype(str).str.strip() == '')
    missing = int(missing_mask.sum())

    pct = (missing / total_comments) * 100.0
    return missing, total_comments, pct


def main() -> None:
    df = load_data(DATA_PATH)

    start_10, end_10, span_10 = last_10_percent_window(df)
    span_days = span_10 / SECONDS_PER_DAY
    print(f"Data loaded: {len(df):,} rows. Last 10% window ≈ {span_days:.2f} days.")

    sample_days, ranges = pick_sample_ranges(start_10, end_10, span_10)
    print(f"Sampling {NUMBER_OF_SAMPLES} windows of {sample_days} day(s) each from the last 10% window.")

    pcts = []
    totals = []
    missings = []

    for i, (s, e) in enumerate(ranges, start=1):
        missing, total, pct = compute_missing_parent_pct(df, s, e)
        missings.append(missing)
        totals.append(total)
        pcts.append(pct)

        if math.isnan(pct):
            print(f"Sample {i}: 0 comments in window — cannot compute percentage.")
        else:
            print(f"Sample {i}: missing parent_user_id in comments = {pct:.2f}% ({missing}/{total})")

    # Aggregate summary
    valid_pcts = [x for x in pcts if not math.isnan(x)]
    total_comments = sum(totals)
    total_missing = sum(missings)

    if valid_pcts:
        avg_pct = sum(valid_pcts) / len(valid_pcts)
        print(
            f"\nSummary: average missing% across {len(valid_pcts)} samples = {avg_pct:.2f}%"
        )
    else:
        print("\nSummary: no comment data present in sampled windows.")

    if total_comments > 0:
        overall_pct = (total_missing / total_comments) * 100.0
        print(
            f"Overall across all samples: {overall_pct:.2f}% missing ({total_missing}/{total_comments})"
        )


if __name__ == '__main__':
    main()
