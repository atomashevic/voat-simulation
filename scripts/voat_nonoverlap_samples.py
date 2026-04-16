#!/usr/bin/env python3
# voat_nonoverlap_samples.py
#
# Generates strictly non-overlapping 30-day samples from the Voat/technology
# community "midlife" (Jan 2016 – Dec 2018), the stable post-peak period.
#
# Sampling strategy:
#   - Sequential, deterministic windows (no randomness)
#   - 30-day sample window + 5-day gap = 35-day stride
#   - Yields ~31 fully independent samples
#
# Output: MADOC/voat-technology-nonoverlap/sample_1/ ... sample_N/
#   (same per-sample artifacts as voat_samples.py)

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_LENGTH_DAYS = 30
GAP_DAYS = 5                          # gap between end of one window and start of next
STRIDE_DAYS = SAMPLE_LENGTH_DAYS + GAP_DAYS   # 35-day stride
SECONDS_PER_DAY = 86_400

# Midlife: stable post-peak period (2015 Reddit-exodus spike excluded)
MIDLIFE_START = datetime(2016, 1, 1)
MIDLIFE_END   = datetime(2018, 12, 31, 23, 59, 59)

OUTPUT_DIR = "MADOC/voat-technology-nonoverlap"
SOURCE_PARQUET = "MADOC/voat-technology/voat_technology_madoc.parquet"

# ---------------------------------------------------------------------------
# Patch voat_samples so its analyze_sample() writes to our OUTPUT_DIR
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
import voat_samples as _vs
_vs.OUTPUT_DIR = OUTPUT_DIR          # redirect all file I/O in imported module

# Re-export helpers used in main
unix_to_date        = _vs.unix_to_date
analyze_sample      = _vs.analyze_sample
save_json           = _vs.save_json


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def create_nonoverlap_samples(df: pd.DataFrame) -> list[dict]:
    """
    Build sequential, non-overlapping 30-day windows across the midlife period.
    Windows are deterministic (sorted by calendar time, no randomness).
    Returns a list of dicts: {'df': ..., 'info': ...}
    """
    start_ts  = MIDLIFE_START.timestamp()
    end_ts    = MIDLIFE_END.timestamp()
    stride_s  = STRIDE_DAYS * SECONDS_PER_DAY
    window_s  = SAMPLE_LENGTH_DAYS * SECONDS_PER_DAY

    samples = []
    sample_id = 1
    cursor = start_ts

    while cursor + window_s <= end_ts:
        window_end = cursor + window_s

        sample_df = df[
            (df['publish_date'] >= cursor) &
            (df['publish_date'] <  window_end)
        ].copy()

        start_label = unix_to_date(cursor)
        end_label   = unix_to_date(window_end)

        # Create output directory for this sample
        sample_dir = os.path.join(OUTPUT_DIR, f"sample_{sample_id}")
        os.makedirs(sample_dir, exist_ok=True)

        parquet_path = os.path.join(sample_dir, f"voat_nonoverlap_{sample_id}.parquet")

        if not os.path.exists(parquet_path) and len(sample_df) > 0:
            sample_df.to_parquet(parquet_path)

        info = {
            'sample_id':        sample_id,
            'start_date':       start_label,
            'end_date':         end_label,
            'start_timestamp':  cursor,
            'end_timestamp':    window_end,
            'record_count':     len(sample_df),
            'file_path':        parquet_path,
        }

        logger.info(
            f"Sample {sample_id:>2}: {start_label} → {end_label}  "
            f"({len(sample_df):,} records)"
        )

        samples.append({'df': sample_df, 'info': info})
        sample_id += 1
        cursor += stride_s          # advance by 30 days + 5-day gap

    logger.info(f"Generated {len(samples)} non-overlapping samples")
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=== voat_nonoverlap_samples.py ===")
    logger.info(
        f"Midlife window: {MIDLIFE_START.date()} → {MIDLIFE_END.date()}  "
        f"(stride = {STRIDE_DAYS}d: {SAMPLE_LENGTH_DAYS}d window + {GAP_DAYS}d gap)"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- load data --------------------------------------------------------
    logger.info(f"Loading {SOURCE_PARQUET}")
    df = pd.read_parquet(SOURCE_PARQUET)
    df['publish_date'] = pd.to_numeric(df['publish_date'])
    df = df.dropna(subset=['publish_date'])

    # normalise interaction_type capitalisation
    df['interaction_type'] = df['interaction_type'].str.lower()
    df.loc[df['interaction_type'] == 'post',    'interaction_type'] = 'posts'
    df.loc[df['interaction_type'] == 'comment', 'interaction_type'] = 'comments'

    logger.info(f"Loaded {len(df):,} rows  "
                f"({unix_to_date(df['publish_date'].min())} → "
                f"{unix_to_date(df['publish_date'].max())})")

    # ---- build samples ----------------------------------------------------
    samples = create_nonoverlap_samples(df)

    # ---- analyse each sample ----------------------------------------------
    summary_stats = []

    for sample in samples:
        sid     = sample['info']['sample_id']
        sdf     = sample['df']
        sinfo   = sample['info']

        if sinfo['record_count'] < 10:
            logger.warning(f"Sample {sid} has only {sinfo['record_count']} records – skipping analysis")
            continue

        logger.info(f"--- Analysing sample {sid} ({sinfo['start_date']} → {sinfo['end_date']}) ---")

        sample_dir = os.path.join(OUTPUT_DIR, f"sample_{sid}")

        # save comments text csv
        try:
            comments_only = sdf[sdf['interaction_type'] == 'comments']
            comments_only[['content']].dropna().to_csv(
                os.path.join(sample_dir, "comments_text.csv"), index=False
            )
        except Exception as e:
            logger.warning(f"Could not save comments_text.csv for sample {sid}: {e}")

        # check if analysis already done
        full_results_path = os.path.join(sample_dir, "full_results.json")
        if os.path.exists(full_results_path):
            logger.info(f"  Analysis already exists for sample {sid}, skipping")
            with open(full_results_path) as f:
                results = json.load(f)
        else:
            results = analyze_sample(sdf, sid, force_rerun=False)

        if results and 'user_dynamics' in results:
            ud = results['user_dynamics']
            summary_stats.append({
                'sample_id':    sid,
                'start_date':   sinfo['start_date'],
                'end_date':     sinfo['end_date'],
                'record_count': sinfo['record_count'],
                'unique_users': ud.get('total_unique_users'),
                'avg_users_per_day': ud.get('avg_users_per_day'),
                'avg_new_users_pct': ud.get('avg_new_users_pct'),
                'avg_churn_pct':     ud.get('avg_churn_pct'),
            })

    # ---- cross-sample summary ---------------------------------------------
    summary_path = os.path.join(OUTPUT_DIR, "summary_statistics.json")
    save_json(summary_stats, summary_path)
    logger.info(f"Saved cross-sample summary → {summary_path}")

    # print compact table
    logger.info("\n=== Summary ===")
    logger.info(f"{'ID':>4} {'Start':>12} {'End':>12} {'Records':>8} {'Users':>7}")
    for s in summary_stats:
        logger.info(
            f"{s['sample_id']:>4} {s['start_date']:>12} {s['end_date']:>12} "
            f"{s['record_count']:>8,} {str(s.get('unique_users','?')):>7}"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
