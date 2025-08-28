#!/usr/bin/env python3
# voat-samples.py
#
# This script:
# 1. Reads `voat/technology.parquet`
# 2. Reads 'voat/toxicity/toxigen/technology.parquet'
# 3. Merges two dfs on post_id
# 4. Makes 10 time samples of 30 days from the middle of timespan of the subreddit data
# 5. For each sample, analyzes user activity, toxicity, and thread patterns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime
from collections import defaultdict, Counter
from tqdm import tqdm
import logging
from urllib.parse import urlparse
import json
import warnings
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set Matplotlib style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Constants
SAMPLE_LENGTH_DAYS = 30
NUMBER_OF_SAMPLES = 10
SECONDS_PER_DAY = 86400  # 24 * 60 * 60
OUTPUT_DIR = "MADOC/voat-technology"

# Create output directory structure
def create_directory_structure():
    """Create the directory structure for output files."""
    try:
        # Create main output directory if it doesn't exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            logger.info(f"Created main output directory: {OUTPUT_DIR}")

        # Create subdirectories for each sample
        for i in range(1, NUMBER_OF_SAMPLES + 1):
            sample_dir = os.path.join(OUTPUT_DIR, f"sample_{i}")
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir, exist_ok=True)
                logger.info(f"Created sample directory: {sample_dir}")
    except Exception as e:
        logger.error(f"Error creating directory structure: {str(e)}")
        raise

def unix_to_date(timestamp):
    """Convert Unix timestamp to human-readable date."""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

# Main functions to be implemented
def load_and_merge_data():
    """Load main data (already includes toxicity) from MADOC/voat-technology."""
    try:
        main_data_path = 'MADOC/voat-technology/voat_technology_madoc.parquet'

        logger.info(f"Loading main data from {main_data_path}")
        if not os.path.exists(main_data_path):
            raise FileNotFoundError(f"Data file not found: {main_data_path}")

        # Load data
        df = pd.read_parquet(main_data_path)
        logger.info(f"Loaded data: {len(df):,} rows")

        # Check for required columns
        required_columns = [
            'post_id', 'publish_date', 'user_id', 'parent_id',
            'content', 'interaction_type', 'toxicity_toxigen'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in data: {missing_columns}")

        # Handle null publish_date values
        null_dates = df['publish_date'].isna().sum()
        if null_dates > 0:
            logger.warning(f"Found {null_dates:,} records with null publish_date")
            logger.info("Dropping records with null publish_date")
            df = df.dropna(subset=['publish_date'])

        # Ensure publish_date is numeric
        df['publish_date'] = pd.to_numeric(df['publish_date'])

        # Check for required interaction types
        unique_interaction_types = df['interaction_type'].unique()
        logger.info(f"Found interaction types: {sorted(unique_interaction_types)}")
        
        required_interaction_types = ['posts', 'comments']
        missing_types = [t for t in required_interaction_types if t not in unique_interaction_types]
        if missing_types:
            # Check if we have uppercase versions instead
            uppercase_types = [t.upper() for t in unique_interaction_types]
            if 'POST' in uppercase_types and 'COMMENT' in uppercase_types:
                logger.warning("Found uppercase interaction types (POST/COMMENT). Converting to lowercase (posts/comments)...")
                df['interaction_type'] = df['interaction_type'].str.lower()
                df.loc[df['interaction_type'] == 'post', 'interaction_type'] = 'posts'
                df.loc[df['interaction_type'] == 'comment', 'interaction_type'] = 'comments'
                logger.info("Converted interaction types to posts/comments")
            else:
                raise ValueError(f"Missing required interaction types: {missing_types}. Found: {sorted(unique_interaction_types)}")

        # Basic dataset info
        logger.info(
            f"Time range: {unix_to_date(df['publish_date'].min())} to {unix_to_date(df['publish_date'].max())}"
        )
        logger.info(f"Unique users: {df['user_id'].nunique():,}")
        logger.info(f"Posts: {df[df['interaction_type']=='posts'].shape[0]:,}")
        logger.info(f"Comments: {df[df['interaction_type']=='comments'].shape[0]:,}")

        return df

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_time_samples(df):
    """
    Create 10 time samples of 30 days each from the middle of the timespan.
    Saves each sample as a parquet file and returns a list of dataframes with metadata.
    If sample files already exist, loads them instead of recreating.
    """
    try:
        # Get min and max timestamps
        min_time = df['publish_date'].min()
        max_time = df['publish_date'].max()
        total_time_span = max_time - min_time

        # Find the middle of the timespan
        mid_point = min_time + (total_time_span / 2)
        logger.info(f"Total time span: {total_time_span / SECONDS_PER_DAY:.1f} days")
        logger.info(f"Middle point: {unix_to_date(mid_point)}")

        # Calculate the total window we need for all samples
        # Since we're taking 10 samples of 30 days each, we need a window of 300 days
        total_window_seconds = NUMBER_OF_SAMPLES * SAMPLE_LENGTH_DAYS * SECONDS_PER_DAY

        # Ensure we have enough data for all our samples
        if total_window_seconds > total_time_span:
            logger.warning(f"Total timespan ({total_time_span / SECONDS_PER_DAY:.1f} days) is less than required for {NUMBER_OF_SAMPLES} samples ({total_window_seconds / SECONDS_PER_DAY:.1f} days)")
            logger.warning(f"Will adjust sample count or length to fit available data")

            # Adjust our approach based on available data
            if total_time_span >= (SAMPLE_LENGTH_DAYS * SECONDS_PER_DAY):
                # We can fit at least one full sample, adjust the number of samples
                adjusted_samples = int(total_time_span / (SAMPLE_LENGTH_DAYS * SECONDS_PER_DAY))
                logger.warning(f"Adjusting to {adjusted_samples} samples of {SAMPLE_LENGTH_DAYS} days each")
                total_window_seconds = adjusted_samples * SAMPLE_LENGTH_DAYS * SECONDS_PER_DAY
            else:
                # We can't even fit one full sample, adjust the sample length
                adjusted_length = int(total_time_span / SECONDS_PER_DAY)
                logger.warning(f"Adjusting to {NUMBER_OF_SAMPLES} samples of {adjusted_length} days each")
                total_window_seconds = NUMBER_OF_SAMPLES * adjusted_length * SECONDS_PER_DAY

        # Start our sampling window from half the total window before the midpoint
        start_time = mid_point - (total_window_seconds / 2)

        # Adjust if we go beyond the available data
        if start_time < min_time:
            logger.warning(f"Adjusted start time to match data availability")
            start_time = min_time

        # Calculate sample intervals
        sample_interval_seconds = SAMPLE_LENGTH_DAYS * SECONDS_PER_DAY

        # Create sample dataframes
        samples = []

        for i in range(NUMBER_OF_SAMPLES):
            sample_id = i + 1
            sample_dir = os.path.join(OUTPUT_DIR, f"sample_{sample_id}")
            sample_file_path = os.path.join(sample_dir, f"voat_sample_{sample_id}.parquet")

            # Check if sample file already exists
            if os.path.exists(sample_file_path):
                logger.info(f"Sample {sample_id} already exists at {sample_file_path}, loading from file")

                try:
                    # Load the sample from existing file
                    sample_df = pd.read_parquet(sample_file_path)

                    # Extract time range from the loaded data
                    sample_start = sample_df['publish_date'].min()
                    sample_end = sample_df['publish_date'].max()

                    # Create sample info
                    sample_info = {
                        'sample_id': sample_id,
                        'start_date': unix_to_date(sample_start),
                        'end_date': unix_to_date(sample_end),
                        'start_timestamp': sample_start,
                        'end_timestamp': sample_end,
                        'record_count': len(sample_df),
                        'file_path': sample_file_path,
                        'loaded_from_file': True
                    }

                    logger.info(f"Loaded sample {sample_id}: {sample_info['start_date']} to {sample_info['end_date']} ({sample_info['record_count']:,} records)")

                    # Store the sample with its metadata
                    samples.append({
                        'df': sample_df,
                        'info': sample_info
                    })

                    continue
                except Exception as e:
                    logger.warning(f"Error loading existing sample file {sample_file_path}: {str(e)}")
                    logger.warning(f"Will recreate sample {sample_id}")

            # If we get here, either the file doesn't exist or loading failed
            # Calculate the time range for this sample
            sample_start = start_time + (i * sample_interval_seconds)
            sample_end = sample_start + sample_interval_seconds

            # Ensure we don't go beyond the available data
            if sample_end > max_time:
                logger.warning(f"Sample {sample_id} end time adjusted to match data availability")
                sample_end = max_time

            # Filter data for this sample
            sample_df = df[(df['publish_date'] >= sample_start) & (df['publish_date'] < sample_end)]

            # Check if we have enough data
            if len(sample_df) < 10:
                logger.warning(f"Sample {sample_id} has very few records ({len(sample_df)}), might not be useful for analysis")

            # Add date information for easier reference
            sample_info = {
                'sample_id': sample_id,
                'start_date': unix_to_date(sample_start),
                'end_date': unix_to_date(sample_end),
                'start_timestamp': sample_start,
                'end_timestamp': sample_end,
                'record_count': len(sample_df),
                'loaded_from_file': False
            }

            logger.info(f"Created sample {sample_id}: {sample_info['start_date']} to {sample_info['end_date']} ({sample_info['record_count']:,} records)")

            # Save the dataframe to parquet
            if len(sample_df) > 0:
                sample_df.to_parquet(sample_file_path)
                logger.info(f"Saved sample {sample_id} to {sample_file_path}")

                # Add file path to sample info
                sample_info['file_path'] = sample_file_path
            else:
                logger.warning(f"Sample {sample_id} has no records, skipping parquet file creation")
                sample_info['file_path'] = None

            # Store the sample with its metadata
            samples.append({
                'df': sample_df,
                'info': sample_info
            })

        return samples

    except Exception as e:
        logger.error(f"Error creating time samples: {str(e)}")
        raise

def analyze_sample(df_sample, sample_id, force_rerun=False):
    """Run all analyses on a given time sample."""
    try:
        logger.info(f"Running comprehensive analysis on sample {sample_id}")

        # Get sample directory
        sample_dir = os.path.join(OUTPUT_DIR, f"sample_{sample_id}")

        # Store results in a dictionary
        results = {}

        # Add sample information
        start_date = unix_to_date(df_sample['publish_date'].min())
        end_date = unix_to_date(df_sample['publish_date'].max())

        results['sample_info'] = {
            'sample_id': sample_id,
            'start_date': start_date,
            'end_date': end_date,
            'record_count': len(df_sample)
        }

        # 1. Analyze active users
        logger.info(f"Analyzing active users for sample {sample_id}")
        active_users_file = os.path.join(sample_dir, "active_users.json")

        # Only use existing file if not forcing rerun
        if os.path.exists(active_users_file) and not force_rerun:
            try:
                logger.info(f"Loading active users data from {active_users_file}")
                with open(active_users_file, 'r') as f:
                    import json
                    active_users_results = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading file {active_users_file}: {str(e)}")
                logger.info(f"Recalculating active users for sample {sample_id}")
                active_users_results = count_active_users(df_sample)
                save_json(active_users_results, active_users_file)
        else:
            # Either file doesn't exist or we're forcing rerun
            active_users_results = count_active_users(df_sample)
            save_json(active_users_results, active_users_file)

        results['active_users'] = active_users_results

        # 2. Analyze user dynamics
        logger.info(f"Analyzing user dynamics for sample {sample_id}")
        user_dynamics_file = os.path.join(sample_dir, "user_dynamics.json")

        # Only use existing file if not forcing rerun
        if os.path.exists(user_dynamics_file) and not force_rerun:
            try:
                logger.info(f"Loading user dynamics data from {user_dynamics_file}")
                with open(user_dynamics_file, 'r') as f:
                    import json
                    user_dynamics_results = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading file {user_dynamics_file}: {str(e)}")
                logger.info(f"Recalculating user dynamics for sample {sample_id}")
                user_dynamics_results = calculate_user_dynamics(df_sample)
                save_json(user_dynamics_results, user_dynamics_file)
        else:
            # Either file doesn't exist or we're forcing rerun
            user_dynamics_results = calculate_user_dynamics(df_sample)
            save_json(user_dynamics_results, user_dynamics_file)

        results['user_dynamics'] = user_dynamics_results

        # 3. Analyze average interactions
        logger.info(f"Analyzing average interactions for sample {sample_id}")
        avg_interactions_file = os.path.join(sample_dir, "avg_interactions.json")

        # Only use existing file if not forcing rerun
        if os.path.exists(avg_interactions_file) and not force_rerun:
            try:
                logger.info(f"Loading average interactions data from {avg_interactions_file}")
                with open(avg_interactions_file, 'r') as f:
                    import json
                    avg_interactions_results = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading file {avg_interactions_file}: {str(e)}")
                logger.info(f"Recalculating average interactions for sample {sample_id}")
                avg_interactions_results = calculate_avg_interactions(df_sample)
                save_json(avg_interactions_results, avg_interactions_file)
        else:
            # Either file doesn't exist or we're forcing rerun
            avg_interactions_results = calculate_avg_interactions(df_sample)
            save_json(avg_interactions_results, avg_interactions_file)

            # Create plots for this analysis
            create_avg_interactions_plots(avg_interactions_results, sample_id, sample_dir, df_sample)

        results['avg_interactions'] = avg_interactions_results

        # 4. Analyze post/comment distribution
        logger.info(f"Analyzing post and comment distribution for sample {sample_id}")
        post_dist_file = os.path.join(sample_dir, "post_distribution.json")

        # Only use existing file if not forcing rerun
        if os.path.exists(post_dist_file) and not force_rerun:
            try:
                logger.info(f"Loading post distribution data from {post_dist_file}")
                with open(post_dist_file, 'r') as f:
                    import json
                    post_dist_results = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading file {post_dist_file}: {str(e)}")
                logger.info(f"Recalculating post distribution for sample {sample_id}")
                post_dist_results = analyze_post_distribution(df_sample)
                save_json(post_dist_results, post_dist_file)
                create_post_distribution_plots(post_dist_results, sample_id, sample_dir)
        else:
            # Either file doesn't exist or we're forcing rerun
            post_dist_results = analyze_post_distribution(df_sample)
            save_json(post_dist_results, post_dist_file)
            create_post_distribution_plots(post_dist_results, sample_id, sample_dir)

        results['post_distribution'] = post_dist_results

        # 5. Analyze toxicity
        logger.info(f"Analyzing toxicity for sample {sample_id}")
        toxicity_file = os.path.join(sample_dir, "toxicity.json")

        # Only use existing file if not forcing rerun
        if os.path.exists(toxicity_file) and not force_rerun:
            try:
                logger.info(f"Loading toxicity data from {toxicity_file}")
                with open(toxicity_file, 'r') as f:
                    import json
                    toxicity_results = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading file {toxicity_file}: {str(e)}")
                logger.info(f"Recalculating toxicity for sample {sample_id}")
                toxicity_results = analyze_toxicity_distribution(df_sample)
                save_json(toxicity_results, toxicity_file)
                create_toxicity_plots(toxicity_results, sample_id, sample_dir)
        else:
            # Either file doesn't exist or we're forcing rerun
            toxicity_results = analyze_toxicity_distribution(df_sample)
            save_json(toxicity_results, toxicity_file)
            create_toxicity_plots(toxicity_results, sample_id, sample_dir)

        results['toxicity'] = toxicity_results

        # 6. Analyze thread length
        logger.info(f"Analyzing thread length for sample {sample_id}")
        thread_length_file = os.path.join(sample_dir, "thread_length.json")

        # Only use existing file if not forcing rerun
        if os.path.exists(thread_length_file) and not force_rerun:
            try:
                logger.info(f"Loading thread length data from {thread_length_file}")
                with open(thread_length_file, 'r') as f:
                    import json
                    thread_length_results = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading file {thread_length_file}: {str(e)}")
                logger.info(f"Recalculating thread length for sample {sample_id}")
                thread_length_results = analyze_thread_length(df_sample)
                save_json(thread_length_results, thread_length_file)
                create_thread_length_plots(thread_length_results, sample_id, sample_dir)
        else:
            # Either file doesn't exist or we're forcing rerun
            thread_length_results = analyze_thread_length(df_sample)
            save_json(thread_length_results, thread_length_file)
            create_thread_length_plots(thread_length_results, sample_id, sample_dir)

        results['thread_length'] = thread_length_results

        # 7. Analyze thread activity
        logger.info(f"Analyzing thread activity for sample {sample_id}")
        thread_activity_file = os.path.join(sample_dir, "thread_activity.json")

        # Only use existing file if not forcing rerun
        if os.path.exists(thread_activity_file) and not force_rerun:
            try:
                logger.info(f"Loading thread activity data from {thread_activity_file}")
                with open(thread_activity_file, 'r') as f:
                    import json
                    thread_activity_results = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading file {thread_activity_file}: {str(e)}")
                logger.info(f"Recalculating thread activity for sample {sample_id}")
                thread_activity_results = analyze_thread_activity(df_sample)
                save_json(thread_activity_results, thread_activity_file)
                create_thread_activity_plots(thread_activity_results, sample_id, sample_dir)
        else:
            # Either file doesn't exist or we're forcing rerun
            thread_activity_results = analyze_thread_activity(df_sample)
            save_json(thread_activity_results, thread_activity_file)
            create_thread_activity_plots(thread_activity_results, sample_id, sample_dir)

        results['thread_activity'] = thread_activity_results

        # 8. Generate summary report
        logger.info(f"Generating summary report for sample {sample_id}")
        report_file = os.path.join(sample_dir, "analysis_report.txt")
        save_results_text(results, report_file)

        # 9. Save full results JSON
        full_results_file = os.path.join(sample_dir, "full_results.json")
        save_json(results, full_results_file)

        logger.info(f"Completed analysis for sample {sample_id}")
        return results

    except Exception as e:
        logger.error(f"Error analyzing sample {sample_id}: {str(e)}")
        raise

def count_active_users(df_sample):
    """
    Count total unique active users in the 30-day period and their activity patterns.
    Returns a dictionary with various user activity metrics.
    """
    try:
        # Get unique user count
        unique_users = df_sample['user_id'].nunique()

        # Group by user_id and count interactions
        user_activity = df_sample.groupby('user_id').agg(
            interactions=pd.NamedAgg(column='post_id', aggfunc='count'),
            first_seen=pd.NamedAgg(column='publish_date', aggfunc='min'),
            last_seen=pd.NamedAgg(column='publish_date', aggfunc='max')
        )

        # Calculate activity span for each user (in days)
        user_activity['activity_span_days'] = (
            (user_activity['last_seen'] - user_activity['first_seen']) / SECONDS_PER_DAY
        )

        # Count users by activity level
        low_activity_users = len(user_activity[user_activity['interactions'] <= 2])
        medium_activity_users = len(user_activity[(user_activity['interactions'] > 2) &
                                                (user_activity['interactions'] <= 10)])
        high_activity_users = len(user_activity[user_activity['interactions'] > 10])

        # Get most active users
        top_users = user_activity.sort_values('interactions', ascending=False).head(10)

        # Calculate one-day-only users (users who appeared only on a single day)
        one_day_users = len(user_activity[user_activity['activity_span_days'] < 1])

        # Prepare results
        results = {
            'unique_users': unique_users,
            'low_activity_users': low_activity_users,
            'medium_activity_users': medium_activity_users,
            'high_activity_users': high_activity_users,
            'one_day_users': one_day_users,
            'one_day_percentage': (one_day_users / unique_users) * 100 if unique_users > 0 else 0,
            'avg_interactions_per_user': user_activity['interactions'].mean(),
            'median_interactions_per_user': user_activity['interactions'].median(),
            'avg_activity_span_days': user_activity['activity_span_days'].mean(),
            'top_users': top_users
        }

        return results

    except Exception as e:
        logger.error(f"Error counting active users: {str(e)}")
        raise

def calculate_user_dynamics(df_sample):
    """
    Track new users and calculate:
    1. Percentage of new users active on each day starting from day 10
    2. Percentage of users from previous days who are never active after each day

    Returns a dictionary with day-by-day user dynamics metrics.
    """
    try:
        # Get the start timestamp from the sample data
        start_timestamp = df_sample['publish_date'].min()

        # Create day buckets for the 30-day period
        days_users = {}  # Maps day number to set of users active on that day

        for day in range(SAMPLE_LENGTH_DAYS):
            day_start = start_timestamp + (day * SECONDS_PER_DAY)
            day_end = day_start + SECONDS_PER_DAY

            # Get users active on this day
            day_users = set(df_sample[
                (df_sample['publish_date'] >= day_start) &
                (df_sample['publish_date'] < day_end)
            ]['user_id'].unique())

            days_users[day] = day_users

        # Initialize an empty set to track all users seen so far
        all_users_so_far = set()

        # Track user dynamics for each day
        user_dynamics = []

        for day in range(SAMPLE_LENGTH_DAYS):
            if day not in days_users or not days_users[day]:
                # No activity on this day, add placeholder
                user_dynamics.append({
                    'day': day + 1,  # 1-indexed for reporting
                    'active_users': 0,
                    'new_users': 0,
                    'new_users_percentage': 0,
                    'churned_users': 0,
                    'churned_percentage': 0
                })
                continue

            # Current day's active users
            current_users = days_users[day]
            active_users_count = len(current_users)

            # Calculate new users (not seen before)
            new_users = current_users - all_users_so_far
            new_users_count = len(new_users)
            new_users_percentage = (new_users_count / active_users_count) * 100 if active_users_count > 0 else 0

            # Calculate churned users (users seen so far who never appear after this day)
            users_after_this_day = set()
            for future_day in range(day + 1, SAMPLE_LENGTH_DAYS):
                if future_day in days_users:
                    users_after_this_day.update(days_users[future_day])

            churned_users = all_users_so_far - users_after_this_day
            churned_users_count = len(churned_users)
            churned_percentage = (churned_users_count / len(all_users_so_far)) * 100 if all_users_so_far else 0

            # Add current day's users to the running set of all users
            all_users_so_far.update(current_users)

            # Store metrics for this day
            user_dynamics.append({
                'day': day + 1,  # 1-indexed for reporting
                'active_users': active_users_count,
                'new_users': new_users_count,
                'new_users_percentage': new_users_percentage,
                'churned_users': churned_users_count,
                'churned_percentage': churned_percentage
            })

        # Filter to days 10+ for final metrics
        days_after_10 = [d for d in user_dynamics if d['day'] >= 10]

        # Calculate average metrics for days 10+
        if days_after_10:
            avg_new_percentage = sum(d['new_users_percentage'] for d in days_after_10) / len(days_after_10)
            avg_churn_percentage = sum(d['churned_percentage'] for d in days_after_10) / len(days_after_10)
        else:
            avg_new_percentage = 0
            avg_churn_percentage = 0

        # Prepare results
        results = {
            'daily_dynamics': user_dynamics,
            'days_10_onwards': days_after_10,
            'avg_new_users_percentage_after_day10': avg_new_percentage,
            'avg_churn_percentage_after_day10': avg_churn_percentage,
            'total_unique_users': len(all_users_so_far)
        }

        return results

    except Exception as e:
        logger.error(f"Error calculating user dynamics: {str(e)}")
        raise

def calculate_avg_interactions(df_sample):
    """Calculate average interactions per user per day."""
    try:
        # Get the start timestamp from the sample data
        start_timestamp = df_sample['publish_date'].min()

        # Create day buckets for the 30-day period
        daily_interactions = {}

        # Initialize data structure to track user interactions per day
        user_interactions_by_day = defaultdict(lambda: defaultdict(int))

        # Track unique active users per day for computing average
        unique_users_by_day = []

        # Process each day in the sample period
        for day in range(SAMPLE_LENGTH_DAYS):
            day_start = start_timestamp + (day * SECONDS_PER_DAY)
            day_end = day_start + SECONDS_PER_DAY

            # Filter data for this day
            day_df = df_sample[(df_sample['publish_date'] >= day_start) &
                             (df_sample['publish_date'] < day_end)]

            # Count interactions per user for this day
            if not day_df.empty:
                user_counts = day_df.groupby('user_id').size()

                # Get unique users for this day
                day_unique_users = day_df['user_id'].nunique()
                unique_users_by_day.append(day_unique_users)

                # Store in our tracking structure
                for user_id, count in user_counts.items():
                    user_interactions_by_day[day][user_id] = count

                # Store overall counts for this day
                daily_interactions[day] = {
                    'total_interactions': len(day_df),
                    'unique_users': len(user_counts),
                    'avg_per_user': len(day_df) / len(user_counts) if len(user_counts) > 0 else 0,
                    'median_per_user': user_counts.median() if len(user_counts) > 0 else 0,
                    'max_per_user': user_counts.max() if len(user_counts) > 0 else 0
                }
            else:
                # No interactions on this day
                unique_users_by_day.append(0)
                daily_interactions[day] = {
                    'total_interactions': 0,
                    'unique_users': 0,
                    'avg_per_user': 0,
                    'median_per_user': 0,
                    'max_per_user': 0
                }

        # Calculate metrics across all days
        days_with_activity = [d for d in daily_interactions.values() if d['unique_users'] > 0]

        if days_with_activity:
            overall_avg_interactions_per_user_per_day = sum([d['avg_per_user'] for d in days_with_activity]) / len(days_with_activity)
            overall_median_interactions_per_user_per_day = np.median([d['median_per_user'] for d in days_with_activity])
        else:
            overall_avg_interactions_per_user_per_day = 0
            overall_median_interactions_per_user_per_day = 0

        # Calculate average unique active users per day
        avg_unique_active_users_per_day = sum(unique_users_by_day) / len(unique_users_by_day) if unique_users_by_day else 0
        median_unique_active_users_per_day = np.median(unique_users_by_day) if unique_users_by_day else 0
        max_unique_active_users_per_day = max(unique_users_by_day) if unique_users_by_day else 0
        min_unique_active_users_per_day = min(unique_users_by_day) if unique_users_by_day else 0

        # Calculate distribution of user activity levels
        user_total_interactions = defaultdict(int)
        user_active_days = defaultdict(int)

        for day, user_data in user_interactions_by_day.items():
            for user_id, count in user_data.items():
                user_total_interactions[user_id] += count
                user_active_days[user_id] += 1

        # Calculate average interactions per active day for each user
        user_avg_per_active_day = {}
        for user_id, total in user_total_interactions.items():
            active_days = user_active_days[user_id]
            user_avg_per_active_day[user_id] = total / active_days if active_days > 0 else 0

        # Convert to series for easier analysis
        avg_interactions_series = pd.Series(user_avg_per_active_day)
        active_days_series = pd.Series(user_active_days)

        # Helper function to calculate optimal bins
        def get_optimal_bins(data_values):
            """Calculate optimal number of bins using multiple methods."""
            if len(data_values) == 0 or len(data_values) < 2:
                return 10

            data_array = np.array(data_values)
            n = len(data_array)

            # Scott's rule
            scott_bins = int(np.ceil((data_array.max() - data_array.min()) / (3.5 * data_array.std() / (n ** (1/3)))))

            # Freedman-Diaconis rule
            q75, q25 = np.percentile(data_array, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                fd_bins = int(np.ceil((data_array.max() - data_array.min()) / (2 * iqr / (n ** (1/3)))))
            else:
                fd_bins = scott_bins

            # Sturges' rule
            sturges_bins = int(np.ceil(np.log2(n) + 1))

            # Use median of the three methods, constrained between 10 and 100
            optimal_bins = int(np.median([scott_bins, fd_bins, sturges_bins]))
            return max(10, min(100, optimal_bins))

        # Create histogram data for distributions with adaptive binning
        interactions_values = list(user_total_interactions.values())
        interactions_bins = get_optimal_bins(interactions_values)
        interactions_distribution = {
            'counts': np.histogram(interactions_values, bins=interactions_bins)[0].tolist(),
            'bin_edges': np.histogram(interactions_values, bins=interactions_bins)[1].tolist()
        }

        active_days_values = list(user_active_days.values())
        active_days_bins = get_optimal_bins(active_days_values)
        active_days_distribution = {
            'counts': np.histogram(active_days_values, bins=active_days_bins)[0].tolist(),
            'bin_edges': np.histogram(active_days_values, bins=active_days_bins)[1].tolist()
        }

        avg_per_day_values = list(user_avg_per_active_day.values())
        avg_per_day_bins = get_optimal_bins(avg_per_day_values)
        avg_per_day_distribution = {
            'counts': np.histogram(avg_per_day_values, bins=avg_per_day_bins)[0].tolist(),
            'bin_edges': np.histogram(avg_per_day_values, bins=avg_per_day_bins)[1].tolist()
        }

        # Create distribution of daily active users
        unique_users_bins = get_optimal_bins(unique_users_by_day)
        unique_users_distribution = {
            'counts': np.histogram(unique_users_by_day, bins=unique_users_bins)[0].tolist(),
            'bin_edges': np.histogram(unique_users_by_day, bins=unique_users_bins)[1].tolist()
        }

        # Prepare final results
        results = {
            'daily_interactions': daily_interactions,
            'overall_avg_interactions_per_user_per_day': overall_avg_interactions_per_user_per_day,
            'overall_median_interactions_per_user_per_day': overall_median_interactions_per_user_per_day,
            'avg_unique_active_users_per_day': avg_unique_active_users_per_day,
            'median_unique_active_users_per_day': median_unique_active_users_per_day,
            'max_unique_active_users_per_day': max_unique_active_users_per_day,
            'min_unique_active_users_per_day': min_unique_active_users_per_day,
            'unique_users_by_day': unique_users_by_day,
            'avg_interactions_per_active_day': {
                'mean': avg_interactions_series.mean(),
                'median': avg_interactions_series.median(),
                'min': avg_interactions_series.min(),
                'max': avg_interactions_series.max()
            },
            'active_days_per_user': {
                'mean': active_days_series.mean(),
                'median': active_days_series.median(),
                'min': active_days_series.min(),
                'max': active_days_series.max()
            },
            'users_with_single_day_activity': sum(1 for days in user_active_days.values() if days == 1),
            'users_with_high_activity': sum(1 for days in user_active_days.values() if days > 10),
            'total_unique_users': len(user_total_interactions),
            'interactions_distribution': interactions_distribution,
            'active_days_distribution': active_days_distribution,
            'avg_per_day_distribution': avg_per_day_distribution,
            'unique_users_distribution': unique_users_distribution
        }

        return results

    except Exception as e:
        logger.error(f"Error calculating average interactions: {str(e)}")
        raise

def analyze_post_distribution(df_sample):
    """Generate distribution of posts and comments per day."""
    try:
        # Get the start timestamp from the sample data
        start_timestamp = df_sample['publish_date'].min()

        # Create data structures to store results
        daily_counts = {}
        post_counts = []
        comment_counts = []

        # Process each day in the sample period
        for day in range(SAMPLE_LENGTH_DAYS):
            day_start = start_timestamp + (day * SECONDS_PER_DAY)
            day_end = day_start + SECONDS_PER_DAY

            # Filter data for this day
            day_df = df_sample[(df_sample['publish_date'] >= day_start) &
                            (df_sample['publish_date'] < day_end)]

            # Get counts for this day
            if not day_df.empty:
                # Count posts and comments
                post_count = day_df[day_df['interaction_type'] == 'posts'].shape[0]
                comment_count = day_df[day_df['interaction_type'] == 'comments'].shape[0]

                # Store for histogram
                post_counts.append(post_count)
                comment_counts.append(comment_count)

                # Store per-user metrics
                user_post_counts = day_df[day_df['interaction_type'] == 'posts'].groupby('user_id').size()
                user_comment_counts = day_df[day_df['interaction_type'] == 'comments'].groupby('user_id').size()

                # Store in results
                daily_counts[day] = {
                    'date': unix_to_date(day_start),
                    'day_num': day + 1,  # 1-indexed for reporting
                    'posts': post_count,
                    'comments': comment_count,
                    'total': post_count + comment_count,
                    'unique_users': day_df['user_id'].nunique(),
                    'users_who_posted': len(user_post_counts),
                    'users_who_commented': len(user_comment_counts),
                    'posts_per_user': post_count / len(user_post_counts) if len(user_post_counts) > 0 else 0,
                    'comments_per_user': comment_count / len(user_comment_counts) if len(user_comment_counts) > 0 else 0,
                    'max_posts_by_user': user_post_counts.max() if len(user_post_counts) > 0 else 0,
                    'max_comments_by_user': user_comment_counts.max() if len(user_comment_counts) > 0 else 0
                }
            else:
                # No activity on this day
                daily_counts[day] = {
                    'date': unix_to_date(day_start),
                    'day_num': day + 1,  # 1-indexed for reporting
                    'posts': 0,
                    'comments': 0,
                    'total': 0,
                    'unique_users': 0,
                    'users_who_posted': 0,
                    'users_who_commented': 0,
                    'posts_per_user': 0,
                    'comments_per_user': 0,
                    'max_posts_by_user': 0,
                    'max_comments_by_user': 0
                }

        # Calculate overall statistics
        daily_metrics = pd.DataFrame(daily_counts).T

        # Get post/comment ratio
        total_posts = daily_metrics['posts'].sum()
        total_comments = daily_metrics['comments'].sum()
        post_comment_ratio = total_comments / total_posts if total_posts > 0 else 0

        # Helper function for optimal binning (reusing from above)
        def get_optimal_bins_daily(data_values):
            """Calculate robust optimal bins for integer-like counts."""
            try:
                import numpy as _np
            except Exception:
                _np = np
            # Coerce to numpy array and drop non-finite
            data_array = _np.asarray(list(data_values), dtype=float)
            data_array = data_array[_np.isfinite(data_array)]
            n = len(data_array)
            if n < 2:
                return 10
            data_min = data_array.min()
            data_max = data_array.max()
            # Sturges as safe baseline
            sturges_bins = int(_np.ceil(_np.log2(n) + 1))
            candidates = [max(1, sturges_bins)]
            # Scott's rule (guard zero std)
            std = float(data_array.std())
            if std > 0:
                width_scott = 3.5 * std / (n ** (1/3))
                if width_scott > 0:
                    scott = (data_max - data_min) / width_scott
                    if _np.isfinite(scott) and scott > 0:
                        candidates.append(int(_np.ceil(scott)))
            # Freedman–Diaconis (guard zero IQR)
            q75, q25 = _np.percentile(data_array, [75, 25])
            iqr = float(q75 - q25)
            if iqr > 0:
                width_fd = 2 * iqr / (n ** (1/3))
                if width_fd > 0:
                    fd = (data_max - data_min) / width_fd
                    if _np.isfinite(fd) and fd > 0:
                        candidates.append(int(_np.ceil(fd)))
            # Median of valid candidates, constrained
            bins = int(_np.median(candidates)) if candidates else 10
            return max(10, min(100, bins))

        # Calculate distributions with adaptive binning
        posts_bins = get_optimal_bins_daily(post_counts)
        posts_distribution = {
            'counts': np.histogram(post_counts, bins=posts_bins)[0].tolist(),
            'bin_edges': np.histogram(post_counts, bins=posts_bins)[1].tolist()
        }

        comments_bins = get_optimal_bins_daily(comment_counts)
        comments_distribution = {
            'counts': np.histogram(comment_counts, bins=comments_bins)[0].tolist(),
            'bin_edges': np.histogram(comment_counts, bins=comments_bins)[1].tolist()
        }

        # Calculate user posting patterns
        user_post_counts = df_sample[df_sample['interaction_type'] == 'posts'].groupby('user_id').size()
        user_comment_counts = df_sample[df_sample['interaction_type'] == 'comments'].groupby('user_id').size()

        # Get distribution of posts per user
        if not user_post_counts.empty:
            user_post_bins = get_optimal_bins_daily(user_post_counts.values)
            posts_per_user_distribution = {
                'counts': np.histogram(user_post_counts, bins=user_post_bins)[0].tolist(),
                'bin_edges': np.histogram(user_post_counts, bins=user_post_bins)[1].tolist()
            }
        else:
            posts_per_user_distribution = {'counts': [], 'bin_edges': []}

        # Get distribution of comments per user
        if not user_comment_counts.empty:
            user_comment_bins = get_optimal_bins_daily(user_comment_counts.values)
            comments_per_user_distribution = {
                'counts': np.histogram(user_comment_counts, bins=user_comment_bins)[0].tolist(),
                'bin_edges': np.histogram(user_comment_counts, bins=user_comment_bins)[1].tolist()
            }
        else:
            comments_per_user_distribution = {'counts': [], 'bin_edges': []}

        # Prepare final results
        results = {
            'daily_counts': daily_counts,
            'total_posts': total_posts,
            'total_comments': total_comments,
            'post_comment_ratio': post_comment_ratio,
            'avg_posts_per_day': total_posts / SAMPLE_LENGTH_DAYS,
            'avg_comments_per_day': total_comments / SAMPLE_LENGTH_DAYS,
            'max_posts_in_a_day': daily_metrics['posts'].max(),
            'max_comments_in_a_day': daily_metrics['comments'].max(),
            'users_who_posted': len(user_post_counts),
            'users_who_commented': len(user_comment_counts),
            'avg_posts_per_user': total_posts / len(user_post_counts) if len(user_post_counts) > 0 else 0,
            'avg_comments_per_user': total_comments / len(user_comment_counts) if len(user_comment_counts) > 0 else 0,
            'max_posts_by_user': user_post_counts.max() if len(user_post_counts) > 0 else 0,
            'max_comments_by_user': user_comment_counts.max() if len(user_comment_counts) > 0 else 0,
            'posts_distribution': posts_distribution,
            'comments_distribution': comments_distribution,
            'posts_per_user_distribution': posts_per_user_distribution,
            'comments_per_user_distribution': comments_per_user_distribution
        }

        return results

    except Exception as e:
        logger.error(f"Error analyzing post distribution: {str(e)}")
        raise

def analyze_toxicity_distribution(df_sample):
    """Generate distribution of toxicity scores."""
    try:
        # Check if toxicity scores are available
        if 'toxicity_toxigen' not in df_sample.columns or df_sample['toxicity_toxigen'].isna().all():
            logger.warning("No toxicity scores available in this sample")
            return {
                'has_toxicity_data': False,
                'message': 'No toxicity scores available in this sample'
            }

        # Get only records with valid toxicity scores
        df_with_toxicity = df_sample.dropna(subset=['toxicity_toxigen'])
        toxicity_coverage = (len(df_with_toxicity) / len(df_sample)) * 100

        logger.info(f"Toxicity score coverage: {toxicity_coverage:.2f}% ({len(df_with_toxicity):,} out of {len(df_sample):,} records)")

        if len(df_with_toxicity) < 10:
            logger.warning("Too few records with toxicity scores for meaningful analysis")
            return {
                'has_toxicity_data': True,
                'message': 'Too few records with toxicity scores for meaningful analysis',
                'toxicity_coverage_pct': toxicity_coverage,
                'records_with_toxicity': len(df_with_toxicity),
                'total_records': len(df_sample)
            }

        # Basic toxicity statistics
        toxicity_stats = {
            'mean': df_with_toxicity['toxicity_toxigen'].mean(),
            'median': df_with_toxicity['toxicity_toxigen'].median(),
            'min': df_with_toxicity['toxicity_toxigen'].min(),
            'max': df_with_toxicity['toxicity_toxigen'].max(),
            'std': df_with_toxicity['toxicity_toxigen'].std()
        }

        # Create toxicity score bins
        bin_edges = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
        toxicity_bins = np.histogram(df_with_toxicity['toxicity_toxigen'], bins=bin_edges)[0]

        # Calculate percentage in each bin
        toxicity_bins_pct = (toxicity_bins / len(df_with_toxicity)) * 100

        # Compare toxicity between posts and comments
        posts_toxicity = df_with_toxicity[df_with_toxicity['interaction_type'] == 'posts']['toxicity_toxigen']
        comments_toxicity = df_with_toxicity[df_with_toxicity['interaction_type'] == 'comments']['toxicity_toxigen']

        # Calculate toxicity distribution by interaction type
        interaction_toxicity = {
            'posts': {
                'count': len(posts_toxicity),
                'mean': posts_toxicity.mean() if len(posts_toxicity) > 0 else None,
                'median': posts_toxicity.median() if len(posts_toxicity) > 0 else None,
                'distribution': np.histogram(posts_toxicity, bins=bin_edges)[0].tolist() if len(posts_toxicity) > 0 else []
            },
            'comments': {
                'count': len(comments_toxicity),
                'mean': comments_toxicity.mean() if len(comments_toxicity) > 0 else None,
                'median': comments_toxicity.median() if len(comments_toxicity) > 0 else None,
                'distribution': np.histogram(comments_toxicity, bins=bin_edges)[0].tolist() if len(comments_toxicity) > 0 else []
            }
        }

        # Calculate high toxicity threshold (e.g., top 10% of scores)
        high_toxicity_threshold = np.percentile(df_with_toxicity['toxicity_toxigen'], 90)
        very_high_toxicity_threshold = np.percentile(df_with_toxicity['toxicity_toxigen'], 95)

        # Count high toxicity posts and comments
        high_toxicity_posts = posts_toxicity[posts_toxicity >= high_toxicity_threshold].count()
        high_toxicity_comments = comments_toxicity[comments_toxicity >= high_toxicity_threshold].count()
        very_high_toxicity_posts = posts_toxicity[posts_toxicity >= very_high_toxicity_threshold].count()
        very_high_toxicity_comments = comments_toxicity[comments_toxicity >= very_high_toxicity_threshold].count()

        # Calculate user toxicity metrics
        user_toxicity = {}
        for user_id, user_data in df_with_toxicity.groupby('user_id'):
            user_toxicity[user_id] = {
                'mean_toxicity': user_data['toxicity_toxigen'].mean(),
                'max_toxicity': user_data['toxicity_toxigen'].max(),
                'interaction_count': len(user_data),
                'high_toxicity_count': len(user_data[user_data['toxicity_toxigen'] >= high_toxicity_threshold])
            }

        # Identify users with consistently high toxicity
        min_interactions = 5  # Minimum interactions to be considered for this metric
        high_toxicity_users = [uid for uid, data in user_toxicity.items()
                              if data['interaction_count'] >= min_interactions and
                              data['mean_toxicity'] >= high_toxicity_threshold]

        # Create distribution of user mean toxicity scores
        user_mean_toxicity = [data['mean_toxicity'] for data in user_toxicity.values()
                             if data['interaction_count'] >= min_interactions]

        user_toxicity_distribution = {}
        if user_mean_toxicity:
            user_toxicity_distribution = {
                'counts': np.histogram(user_mean_toxicity, bins=bin_edges)[0].tolist(),
                'bin_edges': bin_edges.tolist()
            }

        # Calculate daily toxicity averages
        start_timestamp = df_sample['publish_date'].min()
        daily_toxicity = {}

        for day in range(SAMPLE_LENGTH_DAYS):
            day_start = start_timestamp + (day * SECONDS_PER_DAY)
            day_end = day_start + SECONDS_PER_DAY

            day_df = df_with_toxicity[(df_with_toxicity['publish_date'] >= day_start) &
                                    (df_with_toxicity['publish_date'] < day_end)]

            if not day_df.empty:
                daily_toxicity[day] = {
                    'date': unix_to_date(day_start),
                    'day_num': day + 1,  # 1-indexed for reporting
                    'mean_toxicity': day_df['toxicity_toxigen'].mean(),
                    'median_toxicity': day_df['toxicity_toxigen'].median(),
                    'high_toxicity_count': len(day_df[day_df['toxicity_toxigen'] >= high_toxicity_threshold]),
                    'very_high_toxicity_count': len(day_df[day_df['toxicity_toxigen'] >= very_high_toxicity_threshold]),
                    'total_interactions': len(day_df)
                }
            else:
                daily_toxicity[day] = {
                    'date': unix_to_date(day_start),
                    'day_num': day + 1,
                    'mean_toxicity': None,
                    'median_toxicity': None,
                    'high_toxicity_count': 0,
                    'very_high_toxicity_count': 0,
                    'total_interactions': 0
                }

        # Prepare final results
        results = {
            'has_toxicity_data': True,
            'toxicity_coverage_pct': toxicity_coverage,
            'records_with_toxicity': len(df_with_toxicity),
            'total_records': len(df_sample),
            'toxicity_stats': toxicity_stats,
            'toxicity_distribution': {
                'counts': toxicity_bins.tolist(),
                'counts_pct': toxicity_bins_pct.tolist(),
                'bin_edges': bin_edges.tolist()
            },
            'interaction_toxicity': interaction_toxicity,
            'high_toxicity_threshold': high_toxicity_threshold,
            'very_high_toxicity_threshold': very_high_toxicity_threshold,
            'high_toxicity_counts': {
                'posts': high_toxicity_posts,
                'comments': high_toxicity_comments,
                'total': high_toxicity_posts + high_toxicity_comments,
                'pct_of_posts': (high_toxicity_posts / len(posts_toxicity) * 100) if len(posts_toxicity) > 0 else 0,
                'pct_of_comments': (high_toxicity_comments / len(comments_toxicity) * 100) if len(comments_toxicity) > 0 else 0,
                'pct_of_all': ((high_toxicity_posts + high_toxicity_comments) / len(df_with_toxicity) * 100) if len(df_with_toxicity) > 0 else 0
            },
            'very_high_toxicity_counts': {
                'posts': very_high_toxicity_posts,
                'comments': very_high_toxicity_comments,
                'total': very_high_toxicity_posts + very_high_toxicity_comments,
                'pct_of_posts': (very_high_toxicity_posts / len(posts_toxicity) * 100) if len(posts_toxicity) > 0 else 0,
                'pct_of_comments': (very_high_toxicity_comments / len(comments_toxicity) * 100) if len(comments_toxicity) > 0 else 0,
                'pct_of_all': ((very_high_toxicity_posts + very_high_toxicity_comments) / len(df_with_toxicity) * 100) if len(df_with_toxicity) > 0 else 0
            },
            'user_toxicity': {
                'high_toxicity_users_count': len(high_toxicity_users),
                'high_toxicity_users_pct': (len(high_toxicity_users) / len(user_toxicity) * 100) if len(user_toxicity) > 0 else 0,
                'user_toxicity_distribution': user_toxicity_distribution
            },
            'daily_toxicity': daily_toxicity
        }

        return results

    except Exception as e:
        logger.error(f"Error analyzing toxicity distribution: {str(e)}")
        raise

def analyze_thread_length(df_sample):
    """Calculate thread length distribution (post + comments)."""
    try:
        # First, identify all posts (threads)
        posts_df = df_sample[df_sample['interaction_type'] == 'posts']

        if posts_df.empty:
            logger.warning("No posts found in sample, cannot analyze thread lengths")
            return {
                'success': False,
                'message': 'No posts found in sample'
            }

        # Identify all comments
        comments_df = df_sample[df_sample['interaction_type'] == 'comments']

        if comments_df.empty:
            logger.warning("No comments found in sample, all threads have length 1")
            return {
                'success': True,
                'message': 'No comments found in sample, all threads have length 1',
                'thread_count': len(posts_df),
                'avg_thread_length': 1.0,
                'median_thread_length': 1.0,
                'max_thread_length': 1,
                'thread_length_distribution': {
                    'counts': [len(posts_df)],
                    'bin_edges': [1, 2]  # Just one bin for thread length 1
                }
            }

        # Map each comment to its root post (thread)
        # Initialize thread lengths dictionary with all posts (starting length 1 for the post itself)
        thread_lengths = {post_id: 1 for post_id in posts_df['post_id']}

        # Count comments per thread
        # Use parent_id to track replies
        post_id_to_parent_map = {}

        # First pass: map comments directly to posts
        direct_replies = comments_df[comments_df['parent_id'].isin(posts_df['post_id'])]
        for _, comment in direct_replies.iterrows():
            thread_post_id = comment['parent_id']
            if thread_post_id in thread_lengths:
                thread_lengths[thread_post_id] += 1
            else:
                # This can happen if the post is not in our sample
                # but the comment is. We'll track it as its own thread.
                thread_lengths[thread_post_id] = 1

            # Map this comment's ID to its post for use with nested replies
            post_id_to_parent_map[comment['post_id']] = thread_post_id

        # Second pass: map comments that are replies to other comments
        # We may need to do multiple passes to handle deeply nested comments
        nested_comments = comments_df[~comments_df['parent_id'].isin(posts_df['post_id'])]

        # Set a limit to prevent infinite loops
        max_iterations = 10
        iterations = 0
        remaining_comments = len(nested_comments)

        while remaining_comments > 0 and iterations < max_iterations:
            iterations += 1
            mapped_in_this_iteration = 0

            for _, comment in nested_comments.iterrows():
                parent_id = comment['parent_id']

                # If we know which post this comment's parent belongs to
                if parent_id in post_id_to_parent_map:
                    thread_post_id = post_id_to_parent_map[parent_id]

                    # Increment the thread length
                    if thread_post_id in thread_lengths:
                        thread_lengths[thread_post_id] += 1
                    else:
                        thread_lengths[thread_post_id] = 1

                    # Map this comment to its thread for future iterations
                    post_id_to_parent_map[comment['post_id']] = thread_post_id
                    mapped_in_this_iteration += 1

            # Update remaining comments count
            remaining_comments -= mapped_in_this_iteration

            # If we didn't map any comments in this iteration, we're stuck
            if mapped_in_this_iteration == 0 and remaining_comments > 0:
                logger.warning(f"Could not map {remaining_comments} comments to their threads after {iterations} iterations")
                break

        # Convert the thread lengths to a list for histogram calculation
        thread_length_values = list(thread_lengths.values())

        # Calculate basic statistics
        avg_thread_length = sum(thread_length_values) / len(thread_length_values)
        median_thread_length = np.median(thread_length_values)
        max_thread_length = max(thread_length_values)

        # Create histogram bins based on the range of thread lengths
        max_bin = min(50, max_thread_length + 1)  # Limit to 50 bins
        bin_edges = list(range(1, max_bin + 1))

        # Calculate histogram
        hist_counts, _ = np.histogram(thread_length_values, bins=bin_edges)

        # Calculate percentiles for reporting
        percentiles = [50, 75, 90, 95, 99]
        percentile_values = np.percentile(thread_length_values, percentiles)
        percentile_dict = {f'p{p}': percentile_values[i] for i, p in enumerate(percentiles)}

        # Calculate thread length categories
        short_threads = sum(1 for length in thread_length_values if length == 1)
        medium_threads = sum(1 for length in thread_length_values if 1 < length <= 5)
        long_threads = sum(1 for length in thread_length_values if 5 < length <= 20)
        very_long_threads = sum(1 for length in thread_length_values if length > 20)

        # Prepare results
        results = {
            'success': True,
            'thread_count': len(thread_lengths),
            'avg_thread_length': avg_thread_length,
            'median_thread_length': median_thread_length,
            'max_thread_length': max_thread_length,
            'percentiles': percentile_dict,
            'thread_categories': {
                'no_comments': short_threads,
                'few_comments': medium_threads,
                'many_comments': long_threads,
                'very_many_comments': very_long_threads,
                'no_comments_pct': (short_threads / len(thread_lengths)) * 100,
                'few_comments_pct': (medium_threads / len(thread_lengths)) * 100,
                'many_comments_pct': (long_threads / len(thread_lengths)) * 100,
                'very_many_comments_pct': (very_long_threads / len(thread_lengths)) * 100
            },
            'thread_length_distribution': {
                'counts': hist_counts.tolist(),
                'bin_edges': bin_edges
            },
            'unmapped_comments': remaining_comments
        }

        return results

    except Exception as e:
        logger.error(f"Error analyzing thread length: {str(e)}")
        raise

def analyze_thread_activity(df_sample):
    """Calculate distribution of days each thread was active."""
    try:
        # First, identify all posts (threads)
        posts_df = df_sample[df_sample['interaction_type'] == 'posts']

        if posts_df.empty:
            logger.warning("No posts found in sample, cannot analyze thread activity")
            return {
                'success': False,
                'message': 'No posts found in sample'
            }

        # Identify all comments
        comments_df = df_sample[df_sample['interaction_type'] == 'comments']

        if comments_df.empty:
            logger.warning("No comments found in sample, all threads are one-day only")
            # Process just the posts for one-day threads
            post_days = defaultdict(set)
            for _, post in posts_df.iterrows():
                post_day = (post['publish_date'] - df_sample['publish_date'].min()) // SECONDS_PER_DAY
                post_days[post['post_id']].add(post_day)

            thread_days = {pid: len(days) for pid, days in post_days.items()}
            thread_duration_values = list(thread_days.values())

            return {
                'success': True,
                'message': 'No comments found in sample, all threads are active only on their posting day',
                'thread_count': len(thread_days),
                'avg_active_days': np.mean(thread_duration_values),
                'median_active_days': np.median(thread_duration_values),
                'max_active_days': max(thread_duration_values),
                'min_active_days': min(thread_duration_values),
                'active_days_distribution': {
                    'counts': [len(thread_days)],
                    'bin_edges': [1, 2]  # Just one bin for 1 day
                }
            }

        # Create thread ID to post ID mapping
        # We need to map comments back to the root post in a thread
        thread_mapping = {}
        post_to_thread = {}

        # All posts start as their own thread ID
        for _, post in posts_df.iterrows():
            post_id = post['post_id']
            post_to_thread[post_id] = post_id
            thread_mapping[post_id] = post_id

        # Map direct comment replies to their parent posts
        direct_replies = comments_df[comments_df['parent_id'].isin(posts_df['post_id'])]
        for _, comment in direct_replies.iterrows():
            parent_id = comment['parent_id']
            if parent_id in post_to_thread:
                thread_id = post_to_thread[parent_id]
                thread_mapping[comment['post_id']] = thread_id

        # Map nested comments to root thread through multiple passes
        nested_comments = comments_df[~comments_df['parent_id'].isin(posts_df['post_id'])]
        max_iterations = 10
        iterations = 0
        remaining = len(nested_comments)

        while remaining > 0 and iterations < max_iterations:
            iterations += 1
            mapped_count = 0

            for _, comment in nested_comments.iterrows():
                comment_id = comment['post_id']
                if comment_id in thread_mapping:
                    continue  # Already mapped

                parent_id = comment['parent_id']
                if parent_id in thread_mapping:
                    thread_id = thread_mapping[parent_id]
                    thread_mapping[comment_id] = thread_id
                    mapped_count += 1

            remaining -= mapped_count
            if mapped_count == 0:
                break  # No progress made, exit loop

        # Now track activity days for each thread
        thread_days = defaultdict(set)

        # First, add days for posts
        for _, post in posts_df.iterrows():
            post_id = post['post_id']
            thread_id = thread_mapping.get(post_id, post_id)  # Default to self if not mapped
            post_day = (post['publish_date'] - df_sample['publish_date'].min()) // SECONDS_PER_DAY
            thread_days[thread_id].add(post_day)

        # Then add days for comments (both direct and nested)
        for _, comment in comments_df.iterrows():
            comment_id = comment['post_id']
            thread_id = thread_mapping.get(comment_id)
            if thread_id:  # Skip if we couldn't map this comment
                comment_day = (comment['publish_date'] - df_sample['publish_date'].min()) // SECONDS_PER_DAY
                thread_days[thread_id].add(comment_day)

        # Calculate thread durations (number of unique days active)
        thread_duration = {thread_id: len(days) for thread_id, days in thread_days.items()}

        # For each thread, calculate the span from first to last activity
        thread_first_last = {}
        for thread_id, days in thread_days.items():
            if days:  # Skip empty sets
                thread_first_last[thread_id] = {
                    'first_day': min(days),
                    'last_day': max(days),
                    'span_days': max(days) - min(days) + 1,  # +1 because both endpoints are inclusive
                    'active_days': len(days),
                    'activity_density': len(days) / (max(days) - min(days) + 1) if max(days) != min(days) else 1.0
                }

        # Get values as lists for statistics
        duration_values = list(thread_duration.values())
        span_values = [data['span_days'] for data in thread_first_last.values()]
        density_values = [data['activity_density'] for data in thread_first_last.values()]

        # Create histograms with optimal binning
        def get_optimal_bins_thread(data_values):
            """Calculate optimal number of bins for thread activity data."""
            if len(data_values) == 0 or len(data_values) < 2:
                return 10

            data_array = np.array(data_values)
            n = len(data_array)

            # For integer-valued data like days, use different strategy
            unique_values = len(np.unique(data_array))
            max_val = int(data_array.max())
            min_val = int(data_array.min())

            # If we have few unique values, use one bin per value
            if unique_values <= 20:
                return list(range(min_val, max_val + 2))

            # Otherwise use standard optimal binning
            scott_bins = int(np.ceil((max_val - min_val) / (3.5 * data_array.std() / (n ** (1/3)))))
            sturges_bins = int(np.ceil(np.log2(n) + 1))
            optimal_bins = max(10, min(50, int(np.median([scott_bins, sturges_bins]))))

            return optimal_bins

        duration_bin_edges = get_optimal_bins_thread(duration_values)
        duration_hist, duration_bin_edges = np.histogram(duration_values, bins=duration_bin_edges)

        # Calculate percentiles
        percentiles = [50, 75, 90, 95, 99]
        duration_percentiles = np.percentile(duration_values, percentiles) if duration_values else [1] * len(percentiles)
        span_percentiles = np.percentile(span_values, percentiles) if span_values else [1] * len(percentiles)

        # Count threads by duration category
        one_day_threads = sum(1 for d in duration_values if d == 1)
        short_threads = sum(1 for d in duration_values if 1 < d <= 3)
        medium_threads = sum(1 for d in duration_values if 3 < d <= 7)
        long_threads = sum(1 for d in duration_values if d > 7)

        # Prepare results
        results = {
            'success': True,
            'thread_count': len(thread_days),
            'mapped_threads': len(thread_duration),
            'unmapped_comments': remaining,
            'avg_active_days': np.mean(duration_values),
            'median_active_days': np.median(duration_values),
            'max_active_days': max(duration_values),
            'min_active_days': min(duration_values),
            'percentiles': {
                'active_days': {f'p{p}': duration_percentiles[i] for i, p in enumerate(percentiles)},
                'span_days': {f'p{p}': span_percentiles[i] for i, p in enumerate(percentiles)} if span_values else {}
            },
            'avg_span_days': np.mean(span_values) if span_values else 1.0,
            'median_span_days': np.median(span_values) if span_values else 1.0,
            'avg_activity_density': np.mean(density_values) if density_values else 1.0,
            'thread_categories': {
                'one_day_only': one_day_threads,
                'short_duration': short_threads,
                'medium_duration': medium_threads,
                'long_duration': long_threads,
                'one_day_pct': (one_day_threads / len(duration_values)) * 100,
                'short_duration_pct': (short_threads / len(duration_values)) * 100,
                'medium_duration_pct': (medium_threads / len(duration_values)) * 100,
                'long_duration_pct': (long_threads / len(duration_values)) * 100
            },
            'active_days_distribution': {
                'counts': duration_hist.tolist(),
                'bin_edges': duration_bin_edges
            }
        }

        return results

    except Exception as e:
        logger.error(f"Error analyzing thread activity: {str(e)}")
        raise

def save_distribution_plot(data, title, xlabel, ylabel, filename, bins=30):
    """Generate and save enhanced histogram with better visualization options."""
    try:
        # Extract data values for analysis
        if isinstance(data, dict) and 'counts' in data and 'bin_edges' in data:
            counts = np.array(data['counts'])
            bin_edges = np.array(data['bin_edges'])

            # Reconstruct original data for histogram plotting
            data_values = []
            for i in range(len(counts)):
                bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
                data_values.extend([bin_center] * int(counts[i]))
            data_values = np.array(data_values)
        else:
            # If data is just a list of values, use directly
            if not isinstance(data, (list, np.ndarray)):
                try:
                    data = list(data)
                except:
                    raise ValueError(f"Data must be a list, array, or histogram dict, got {type(data)}")

            if not data:
                logger.warning(f"No data to plot for {filename}, skipping")
                return

            data_values = np.array(data)

        # Check if this is a toxicity plot to apply special styling
        is_toxicity_plot = 'toxicity' in title.lower() or 'toxicity' in xlabel.lower()

        if is_toxicity_plot:
            # Use the specific style for toxicity plots with minimalistic theme
            plt.figure(figsize=(10, 6))
            plt.style.use('default')  # Reset to default style
            sns.histplot(data_values, kde=True, bins=bins, color='darkblue')
            plt.title('r/technology sample', fontsize=15)
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.axvline(x=0.5, color='red', linestyle='--', label='Toxic Threshold (0.5)')
            plt.axvline(x=0.25, color='orange', linestyle='--', label='Mild Threshold (0.25)')
            plt.legend()
            # Add mean toxicity text below the legend
            mean_toxicity = np.mean(data_values)
            plt.text(0.7, 250000, f'Mean Toxicity: {mean_toxicity:.4f}',
                    fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            # Remove all grid lines for minimalistic look
            plt.grid(False)
            plt.gca().set_axisbelow(True)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            # Create figure with subplots for histogram and CDF for non-toxicity plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Determine if we should use log scale based on data distribution
            data_range = data_values.max() - data_values.min()
            skewness = abs(np.mean(data_values) - np.median(data_values)) / np.std(data_values) if np.std(data_values) > 0 else 0
            use_log = skewness > 1 and data_range > 100 and data_values.min() > 0

            if use_log:
                # Use log-spaced bins for highly skewed data
                log_bins = np.logspace(np.log10(data_values.min()), np.log10(data_values.max()), bins)
                counts, bin_edges, _ = ax1.hist(data_values, bins=log_bins, alpha=0.7,
                                              edgecolor='black', linewidth=0.5)
                ax1.set_xscale('log')
            else:
                counts, bin_edges, _ = ax1.hist(data_values, bins=bins, alpha=0.7,
                                              edgecolor='black', linewidth=0.5)
            max_count = max(counts) if counts.size > 0 else 0

            # Histogram styling
            ax1.set_xlabel(xlabel, fontsize=12)
            ax1.set_ylabel(ylabel, fontsize=12)
            ax1.set_title(f"{title} - Distribution", fontsize=13, fontweight='bold')
            ax1.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
            ax1.grid(axis='y', linestyle='--', alpha=0.3)
            ax1.set_ylim(0, max_count * 1.05)

            # Add statistics text box on histogram
            if len(data_values) > 0:
                stats_text = f'n = {len(data_values):,}\n'
                stats_text += f'Mean = {np.mean(data_values):.2f}\n'
                stats_text += f'Median = {np.median(data_values):.2f}\n'
                stats_text += f'Std = {np.std(data_values):.2f}'
                ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # CDF plot
            if len(data_values) > 0:
                sorted_data = np.sort(data_values)
                y_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                ax2.plot(sorted_data, y_cdf, linewidth=2, color='darkblue')
                ax2.set_xlabel(xlabel, fontsize=12)
                ax2.set_ylabel('Cumulative Probability', fontsize=12)
                ax2.set_title(f"{title} - Cumulative Distribution", fontsize=13, fontweight='bold')
                ax2.grid(True, linestyle='--', alpha=0.3)
                ax2.set_ylim(0, 1)

                # Add percentile lines
                for p in [25, 50, 75, 90, 95]:
                    percentile_val = np.percentile(data_values, p)
                    ax2.axvline(percentile_val, color='red', linestyle=':', alpha=0.7, linewidth=1)
                    ax2.text(percentile_val, p/100, f'P{p}', rotation=90, fontsize=8,
                            verticalalignment='bottom', horizontalalignment='right')

            plt.tight_layout()

            # Save the figure
            plt.savefig(filename, dpi=300)
            plt.close()

        logger.info(f"Saved distribution plot to {filename}")

    except Exception as e:
        logger.error(f"Error saving distribution plot: {str(e)}")
        # Don't re-raise - plotting errors shouldn't stop the script


def save_time_series_plot(data, dates, title, xlabel, ylabel, filename):
    """Generate and save a time series plot."""
    try:
        plt.figure(figsize=(12, 6))

        # Create the line plot
        plt.plot(range(len(data)), data, marker='o', linestyle='-', alpha=0.7, markersize=4)

        # Add labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        # Format y-axis with comma separators for large numbers
        plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)

        # If we have dates for the x-axis, use them
        if dates and len(dates) == len(data):
            # Plot only a subset of tick labels to avoid overcrowding
            step = max(1, len(dates) // 10)  # Show ~10 tick labels
            plt.xticks(
                range(0, len(dates), step),
                [dates[i] for i in range(0, len(dates), step)],
                rotation=45,
                ha='right'
            )

        # Set a reasonable y limit
        max_val = max(data) if data else 0
        plt.ylim(0, max_val * 1.1)  # Add 10% padding

        # Improve layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(filename, dpi=300)
        plt.close()

        logger.info(f"Saved time series plot to {filename}")

    except Exception as e:
        logger.error(f"Error saving time series plot: {str(e)}")
        # Don't re-raise - plotting errors shouldn't stop the script


def save_time_series_with_ma_plot(data, dates, window_size, title, xlabel, ylabel, filename):
    """Generate and save a time series plot with moving average."""
    try:
        plt.figure(figsize=(12, 6))

        # Calculate the moving average if we have enough data points
        if len(data) >= window_size:
            # Calculate the moving average
            ma_data = []
            for i in range(len(data)):
                if i < window_size - 1:
                    # Not enough data points yet for the full window
                    ma_data.append(None)
                else:
                    # Calculate average of the window
                    window = data[i - window_size + 1:i + 1]
                    ma_data.append(sum(window) / window_size)

            # Create the line plots
            plt.plot(range(len(data)), data, marker='o', linestyle='-', alpha=0.5, markersize=3, label='Daily')

            # Plot the moving average, skipping the initial None values
            valid_indices = [i for i, val in enumerate(ma_data) if val is not None]
            valid_values = [ma_data[i] for i in valid_indices]
            plt.plot(valid_indices, valid_values, linewidth=2, alpha=0.8,
                     label=f'{window_size}-Day Moving Average')

            plt.legend()
        else:
            # Not enough data for moving average, just plot the raw data
            plt.plot(range(len(data)), data, marker='o', linestyle='-', alpha=0.7, markersize=4)

        # Add labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        # Format y-axis with comma separators for large numbers
        plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)

        # If we have dates for the x-axis, use them
        if dates and len(dates) == len(data):
            # Plot only a subset of tick labels to avoid overcrowding
            step = max(1, len(dates) // 10)  # Show ~10 tick labels
            plt.xticks(
                range(0, len(dates), step),
                [dates[i] for i in range(0, len(dates), step)],
                rotation=45,
                ha='right'
            )

        # Set a reasonable y limit
        max_val = max(data) if data else 0
        plt.ylim(0, max_val * 1.1)  # Add 10% padding

        # Improve layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(filename, dpi=300)
        plt.close()

        logger.info(f"Saved time series with MA plot to {filename}")

    except Exception as e:
        logger.error(f"Error saving time series with MA plot: {str(e)}")
        # Don't re-raise - plotting errors shouldn't stop the script

def save_results_text(results, filename):
    """Save analysis results to text file."""
    try:
        # Safe formatting helpers
        def fmtf(x, nd=4):
            try:
                import math
                v = float(x)
                return f"{v:.{nd}f}" if not math.isnan(v) else "n/a"
            except Exception:
                return "n/a"

        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Voat Technology Analysis Results\n")
            f.write("=" * 80 + "\n\n")

            # General sample information
            if 'sample_info' in results:
                sample_info = results['sample_info']
                f.write(f"Sample Period: {sample_info.get('start_date', 'Unknown')} to {sample_info.get('end_date', 'Unknown')}\n")
                f.write(f"Total Records: {sample_info.get('record_count', 0):,}\n\n")

            # Active Users Analysis
            if 'active_users' in results:
                users = results['active_users']
                f.write("===== ACTIVE USERS =====\n")
                f.write(f"Unique Users: {users.get('unique_users', 0):,}\n")
                f.write(f"Low Activity Users (1-2 interactions): {users.get('low_activity_users', 0):,} ")
                f.write(f"({users.get('low_activity_users', 0) / users.get('unique_users', 1) * 100:.1f}%)\n")
                f.write(f"Medium Activity Users (3-10 interactions): {users.get('medium_activity_users', 0):,} ")
                f.write(f"({users.get('medium_activity_users', 0) / users.get('unique_users', 1) * 100:.1f}%)\n")
                f.write(f"High Activity Users (>10 interactions): {users.get('high_activity_users', 0):,} ")
                f.write(f"({users.get('high_activity_users', 0) / users.get('unique_users', 1) * 100:.1f}%)\n")
                f.write(f"One-Day-Only Users: {users.get('one_day_users', 0):,} ")
                f.write(f"({users.get('one_day_percentage', 0):.1f}%)\n")
                f.write(f"Average Interactions per User: {users.get('avg_interactions_per_user', 0):.2f}\n")
                f.write(f"Median Interactions per User: {users.get('median_interactions_per_user', 0):.2f}\n")
                f.write(f"Average Activity Span: {users.get('avg_activity_span_days', 0):.2f} days\n\n")

            # User Dynamics Analysis
            if 'user_dynamics' in results:
                dynamics = results['user_dynamics']
                f.write("===== USER DYNAMICS =====\n")
                f.write(f"Average New Users Percentage (days 10+): {dynamics.get('avg_new_users_percentage_after_day10', 0):.2f}%\n")
                f.write(f"Average Churn Percentage (days 10+): {dynamics.get('avg_churn_percentage_after_day10', 0):.2f}%\n")
                f.write(f"Total Unique Users: {dynamics.get('total_unique_users', 0):,}\n\n")

            # Average Interactions Analysis
            if 'avg_interactions' in results:
                interactions = results['avg_interactions']
                f.write("===== INTERACTIONS PER USER =====\n")
                f.write(f"Overall Average Interactions per User per Day: {interactions.get('overall_avg_interactions_per_user_per_day', 0):.2f}\n")
                f.write(f"Overall Median Interactions per User per Day: {interactions.get('overall_median_interactions_per_user_per_day', 0):.2f}\n\n")

                # Add a dedicated section for unique active users per day
                f.write("===== UNIQUE ACTIVE USERS PER DAY =====\n")
                f.write(f"Average Unique Active Users per Day: {interactions.get('avg_unique_active_users_per_day', 0):.2f}\n")
                f.write(f"Median Unique Active Users per Day: {interactions.get('median_unique_active_users_per_day', 0):.2f}\n")
                f.write(f"Maximum Unique Active Users in a Day: {interactions.get('max_unique_active_users_per_day', 0):.2f}\n")
                f.write(f"Minimum Unique Active Users in a Day: {interactions.get('min_unique_active_users_per_day', 0):.2f}\n\n")

                if 'avg_interactions_per_active_day' in interactions:
                    avg_per_day = interactions['avg_interactions_per_active_day']
                    f.write(f"Average Interactions on Active Days: {avg_per_day.get('mean', 0):.2f}\n")
                    f.write(f"Median Interactions on Active Days: {avg_per_day.get('median', 0):.2f}\n")
                    f.write(f"Maximum Interactions on Active Days: {avg_per_day.get('max', 0):.2f}\n")

                if 'active_days_per_user' in interactions:
                    active_days = interactions['active_days_per_user']
                    f.write(f"Average Active Days per User: {active_days.get('mean', 0):.2f}\n")
                    f.write(f"Median Active Days per User: {active_days.get('median', 0):.2f}\n")
                    f.write(f"Maximum Active Days per User: {active_days.get('max', 0):.2f}\n")

                f.write(f"Users with Single Day Activity: {interactions.get('users_with_single_day_activity', 0):,}\n")
                f.write(f"Users with High Activity (>10 days): {interactions.get('users_with_high_activity', 0):,}\n\n")

            # Post Distribution Analysis
            if 'post_distribution' in results:
                posts = results['post_distribution']
                f.write("===== POSTS AND COMMENTS =====\n")
                f.write(f"Total Posts: {posts.get('total_posts', 0):,}\n")
                f.write(f"Total Comments: {posts.get('total_comments', 0):,}\n")
                f.write(f"Post-Comment Ratio: 1:{posts.get('post_comment_ratio', 0):.2f}\n")
                f.write(f"Average Posts per Day: {posts.get('avg_posts_per_day', 0):.2f}\n")
                f.write(f"Average Comments per Day: {posts.get('avg_comments_per_day', 0):.2f}\n")
                f.write(f"Maximum Posts in a Day: {posts.get('max_posts_in_a_day', 0):,}\n")
                f.write(f"Maximum Comments in a Day: {posts.get('max_comments_in_a_day', 0):,}\n")
                f.write(f"Unique Users Who Posted: {posts.get('users_who_posted', 0):,}\n")
                f.write(f"Unique Users Who Commented: {posts.get('users_who_commented', 0):,}\n")
                f.write(f"Average Posts per User: {posts.get('avg_posts_per_user', 0):.2f}\n")
                f.write(f"Average Comments per User: {posts.get('avg_comments_per_user', 0):.2f}\n")
                f.write(f"Maximum Posts by a Single User: {posts.get('max_posts_by_user', 0):,}\n")
                f.write(f"Maximum Comments by a Single User: {posts.get('max_comments_by_user', 0):,}\n\n")

            # Toxicity Analysis
            if 'toxicity' in results:
                toxicity = results['toxicity']

                # Check if we have toxicity data
                if not toxicity.get('has_toxicity_data', False):
                    f.write("===== TOXICITY ANALYSIS =====\n")
                    f.write(f"No toxicity data available for this sample.\n\n")
                else:
                    f.write("===== TOXICITY ANALYSIS =====\n")
                    f.write(f"Toxicity Coverage: {toxicity.get('toxicity_coverage_pct', 0):.2f}% ")
                    f.write(f"({toxicity.get('records_with_toxicity', 0):,} out of {toxicity.get('total_records', 0):,} records)\n")

                    if 'toxicity_stats' in toxicity:
                        stats = toxicity['toxicity_stats']
                        f.write(f"Mean Toxicity Score: {fmtf(stats.get('mean'))}\n")
                        f.write(f"Median Toxicity Score: {fmtf(stats.get('median'))}\n")
                        f.write(f"Min Toxicity Score: {fmtf(stats.get('min'))}\n")
                        f.write(f"Max Toxicity Score: {fmtf(stats.get('max'))}\n")
                        f.write(f"Standard Deviation: {fmtf(stats.get('std'))}\n")

                    if 'interaction_toxicity' in toxicity:
                        interaction = toxicity['interaction_toxicity']
                        f.write("\nToxicity by Interaction Type:\n")
                        posts_stats = interaction.get('posts', {})
                        comments_stats = interaction.get('comments', {})
                        f.write(f"  Posts: Mean {fmtf(posts_stats.get('mean'))}, ")
                        f.write(f"Median {fmtf(posts_stats.get('median'))}, ")
                        f.write(f"Count {posts_stats.get('count', 0):,}\n")
                        f.write(f"  Comments: Mean {fmtf(comments_stats.get('mean'))}, ")
                        f.write(f"Median {fmtf(comments_stats.get('median'))}, ")
                        f.write(f"Count {comments_stats.get('count', 0):,}\n")

                    if 'high_toxicity_counts' in toxicity:
                        high_tox = toxicity['high_toxicity_counts']
                        very_high_tox = toxicity.get('very_high_toxicity_counts', {})
                        f.write("\nHigh Toxicity Content:\n")
                        f.write(f"  High Toxicity Threshold: {toxicity.get('high_toxicity_threshold', 0):.4f}\n")
                        f.write(f"  High Toxicity Posts: {high_tox.get('posts', 0):,} ({high_tox.get('pct_of_posts', 0):.2f}%)\n")
                        f.write(f"  High Toxicity Comments: {high_tox.get('comments', 0):,} ({high_tox.get('pct_of_comments', 0):.2f}%)\n")
                        f.write(f"  Total High Toxicity: {high_tox.get('total', 0):,} ({high_tox.get('pct_of_all', 0):.2f}%)\n")

                        if very_high_tox:
                            f.write(f"\n  Very High Toxicity Threshold: {toxicity.get('very_high_toxicity_threshold', 0):.4f}\n")
                            f.write(f"  Very High Toxicity Posts: {very_high_tox.get('posts', 0):,} ({very_high_tox.get('pct_of_posts', 0):.2f}%)\n")
                            f.write(f"  Very High Toxicity Comments: {very_high_tox.get('comments', 0):,} ({very_high_tox.get('pct_of_comments', 0):.2f}%)\n")
                            f.write(f"  Total Very High Toxicity: {very_high_tox.get('total', 0):,} ({very_high_tox.get('pct_of_all', 0):.2f}%)\n")

                    if 'user_toxicity' in toxicity:
                        user_tox = toxicity['user_toxicity']
                        f.write("\nUser Toxicity Metrics:\n")
                        f.write(f"  Users with Consistently High Toxicity: {user_tox.get('high_toxicity_users_count', 0):,} ")
                        f.write(f"({user_tox.get('high_toxicity_users_pct', 0):.2f}%)\n\n")

            # Thread Length Analysis
            if 'thread_length' in results:
                threads = results['thread_length']

                if not threads.get('success', False):
                    f.write("===== THREAD LENGTH ANALYSIS =====\n")
                    f.write(f"Thread analysis failed: {threads.get('message', 'Unknown error')}\n\n")
                else:
                    f.write("===== THREAD LENGTH ANALYSIS =====\n")
                    f.write(f"Total Threads: {threads.get('thread_count', 0):,}\n")
                    f.write(f"Average Thread Length: {threads.get('avg_thread_length', 0):.2f} interactions\n")
                    f.write(f"Median Thread Length: {threads.get('median_thread_length', 0):.2f} interactions\n")
                    f.write(f"Maximum Thread Length: {threads.get('max_thread_length', 0):,} interactions\n")

                    if 'percentiles' in threads:
                        percentiles = threads['percentiles']
                        f.write("\nThread Length Percentiles:\n")
                        for p, val in percentiles.items():
                            f.write(f"  {p}: {val:.1f} interactions\n")

                    if 'thread_categories' in threads:
                        categories = threads['thread_categories']
                        f.write("\nThread Categories:\n")
                        f.write(f"  Posts with no comments: {categories.get('no_comments', 0):,} ({categories.get('no_comments_pct', 0):.2f}%)\n")
                        f.write(f"  Short threads (2-5): {categories.get('few_comments', 0):,} ({categories.get('few_comments_pct', 0):.2f}%)\n")
                        f.write(f"  Medium threads (6-20): {categories.get('many_comments', 0):,} ({categories.get('many_comments_pct', 0):.2f}%)\n")
                        f.write(f"  Long threads (>20): {categories.get('very_many_comments', 0):,} ({categories.get('very_many_comments_pct', 0):.2f}%)\n\n")

            # Thread Activity Analysis
            if 'thread_activity' in results:
                activity = results['thread_activity']

                if not activity.get('success', False):
                    f.write("===== THREAD ACTIVITY ANALYSIS =====\n")
                    f.write(f"Thread activity analysis failed: {activity.get('message', 'Unknown error')}\n\n")
                else:
                    f.write("===== THREAD ACTIVITY ANALYSIS =====\n")
                    f.write(f"Threads Analyzed: {activity.get('thread_count', 0):,}\n")
                    f.write(f"Average Active Days per Thread: {activity.get('avg_active_days', 0):.2f} days\n")
                    f.write(f"Median Active Days per Thread: {activity.get('median_active_days', 0):.2f} days\n")
                    f.write(f"Maximum Active Days for a Thread: {activity.get('max_active_days', 0):,} days\n")

                    if 'avg_span_days' in activity:
                        f.write(f"Average Thread Lifespan: {activity.get('avg_span_days', 0):.2f} days\n")
                        f.write(f"Median Thread Lifespan: {activity.get('median_span_days', 0):.2f} days\n")
                        f.write(f"Average Activity Density: {activity.get('avg_activity_density', 0):.2f}\n")

                    if 'percentiles' in activity and 'active_days' in activity['percentiles']:
                        active_percentiles = activity['percentiles']['active_days']
                        f.write("\nActive Days Percentiles:\n")
                        for p, val in active_percentiles.items():
                            f.write(f"  {p}: {val:.1f} days\n")

                    if 'thread_categories' in activity:
                        categories = activity['thread_categories']
                        f.write("\nThread Duration Categories:\n")
                        f.write(f"  One-day only: {categories.get('one_day_only', 0):,} ({categories.get('one_day_pct', 0):.2f}%)\n")
                        f.write(f"  Short duration (2-3 days): {categories.get('short_duration', 0):,} ({categories.get('short_duration_pct', 0):.2f}%)\n")
                        f.write(f"  Medium duration (4-7 days): {categories.get('medium_duration', 0):,} ({categories.get('medium_duration_pct', 0):.2f}%)\n")
                        f.write(f"  Long duration (>7 days): {categories.get('long_duration', 0):,} ({categories.get('long_duration_pct', 0):.2f}%)\n\n")

            # Footer
            f.write("=" * 80 + "\n")
            f.write("KEY METRICS SUMMARY\n")
            f.write("-" * 30 + "\n")

            if 'user_dynamics' in results and 'total_unique_users' in results['user_dynamics']:
                f.write(f"Total Unique Users: {results['user_dynamics']['total_unique_users']:,}\n")

            if 'avg_interactions' in results:
                f.write(f"Avg. Unique Active Users per Day: {results['avg_interactions'].get('avg_unique_active_users_per_day', 0):.2f}\n")
                f.write(f"Avg. Interactions per User per Day: {results['avg_interactions'].get('overall_avg_interactions_per_user_per_day', 0):.2f}\n")

            if 'post_distribution' in results:
                f.write(f"Total Posts: {results['post_distribution'].get('total_posts', 0):,}\n")
                f.write(f"Total Comments: {results['post_distribution'].get('total_comments', 0):,}\n")
                f.write(f"Post-Comment Ratio: 1:{results['post_distribution'].get('post_comment_ratio', 0):.2f}\n")

            if 'thread_length' in results and results['thread_length'].get('success', False):
                f.write(f"Avg. Thread Length: {results['thread_length'].get('avg_thread_length', 0):.2f} interactions\n")

            if 'thread_activity' in results and results['thread_activity'].get('success', False):
                f.write(f"Avg. Thread Active Days: {results['thread_activity'].get('avg_active_days', 0):.2f} days\n")

            if 'toxicity' in results and results['toxicity'].get('has_toxicity_data', False):
                f.write(f"Avg. Toxicity Score: {results['toxicity'].get('toxicity_stats', {}).get('mean', 0):.4f}\n")
                f.write(f"Toxicity Coverage: {results['toxicity'].get('toxicity_coverage_pct', 0):.2f}%\n")

            f.write("=" * 80 + "\n")
            f.write(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Saved results text to {filename}")

    except Exception as e:
        logger.error(f"Error saving results text: {str(e)}")
        raise

def save_json(data, filepath):
    """Helper function to save data to a JSON file."""
    import json

    class NumpyEncoder(json.JSONEncoder):
        """Custom encoder for numpy data types."""
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64, np.uint8,
                               np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            return json.JSONEncoder.default(self, obj)

    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)

    logger.info(f"Results saved to {filepath}")


def is_valid_json_file(filepath):
    """Check if a file contains valid JSON."""
    import json
    try:
        with open(filepath, 'r') as f:
            json.load(f)
        return True
    except Exception as e:
        logger.warning(f"Invalid JSON file detected at {filepath}: {str(e)}")
        return False


def create_avg_interactions_plots(avg_interactions_results, sample_id, sample_dir, df_sample):
    """Create plots for the average interactions analysis."""
    try:
        # Create plot for interactions distribution
        plot_file = os.path.join(sample_dir, "interactions_distribution.png")
        save_distribution_plot(
            avg_interactions_results['interactions_distribution'],
            f"Distribution of Interactions per User (Sample {sample_id})",
            "Number of Interactions",
            "Number of Users",
            plot_file
        )

        # Create plot for active days distribution
        plot_file = os.path.join(sample_dir, "active_days_distribution.png")
        save_distribution_plot(
            avg_interactions_results['active_days_distribution'],
            f"Distribution of Active Days per User (Sample {sample_id})",
            "Number of Active Days",
            "Number of Users",
            plot_file
        )

        # Create plot for unique active users per day
        plot_file = os.path.join(sample_dir, "unique_active_users_distribution.png")
        save_distribution_plot(
            avg_interactions_results['unique_users_distribution'],
            f"Distribution of Unique Active Users per Day (Sample {sample_id})",
            "Number of Unique Active Users",
            "Number of Days",
            plot_file
        )

        # Create time series plot of unique active users per day
        if 'unique_users_by_day' in avg_interactions_results and avg_interactions_results['unique_users_by_day']:
            # Get the dates for the days (convert from day number to date string)
            start_timestamp = df_sample['publish_date'].min()

            # Generate date strings for each day
            dates = []
            for day in range(len(avg_interactions_results['unique_users_by_day'])):
                day_timestamp = start_timestamp + (day * SECONDS_PER_DAY)
                dates.append(unix_to_date(day_timestamp))

            # Create the time series plot
            plot_file = os.path.join(sample_dir, "unique_active_users_timeseries.png")
            save_time_series_plot(
                avg_interactions_results['unique_users_by_day'],
                dates,
                f"Unique Active Users per Day Over Time (Sample {sample_id})",
                "Date",
                "Number of Unique Active Users",
                plot_file
            )

            # Create the time series plot with 7-day moving average
            plot_file = os.path.join(sample_dir, "unique_active_users_ma.png")
            save_time_series_with_ma_plot(
                avg_interactions_results['unique_users_by_day'],
                dates,
                7,  # 7-day moving average
                f"Unique Active Users with 7-Day Moving Average (Sample {sample_id})",
                "Date",
                "Number of Unique Active Users",
                plot_file
            )
    except Exception as e:
        logger.warning(f"Error creating avg interactions plots: {str(e)}")


def create_post_distribution_plots(post_dist_results, sample_id, sample_dir):
    """Create plots for the post distribution analysis."""
    try:
        # Create plot for posts per day
        plot_file = os.path.join(sample_dir, "posts_per_day.png")
        save_distribution_plot(
            post_dist_results['posts_distribution'],
            f"Distribution of Posts per Day (Sample {sample_id})",
            "Number of Posts",
            "Number of Days",
            plot_file
        )

        # Create plot for comments per day
        plot_file = os.path.join(sample_dir, "comments_per_day.png")
        save_distribution_plot(
            post_dist_results['comments_distribution'],
            f"Distribution of Comments per Day (Sample {sample_id})",
            "Number of Comments",
            "Number of Days",
            plot_file
        )

        # Create plot for posts per user
        plot_file = os.path.join(sample_dir, "posts_per_user.png")
        save_distribution_plot(
            post_dist_results['posts_per_user_distribution'],
            f"Distribution of Posts per User (Sample {sample_id})",
            "Number of Posts",
            "Number of Users",
            plot_file
        )

        # Create plot for comments per user
        plot_file = os.path.join(sample_dir, "comments_per_user.png")
        save_distribution_plot(
            post_dist_results['comments_per_user_distribution'],
            f"Distribution of Comments per User (Sample {sample_id})",
            "Number of Comments",
            "Number of Users",
            plot_file
        )
    except Exception as e:
        logger.warning(f"Error creating post distribution plots: {str(e)}")


def create_toxicity_plots(toxicity_results, sample_id, sample_dir):
    """Create plots for the toxicity analysis."""
    try:
        # Check if we have toxicity data to plot
        if toxicity_results.get('has_toxicity_data', False):
            # Create toxicity distribution plot
            plot_file = os.path.join(sample_dir, "toxicity_distribution.png")
            save_distribution_plot(
                toxicity_results['toxicity_distribution'],
                f"Distribution of Toxicity Scores (Sample {sample_id})",
                "Toxicity Score",
                "Number of Posts/Comments",
                plot_file
            )
    except Exception as e:
        logger.warning(f"Error creating toxicity plots: {str(e)}")


def create_thread_length_plots(thread_length_results, sample_id, sample_dir):
    """Create plots for the thread length analysis."""
    try:
        # Create thread length distribution plot if analysis was successful
        if thread_length_results.get('success', False):
            plot_file = os.path.join(sample_dir, "thread_length_distribution.png")
            save_distribution_plot(
                thread_length_results['thread_length_distribution'],
                f"Distribution of Thread Lengths (Sample {sample_id})",
                "Thread Length (# of interactions)",
                "Number of Threads",
                plot_file
            )
    except Exception as e:
        logger.warning(f"Error creating thread length plots: {str(e)}")


def create_thread_activity_plots(thread_activity_results, sample_id, sample_dir):
    """Create plots for the thread activity analysis."""
    try:
        # Create thread activity distribution plot if analysis was successful
        if thread_activity_results.get('success', False):
            plot_file = os.path.join(sample_dir, "thread_active_days_distribution.png")
            save_distribution_plot(
                thread_activity_results['active_days_distribution'],
                f"Distribution of Thread Active Days (Sample {sample_id})",
                "Number of Active Days",
                "Number of Threads",
                plot_file
            )
    except Exception as e:
        logger.warning(f"Error creating thread activity plots: {str(e)}")

def extract_external_urls(df, content_col='content'):
    """
    Extract external (non-Reddit) URLs from the content column of a DataFrame.
    Returns a list of URLs (strings).
    """
    url_pattern = re.compile(r'https?://\S+')
    reddit_domains = {'voat.co', 'voat.co'}

    urls = []
    for idx, row in df.iterrows():
        text = row.get(content_col, '')
        if not isinstance(text, str):
            continue
        found_urls = url_pattern.findall(text)
        for url in found_urls:
            try:
                hostname = urlparse(url).hostname or ''
                if any(domain in hostname for domain in reddit_domains):
                    continue
                urls.append(url)
            except Exception:
                continue
    return urls

def main():
    """Main execution function."""
    try:
        logger.info("Starting voat-samples.py script")
        create_directory_structure()

        # Global option to rerun all analyses
        global_rerun = None

        # Load and prepare data
        df = load_and_merge_data()
        samples = create_time_samples(df)

        logger.info(f"Processing {len(samples)} samples")

        # Lists to collect summary statistics across all samples
        all_users_count = []
        all_users_per_day_avg = []
        all_new_users_pct_avg = []
        all_churn_pct_avg = []
        all_interaction_stats = []
        all_post_comment_ratios = []
        all_toxicity_stats = []
        all_thread_length_stats = []
        all_thread_activity_stats = []

        # Process each sample
        for sample_idx, sample in enumerate(samples):
            sample_id = sample_idx + 1
            sample_df = sample['df']
            sample_info = sample['info']

            logger.info(f"Analyzing sample {sample_id}: {sample_info['start_date']} to {sample_info['end_date']} ({sample_info['record_count']:,} records)")

            # Skip samples with very few records
            if sample_info['record_count'] < 10:
                logger.warning(f"Sample {sample_id} has too few records ({sample_info['record_count']}), skipping analysis")
                continue

            # Create output directory path for this sample
            sample_dir = os.path.join(OUTPUT_DIR, f"sample_{sample_id}")

            # --- SAVE CSV WITH JUST TEXT OF EACH COMMENT IN THE SAMPLE ---
            try:
                comments_only = sample_df[sample_df['interaction_type'] == 'comments']
                comments_text_csv = os.path.join(sample_dir, "comments_text.csv")
                comments_only[['content']].dropna().to_csv(comments_text_csv, index=False)
                logger.info(f"Saved comments text CSV for sample {sample_id} to {comments_text_csv}")
            except Exception as e:
                logger.warning(f"Failed to save comments text CSV for sample {sample_id}: {str(e)}")

            # Check for existing analyses for this sample
            full_results_file = os.path.join(sample_dir, "full_results.json")

            # Check if analysis already exists and prompt for rerun if needed
            if os.path.exists(full_results_file) and global_rerun is None:
                global_rerun_prompt = input("Some analyses already exist. Do you want to rerun all analyses? (y/n/ask): ").lower().strip()
                if global_rerun_prompt in ['y', 'n']:
                    global_rerun = (global_rerun_prompt == 'y')
                else:  # 'ask' or anything else
                    global_rerun = 'ask'

            # Determine whether to run analysis
            run_analysis = True
            if os.path.exists(full_results_file) and global_rerun is not True:  # Skip only if global_rerun is NOT True
                if global_rerun is False:
                    logger.info(f"Using existing analysis for sample {sample_id} as per global setting")
                    run_analysis = False
                elif global_rerun == 'ask':  # Only ask if global_rerun is 'ask'
                    rerun_prompt = input(f"Analysis already exists for sample {sample_id}. Run again? (y/n): ").lower().strip()
                    run_analysis = (rerun_prompt == 'y')
                    if not run_analysis:
                        logger.info(f"Using existing analysis for sample {sample_id}")
            elif global_rerun is True and os.path.exists(full_results_file):
                logger.info(f"Rerunning analysis for sample {sample_id} as per global setting")

            # Run or load analysis
            sample_results = None
            if run_analysis:
                # Run full analysis on this sample
                force_rerun = (global_rerun is True)  # Force rerun if global_rerun is True
                sample_results = analyze_sample(sample_df, sample_id, force_rerun=force_rerun)
            else:
                # Load existing analysis
                try:
                    import json
                    with open(full_results_file, 'r') as f:
                        sample_results = json.load(f)
                    logger.info(f"Loaded existing analysis from {full_results_file}")
                except Exception as e:
                    logger.error(f"Error loading existing analysis for sample {sample_id}: {str(e)}")
                    logger.info(f"Will run analysis for sample {sample_id} again")
                    # Run with force_rerun=True in case of errors in existing files
                    sample_results = analyze_sample(sample_df, sample_id, force_rerun=True)

            # Collect statistics for cross-sample analysis if results are available
            if sample_results:
                # User count stats
                if 'user_dynamics' in sample_results and 'total_unique_users' in sample_results['user_dynamics']:
                    all_users_count.append(sample_results['user_dynamics']['total_unique_users'])
                    all_new_users_pct_avg.append(sample_results['user_dynamics'].get('avg_new_users_percentage_after_day10', 0))
                    all_churn_pct_avg.append(sample_results['user_dynamics'].get('avg_churn_percentage_after_day10', 0))

                # User activity per day
                if 'avg_interactions' in sample_results:
                    all_interaction_stats.append({
                        'sample_id': sample_id,
                        'avg_per_user_per_day': sample_results['avg_interactions'].get('overall_avg_interactions_per_user_per_day', 0),
                        'median_per_user_per_day': sample_results['avg_interactions'].get('overall_median_interactions_per_user_per_day', 0),
                        'avg_unique_active_users': sample_results['avg_interactions'].get('avg_unique_active_users_per_day', 0),
                        'median_unique_active_users': sample_results['avg_interactions'].get('median_unique_active_users_per_day', 0)
                    })

                # Post/comment stats
                if 'post_distribution' in sample_results:
                    all_post_comment_ratios.append({
                        'sample_id': sample_id,
                        'posts': sample_results['post_distribution'].get('total_posts', 0),
                        'comments': sample_results['post_distribution'].get('total_comments', 0),
                        'ratio': sample_results['post_distribution'].get('post_comment_ratio', 0)
                    })

                # Toxicity stats
                if 'toxicity' in sample_results and sample_results['toxicity'].get('has_toxicity_data', False):
                    if 'toxicity_stats' in sample_results['toxicity']:
                        all_toxicity_stats.append({
                            'sample_id': sample_id,
                            'mean': sample_results['toxicity']['toxicity_stats'].get('mean', 0),
                            'median': sample_results['toxicity']['toxicity_stats'].get('median', 0),
                            'coverage': sample_results['toxicity'].get('toxicity_coverage_pct', 0)
                        })

                # Thread length stats
                if 'thread_length' in sample_results and sample_results['thread_length'].get('success', False):
                    all_thread_length_stats.append({
                        'sample_id': sample_id,
                        'avg_length': sample_results['thread_length'].get('avg_thread_length', 0),
                        'median_length': sample_results['thread_length'].get('median_thread_length', 0),
                        'max_length': sample_results['thread_length'].get('max_thread_length', 0)
                    })

                # Thread activity stats
                if 'thread_activity' in sample_results and sample_results['thread_activity'].get('success', False):
                    all_thread_activity_stats.append({
                        'sample_id': sample_id,
                        'avg_active_days': sample_results['thread_activity'].get('avg_active_days', 0),
                        'median_active_days': sample_results['thread_activity'].get('median_active_days', 0),
                        'max_active_days': sample_results['thread_activity'].get('max_active_days', 0)
                    })

            # --- EXTRACT EXTERNAL URLS AND SAVE TO TXT ---
            try:
                external_urls = extract_external_urls(sample_df)
                external_urls_file = os.path.join(sample_dir, "external_urls.txt")
                with open(external_urls_file, "w") as f:
                    for url in external_urls:
                        f.write(url + "\n")
                logger.info(f"Extracted {len(external_urls)} external URLs for sample {sample_id} to {external_urls_file}")
            except Exception as e:
                logger.warning(f"Failed to extract/save external URLs for sample {sample_id}: {str(e)}")

        # Print overall summary statistics across all samples
        logger.info("\n==== SUMMARY STATISTICS ACROSS ALL SAMPLES ====")
        if all_users_count:
            logger.info(f"Average total unique users per sample: {sum(all_users_count) / len(all_users_count):.2f}")
            logger.info(f"Minimum unique users in a sample: {min(all_users_count):,}")
            logger.info(f"Maximum unique users in a sample: {max(all_users_count):,}")
            logger.info(f"Total unique users across samples (may include duplicates): {sum(all_users_count):,}")

        if all_new_users_pct_avg:
            logger.info(f"Average percentage of new users per day (days 10+): {sum(all_new_users_pct_avg) / len(all_new_users_pct_avg):.2f}%")

        if all_churn_pct_avg:
            logger.info(f"Average churn percentage per day (days 10+): {sum(all_churn_pct_avg) / len(all_churn_pct_avg):.2f}%")

        if all_interaction_stats:
            avg_interactions = sum(s['avg_per_user_per_day'] for s in all_interaction_stats) / len(all_interaction_stats)
            avg_unique_users = sum(s['avg_unique_active_users'] for s in all_interaction_stats) / len(all_interaction_stats)
            logger.info(f"Average interactions per user per day across samples: {avg_interactions:.2f}")
            logger.info(f"Average unique active users per day across samples: {avg_unique_users:.2f}")

        if all_post_comment_ratios:
            avg_ratio = sum(s['ratio'] for s in all_post_comment_ratios) / len(all_post_comment_ratios)
            logger.info(f"Average post-comment ratio across samples: 1:{avg_ratio:.2f}")

        if all_toxicity_stats:
            avg_toxicity = sum(s['mean'] for s in all_toxicity_stats) / len(all_toxicity_stats)
            logger.info(f"Average toxicity score across samples: {avg_toxicity:.4f}")

        if all_thread_length_stats:
            avg_thread_length = sum(s['avg_length'] for s in all_thread_length_stats) / len(all_thread_length_stats)
            logger.info(f"Average thread length across samples: {avg_thread_length:.2f} interactions")

        if all_thread_activity_stats:
            avg_thread_days = sum(s['avg_active_days'] for s in all_thread_activity_stats) / len(all_thread_activity_stats)
            logger.info(f"Average thread active days across samples: {avg_thread_days:.2f} days")

        # Save summary statistics to a JSON file
        summary_stats = {
            "avg_unique_users_per_sample": sum(all_users_count) / len(all_users_count) if all_users_count else None,
            "min_unique_users": min(all_users_count) if all_users_count else None,
            "max_unique_users": max(all_users_count) if all_users_count else None,
            "total_unique_users_with_duplicates": sum(all_users_count) if all_users_count else None,
            "avg_new_users_percentage": sum(all_new_users_pct_avg) / len(all_new_users_pct_avg) if all_new_users_pct_avg else None,
            "avg_churn_percentage": sum(all_churn_pct_avg) / len(all_churn_pct_avg) if all_churn_pct_avg else None,
            "avg_interactions_per_user_per_day": avg_interactions if all_interaction_stats else None,
            "avg_unique_active_users_per_day": avg_unique_users if all_interaction_stats else None,
            "avg_post_comment_ratio": avg_ratio if all_post_comment_ratios else None,
            "avg_toxicity": avg_toxicity if all_toxicity_stats else None,
            "avg_thread_length": avg_thread_length if all_thread_length_stats else None,
            "avg_thread_active_days": avg_thread_days if all_thread_activity_stats else None,
            "sample_count": len(samples),
            "samples_with_valid_data": len(all_users_count),
            "interaction_stats_by_sample": all_interaction_stats,
            "post_comment_ratios_by_sample": all_post_comment_ratios,
            "toxicity_stats_by_sample": all_toxicity_stats,
            "thread_length_stats_by_sample": all_thread_length_stats,
            "thread_activity_stats_by_sample": all_thread_activity_stats
        }

        summary_file = os.path.join(OUTPUT_DIR, "summary_statistics.json")
        save_json(summary_stats, summary_file)
        logger.info(f"Summary statistics saved to {summary_file}")

        logger.info("Script completed successfully")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
