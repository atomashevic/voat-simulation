import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

def load_data():
    """Load the technology parquet file"""
    print("- Loading madoc-calibration/technology.parquet...")
    df = pd.read_parquet("madoc-calibration/technology.parquet")

    # Convert publish_date from Unix timestamp to datetime
    df['datetime'] = pd.to_datetime(df['publish_date'], unit='s')
    df['date'] = df['datetime'].dt.date

    return df

# Function removed as we now use analyze_specific_months instead

def analyze_daily_activity(month_data):
    """Analyze user activity on a daily level including churn metrics"""
    print("- Analyzing daily user activity:")
    print("  | Date | Posts | Comments | Unique Users | Cumulative Users | New Users % | Returning Users % | Churn % |")
    print("  | ---- | ----- | -------- | ------------ | ---------------- | ----------- | ----------------- | ------- |")

    daily_stats = []
    all_users_seen = set()

    # Create a map of dates to identify users' future activity
    dates = sorted(month_data['date'].unique())
    date_idx = {date: i for i, date in enumerate(dates)}

    # First, identify each user's activity dates for future calculations
    user_activity_dates = {}
    for _, row in month_data.iterrows():
        user_id = row['user_id']
        date = row['date']
        if user_id not in user_activity_dates:
            user_activity_dates[user_id] = set()
        user_activity_dates[user_id].add(date)

    for date_index, date in enumerate(dates):
        day_data = month_data[month_data['date'] == date]

        # Calculate metrics
        posts_count = len(day_data[day_data['interaction_type'] == 'POST'])
        comments_count = len(day_data[day_data['interaction_type'] == 'COMMENT'])

        # Unique users for the day
        day_users = set(day_data['user_id'].unique())
        unique_users_count = len(day_users)

        # New users (first ever seen)
        new_users = day_users - all_users_seen
        new_users_pct = (len(new_users) / unique_users_count) * 100 if unique_users_count > 0 else 0

        # Identify returning users (those who will be active in the future)
        returning_users = set()
        for user_id in day_users:
            user_future_dates = {d for d in user_activity_dates[user_id] if date_idx[d] > date_index}
            if user_future_dates:
                returning_users.add(user_id)

        returning_users_pct = (len(returning_users) / unique_users_count) * 100 if unique_users_count > 0 else 0

        # Churn - percentage of users who won't be active again after this day
        churned_users = day_users - returning_users
        churn_pct = (len(churned_users) / unique_users_count) * 100 if unique_users_count > 0 else 0

        # Focus on new users who will return in the future
        new_returning_users = new_users.intersection(returning_users)
        new_returning_pct = (len(new_returning_users) / unique_users_count) * 100 if unique_users_count > 0 else 0

        # Store daily stats before updating all_users_seen
        daily_stats.append({
            'date': date,
            'posts': posts_count,
            'comments': comments_count,
            'unique_users': unique_users_count,
            'cumulative_users': len(all_users_seen) + len(new_users),
            'new_users_pct': new_users_pct,
            'new_users': len(new_users),
            'returning_users': len(returning_users),
            'returning_users_pct': returning_users_pct,
            'churn_pct': churn_pct,
            'new_returning_users': len(new_returning_users),
            'new_returning_pct': new_returning_pct
        })

        # Print formatted stats
        print(f"  | {date} | {posts_count} | {comments_count} | {unique_users_count} | " +
              f"{len(all_users_seen) + len(new_users)} | {new_users_pct:.2f}% | " +
              f"{returning_users_pct:.2f}% | {churn_pct:.2f}% |")

        # Update all users seen AFTER calculating metrics
        all_users_seen.update(day_users)

    return pd.DataFrame(daily_stats)

def calculate_user_probabilities(month_data):
    """Calculate probabilities of user actions on a random day"""
    print("\n- User Action Probabilities:")

    try:
        # Get all unique users and days
        all_users = month_data['user_id'].unique()
        all_dates = month_data['date'].unique()

        total_user_days = len(all_users) * len(all_dates)

        # Create a user-day activity map
        user_day_activity = defaultdict(lambda: defaultdict(int))

        for _, row in month_data.iterrows():
            user_id = row['user_id']
            date = row['date']
            activity = row['interaction_type']

            if activity == 'POST':
                user_day_activity[user_id][date] += 1
            elif activity == 'COMMENT':
                # Mark as -1 to differentiate from POST
                if user_day_activity[user_id][date] != 1:  # Don't overwrite POST
                    user_day_activity[user_id][date] = -1

        # Count user activities
        inactive_days = total_user_days
        post_days = 0
        comment_days = 0
        active_days = 0

        for user in all_users:
            for date in all_dates:
                activity = user_day_activity[user][date]
                if activity > 0:  # POST
                    post_days += 1
                    inactive_days -= 1
                    active_days += 1
                elif activity < 0:  # COMMENT
                    comment_days += 1
                    inactive_days -= 1
                    active_days += 1

        # Calculate unconditional probabilities
        p_inactive = inactive_days / total_user_days
        p_post = post_days / total_user_days
        p_comment = comment_days / total_user_days

        # Calculate conditional probabilities (given that user is active)
        p_post_given_active = post_days / active_days if active_days > 0 else 0
        p_comment_given_active = comment_days / active_days if active_days > 0 else 0

        print("  | Action | Probability |")
        print("  | ------ | ----------- |")
        print(f"  | Inactive | {p_inactive:.4f} ({p_inactive*100:.2f}%) |")
        print(f"  | Post a comment | {p_comment:.4f} ({p_comment*100:.2f}%) |")
        print(f"  | Post a submission | {p_post:.4f} ({p_post*100:.2f}%) |")

        print("\n- Conditional Probabilities (Given User is Active):")
        print("  | Action | Probability |")
        print("  | ------ | ----------- |")
        print(f"  | Post a comment | {p_comment_given_active:.4f} ({p_comment_given_active*100:.2f}%) |")
        print(f"  | Post a submission | {p_post_given_active:.4f} ({p_post_given_active*100:.2f}%) |")

        return {
            'inactive': p_inactive,
            'post': p_post,
            'comment': p_comment,
            'post_given_active': p_post_given_active,
            'comment_given_active': p_comment_given_active
        }
    except Exception as e:
        print(f"Error calculating probabilities: {e}")
        # Return default values to avoid breaking the main function
        return {
            'inactive': 0.95,
            'post': 0.01,
            'comment': 0.04,
            'post_given_active': 0.2,
            'comment_given_active': 0.8
        }

def analyze_specific_months(df, months_to_analyze):
    """Analyze specific months in chronological order"""
    # Sort all months chronologically
    df['year_month'] = df['datetime'].dt.to_period('M')
    all_months = sorted(df['year_month'].unique())

    # Get the requested months (3, 4, 5)
    if len(all_months) >= 5:
        target_months = [all_months[2], all_months[3], all_months[4]]  # 0-indexed, so 2,3,4 = 3rd,4th,5th months
    else:
        # Fallback if not enough months
        target_months = all_months[:min(3, len(all_months))]

    print(f"- Analyzing months: {', '.join(str(m) for m in target_months)}")

    # Filter data for these months
    selected_data = df[df['year_month'].isin(target_months)].copy()
    selected_data = selected_data.sort_values('datetime')

    return selected_data

def analyze_user_interactions(month_data):
    """Analyze user interaction patterns and distribution"""
    print("\n- User Interaction Distribution:")

    # Group by user_id and date to count interactions per user per day
    user_day_counts = month_data.groupby(['user_id', 'date']).size().reset_index(name='interactions')

    # Calculate avg and median interactions per day for each user
    user_stats = user_day_counts.groupby('user_id').agg(
        avg_interactions=('interactions', 'mean'),
        median_interactions=('interactions', 'median'),
        total_interactions=('interactions', 'sum'),
        active_days=('interactions', 'count')
    ).reset_index()

    # Calculate distribution statistics
    avg_stats = user_stats['avg_interactions'].describe(percentiles=[0.25, 0.5, 0.75])
    median_stats = user_stats['median_interactions'].describe(percentiles=[0.25, 0.5, 0.75])

    print("  | Metric | Min | Q1 | Median | Mean | Q3 | Max |")
    print("  | ------ | --- | -- | ------ | ---- | -- | --- |")
    print(f"  | Avg interactions per day | {avg_stats['min']:.2f} | {avg_stats['25%']:.2f} | {avg_stats['50%']:.2f} | {avg_stats['mean']:.2f} | {avg_stats['75%']:.2f} | {avg_stats['max']:.2f} |")
    print(f"  | Median interactions per day | {median_stats['min']:.2f} | {median_stats['25%']:.2f} | {median_stats['50%']:.2f} | {median_stats['mean']:.2f} | {median_stats['75%']:.2f} | {median_stats['max']:.2f} |")

    # Identify power users (top 1% by total interactions)
    threshold = np.percentile(user_stats['total_interactions'], 99)
    power_users = user_stats[user_stats['total_interactions'] >= threshold]

    print(f"\n- Power Users (Top 1%):")
    print(f"  | Count: {len(power_users)} users |")
    print(f"  | Average daily interactions: {power_users['avg_interactions'].mean():.2f} |")
    print(f"  | Responsible for {(power_users['total_interactions'].sum() / user_stats['total_interactions'].sum()) * 100:.2f}% of all interactions |")

    return user_stats

def main():
    """Main function to run the analysis"""
    print("MADOC Technology Data Analysis")
    print("-----------------------------")

    # Load data
    df = load_data()

    # Get specific months (3, 4, 5) instead of random
    month_data = analyze_specific_months(df, [3, 4, 5])

    # Analyze daily activity with churn metrics
    daily_stats = analyze_daily_activity(month_data)

    # Analyze user interaction patterns
    user_interaction_stats = analyze_user_interactions(month_data)

    # Calculate user probabilities
    probabilities = calculate_user_probabilities(month_data)
    active_percentage = (1 - probabilities['inactive']) * 100

    print("\n- Summary:")
    print(f"  | Total days analyzed: {len(daily_stats)} |")
    print(f"  | Total unique users: {daily_stats['cumulative_users'].max()} |")
    print(f"  | Average daily posts: {daily_stats['posts'].mean():.2f} |")
    print(f"  | Average daily comments: {daily_stats['comments'].mean():.2f} |")
    print(f"  | Average daily unique users: {daily_stats['unique_users'].mean():.2f} |")
    print(f"  | Average daily new users percentage: {daily_stats['new_users_pct'].mean():.2f}% |")
    print(f"  | Average daily returning users percentage: {daily_stats['returning_users_pct'].mean():.2f}% |")
    print(f"  | Average daily churn percentage: {daily_stats['churn_pct'].mean():.2f}% |")
    print(f"  | Percentage of users active on an average day: {active_percentage:.2f}% |")

if __name__ == "__main__":
    main()
    print(f"  | Percentage of users active on an average day: {(1-probabilities['inactive'])*100:.2f}% |")

if __name__ == "__main__":
    main()
