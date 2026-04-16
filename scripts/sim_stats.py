"""
sim-stats.py - Simulation Data Analysis Script
Analyzes Reddit simulation data from SQLite database.
"""

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import os
import sys
import numpy as np

def print_header(simulation_name, simulation_date):
    """Print script header with simulation metadata"""
    print(f"\n{'-'*60}")
    print(f"Analyzing Simulation: {simulation_name}")
    print(f"{'-'*60}\n")

def load_simulation_data(db_path):
    """Load simulation data from SQLite database"""
    print("[1/6] Connecting to database...")
    conn = sqlite3.connect(db_path)

    print("[2/6] Loading post data...")
    posts_df = pd.read_sql_query("SELECT * FROM post", conn)

    print("[3/6] Loading news data...")
    news_df = pd.read_sql_query("SELECT * FROM articles", conn)

    print("[4/6] Loading user data...")
    users_df = pd.read_sql_query("SELECT * FROM user_mgmt", conn)

    return conn, posts_df, news_df, users_df

def generate_basic_stats(df):
    """Generate basic statistics from dataframe"""
    print("[5/6] Calculating basic statistics...")

    # Count comments (posts where comment_to is not -1)
    num_comments = len(df[df['comment_to'] != -1])

    # Calculate number of unique users
    num_users = df['user_id'].nunique()

    stats = {
        'num_posts': len(df),
        'num_comments': num_comments,
        'num_root_posts': len(df[df['comment_to'] == -1]),
        'num_users': num_users,
        'avg_thread_length': df.groupby('thread_id').size().mean() if len(df) > 0 else 0,
        'num_news_ids': df['news_id'].nunique(),
        'posts_per_user_total': len(df) / num_users if num_users > 0 else float('nan')
    }

    return stats

def generate_visualizations(df, simulation_name, output_dir):
    """Generate and save visualizations"""
    print("[6/6] Creating visualizations...")

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Distribution of posts per user
    plt.figure(figsize=(12, 7))
    posts_per_user = df['user_id'].value_counts()
    # Create histogram of how many users have X posts
    plt.hist(posts_per_user.values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    plt.title(f'Reddit Technology - Distribution of Posts per User\nN={len(posts_per_user)} Users')
    plt.xlabel('Number of Posts')
    plt.ylabel('Number of Users')
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/posts_distribution.png")
    plt.close()

    # Plot 2: Distribution of activity days per user
    plt.figure(figsize=(12, 7))
    days_active = df.groupby('user_id')['round'].nunique()
    plt.hist(days_active.values, bins=30, alpha=0.7, color='forestgreen', edgecolor='black')
    plt.title(f'Reddit Technology - Distribution of Days Active per User\nN={len(days_active)} Users')
    plt.xlabel('Number of Days Active')
    plt.ylabel('Number of Users')
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/activity_days_distribution.png")
    plt.close()

    # Plot 3: Log-scale distribution of posts per user (for better visibility with skewed data)
    plt.figure(figsize=(12, 7))
    plt.hist(posts_per_user.values, bins=30, alpha=0.7, color='steelblue', edgecolor='black', log=True)
    plt.title(f'Reddit Technology - Distribution of Posts per User (Log Scale)\nN={len(posts_per_user)} Users')
    plt.xlabel('Number of Posts')
    plt.ylabel('Number of Users (Log Scale)')
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/posts_distribution_log.png")
    plt.close()

    # Plot 4: Cumulative distribution of posts per user
    plt.figure(figsize=(12, 7))
    sorted_counts = sorted(posts_per_user.values)
    y = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    plt.plot(sorted_counts, y, marker='.', linestyle='none', alpha=0.5)
    plt.title(f'Reddit Technology - Cumulative Distribution of Posts per User\nN={len(posts_per_user)} Users')
    plt.xlabel('Number of Posts')
    plt.ylabel('Cumulative Proportion of Users')
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/posts_cumulative_distribution.png")
    plt.close()

    # Plot 5: Log-log scatter plot of post frequency distribution
    plt.figure(figsize=(12, 7))
    # Count frequency of each post count
    post_count_freq = posts_per_user.value_counts().sort_index()
    # Convert to DataFrame for easier plotting
    post_dist_df = pd.DataFrame({
        'post_count': post_count_freq.index,
        'num_users': post_count_freq.values
    })
    # Plot on log-log scale
    plt.loglog(post_dist_df['post_count'], post_dist_df['num_users'],
               marker='o', linestyle='none', alpha=0.7, color='purple')

    # Add a best fit line to check linearity
    if len(post_dist_df) > 1:  # Only if we have enough points
        # Take log of both axes for linear regression
        x_log = np.log10(post_dist_df['post_count'])
        y_log = np.log10(post_dist_df['num_users'])
        # Linear regression
        slope, intercept = np.polyfit(x_log, y_log, 1)
        # Generate points for the line
        x_line = np.logspace(min(x_log), max(x_log), 100, base=10)
        y_line = 10**(slope * np.log10(x_line) + intercept)
        # Plot the line
        plt.loglog(x_line, y_line, 'r--', alpha=0.7,
                   label=f'Slope: {slope:.2f}')
        plt.legend()

    plt.title(f'Reddit Technology - Log-Log Plot of Post Distribution\nN={len(posts_per_user)} Users')
    plt.xlabel('Number of Posts (Log Scale)')
    plt.ylabel('Number of Users with X Posts (Log Scale)')
    plt.grid(alpha=0.3, which='both')
    plt.savefig(f"{output_dir}/posts_loglog_distribution.png")
    plt.close()

def export_to_csv(posts_df, news_df, users_df, output_dir):
    """Export posts, news, and user data to CSV"""
    print("Exporting data to CSV...")

    # Export posts data
    posts_csv_path = os.path.join(output_dir, 'posts.csv')
    posts_df.to_csv(posts_csv_path, index=False)

    # Export news data
    news_csv_path = os.path.join(output_dir, 'news.csv')
    news_df.to_csv(news_csv_path, index=False)

    # Export user data
    users_csv_path = os.path.join(output_dir, 'users.csv')
    users_df.to_csv(users_csv_path, index=False)

    return posts_csv_path, news_csv_path, users_csv_path

def main(db_path):
    """Main analysis workflow"""
    try:
        # Extract simulation name from the db file name
        sim_name = os.path.splitext(os.path.basename(db_path))[0]

        # Clean up simulation name for display
        display_name = sim_name.replace('-', ' ').title()

        # Create output directory path (keep original sim_name for paths)
        output_dir = os.path.join('results', sim_name)
        os.makedirs(output_dir, exist_ok=True)

        # Redirect stdout to both console and file
        summary_path = os.path.join(output_dir, 'summary.txt')
        original_stdout = sys.stdout
        with open(summary_path, 'w') as f:
            class TeeOutput:
                def write(self, text):
                    original_stdout.write(text)
                    f.write(text)
                def flush(self):
                    original_stdout.flush()
                    f.flush()

            sys.stdout = TeeOutput()

            # Run analysis
            print_header(display_name, datetime.now().strftime('%Y-%m-%d'))

            # Data loading
            conn, posts_df, news_df, users_df = load_simulation_data(db_path)

            # Analysis
            stats = generate_basic_stats(posts_df)

            # Export to CSV
            posts_csv_path, news_csv_path, users_csv_path = export_to_csv(posts_df, news_df, users_df, output_dir)

            # Visualization
            generate_visualizations(posts_df, sim_name, output_dir)

            # Close database connection
            conn.close()

            # Print results
            print("\nAnalysis Complete!")
            print("Basic Statistics:")
            for k, v in stats.items():
                print(f"- {k.replace('_', ' ').title()}: {v:.2f}")

            print(f"\nVisualizations saved to: {output_dir}/")
            print(f"Posts data exported to: {posts_csv_path}")
            print(f"News data exported to: {news_csv_path}")
            print(f"Users data exported to: {users_csv_path}")
            print(f"Total news articles: {len(news_df)}")
            print(f"Total users: {len(users_df)}")

            # Restore original stdout
            sys.stdout = original_stdout

    except Exception as e:
        if 'sys.stdout' in locals() and sys.stdout != original_stdout:
            sys.stdout = original_stdout
        print(f"\nError during analysis: {str(e)}")
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Reddit simulation data')
    parser.add_argument('db_path', type=str, help='Path to SQLite database file')
    args = parser.parse_args()

    main(args.db_path)
