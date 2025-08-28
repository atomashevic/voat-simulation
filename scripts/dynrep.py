# -*- coding: utf-8 -*-
"""Dynamical Reputation Module for Reddit Simulation Data

This module holds functions for calculating dynamical reputation
based on Reddit simulation data, adapted from Stack Exchange interactions.

Two types of dynamical reputation are included: popularity and engagement.
The type of reputation is specified as argument to ``prepare_reddit_interactions`` function.

Two methods for calculating dynamical reputation are provided: linear and parallel.

Note: In Reddit simulation data, 1 round = 1 hour, and 24 rounds = 1 day.
The reputation calculations and visualizations work on a daily basis.

Examples:

    Calculate popularity reputation using linear method::

        import pandas as pd
        import dynamical_reputation as dr

        # Load Reddit simulation data
        posts_df = pd.read_csv('results/reddit-tech/posts.csv')
        
        # Prepare data for reputation calculation (popularity)
        # Note: 180 rounds = 7.5 days
        data = dr.prepare_reddit_interactions(posts_df, 0, 180, 'pop')
        dr_reddit_pop = dr.calculate_dynamical_reputation(data, beta=0.999, Ib=1, alpha=2, decay_per_day="True")

        # Prepare data for reputation calculation (engagement)
        data = dr.prepare_reddit_interactions(posts_df, 0, 180, 'eng')
        dr_reddit_eng = dr.calculate_dynamical_reputation(data, beta=0.999, Ib=1, alpha=2, decay_per_day="True")

    Calculate popularity reputation using parallel method::
        import pandas as pd
        import dynamical_reputation as dr

        # Load Reddit simulation data
        posts_df = pd.read_csv('results/reddit-tech/posts.csv')
        
        # Prepare data and calculate using parallel method
        # Note: 1000 rounds ≈ 41.7 days
        data = dr.prepare_reddit_interactions(posts_df, 0, 1000, 'pop')
        reddit_pop = dr.calculate_dynamical_reputation_paralel(data, beta=0.999, Ib=1, alpha=2, decay_per_day="True")

    Generate reputation statistics and visualizations::
        import pandas as pd
        import dynamical_reputation as dr

        # Load Reddit simulation data
        posts_df = pd.read_csv('results/reddit-tech/posts.csv')
        
        # Calculate both reputation types
        # Note: 720 rounds = 30 days (1 month)
        pop_data = dr.prepare_reddit_interactions(posts_df, 0, 720, 'pop')
        eng_data = dr.prepare_reddit_interactions(posts_df, 0, 720, 'eng')
        
        pop_rep = dr.calculate_dynamical_reputation(pop_data, beta=0.999, Ib=1, alpha=2, decay_per_day="True")
        eng_rep = dr.calculate_dynamical_reputation(eng_data, beta=0.999, Ib=1, alpha=2, decay_per_day="True")
        
        # Generate reputation statistics and visualizations
        dr.plot_reputation_statistics(pop_rep, 'popularity', 'results/reddit-tech/')
        dr.plot_reputation_statistics(eng_rep, 'engagement', 'results/reddit-tech/')
"""

from functools import partial
from datetime import timedelta
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_dynamical_reputation(interactions, beta=0.999, Ib=1, alpha=2, decay_per_day = "True"):
    """Calculate time series of dynamical reputation for each user given a set of interactions.
    This is a linear method and does not use parallel computing.

    Args:
        interactions (DataFrame): Pandas dataframe of merged interactions.
        The type of merged interactions determines the type of reputation (popularity or engagement)
        beta (float, optional): Reputation decay parameter. Defaults to 0.999.
        Ib (float, optional): Basic reputational value of a single interaction. Defaults to 1.
        alpha (float, optional): Cumulation parameter. Defaults to 2.
        decay_per_day (str, optional): Perform decay in each day of inactivity? Defaults to "True".

    Returns:
        (DataFrame): DF with values of reputation for each user (row) each day (column)
    """

    TReputation = {}
    users = interactions['UserId'].unique()
    day_max = max(interactions['days'])
    d_min = min(interactions['Time'])
    for u in users:
        dates = interactions[interactions['UserId']==u][['Time', 'days']]
        Ru = calculate_user_reputation(dates, d_min, day_max, beta=beta, Ib=Ib, alpha=alpha, decay_per_day = decay_per_day )
        TReputation[int(u)] = Ru
    df = pd.DataFrame(TReputation).T
    return df

def calculate_dynamical_reputation_paralel(interactions, beta = 0.999, Ib=1., alpha = 2, decay_per_day = 'True'):
    """Calculate time series of dynamical reputation for each user given a set of interactions.
    This is a linear method and does not use parallel computing.

    Args:
        interactions (Pandas DataFrame): Pandas dataframe of merged interactions.
        The type of merged interactions determines the type of reputation (popularity or engagement)
        beta (float, optional): Reputation decay parameter. Defaults to 0.999.
        Ib (float, optional): Basic reputational value of a single interaction. Defaults to 1.
        alpha (float, optional): Cumulation parameter. Defaults to 2.
        decay_per_day (str, optional): Perform decay in each day of inactivity? Defaults to "True".

    Returns:
        (DataFrame): DF with values of reputation for each user (row) each day (column)
    """
    TReputation = {}
    users = interactions['UserId'].unique()
    day_max = max(interactions['days'])
    d_min = min(interactions['Time'])
    data = [(u, interactions[interactions['UserId']==u][['Time', 'days']]) for u in users ]
    pool = Pool(cpu_count())
    func = partial(mpi_run, d_min, day_max, beta, Ib, alpha, decay_per_day)
    results = pool.map(func, data)
    for u, rep in results:
        TReputation[int(u)]=rep
    df = pd.DataFrame(TReputation).T
    del data
    return df

def mpi_run(d_min, day_max, beta, Ib, alpha, decay_per_day, data ):
    """Multiprocessing instance run of ``calculate_user_reputation`` function.

    Args:
        d_min (datetime): Date-time stamp of the first interaction: 'YYYY-MM-DDTHH:MM:SS.SSS'
        day_max (int): Upper limit for days after first interaction which are counted.
        beta (float): Reputation decay parameter.
        Ib (float): Basic reputational value of a single interaction.
        alpha (float): Cumulation parameter.
        decay_per_day (str): Perform decay in each day of inactivity? 'True' or 'False'
        data (Pandas DataFrame / dictionary ???): Sorted interactions for a single user

    Returns:
        (tuple): A tuple of user ID and dictionary of user's reputation
    """
    u, dates = data[0], data[1]
    Ru = calculate_user_reputation(dates, d_min, day_max, beta=beta, Ib=Ib, alpha=alpha, decay_per_day = decay_per_day )
    return u, Ru

def calculate_user_reputation(dates, d_min, day_max, beta=0.999, Ib=1, alpha=2, decay_per_day = "True" ):
    """Returns a dictionary of user's reputational values for each day

    Args:
        dates (DataFrame): Pandas DataFrame with timestamp and day columns.
        Each row is a single interaction by the same user.
        d_min (datetime): Date-time stamp of the first interaction: 'YYYY-MM-DDTHH:MM:SS.SSS'
        day_max (int): Upper limit for days after first interaction which are counted.
        beta (float, optional): Reputation decay parameter. Defaults to 0.999.
        Ib (float, optional): Basic reputational value of a single interaction. Defaults to 1.
        alpha (float, optional): Cumulation parameter. Defaults to 2.
        decay_per_day (str, optional): Perform decay in each day of inactivity? Defaults to "True".

    Returns:
        (dict): A dictionary of user's reputational values for each day
    """
    dates = dates.sort_values(by='Time')
    Ru = {}
    first_day = dates.iloc[0].days
    first_date = dates.iloc[0].Time
    if first_day > 0:
        for day in range(first_day):
            Ru[day] = 0.
    A = 1
    Ru[first_day] = Ib + Ib*alpha*(1.-1./(A+1))
    last_day = first_day
    last_activity_date = first_date
    last_activity_day = first_day
    for i in range(1,len(dates)):
        curr_day = dates.iloc[i].days
        curr_activity_date = dates.iloc[i].Time
        if curr_day > (last_day+1):
            for i in range(curr_day-last_day - 1):
                inactive_date = pd.to_datetime(d_min) + timedelta(days=int(last_day)+2)
                update_reputation_inactive(Ru, inactive_date, last_activity_date, last_activity_day,  last_day, beta=beta, decay_per_day = decay_per_day)
                last_day = last_day + 1
                A = 0
        A+=1
        update_reputation(Ru, A, curr_activity_date, last_activity_date, last_day, last_activity_day, curr_day, beta=beta,  Ib=Ib, alpha=alpha, decay_per_day=decay_per_day)
        last_activity_date = curr_activity_date
        last_activity_day = curr_day
        last_day = curr_day
    rest_days = day_max - last_day
    for i in range(rest_days):
        inactive_date = pd.to_datetime(d_min) + timedelta(days=int(last_day)+2)
        update_reputation_inactive(Ru, inactive_date, last_activity_date, last_activity_day,  last_day, beta=beta, decay_per_day = decay_per_day)
        last_day = last_day + 1
    return Ru

def update_reputation_inactive(R, inactive_date, last_activity_date, last_activity_day, last_day, beta=0.999, decay_per_day = 'True'):
    """Performs the decay of user's reputation during a period of inactivity."""
    dt = (pd.to_datetime(inactive_date) - pd.to_datetime(last_activity_date)) / np.timedelta64(1, 'D')
    dt = float(dt)
    if decay_per_day == 'True':
        D = R[last_day] * np.power(beta, dt)
    else:
        D = R[last_activity_day] * np.power(beta, dt)
    # No bonus for inactivity, just decay
    R[last_day + 1] = D

def update_reputation(R, A, curr_date, last_activity_date, last_day, last_activity_day, curr_day, beta=0.999, Ib=1, alpha=2, decay_per_day='True'):
    """Performs the update of user's reputation during a period of activity."""
    dt = (pd.to_datetime(curr_date) - pd.to_datetime(last_activity_date)) / np.timedelta64(1, 'D')
    dt = float(dt)
    In = Ib + Ib * alpha * (1. - 1. / float(A + 1))
    if (decay_per_day == 'False') and (A == 1):
        D = R[last_activity_day] * np.power(beta, dt)
    else:
        D = R[last_day] * np.power(beta, dt)
    # Correct update: add bonus after decay
    R[curr_day] = D + In

def prepare_reddit_interactions(posts_df, ltlim, htlim, reputation):
    """Prepare Reddit simulation data for reputation calculation.
    
    Adapts the Reddit post/comment structure to the dynamical reputation interaction model.
    For Reddit data, we consider:
    - Root posts (comment_to == -1) as questions
    - Comments (comment_to != -1) as answers/comments
    
    Note: In the Reddit simulation data, 1 round = 1 hour, and 24 rounds = 1 day.
    This function converts rounds to days for the reputation calculation.
    
    Args:
        posts_df (DataFrame): DataFrame containing Reddit posts and comments
        ltlim (int): Lower time limit in rounds (hours)
        htlim (int): Upper time limit in rounds (hours)
        reputation (str): "pop" for popularity reputation or "eng" for engagement reputation
        
    Returns:
        (DataFrame): DataFrame with three columns: user ID, time-stamp and day of interaction.
    """
    # Check required columns
    required_columns = ['user_id', 'round', 'comment_to', 'id']
    for col in required_columns:
        if col not in posts_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset. Available columns: {posts_df.columns.tolist()}")
    
    # Check data format and handle various possible column names
    time_column = None
    for possible_time_column in ['created_utc', 'created_at', 'timestamp', 'time', 'created']:
        if possible_time_column in posts_df.columns:
            time_column = possible_time_column
            break
    
    if time_column is None:
        # If no timestamp column exists, create a synthetic one using round values
        print("Warning: No timestamp column found. Using round values to create synthetic timestamps.")
        time_column = 'synthetic_time'
        # Create a synthetic timestamp using round as hours since epoch
        from datetime import datetime, timedelta
        start_date = datetime(2022, 1, 1)  # Default start date
        posts_df[time_column] = posts_df['round'].apply(lambda r: start_date + timedelta(hours=int(r)))
    
    # Filter by time range
    filtered_df = posts_df[(posts_df['round'] >= ltlim) & (posts_df['round'] < htlim)]
    
    # Create necessary columns for reputation calculation
    interactions = pd.DataFrame()
    
    # For reputation calculation, we need:
    # - UserId: The user receiving reputation
    # - Time: Timestamp of interaction
    # - days: Days since start (convert from rounds/hours: round // 24)
    
    if reputation == 'pop':
        # Popularity: Users get reputation when others interact with their content
        # Root posts (comment_to == -1)
        #root_posts = filtered_df[filtered_df['comment_to'] == -1]
        
        # Comments to posts (comment_to != -1)
        comments = filtered_df[filtered_df['comment_to'] != -1]
        
        # For each comment, find the original post author
        comments_with_targets = []
        for _, comment in comments.iterrows():
            # Find the post this comment is responding to
            target_id = comment['comment_to']
            
            # Find the user who created that post
            target_post = filtered_df[filtered_df['id'] == target_id]
            if not target_post.empty:
                target_user_id = target_post.iloc[0]['user_id']
                
                # Only count interactions between different users
                if target_user_id != comment['user_id']:
                    comments_with_targets.append({
                        'UserId': target_user_id,  # User receiving reputation
                        'Time': comment[time_column],  # Time of interaction
                        'days': comment['round'] // 24  # Convert rounds (hours) to days
                    })
        
        # Convert list of interactions to DataFrame
        if comments_with_targets:
            interactions = pd.DataFrame(comments_with_targets)
            print(f"      Found {len(interactions)} interactions for popularity reputation")
        else:
            print("Warning: No interactions found for popularity reputation calculation")
        
    elif reputation == 'eng':
        # Engagement: Users get reputation when they create content
        interactions = pd.DataFrame({
            'UserId': filtered_df['user_id'],
            'Time': filtered_df[time_column],
            'days': filtered_df['round'] // 24  # Convert rounds (hours) to days
        })
        print(f"      Found {len(interactions)} interactions for engagement reputation")
    
    # Return sorted interactions
    return interactions.dropna().sort_values(by='Time')


def plot_reputation_statistics(reputation_df, rep_type, output_dir):
    """Generate visualizations for reputation statistics.
    
    Creates plots showing mean, median, and total reputation over time.
    
    Args:
        reputation_df (DataFrame): DataFrame with user reputation per day
        rep_type (str): Type of reputation ('popularity' or 'engagement')
        output_dir (str): Directory to save the visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all unique days/rounds
    days = sorted([col for col in reputation_df.columns if isinstance(col, int) or (isinstance(col, str) and col.isdigit())])
    # Convert string columns to int if needed
    days = [int(day) for day in days]

    # Calculate statistics for each day, applying threshold (reputation < 1 -> 0)
    mean_rep = []
    median_rep = []
    total_rep = []
    
    for day in days:
        if day in reputation_df.columns:
            # Apply threshold: set values < 1 to 0
            day_values = reputation_df[day].copy()
            day_values = day_values.apply(lambda x: 0 if x < 1 else x)
            
            # Calculate statistics
            mean_rep.append(day_values.mean())
            median_rep.append(day_values.median())
            total_rep.append(day_values.sum())
        else:
            mean_rep.append(0)
            median_rep.append(0)
            total_rep.append(0)
    
    # Convert to pandas DataFrame for easy export
    stats_df = pd.DataFrame({
        'day': days,
        'mean_reputation': mean_rep,
        'median_reputation': median_rep,
        'total_reputation': total_rep
    })
    
    # Save statistics to CSV
    stats_df.to_csv(f"{output_dir}/{rep_type}_reputation_stats.csv", index=False)
    
    # Generate individual plots
    
    # 1. Mean Reputation Over Time
    plt.figure(figsize=(12, 7))
    plt.plot(days, mean_rep, 'b-', linewidth=2)
    plt.title(f'Mean {rep_type.title()} Reputation Over Time')
    plt.xlabel('Day (Round)')
    plt.ylabel('Mean Reputation')
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/{rep_type}_mean_reputation.png")
    plt.close()
    
    # 2. Median Reputation Over Time
    plt.figure(figsize=(12, 7))
    plt.plot(days, median_rep, 'g-', linewidth=2)
    plt.title(f'Median {rep_type.title()} Reputation Over Time')
    plt.xlabel('Day (Round)')
    plt.ylabel('Median Reputation')
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/{rep_type}_median_reputation.png")
    plt.close()
    
    # 3. Total Reputation Over Time
    plt.figure(figsize=(12, 7))
    plt.plot(days, total_rep, 'r-', linewidth=2)
    plt.title(f'Total {rep_type.title()} Reputation Over Time')
    plt.xlabel('Day (Round)')
    plt.ylabel('Total Reputation')
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/{rep_type}_total_reputation.png")
    plt.close()
    
    # 4. Combined Statistics Plot
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot mean and median on left y-axis
    ax1.plot(days, mean_rep, 'b-', linewidth=2, label='Mean')
    ax1.plot(days, median_rep, 'g-', linewidth=2, label='Median')
    ax1.set_xlabel('Day (Round)')
    ax1.set_ylabel('Reputation Value')
    ax1.tick_params(axis='y')
    
    # Create second y-axis for total
    ax2 = ax1.twinx()
    ax2.plot(days, total_rep, 'r-', linewidth=2, label='Total')
    ax2.set_ylabel('Total Reputation')
    ax2.tick_params(axis='y')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f'{rep_type.title()} Reputation Statistics Over Time')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{rep_type}_combined_reputation_stats.png")
    plt.close()
    
    # 5. Reputation Distribution at select times
    # Choose a few days to show distribution (start, 1/4, 1/2, 3/4, end)
    if len(days) >= 5:
        selected_days = [days[0], days[len(days)//4], days[len(days)//2], days[3*len(days)//4], days[-1]]
    else:
        selected_days = days
    
    plt.figure(figsize=(14, 8))
    for day in selected_days:
        if day in reputation_df.columns:
            # Apply threshold: set values < 1 to 0
            day_values = reputation_df[day].copy()
            day_values = day_values.apply(lambda x: 0 if x < 1 else x)
            plt.hist(day_values.values, bins=30, alpha=0.3, label=f'Day {day}')
    
    plt.title(f'{rep_type.title()} Reputation Distribution')
    plt.xlabel('Reputation Value')
    plt.ylabel('Number of Users')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/{rep_type}_reputation_distribution.png")
    plt.close()
    
    # 6. Top Users Reputation Trajectory
    # Get top 10 users by final reputation (after applying threshold)
    if days and days[-1] in reputation_df.columns:
        last_day = days[-1]
        # Apply threshold before selecting top users
        last_day_values = reputation_df[last_day].copy()
        last_day_values = last_day_values.apply(lambda x: 0 if x < 1 else x)
        top_users = last_day_values.nlargest(10).index
        
        plt.figure(figsize=(14, 8))
        for user in top_users:
            # Apply threshold to each day's value
            user_rep = []
            for day in days:
                if day in reputation_df.columns:
                    value = reputation_df.loc[user, day]
                    user_rep.append(0 if value < 1 else value)
                else:
                    user_rep.append(0)
            plt.plot(days, user_rep, marker='.', label=f'User {user}')
        
        plt.title(f'Top 10 Users {rep_type.title()} Reputation Over Time')
        plt.xlabel('Day (Round)')
        plt.ylabel('Reputation')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"{output_dir}/{rep_type}_top_users_reputation.png")
        plt.close()


def export_reputation_data(reputation_df, rep_type, output_dir):
    """Export reputation data to CSV files.
    
    Args:
        reputation_df (DataFrame): DataFrame with user reputation per day
        rep_type (str): Type of reputation ('popularity' or 'engagement')
        output_dir (str): Directory to save the CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the full reputation DataFrame
    reputation_df.to_csv(f"{output_dir}/{rep_type}_reputation.csv")
    
    # Create a long-format version of the data for easier analysis
    # Melt the dataframe to have columns: user_id, day, reputation
    days = [col for col in reputation_df.columns if isinstance(col, int) or (isinstance(col, str) and str(col).isdigit())]
    days = [int(day) for day in days]
    reputation_long = pd.melt(
        reputation_df.reset_index(),
        id_vars=['index'],
        value_vars=days,
        var_name='day',
        value_name='reputation'
    )
    reputation_long = reputation_long.rename(columns={'index': 'user_id'})
    
    # Save long-format data
    reputation_long.to_csv(f"{output_dir}/{rep_type}_reputation_long.csv", index=False)


def main():
    """Main function to run the dynamical reputation calculation directly from command line.
    
    This function takes a posts.csv file as input and calculates both popularity and
    engagement reputation metrics. Results are saved in the same directory as the input file,
    with appropriate subfolders.
    
    Note: In the Reddit simulation data, 1 round = 1 hour, and 24 rounds = 1 day.
    All inputs and outputs are in terms of rounds (hours), but reputation calculations
    and visualizations are on a daily basis.
    
    Usage:
        python dynrep.py path/to/posts.csv [--min-round MIN] [--max-round MAX]
    
    Example:
        python dynrep.py results/reddit-tech/posts.csv --min-round 0 --max-round 720
    """
    import argparse
    import os
    import sys
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Calculate dynamical reputation for Reddit simulation data')
    parser.add_argument('posts_file', type=str, help='Path to the posts.csv file')
    parser.add_argument('--min-round', type=int, default=0, help='Minimum round to include (in hours, default: 0)')
    parser.add_argument('--max-round', type=int, default=None, help='Maximum round to include (in hours, default: max round in data)')
    parser.add_argument('--beta', type=float, default=0.999, help='Reputation decay parameter (default: 0.999)')
    parser.add_argument('--alpha', type=float, default=2.0, help='Cumulation parameter (default: 2.0)')
    parser.add_argument('--Ib', type=float, default=1.0, help='Basic interaction value (default: 1.0)')
    
    args = parser.parse_args()
    
    # Print header
    print(f"\n{'-'*60}")
    print(f"Dynamical Reputation Analysis: {os.path.basename(args.posts_file)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'-'*60}\n")
    
    try:
        # Load data
        print(f"[1/5] Loading data from {args.posts_file}...")
        try:
            posts_df = pd.read_csv(args.posts_file)
            print(f"      Loaded {len(posts_df)} posts and comments")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            print(f"Check that the file {args.posts_file} exists and is a valid CSV file.")
            sys.exit(1)
        
        # Print available columns to help with debugging
        print(f"      Available columns: {', '.join(posts_df.columns)}")
        
        # Determine output directory (same as input file)
        output_dir = os.path.dirname(args.posts_file)
        reputation_dir = os.path.join(output_dir, 'reputation')
        os.makedirs(reputation_dir, exist_ok=True)
        
        # Determine actual max round if not specified
        if args.max_round is None:
            args.max_round = int(posts_df['round'].max()) + 1
        
        # Print time range in both rounds (hours) and days
        min_day = args.min_round // 24
        max_day = args.max_round // 24
        print(f"[2/5] Preparing data for reputation calculation...")
        print(f"      Time range: rounds {args.min_round} to {args.max_round} (hours)")
        print(f"                  days {min_day} to {max_day}")
        
        # Prepare data for popularity reputation
        print(f"      Preparing popularity data...")
        try:
            pop_data = prepare_reddit_interactions(posts_df, args.min_round, args.max_round, 'pop')
        except Exception as e:
            print(f"Error preparing popularity data: {str(e)}")
            pop_data = pd.DataFrame()
        
        # Prepare data for engagement reputation
        print(f"      Preparing engagement data...")
        try:
            eng_data = prepare_reddit_interactions(posts_df, args.min_round, args.max_round, 'eng')
        except Exception as e:
            print(f"Error preparing engagement data: {str(e)}")
            eng_data = pd.DataFrame()
        
        print(f"[3/5] Calculating reputation metrics...")
        
        # Calculate popularity reputation
        pop_rep = None
        if not pop_data.empty:
            print(f"      Calculating popularity reputation...")
            try:
                pop_rep = calculate_dynamical_reputation(
                    pop_data, beta=args.beta, Ib=args.Ib, alpha=args.alpha, decay_per_day="True")
                print(f"      Calculated popularity reputation for {len(pop_rep)} users")
            except Exception as e:
                print(f"Error calculating popularity reputation: {str(e)}")
        else:
            print("      Skipping popularity reputation calculation (no data)")
        
        # Calculate engagement reputation
        eng_rep = None
        if not eng_data.empty:
            print(f"      Calculating engagement reputation...")
            try:
                eng_rep = calculate_dynamical_reputation(
                    eng_data, beta=args.beta, Ib=args.Ib, alpha=args.alpha, decay_per_day="True")
                print(f"      Calculated engagement reputation for {len(eng_rep)} users")
            except Exception as e:
                print(f"Error calculating engagement reputation: {str(e)}")
        else:
            print("      Skipping engagement reputation calculation (no data)")
        
        # Generate reputation visualizations and export data
        print(f"[4/5] Generating visualizations...")
        
        # Generate popularity reputation visualizations
        if pop_rep is not None and not pop_rep.empty:
            print(f"      Generating popularity reputation visualizations...")
            pop_dir = os.path.join(reputation_dir, 'popularity')
            plot_reputation_statistics(pop_rep, 'popularity', pop_dir)
        else:
            print("      Skipping popularity visualizations (no data)")
        
        # Generate engagement reputation visualizations
        if eng_rep is not None and not eng_rep.empty:
            print(f"      Generating engagement reputation visualizations...")
            eng_dir = os.path.join(reputation_dir, 'engagement')
            plot_reputation_statistics(eng_rep, 'engagement', eng_dir)
        else:
            print("      Skipping engagement visualizations (no data)")
        
        # Export data files
        print(f"[5/5] Exporting data files...")
        
        if pop_rep is not None and not pop_rep.empty:
            print(f"      Exporting popularity reputation data...")
            export_reputation_data(pop_rep, 'popularity', reputation_dir)
        else:
            print("      Skipping popularity data export (no data)")
        
        if eng_rep is not None and not eng_rep.empty:
            print(f"      Exporting engagement reputation data...")
            export_reputation_data(eng_rep, 'engagement', reputation_dir)
        else:
            print("      Skipping engagement data export (no data)")
        
        # Print completion message
        print(f"\nAnalysis complete!")
        print(f"Results saved to: {reputation_dir}/")
        print(f"\nPopularity Reputation: {len(pop_rep) if pop_rep is not None else 0} users")
        print(f"Engagement Reputation: {len(eng_rep) if eng_rep is not None else 0} users")
        
    except Exception as e:
        print(f"\nUnexpected error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
