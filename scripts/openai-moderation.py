"""
openai-moderation.py - Content Moderation Analysis
Uses OpenAI's moderation API to analyze social media posts from an SQLite database.
"""

import pandas as pd
import sqlite3
import argparse
import os
from openai import OpenAI

# Get OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required but not set")

def load_tweets(db_path):
    """Load tweets from SQLite database"""
    print("[1/3] Connecting to database...")
    conn = sqlite3.connect(db_path)
    
    print("[2/3] Loading tweet data...")
    df = pd.read_sql_query("SELECT * FROM post", conn)
    
    # Add post type column
    df['post_type'] = 'regular_post'  # Default type
    df.loc[(~df['news_id'].isna()) & (df['comment_to'] == -1), 'post_type'] = 'news_share'
    df.loc[df['comment_to'] != -1, 'post_type'] = 'comment'
    
    return conn, df

def moderate_tweets(df):
    """Run posts through OpenAI moderation API"""
    print("[3/3] Running moderation checks...")
    client = OpenAI()
    
    moderation_results = []
    for i, tweet in enumerate(df['tweet']):
        print(f"Processing post {i+1}/{len(df)}...")
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=tweet,
        )
        # Convert boolean categories to 1/0 integers
        categories = response.results[0].categories
        result = {category: 1 if getattr(categories, category) else 0 
                 for category in categories.__dict__.keys()}
        result['post_type'] = df.iloc[i]['post_type']  # Add post type to results
        moderation_results.append(result)

    return pd.DataFrame(moderation_results).astype({col: int for col in moderation_results[0].keys() if col != 'post_type'})

def main(db_path):
    """Main moderation workflow"""
    try:
        # Load tweets
        conn, df = load_tweets(db_path)
        
        # Run moderation
        moderation_df = moderate_tweets(df)
        
        # Add tweet text and ID to results
        moderation_df['text'] = df['tweet']
        moderation_df['id'] = df['id']
        
        # Set specific output directory
        output_dir = "results/reddit-technology-4"
        output_csv = f"{output_dir}/moderation.csv"
        summary_file = f"{output_dir}/moderation_summary.txt"

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results CSV with all flags
        moderation_columns = [col for col in moderation_df.columns 
                            if col not in ['text'] and col != 'text']  # Keep all columns except 'text'
        results_to_save = moderation_df[moderation_columns].copy()
        
        # Ensure columns are in a clear order
        column_order = ['id', 'post_type']  # Start with id and post_type
        flag_columns = [col for col in results_to_save.columns 
                       if col not in ['id', 'post_type']]  # Get all flag columns
        column_order.extend(sorted(flag_columns))  # Add sorted flag columns
        
        # Reorder columns and save to CSV
        results_to_save = results_to_save[column_order]
        results_to_save.to_csv(output_csv, index=False)
        print(f"\nDetailed moderation results saved to: {output_csv}")
        print(f"Columns saved: {', '.join(column_order)}")
        
        # Calculate overall statistics
        total_posts = len(moderation_df)
        category_counts = moderation_df.drop(['text', 'id', 'post_type'], axis=1).sum().astype(int)
        category_percentages = (category_counts / total_posts * 100).round(2)
        
        # Create overall frequency table
        freq_table = pd.DataFrame({
            'Count': category_counts,
            'Percentage': category_percentages
        })
        
        # Save summary to file
        with open(summary_file, 'w') as f:
            f.write("Content Moderation Analysis Summary\n")
            f.write("=================================\n\n")
            f.write(f"Total posts analyzed: {total_posts}\n\n")
            
            f.write("Overall Moderation Category Frequencies:\n")
            f.write("=====================================\n")
            f.write(freq_table.to_string())
            f.write("\n\n")
            
            # Add breakdown by post type
            f.write("Breakdown by Post Type\n")
            f.write("=====================\n")
            
            for post_type in ['regular_post', 'news_share', 'comment']:
                category_data = moderation_df[moderation_df['post_type'] == post_type]
                if len(category_data) > 0:
                    f.write(f"\n{post_type.replace('_', ' ').title()}:\n")
                    f.write(f"Count: {len(category_data)}\n")
                    
                    # Calculate category frequencies for this post type
                    cat_counts = category_data.drop(['text', 'id', 'post_type'], axis=1).sum().astype(int)
                    cat_percentages = (cat_counts / len(category_data) * 100).round(2)
                    
                    cat_freq = pd.DataFrame({
                        'Count': cat_counts,
                        'Percentage': cat_percentages
                    })
                    f.write("\nCategory frequencies:\n")
                    f.write(cat_freq.to_string())
                    f.write("\n")
                    
                    # Add example posts for each moderation flag
                    f.write("\nExample posts by moderation flag:\n")
                    f.write("==============================\n")
                    moderation_flags = category_data.drop(['text', 'id', 'post_type'], axis=1).columns
                    
                    # First show non-flagged example
                    non_flagged = category_data[category_data[moderation_flags].sum(axis=1) == 0]
                    if not non_flagged.empty:
                        f.write("\nNon-flagged example:\n")
                        sample = non_flagged.sample(n=1, random_state=42)
                        f.write(f"Post ID: {sample['id'].iloc[0]}\n")
                        f.write(f"Text: {sample['text'].iloc[0]}\n")
                    
                    # Then show examples for each flag
                    for flag in sorted(moderation_flags):
                        flagged = category_data[category_data[flag] == 1]
                        if not flagged.empty:
                            f.write(f"\n{flag}:\n")
                            # Get up to 3 examples for each flag
                            samples = flagged.sample(n=min(3, len(flagged)), random_state=42)
                            for _, sample in samples.iterrows():
                                f.write(f"Post ID: {sample['id']}\n")
                                f.write(f"Text: {sample['text']}\n")
                                other_flags = [f for f in moderation_flags 
                                             if f != flag and sample[f] == 1]
                                if other_flags:
                                    f.write(f"Additional flags: {', '.join(other_flags)}\n")
                                f.write("\n")
                    f.write("\n")
            
        print(f"\nSummary saved to: {summary_file}")
        
        # Close database connection
        conn.close()
        
    except Exception as e:
        print(f"\nError during moderation: {str(e)}")
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Moderate social media posts using OpenAI API')
    parser.add_argument('db_path', type=str, help='Path to SQLite database file')
    args = parser.parse_args()
    
    main(args.db_path)
