#!/usr/bin/env python3
"""
Preprocess simulation data to extract clean text for topic modeling.
Handles the special "TITLE: ..." format and filters comments.
Fixes common issues in simulated text like missing spaces after periods.
"""

import pandas as pd
import re
from pathlib import Path
import argparse


def extract_clean_posts(df: pd.DataFrame, text_column: str = 'tweet') -> pd.DataFrame:
    """Extract and clean posts from simulation data."""
    
    # Create a copy to work with
    df = df.copy()
    
    # Separate posts with titles from comments
    df['has_title'] = df[text_column].str.startswith('TITLE:', na=False)
    
    # Extract title and body for posts with titles
    def extract_parts(text):
        if pd.isna(text):
            return '', ''
        
        if text.startswith('TITLE:'):
            # Split on first newline after title
            parts = text.split('\n', 1)
            if len(parts) == 2:
                title = parts[0].replace('TITLE:', '').strip()
                body = parts[1].strip()
            else:
                # No body, just title
                title = parts[0].replace('TITLE:', '').strip()
                body = ''
            return title, body
        else:
            # It's a comment or reply - no separate title
            return '', text.strip()
    
    df[['title', 'body']] = df[text_column].apply(
        lambda x: pd.Series(extract_parts(x))
    )
    
    # Clean the text
    def clean_text(text):
        # Fix missing spaces after periods (common artifact in simulated data)
        # This handles: "sentence.Another" -> "sentence. Another"
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove usernames/mentions at start of comments
        # Handles: ".StanleyAyala", "@Dr.StanleyAyala", "@username", etc.
        text = re.sub(r'^[.@]\S+\s+', '', text)
        
        # Fix any double spaces created
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    df['title'] = df['title'].apply(clean_text)
    df['body'] = df['body'].apply(clean_text)
    
    # Combine title and body for full text
    df['full_text'] = df.apply(
        lambda row: f"{row['title']} {row['body']}".strip() if row['title'] 
                    else row['body'],
        axis=1
    )
    
    # Filter out very short texts
    df['text_length'] = df['full_text'].str.len()
    df = df[df['text_length'] >= 50]
    
    # Add post type
    df['post_type'] = df['has_title'].apply(lambda x: 'post' if x else 'comment')
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess simulation data")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file")
    parser.add_argument("--text-column", type=str, default="tweet", help="Column containing text")
    parser.add_argument("--posts-only", action="store_true", help="Keep only posts (not comments)")
    parser.add_argument("--min-length", type=int, default=50, help="Minimum text length")
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    
    # Process
    df_clean = extract_clean_posts(df, args.text_column)
    
    # Filter if requested
    if args.posts_only:
        df_clean = df_clean[df_clean['post_type'] == 'post']
        print(f"Filtered to {len(df_clean)} posts (excluded comments)")
    
    # Additional length filtering if specified
    if args.min_length > 50:
        df_clean = df_clean[df_clean['text_length'] >= args.min_length]
        print(f"Filtered to {len(df_clean)} texts with length >= {args.min_length}")
    
    # Save
    columns_to_save = ['id', 'title', 'body', 'full_text', 'post_type', 'text_length']
    # Only include columns that exist
    columns_to_save = [col for col in columns_to_save if col in df_clean.columns]
    
    df_clean[columns_to_save].to_csv(args.output, index=False)
    print(f"Saved {len(df_clean)} processed rows to {args.output}")
    
    # Print statistics
    print("\nStatistics:")
    print(f"  Posts with titles: {(df_clean['post_type'] == 'post').sum()}")
    print(f"  Comments: {(df_clean['post_type'] == 'comment').sum()}")
    print(f"  Avg text length: {df_clean['text_length'].mean():.0f} chars")
    print(f"  Median text length: {df_clean['text_length'].median():.0f} chars")


if __name__ == "__main__":
    main()
