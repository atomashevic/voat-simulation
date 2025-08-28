#!/usr/bin/env python3
"""
Extract external URLs from Reddit MADOC parquet files and save curated outputs.

- Input: Reddit sample parquet files in MADOC/reddit-technology/ (expects a 'url' column)
- Output: MADOC/reddit-technology/external_urls.csv (url, domain, count)
- Optional: top-N URLs text file if --txt-out is provided (not written by default)

Usage:
  # Process all sample files
  python scripts/extract_reddit_urls.py \
    --input-dir MADOC/reddit-technology \
    --csv-out MADOC/reddit-technology/external_urls.csv \
    [--txt-out simulation1/config/urls.txt] \
    --top 500

  # Process single file
  python scripts/extract_reddit_urls.py \
    --input MADOC/reddit-technology/sample_1/technology_sample_1.parquet \
    --csv-out MADOC/reddit-technology/external_urls.csv \
    --top 500
"""

import argparse
import os
import glob
from collections import Counter
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd

TRACKING_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "gclid",
    "fbclid",
    "scid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "ref_source",
}

INTERNAL_DOMAINS = {
    "reddit.com", 
    "www.reddit.com", 
    "old.reddit.com", 
    "new.reddit.com", 
    "m.reddit.com"
}

DEFAULT_SHORTENERS = {
    "t.co",
    "bit.ly",
    "goo.gl",
    "tinyurl.com",
    "ow.ly",
    "buff.ly",
    "ift.tt",
    "trib.al",
    "dlvr.it",
    "lnkd.in",
    "fb.me",
}

# Known media/video/image hosts to exclude
MEDIA_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "youtu.be",
    "vimeo.com",
    "streamable.com",
    "twitch.tv",
    "clips.twitch.tv",
    "dailymotion.com",
    "gfycat.com",
    "tenor.com",
    "giphy.com",
    "imgur.com",
    "i.imgur.com",
    "redd.it",
    "i.redd.it",
    "v.redd.it",
    "redditmedia.com",
    "i.redditmedia.com",
    "twimg.com",
    "pbs.twimg.com",
}

MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".gifv", ".webp",
    ".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v",
    ".mp3", ".m4a", ".wav", ".flac",
}


def normalize_url(u: str) -> str | None:
    """Normalize a URL: lowercase scheme/host, strip www, tracking params, fragments, trailing punctuation."""
    try:
        # strip trailing punctuation commonly attached in text
        u = u.rstrip(").,!?;'\"")
        p = urlparse(u)
        if not p.scheme or not p.netloc:
            return None
        scheme = p.scheme.lower()
        netloc = p.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        # remove tracking params but keep others
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k not in TRACKING_KEYS]
        query = urlencode(q, doseq=True)
        # drop fragments entirely
        return urlunparse((scheme, netloc, p.path or "", "", query, ""))
    except Exception:
        return None


def is_media_url(u: str) -> bool:
    """Heuristic filter for media links (domains and file extensions)."""
    try:
        p = urlparse(u)
        domain = p.netloc.lower()
        if domain in MEDIA_DOMAINS:
            return True
        # extension check on path (case-insensitive)
        path_lower = (p.path or "").lower()
        return any(path_lower.endswith(ext) for ext in MEDIA_EXTENSIONS)
    except Exception:
        return False


def extract_urls_from_parquet(file_path: str) -> list[str]:
    """Extract URLs directly from the 'url' column of a parquet file."""
    try:
        df = pd.read_parquet(file_path)
        if 'url' not in df.columns:
            print(f"Warning: No 'url' column found in {file_path}")
            return []
        
        urls = df['url'].dropna().astype(str).tolist()
        return [url for url in urls if url and url != 'nan']
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def find_sample_files(input_dir: str) -> list[str]:
    """Find all sample parquet files in the input directory."""
    patterns = [
        os.path.join(input_dir, "sample_*/technology_sample_*.parquet"),
        os.path.join(input_dir, "early_sample_*/technology_sample_*.parquet"),
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    
    return sorted(files)


def save_csv(counts: Counter, csv_out: str) -> None:
    rows = [
        {"url": u, "domain": urlparse(u).netloc, "count": c}
        for u, c in counts.most_common()
    ]
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_out, index=False)


def save_txt(urls: list[str], txt_out: str, top: int) -> None:
    os.makedirs(os.path.dirname(txt_out), exist_ok=True)
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(urls[:top]))


def main():
    ap = argparse.ArgumentParser(description="Extract external URLs from Reddit parquet files")
    ap.add_argument("--input", help="Single parquet file to process")
    ap.add_argument("--input-dir", default="MADOC/reddit-technology", 
                    help="Directory containing sample parquet files")
    ap.add_argument("--csv-out", default="MADOC/reddit-technology/external_urls.csv")
    # Do not write urls.txt by default; only if provided explicitly
    ap.add_argument("--txt-out", default=None, 
                    help="Optional path to write top URLs (not written by default)")
    ap.add_argument("--top", type=int, default=500, 
                    help="Top-N unique URLs for txt list if --txt-out is used")
    ap.add_argument(
        "--keep-shorteners",
        action="store_true",
        help="Keep known URL shorteners (default is to drop them)",
    )
    ap.add_argument(
        "--whitelist-domains",
        default=None,
        help="Optional path to a newline-separated domain whitelist (keep only these domains)",
    )
    args = ap.parse_args()

    # Determine input files
    if args.input:
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input parquet not found: {args.input}")
        input_files = [args.input]
    else:
        input_files = find_sample_files(args.input_dir)
        if not input_files:
            raise FileNotFoundError(f"No sample parquet files found in {args.input_dir}")

    print(f"Processing {len(input_files)} files...")
    
    all_urls = []
    file_stats = []
    
    for file_path in input_files:
        print(f"Processing: {file_path}")
        raw_urls = extract_urls_from_parquet(file_path)
        
        # Normalize URLs
        norm_urls = list(filter(None, (normalize_url(u) for u in raw_urls)))
        
        # Filter internal Reddit links
        norm_urls = [u for u in norm_urls if urlparse(u).netloc not in INTERNAL_DOMAINS]
        
        # Optionally drop known shorteners
        if not args.keep_shorteners:
            norm_urls = [u for u in norm_urls if urlparse(u).netloc not in DEFAULT_SHORTENERS]
        
        # Drop media links (pics, gifs, video platforms, audio)
        norm_urls = [u for u in norm_urls if not is_media_url(u)]
        
        all_urls.extend(norm_urls)
        file_stats.append({
            'file': file_path,
            'raw_urls': len(raw_urls),
            'filtered_urls': len(norm_urls)
        })

    # Optional whitelist by domain
    if args.whitelist_domains:
        with open(args.whitelist_domains, "r", encoding="utf-8") as f:
            whitelist = {line.strip().lower() for line in f if line.strip()}
        all_urls = [u for u in all_urls if urlparse(u).netloc in whitelist]

    counts = Counter(all_urls)

    # Save detailed CSV (all URLs)
    save_csv(counts, args.csv_out)

    # Save top-N unique URLs for simulation config only if path is provided
    if args.txt_out:
        top_urls = [u for u, _ in counts.most_common(args.top)]
        save_txt(top_urls, args.txt_out, args.top)

    # Print summary statistics
    total_raw = sum(stat['raw_urls'] for stat in file_stats)
    total_filtered = sum(stat['filtered_urls'] for stat in file_stats)
    
    print(f"\nSummary:")
    print(f"Files processed: {len(input_files)}")
    print(f"Total raw URLs: {total_raw:,}")
    print(f"URLs after filtering: {total_filtered:,}")
    print(f"Unique URLs: {len(counts):,}")
    print(f"Wrote CSV -> {args.csv_out}")
    if args.txt_out:
        print(f"Wrote top {min(args.top, len(counts))} URLs -> {args.txt_out}")
    
    # Per-file breakdown
    print(f"\nPer-file breakdown:")
    for stat in file_stats:
        filename = os.path.basename(stat['file'])
        print(f"  {filename}: {stat['raw_urls']} raw → {stat['filtered_urls']} filtered")


if __name__ == "__main__":
    main()