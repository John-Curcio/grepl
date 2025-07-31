"""DEPRE
* Get all timestamped video clips from SQLite with timestamps and a #footage tag
* Download each clip
"""

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm
from grepl.scrape.video_clip_download import download_clip_from_timestamped_url

CSV_PATH = "/home/rocus/Documents/john/grepl/parsed_page_enriched.csv"

def get_all_footage_urls() -> list[str]:
    """Get all timestamped video URLs that have a #footage tag."""
    df = pd.read_csv(CSV_PATH)
    df = df[df["tags"].str.contains("#footage", na=False)]
    df = df[df["timestamped_url"].notna() & (df["timestamped_url"] != '')]
    return df["timestamped_url"].tolist()


def _download(url: str, duration: int = 30) -> bool:
    """Wrapper for single clip download with error handling."""
    try:
        download_clip_from_timestamped_url(url, duration=duration)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Bulk download timestamped YouTube clips in parallel")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel download threads")
    parser.add_argument("--skip", type=int, default=0, help="Skip the first N URLs (after sorting)")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds to download from each clip")
    parser.add_argument("--limit", type=int, default=None, help="Download at most N URLs (after skipping)")
    args = parser.parse_args()

    urls = sorted(get_all_footage_urls())
    if args.skip:
        urls = urls[args.skip:]
    if args.limit is not None:
        urls = urls[:args.limit]

    total = len(urls)
    if total == 0:
        print("No URLs to download – exiting.")
        return

    successes = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_download, url, duration=args.duration): url for url in urls}
        for fut in tqdm(as_completed(futures), total=total):
            if fut.result():
                successes += 1

    print(f"Finished. Success: {successes}/{total} clips.")


if __name__ == '__main__':
    main()