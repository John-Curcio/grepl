#!/usr/bin/env python3
"""
Bulk‑download timestamped YouTube clips — fast version.
Input CSV must contain:  url,start,duration   (seconds, floats OK)
Each clip is written as:  <outdir>/<video_id>_<start>_<duration>.mp4
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func
from tqdm import tqdm

# ---------- CLI ----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="bulk_video_clip_download",
            description="Download many short YouTube clips quickly.")
    p.add_argument("csv", help="CSV with url,start,duration columns")
    p.add_argument("-o", "--outdir", default="clips", help="Destination directory")
    p.add_argument("-w", "--workers", type=int,
                   default=min(32, (os.cpu_count() or 4) * 4),
                   help="Parallel downloads in flight (default: min(32, 4×CPU))")
    p.add_argument("-c", "--cf", type=int, default=4,
                   help="Concurrent fragment downloads per clip (default: 4)")
    p.add_argument("--retry", type=int, default=1, help="Retry failed downloads")
    return p.parse_args()

# ---------- Core ----------------------------------------------------------------

FMT_240P_LOW_FPS = (
    "(bv*[height<=240][fps<=15][vcodec^=avc1]/"
    "bv*[height<=240][fps<=15]/"
    "bv*[height<=240]/"
    "bv*[height<=360][fps<=15][vcodec^=avc1]/"
    "bv*[height<=360][fps<=15]/"
    "bv*[height<=360]/"
    "bestvideo[height<=360]/"
    "bestvideo)"
)



def clip_outtmpl(outdir: str, video_id: str, start: int, dur: int) -> str:
    """Generate the output template for a video clip.

    Args:
        outdir (str): Output directory for the clips.
        video_id (str): YouTube video ID.
        start (int): Start time of the clip (in seconds).
        dur (int): Duration of the clip (in seconds).

    Returns:
        str: Output file path template for the video clip.
    """
    return os.path.join(outdir, f"{video_id}_{start}_{dur}.mp4")


def build_ydl_opts(outtmpl: str, start: int, dur: int, cf: int) -> dict:
    """Build options for youtube-dl.

    Args:
        outtmpl (str): Output file path template.
        start (int): Start time of the clip (in seconds).
        dur (int): Duration of the clip (in seconds).
        cf (int): Number of concurrent fragment downloads.

    Returns:
        dict: Options for youtube-dl.
    """
    end = start + dur
    return {
        "format": FMT_240P_LOW_FPS,
        "download_ranges": download_range_func(None, [(start, end)]),
        "merge_output_format": "mp4",
        "force_keyframes_at_cuts": True,
        "concurrent_fragment_downloads": cf,
        "outtmpl": outtmpl,
        "quiet": True,
        "noprogress": True,
        "socket_timeout": 20,
        # Disable most throttling mitigations – we want speed
        "ratelimit": None,
    }

def download_clip_from_task(task: Tuple[str, int, int, str, int]) -> Tuple[bool, str]:
    """Wrapper for the download function to handle errors.

    Args:
        task (Tuple[str, int, int, str, int]): Download task parameters.
        These include:
            - url: YouTube video URL
            - start: Start time of the clip (in seconds)
            - duration: Duration of the clip (in seconds)
            - outdir: Output directory for the clips
            - cf: Number of concurrent fragment downloads

    Returns:
        Tuple[bool, str]: (success, outpath or error message).
    """
    url, start, dur, outdir, cf = task
    return download_clip(url, start, dur, outdir, cf)

def download_clip(url: str, start: int, dur: int, outdir: str, cf: int) -> Tuple[bool, str]:
    """Worker wrapper: returns (success, outpath).

    Args:
        - url: YouTube video URL
        - start: Start time of the clip (in seconds)
        - duration: Duration of the clip (in seconds)
        - outdir: Output directory for the clips
        - cf: Number of concurrent fragment downloads

    Returns:
        Tuple[bool, str]: (success, outpath or error message).
    """
    video_id = url.split("v=")[-1].split("&")[0]
    outpath = clip_outtmpl(outdir, video_id, start, dur)

    if os.path.exists(outpath):
        if os.path.getsize(outpath) > 0:
            # If the file already exists and is non-empty, skip downloading
            print(f"Skipping {outpath} (already exists)")
            return True, outpath
        else:
            print(f"Removing empty file {outpath}")
            os.remove(outpath)

    os.makedirs(outdir, exist_ok=True)
    opts = build_ydl_opts(outpath, start, dur, cf)

    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([url])
        return True, outpath
    except Exception as e:               # noqa: BLE001
        return False, f"{outpath}  ::  {e}"


# ---------- Helpers -------------------------------------------------------------

def load_tasks(csv_path: str, outdir: str, cf: int) -> List[Tuple]:
    """Load download tasks from a CSV file. Expects columns: url, start, duration.

    Args:
        csv_path (str): Path to the input CSV file.
        outdir (str): Output directory for the clips.
        cf (int): Number of concurrent fragment downloads.

    Returns:
        List[Tuple]: List of download tasks.
    """
    tasks = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"url", "start", "duration"}
        if not required.issubset(reader.fieldnames or []):
            print("Found columns:", reader.fieldnames)
            sys.exit(f"CSV must have columns: {', '.join(required)}")

        # Pre‑sort by URL to maximise extractor cache hits
        for row in sorted(reader, key=lambda r: r["url"]):
            tasks.append(
                (
                    row["url"].strip(),
                    int(float(row["start"])),
                    int(float(row["duration"])),
                    outdir,
                    cf,
                )
            )
    return tasks


def run_pool(tasks: List[Tuple], workers: int, retry: int) -> None:
    fails = []

    def submit_pool(tsk_list: List[Tuple], desc: str):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(download_clip_from_task, t): t for t in tsk_list}
            for f in tqdm(as_completed(futures), total=len(tsk_list), desc=desc):
                ok, msg = f.result()
                if not ok:
                    fails.append(msg)

    submit_pool(tasks, "Downloading")
    for attempt in range(1, retry + 1):
        if not fails:
            break
        print(f"\nRetry pass {attempt}   (remaining: {len(fails)})")
        retry_tasks = []
        for fail_msg in fails.copy():
            # Rebuild task tuple from the outpath in the error string
            outpath = fail_msg.split("::")[0].strip()
            video_id, start, dur_mp4 = os.path.basename(outpath).split("_")
            dur = int(dur_mp4.split(".")[0])
            url = f"https://www.youtube.com/watch?v={video_id}"
            retry_tasks.append((url, int(start), dur, os.path.dirname(outpath), cf))
            fails.remove(fail_msg)

        submit_pool(retry_tasks, f"Retry {attempt}")

    if fails:
        print("\nUn‑downloadable clips:")
        for f in fails:
            print("  ", f)


# ---------- Main ----------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    cf = max(1, args.cf)
    tasks = load_tasks(args.csv, args.outdir, cf)

    t0 = time.time()
    run_pool(tasks, args.workers, args.retry)
    print(f"\nDone in {time.time() - t0:,.1f} s.")
