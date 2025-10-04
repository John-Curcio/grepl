"""Shared repository constants for paths and processing defaults."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Core directories
CLIPS_DIR = REPO_ROOT / "clips"
CLIP_FRAMES_DIR = REPO_ROOT / "clip_frames"

# Metadata assets
PARSED_PAGE_CSV = REPO_ROOT / "parsed_page_enriched.csv"
LABELED_TAGS_FILE = REPO_ROOT / "labeled_tags.txt"

# Frame extraction defaults
FRAME_TEMPLATE = "frame_{:012d}.jpg"
FRAME_WIDTH = 398
FRAME_HEIGHT = 224
FRAME_FPS = 10
VIDEO_EXTENSION = ".mp4"
NUM_EXTRACTION_THREADS = 15

# Annotation filenames
ANNOTATION_FILENAME = "annotations.txt"
ANNOTATION_CSV_FILENAME = "annotations.csv"
LABEL_ENCODER_FILENAME = "label_encoder.pkl"
FILTERED_LABEL_ENCODER_FILENAME = "filtered_label_encoder.pkl"
