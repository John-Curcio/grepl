# clip_frames Agent Notes

## Purpose
- `clip_frames/` stores JPEGs extracted from BJJ sparring videos at a fixed 10 FPS and resized to 398x224 (w x h).
- Each subdirectory corresponds to one source clip and feeds downstream labeling, annotation, and training jobs.
- Filenames follow `frame_{:012d}.jpg`, which matches the expectations of the ViTPose labeling pipeline and keeps frame ordering lexicographically stable.

## Creating `clip_frames/`
- Script: `src/grepl/processing/videos_to_frames.py`
- Dependencies: `opencv-python`, `av` (PyAV), `numpy`, `pillow`, `ffmpeg` on the system PATH, plus `threading`/`queue` from the standard library.
- Configuration lives in `src/grepl/constants.py` (e.g., `FRAME_FPS`, `FRAME_TEMPLATE`, `CLIPS_DIR`, `CLIP_FRAMES_DIR`). Tweak those constants or layer your own wrapper script if you need machine-specific overrides.
- Processing flow:
  - For each video file matching `FILE_EXTENSION`, spawn a worker thread that calls `video_to_rgb`.
  - `video_to_rgb` tries OpenCV extraction first (fast path). If decoding fails, it falls back to PyAV/FFmpeg.
  - Frames are saved every `fps_in / FPS_OUT` input frames, ensuring a roughly uniform 10 FPS sampling regardless of the source frame rate.
  - Output directories are created lazily (`clip_frames/<video_basename>/`), and frames are resized to the configured width/height before encoding as JPEG.
- How to run (from repo root):
  ```bash
  uv sync --group dev  # first-time setup
  uv run python src/grepl/processing/videos_to_frames.py
  ```
  Set env vars in the same shell invocation to change source/destination paths or extraction parameters.
- Monitoring progress:
  - Stdout log will print extraction counts per video.
  - Inspect `clip_frames/<some_video>/` to confirm sequential filenames and consistent dimensions.

## Directory Structure & Naming
- `<repo>/clip_frames/` (configurable via `RGB_OUT_PATH`)
  - `<youtube_id>_<start_second>_<duration>[_bw]/`
    - `frame_000000000000.jpg`
    - `frame_000000000001.jpg`
    - ...
- This naming convention matches the CSV metadata in `parsed_page_enriched.csv` and is reused by annotation scripts.

## Downstream Consumers
- `src/grepl/processing/generate_annotations.py`
  - Reads `clip_frames/` to confirm clip availability, count frames, and write `annotations*.txt` alongside label encoder pickles.
  - Imports paths and filenames from `grepl.constants`; adjust that module if assets move.
- `src/grepl/processing/video_dataset.py`
  - Generic dataset loader copied from an external repo. When instantiating `VideoFrameDataset`, pass `imagefile_template='frame_{:012d}.jpg'` and `root_path=RGB_OUT_PATH` to align with this repository's frames.
- Notebook workflows (`src/grepl/2025_*`) load frames directly from `clip_frames/` for visualization and model experiments.

## Sampling & Task Queue
- SQLite location: `grepl.constants.LABELING_DB_PATH` (defaults to `<repo>/data/labeling.sqlite3`).
- Schema + helpers: `src/grepl/labeling/db.py` and `src/grepl/labeling/queue.py` manage frame tasks with statuses (`pending`, `in_progress`, `completed`, `skipped`).
- CLI usage (run from repo root):
  ```bash
  uv run python -m grepl.labeling.cli init-db
  uv run python -m grepl.labeling.cli enqueue-random 200 --seed 7
  uv run python -m grepl.labeling.cli next
  uv run python -m grepl.labeling.cli update 1 completed
  uv run python -m grepl.labeling.cli stats
  ```
- Frame selection defaults to uniform random sampling across `clip_frames/`. Extend `FrameTaskQueue.enqueue_frames` for new strategies (uncertainty, curriculum) or adjust priority scores in the SQLite table.

## ViTPose Inference
- Module: `src/grepl/inference/vitpose_service.py` loads Hugging Face `VitPoseForKeypointDetection` (defaults defined in `grepl.constants`).
- Use the singleton helper for repeated calls:
  ```python
  from grepl.inference import get_vitpose_service

  service = get_vitpose_service()
  prediction = service.predict("clip_frames/some_clip/frame_000000000010.jpg")
  for person in prediction.people:
      print(person.score, person.bbox)
  ```
- Structured outputs include per-keypoint confidence/occlusion flags and raw keypoint tensors for downstream storage.
- Alternative: `src/grepl/inference/mmpose_vitpose_service.py` performs the same formatting using MMPose runtimes. Defaults expect the config/checkpoint under `models/vitpose/`; override via `get_mmpose_vitpose_service(config_path=..., checkpoint_path=...)` if needed.

## Maintenance & Troubleshooting
- Disk usage grows quickly (~10 FPS * duration * JPEG size). Periodically archive or prune unused clip folders.
- If OpenCV extraction silently fails (empty directories), confirm the codec is supported. The script already falls back to PyAV, but ensure `ffmpeg` is installed.
- Re-running the script will skip existing directories; delete a folder first if you need to regenerate it with new settings.
- Keep `clip_frames_60s.zip` updated if distributing precomputed frames—document the date/version when refreshing.

## Extending the Pipeline
- Introduce alternate sampling strategies by wrapping `video_to_rgb` or adding a pre-filter on candidate clips before populating the worker queue.
- Hook additional metadata (e.g., frame timestamps) by augmenting the saved JPEG filenames or writing a sidecar JSON/CSV during extraction.
- For the upcoming labeling UI, expose `clip_frames` via an API that retrieves random folders, loads JPEG bytes, and serves initial ViTPose predictions.
