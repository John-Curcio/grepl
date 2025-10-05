# Labeling Utilities

This package contains the scaffolding for sampling frames from `clip_frames/`,
recording them in a lightweight SQLite queue, and exposing a CLI for day-to-day
operations.

## Modules
- `db.py`: wraps the SQLite database at `grepl.constants.LABELING_DB_PATH`,
  ensures schema creation (`frame_tasks` table), and exposes status helpers.
- `queue.py`: provides `FrameTaskQueue` for enumerating frame files, enqueueing
  random selections, and updating task status (`pending`, `in_progress`,
  `completed`, `skipped`).
- `cli.py`: command-line entry point (invoked via
  `poetry run python -m grepl.labeling.cli`) that offers commands such as
  `init-db`, `enqueue-random`, `next`, `update`, `reset`, `stats`, and `list`.

## Quick Start
```bash
poetry run python -m grepl.labeling.cli init-db
poetry run python -m grepl.labeling.cli enqueue-random 200 --seed 42
poetry run python -m grepl.labeling.cli stats
poetry run python -m grepl.labeling.cli next
```

## Extending
- To support smarter sampling, extend `FrameTaskQueue.enqueue_frames` or adjust
  the `priority` column in the `frame_tasks` table.
- Additional metadata can be stored by altering the schema in `db.py`; remember
  to add migration scripts once the structure stabilises.
- When integrating with CVAT, use `FrameTaskQueue.next()` to feed frames into the
  importer and `update_task_status` once annotations return.
