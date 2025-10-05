"""Frame sampling and task queue operations."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PIL import Image

from grepl import constants
from .db import (
    LabelingDB,
    STATUS_COMPLETED,
    STATUS_IN_PROGRESS,
    STATUS_PENDING,
    STATUS_SKIPPED,
    VALID_STATUSES,
)

FRAME_REGEX = re.compile(r"frame_(\d+)\.(?:jpg|jpeg)", re.IGNORECASE)


@dataclass(frozen=True)
class FrameTask:
    task_id: int
    clip_id: str
    frame_filename: str
    frame_path: Path
    frame_index: int
    width: Optional[int]
    height: Optional[int]
    timestamp: Optional[float]
    status: str
    priority: float
    sample_strategy: str

    @classmethod
    def from_row(cls, row) -> "FrameTask":
        return cls(
            task_id=row["task_id"],
            clip_id=row["clip_id"],
            frame_filename=row["frame_filename"],
            frame_path=Path(row["frame_path"]),
            frame_index=row["frame_index"],
            width=row["width"],
            height=row["height"],
            timestamp=row["timestamp"],
            status=row["status"],
            priority=row["priority"],
            sample_strategy=row["sample_strategy"],
        )


class FrameTaskQueue:
    """High-level API for sampling frames and managing the SQLite-backed queue."""

    def __init__(self, db: Optional[LabelingDB] = None) -> None:
        self._db = db or LabelingDB.default()
        self._db.initialize()
        self._clip_frames_dir = constants.CLIP_FRAMES_DIR

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def _iter_frame_paths(self) -> Iterable[Path]:
        root: Path = self._clip_frames_dir
        if not root.exists():
            return
        for clip_dir in sorted(root.iterdir()):
            if not clip_dir.is_dir():
                continue
            for frame_path in sorted(clip_dir.glob("frame_*")):
                if frame_path.suffix.lower() not in {".jpg", ".jpeg"}:
                    continue
                yield frame_path

    @staticmethod
    def _frame_index_from_path(frame_path: Path) -> Optional[int]:
        match = FRAME_REGEX.match(frame_path.name)
        if not match:
            return None
        return int(match.group(1))

    def enqueue_random_frames(
        self,
        count: int,
        *,
        seed: Optional[int] = None,
        strategy: str = "random",
    ) -> int:
        """Sample `count` frames at random and enqueue them.

        Returns the number of new tasks inserted (existing tasks are skipped).
        """

        frame_paths = list(self._iter_frame_paths())
        if not frame_paths:
            return 0

        rng = random.Random(seed)
        sample_size = min(count, len(frame_paths))
        selected_paths = rng.sample(frame_paths, sample_size)
        return self.enqueue_frames(selected_paths, strategy=strategy)

    def enqueue_frames(self, frame_paths: Sequence[Path], *, strategy: str) -> int:
        """Insert provided frame paths into the task queue."""

        if not frame_paths:
            return 0

        inserted = 0
        with self._db.connect() as conn:
            for frame_path in frame_paths:
                clip_id = frame_path.parent.name
                frame_filename = frame_path.name
                frame_index = self._frame_index_from_path(frame_path)
                timestamp = (
                    frame_index / constants.FRAME_FPS if frame_index is not None else None
                ) # TODO (JC): watch out: the clips themselves are w.r.t. a timestamp, so 
                # this could be ambiguous.

                width = height = None
                try:
                    with Image.open(frame_path) as img:
                        width, height = img.size
                except FileNotFoundError:
                    continue

                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO frame_tasks (
                        clip_id,
                        frame_filename,
                        frame_path,
                        frame_index,
                        width,
                        height,
                        timestamp,
                        status,
                        priority,
                        sample_strategy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        clip_id,
                        frame_filename,
                        str(frame_path.resolve()),
                        frame_index if frame_index is not None else -1,
                        width,
                        height,
                        timestamp,
                        STATUS_PENDING,
                        0.0,
                        strategy,
                    ),
                )
                if cursor.rowcount:
                    inserted += 1
            conn.commit()
        return inserted

    # ------------------------------------------------------------------
    # Task consumption
    # ------------------------------------------------------------------
    def next_task(self) -> Optional[FrameTask]:
        """Pop the highest-priority pending task and mark it in-progress."""

        with self._db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            row = conn.execute(
                """
                SELECT * FROM frame_tasks
                WHERE status = ?
                ORDER BY priority DESC, inserted_at ASC
                LIMIT 1
                """,
                (STATUS_PENDING,),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            conn.execute(
                "UPDATE frame_tasks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                (STATUS_IN_PROGRESS, row["task_id"]),
            )
            conn.commit()
            row = dict(row)
            row["status"] = STATUS_IN_PROGRESS
            return FrameTask.from_row(row)

    def update_task_status(self, task_id: int, status: str) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        with self._db.connect() as conn:
            conn.execute(
                "UPDATE frame_tasks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                (status, task_id),
            )
            conn.commit()

    def reset_stale_tasks(self) -> int:
        """Move in-progress tasks back to pending (for crash recovery)."""

        with self._db.connect() as conn:
            cursor = conn.execute(
                "UPDATE frame_tasks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE status = ?",
                (STATUS_PENDING, STATUS_IN_PROGRESS),
            )
            conn.commit()
            return cursor.rowcount

    def count(self, status: Optional[str] = None) -> int:
        with self._db.connect() as conn:
            if status is None:
                row = conn.execute("SELECT COUNT(*) AS c FROM frame_tasks").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM frame_tasks WHERE status = ?",
                    (status,),
                ).fetchone()
            return int(row["c"]) if row else 0

    def list_tasks(self, limit: int = 20, status: Optional[str] = None) -> List[FrameTask]:
        with self._db.connect() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM frame_tasks ORDER BY inserted_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM frame_tasks WHERE status = ? ORDER BY inserted_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            return [FrameTask.from_row(row) for row in rows]
