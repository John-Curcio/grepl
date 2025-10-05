"""SQLite helpers for the labeling task queue."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from grepl import constants


CREATE_STATEMENTS: Iterable[str] = (
    "PRAGMA journal_mode=WAL;",
    """
    CREATE TABLE IF NOT EXISTS frame_tasks (
        task_id INTEGER PRIMARY KEY AUTOINCREMENT,
        clip_id TEXT NOT NULL,
        frame_filename TEXT NOT NULL,
        frame_path TEXT NOT NULL UNIQUE,
        frame_index INTEGER NOT NULL,
        width INTEGER,
        height INTEGER,
        timestamp REAL,
        status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','in_progress','completed','skipped')),
        priority REAL NOT NULL DEFAULT 0.0,
        sample_strategy TEXT NOT NULL DEFAULT 'random',
        inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_frame_tasks_status_priority ON frame_tasks(status, priority DESC, inserted_at ASC);",
    "CREATE INDEX IF NOT EXISTS idx_frame_tasks_clip ON frame_tasks(clip_id);",
)


@dataclass(frozen=True)
class LabelingDB:
    """Small wrapper around the SQLite database used for labeling tasks."""

    path: Path

    @classmethod
    def default(cls) -> "LabelingDB":
        """Resolve the default database location using repository constants."""

        path = constants.LABELING_DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(path=path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        """Ensure schema exists."""

        with self.connect() as conn:
            for statement in CREATE_STATEMENTS:
                conn.execute(statement)
            conn.commit()


STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_SKIPPED = "skipped"

VALID_STATUSES = {STATUS_PENDING, STATUS_IN_PROGRESS, STATUS_COMPLETED, STATUS_SKIPPED}
