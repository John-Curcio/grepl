"""Command-line helpers for the frame task queue."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from .db import (
    LabelingDB,
    STATUS_COMPLETED,
    STATUS_IN_PROGRESS,
    STATUS_PENDING,
    STATUS_SKIPPED,
)
from .queue import FrameTaskQueue


def _task_to_dict(task) -> dict:
    return {
        "task_id": task.task_id,
        "clip_id": task.clip_id,
        "frame_filename": task.frame_filename,
        "frame_path": str(task.frame_path),
        "frame_index": task.frame_index,
        "width": task.width,
        "height": task.height,
        "timestamp": task.timestamp,
        "status": task.status,
        "priority": task.priority,
        "sample_strategy": task.sample_strategy,
    }


def cmd_init_db(_: argparse.Namespace) -> None:
    LabelingDB.default().initialize()
    print("Initialized labeling database.")


def cmd_enqueue_random(args: argparse.Namespace) -> None:
    queue = FrameTaskQueue()
    inserted = queue.enqueue_random_frames(args.count, seed=args.seed, strategy=args.strategy)
    print(f"Inserted {inserted} new frame task(s).")


def cmd_next(_: argparse.Namespace) -> None:
    queue = FrameTaskQueue()
    task = queue.next_task()
    if task is None:
        print("No pending tasks.")
    else:
        print(json.dumps(_task_to_dict(task), indent=2, sort_keys=True))


def cmd_update(args: argparse.Namespace) -> None:
    queue = FrameTaskQueue()
    queue.update_task_status(args.task_id, args.status)
    print(f"Task {args.task_id} updated to {args.status}.")


def cmd_reset(_: argparse.Namespace) -> None:
    queue = FrameTaskQueue()
    count = queue.reset_stale_tasks()
    print(f"Reset {count} task(s) to pending.")


def cmd_stats(_: argparse.Namespace) -> None:
    queue = FrameTaskQueue()
    stats = {
        "total": queue.count(),
        STATUS_PENDING: queue.count(STATUS_PENDING),
        STATUS_IN_PROGRESS: queue.count(STATUS_IN_PROGRESS),
        STATUS_COMPLETED: queue.count(STATUS_COMPLETED),
        STATUS_SKIPPED: queue.count(STATUS_SKIPPED),
    }
    print(json.dumps(stats, indent=2, sort_keys=True))


def cmd_list(args: argparse.Namespace) -> None:
    queue = FrameTaskQueue()
    tasks = queue.list_tasks(limit=args.limit, status=args.status)
    print(json.dumps([_task_to_dict(task) for task in tasks], indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frame task queue management")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-db", help="Create the labeling database schema")
    p_init.set_defaults(func=cmd_init_db)

    p_enqueue = sub.add_parser("enqueue-random", help="Sample random frames and enqueue tasks")
    p_enqueue.add_argument("count", type=int, help="Number of frames to sample")
    p_enqueue.add_argument("--seed", type=int, default=None, help="Random seed for deterministic sampling")
    p_enqueue.add_argument("--strategy", default="random", help="Sampling strategy label to store")
    p_enqueue.set_defaults(func=cmd_enqueue_random)

    p_next = sub.add_parser("next", help="Pop the next pending task")
    p_next.set_defaults(func=cmd_next)

    p_update = sub.add_parser("update", help="Update a task's status")
    p_update.add_argument("task_id", type=int)
    p_update.add_argument("status", choices=[
        STATUS_PENDING,
        STATUS_IN_PROGRESS,
        STATUS_COMPLETED,
        STATUS_SKIPPED,
    ])
    p_update.set_defaults(func=cmd_update)

    p_reset = sub.add_parser("reset", help="Move all in-progress tasks back to pending")
    p_reset.set_defaults(func=cmd_reset)

    p_stats = sub.add_parser("stats", help="Print task counts by status")
    p_stats.set_defaults(func=cmd_stats)

    p_list = sub.add_parser("list", help="List most recent tasks")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--status", choices=[
        STATUS_PENDING,
        STATUS_IN_PROGRESS,
        STATUS_COMPLETED,
        STATUS_SKIPPED,
    ], default=None)
    p_list.set_defaults(func=cmd_list)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
