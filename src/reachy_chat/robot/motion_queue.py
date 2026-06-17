"""Thread-safe motion command queue and control-loop worker.

This mirrors the architecture used by the official conversation app
(``ai-integration.md``): the LLM never talks to the motors directly. Instead it
calls a tool, the tool *enqueues* a command, and a dedicated control-loop worker
thread dequeues and executes commands smoothly via the REST client.

::

    LLM -> tool call -> MotionQueue.enqueue(...) -> control loop -> ReachyClient

Benefits of the indirection:

* the control loop stays clean and never blocks on LLM latency;
* commands are serialised, avoiding races between concurrent tool calls;
* blocking moves (goto / recorded emotions) run to completion before the next
  command starts, while the queue keeps accepting new commands.

Two command flavours are supported:

* :class:`GotoCommand`    - a smooth interpolated head/body move.
* :class:`PlayMoveCommand`- a recorded move (emotion / dance) from a dataset.

The worker waits for each blocking move to finish (polling
``/api/move/running``) before pulling the next command, so motions never stomp
on each other.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Optional, Union

from .client import HeadPose, InterpolationTechnique, ReachyClient, ReachyClientError

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Command types
# --------------------------------------------------------------------------- #
@dataclass
class GotoCommand:
    """A smooth interpolated move to a head pose and/or body yaw."""

    head_pose: Optional[HeadPose] = None
    body_yaw_deg: Optional[float] = None
    antennas: Optional[tuple[float, float]] = None
    duration: float = 1.0
    interpolation: InterpolationTechnique = "minjerk"


@dataclass
class PlayMoveCommand:
    """A recorded move (emotion / dance) played from a HF dataset."""

    move_name: str
    dataset: Optional[str] = None  # None -> client default emotion dataset


Command = Union[GotoCommand, PlayMoveCommand]


@dataclass
class _QueueItem:
    command: Command
    # Whether to drain (clear) the queue before running this command. Used by
    # "interrupting" actions so a freshly requested emotion pre-empts a backlog.
    interrupt: bool = False


class MotionQueue:
    """A thread-safe command queue driven by a background control loop.

    Args:
        client: A connected :class:`ReachyClient`.
        poll_interval: How often (s) the worker polls for move completion.
    """

    def __init__(self, client: ReachyClient, poll_interval: float = 0.05) -> None:
        self._client = client
        self._poll_interval = poll_interval
        self._queue: "Queue[_QueueItem]" = Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the control-loop worker thread (idempotent)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, name="reachy-motion-queue", daemon=True
            )
            self._thread.start()
            logger.info("MotionQueue control loop started")

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the worker to stop and wait for it to exit."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None
        logger.info("MotionQueue control loop stopped")

    def __enter__(self) -> "MotionQueue":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # Enqueue API (called by tool handlers / the LLM layer)
    # ------------------------------------------------------------------ #
    def enqueue(self, command: Command, *, interrupt: bool = False) -> None:
        """Enqueue a command for the control loop.

        Args:
            command: A :class:`GotoCommand` or :class:`PlayMoveCommand`.
            interrupt: If ``True``, clear any pending commands and stop the
                currently running move before this one executes.
        """
        if interrupt:
            self.clear()
            try:
                self._client.stop_all_moves()
            except ReachyClientError:
                logger.debug("stop_all_moves failed during interrupt", exc_info=True)
        self._queue.put(_QueueItem(command=command, interrupt=interrupt))

    def clear(self) -> None:
        """Drop all pending (not-yet-started) commands."""
        try:
            while True:
                self._queue.get_nowait()
        except Empty:
            pass

    @property
    def pending(self) -> int:
        """Number of commands waiting in the queue."""
        return self._queue.qsize()

    # ------------------------------------------------------------------ #
    # Worker
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._execute(item.command)
            except ReachyClientError as exc:
                logger.warning("Motion command failed: %s", exc)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Unexpected error executing motion command")

    def _execute(self, command: Command) -> None:
        if isinstance(command, GotoCommand):
            self._client.goto(
                head_pose=command.head_pose,
                body_yaw_deg=command.body_yaw_deg,
                antennas=command.antennas,
                duration=command.duration,
                interpolation=command.interpolation,
            )
            # Wait roughly for the interpolation to finish so subsequent commands
            # don't get queued on top mid-motion.
            self._wait_for_idle(min_wait=command.duration)
        elif isinstance(command, PlayMoveCommand):
            if command.dataset is not None:
                self._client.play_recorded_move(command.move_name, dataset=command.dataset)
            else:
                # Fall back to the client's default emotion dataset.
                self._client.play_recorded_move(command.move_name)
            self._wait_for_idle()
        else:  # pragma: no cover - exhaustive
            logger.warning("Unknown command type: %r", command)

    def _wait_for_idle(self, min_wait: float = 0.0) -> None:
        """Block until no move is running (or the worker is told to stop).

        ``min_wait`` provides a floor for goto moves, since a freshly issued
        goto may not register as "running" the instant it is posted.
        """
        deadline_floor = time.monotonic() + min_wait
        # Small initial settle so the just-posted move shows up as running.
        time.sleep(min(self._poll_interval, 0.05))
        while not self._stop_event.is_set():
            try:
                moving = self._client.is_moving()
            except ReachyClientError:
                moving = False
            if not moving and time.monotonic() >= deadline_floor:
                return
            self._stop_event.wait(self._poll_interval)
