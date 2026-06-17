#!/usr/bin/env python
"""Smoke test for the Reachy Mini robot-integration layer.

Connects to a running daemon/simulator (default http://localhost:8000), prints
state and DoA, then runs a few safe motions through the decoupled motion queue
(nod, look left/right, an emotion if the dataset is reachable) and exits cleanly.

Run::

    .venv/bin/python scripts/smoke_robot.py
    .venv/bin/python scripts/smoke_robot.py --base-url http://localhost:8000

Note: under the project's command sandbox, HTTP to localhost needs the sandbox
disabled.
"""

from __future__ import annotations

import argparse
import sys
import time

# Make the script runnable without installing the package.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachy_chat.robot import (  # noqa: E402
    GotoCommand,
    HeadPose,
    LookAtSpeaker,
    MotionQueue,
    ReachyClient,
    ReachyClientError,
    RobotTools,
)


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    client = ReachyClient(base_url=args.base_url, timeout=args.timeout)

    banner("Connectivity")
    if not client.ping():
        print(f"FAIL: daemon not reachable at {args.base_url}")
        return 1
    print(f"Connected to daemon at {args.base_url}")

    banner("State")
    try:
        pose = client.get_present_head_pose()
        body = client.get_present_body_yaw_deg()
        mode = client.get_motor_mode()
        print(f"motor mode      : {mode}")
        print(
            "head pose (deg) : "
            f"roll={pose.roll:+.2f} pitch={pose.pitch:+.2f} yaw={pose.yaw:+.2f}"
        )
        print(f"body yaw (deg)  : {body:+.2f}")
    except ReachyClientError as exc:
        print(f"FAIL reading state: {exc}")
        return 1

    banner("DoA")
    doa = client.get_doa()
    if doa is None:
        print("DoA: null (no mic array / no source on this simulator)")
    else:
        print(
            f"DoA: angle={doa.angle_rad:.3f} rad ({doa.angle_deg:.1f} deg), "
            f"speech_detected={doa.speech_detected}"
        )

    banner("Recorded moves")
    moves = client.list_recorded_moves()
    if moves:
        print(f"{len(moves)} moves available, e.g. {moves[:8]}")
    else:
        print("No recorded moves listed (dataset unreachable?)")

    # Wire up the decoupled motion stack.
    look_at = LookAtSpeaker(client)
    with MotionQueue(client) as queue:
        tools = RobotTools(client, queue, look_at)

        banner("Motion: nod")
        print(tools.nod(times=2))
        _drain(queue)

        banner("Motion: look left / right / front")
        print(tools.move_head("left"))
        _drain(queue)
        print(tools.move_head("right"))
        _drain(queue)
        print(tools.move_head("front"))
        _drain(queue)

        banner("Motion: look_at azimuth")
        print(tools.look_at(30))
        time.sleep(1.0)
        print(tools.look_at(0))
        time.sleep(1.0)

        if moves:
            banner("Motion: play emotion")
            print(tools.play_emotion("happy"))
            _drain(queue, max_wait=8.0)

        banner("Reset to neutral")
        queue.enqueue(GotoCommand(head_pose=HeadPose(), body_yaw_deg=0.0, duration=0.8))
        _drain(queue)

    print("\nSmoke test completed successfully.")
    return 0


def _drain(queue: MotionQueue, max_wait: float = 6.0) -> None:
    """Wait until the queue is empty and the robot is idle (best effort)."""
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        if queue.pending == 0:
            break
        time.sleep(0.1)
    # Give the in-flight move a moment to finish.
    time.sleep(0.6)


if __name__ == "__main__":
    raise SystemExit(main())
