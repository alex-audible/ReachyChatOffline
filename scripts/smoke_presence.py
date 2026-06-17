#!/usr/bin/env python
"""Smoke test for the embodiment / presence layer (:class:`RobotPresence`).

Connects to a running Reachy Mini daemon/simulator (default
``http://localhost:8000``), starts the decoupled presence loop, and cycles
through the conversation states the way the real interaction machine would::

    idle -> listening -> thinking -> speaking -> idle

with realistic dwell times. It then calls :meth:`RobotPresence.look_at` with a
couple of mock azimuths. Throughout, it polls ``/api/move/running`` and watches
the present head pose to *confirm motions actually execute on the sim* (rather
than just trusting that commands were enqueued).

Run::

    .venv/bin/python scripts/smoke_presence.py
    .venv/bin/python scripts/smoke_presence.py --base-url http://localhost:8000

Note: under the project's command sandbox, HTTP to localhost needs the sandbox
disabled.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the script runnable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachy_chat.robot import ReachyClient, ReachyClientError  # noqa: E402
from reachy_chat.robot.presence import RobotPresence  # noqa: E402


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def _pose_str(client: ReachyClient) -> str:
    try:
        p = client.get_present_head_pose()
        b = client.get_present_body_yaw_deg()
        return (
            f"head roll={p.roll:+5.1f} pitch={p.pitch:+5.1f} yaw={p.yaw:+5.1f} "
            f"| body={b:+5.1f}"
        )
    except ReachyClientError as exc:
        return f"<state unavailable: {exc}>"


def observe(client: ReachyClient, label: str, dwell_s: float, period: float = 0.4):
    """Hold for ``dwell_s`` while sampling pose + running-move state.

    Returns ``(saw_running, saw_pose_change)`` so the caller can assert that the
    presence layer drove the sim during this window.
    """
    print(f"\n[{label}] dwell {dwell_s:.1f}s")
    deadline = time.monotonic() + dwell_s
    saw_running = False
    poses: set[tuple[int, int, int]] = set()
    while time.monotonic() < deadline:
        try:
            running = client.list_running_moves()
            p = client.get_present_head_pose()
            poses.add((round(p.roll), round(p.pitch), round(p.yaw)))
            flag = "MOVING" if running else "idle  "
            if running:
                saw_running = True
            print(f"    {flag}  {_pose_str(client)}")
        except ReachyClientError as exc:
            print(f"    <poll failed: {exc}>")
        time.sleep(period)
    saw_pose_change = len(poses) > 1
    return saw_running, saw_pose_change


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
    print(f"start  {_pose_str(client)}")

    # Track whether ANY state window produced observable motion on the sim.
    results: dict[str, tuple[bool, bool]] = {}

    banner("Presence loop")
    presence = RobotPresence(client)
    presence.start()
    try:
        # Begin in idle so we have a baseline, then cycle like a real turn.
        presence.on_state("idle")
        results["idle(initial)"] = observe(client, "idle", dwell_s=7.0)

        presence.on_state("listening")
        results["listening"] = observe(client, "listening", dwell_s=3.5)

        # Mock pointing at a sound source (DoA is null on the bare sim, so we
        # feed manual azimuths the way the host app would on hardware).
        banner("look_at mock azimuths")
        for az in (35.0, -40.0, 0.0):
            print(f"  look_at({az:+.0f})")
            presence.look_at(az)
            r = observe(client, f"look_at {az:+.0f}", dwell_s=1.6)
            results[f"look_at {az:+.0f}"] = r

        presence.on_state("thinking")
        results["thinking"] = observe(client, "thinking", dwell_s=2.5)

        presence.on_state("speaking")
        # Long enough to see at least two periodic nods.
        results["speaking"] = observe(client, "speaking", dwell_s=7.0)

        presence.on_state("idle")
        results["idle(return)"] = observe(client, "idle", dwell_s=3.0)
    finally:
        banner("Stop")
        presence.stop()
        print(f"final  {_pose_str(client)}")

    banner("Results")
    any_motion = False
    for label, (saw_running, saw_change) in results.items():
        moved = saw_running or saw_change
        any_motion = any_motion or moved
        marker = "drove sim" if moved else "no motion observed"
        print(f"  {label:18s}: running={saw_running!s:5s} pose_changed={saw_change!s:5s}  -> {marker}")

    if any_motion:
        print("\nPASS: RobotPresence drove the simulator (no exceptions).")
        return 0
    print("\nFAIL: no motion observed on the simulator during any state.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
