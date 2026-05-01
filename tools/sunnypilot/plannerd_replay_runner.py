#!/usr/bin/env python3
"""Single-segment plannerd replay runner.

Replays `plannerd` against a sunnypilot-logs segment using the version of
`long_mpc.py` resolved from the current sys.path / cwd. Dumps every
`longitudinalPlan` message it observes to a JSONL file for the parent
harness (replay_veryaggressive_validation.py) to diff.

Why a separate runner? Step 3b of the veryAggressive tuning plan must
exercise both the baseline and candidate `long_mpc.py` constants. The
candidate constants live in a temporary git worktree; running this script
with `cwd=<worktree>` ensures plannerd loads the worktree's copy.

Usage:
  uv run tools/sunnypilot/plannerd_replay_runner.py <route--seg> --out <jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_LOG_ROOTS = [
    Path(os.path.expanduser("~/sunnypilot-logs")),  # macOS / dev laptop
    Path("/data/media/0/realdata"),                 # comma device
]


def _resolve_log_root(override: str | None) -> Path:
    if override:
        return Path(os.path.expanduser(override))
    for p in DEFAULT_LOG_ROOTS:
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f"no log root found; tried {', '.join(str(p) for p in DEFAULT_LOG_ROOTS)}. "
        "pass --log-root explicitly."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("segment", help="route--seg directory name (e.g. 000000d3--ae7d119453--18)")
    ap.add_argument("--out", required=True, help="JSONL file to write longitudinalPlan records to")
    ap.add_argument("--log-root", default=None,
                    help="override log root (default: ~/sunnypilot-logs on dev, /data/media/0/realdata on comma)")
    args = ap.parse_args()

    log_root = _resolve_log_root(args.log_root)
    rlog = log_root / args.segment / "rlog.zst"
    if not rlog.exists():
        print(f"[runner] rlog not found: {rlog}", file=sys.stderr)
        return 2

    # Imported here so a missing import doesn't crash before we can report.
    from openpilot.tools.lib.logreader import LogReader
    from openpilot.selfdrive.test.process_replay.process_replay import replay_process_with_name

    lr = list(LogReader(str(rlog)))
    print(f"[runner] {args.segment}: {len(lr)} input messages", file=sys.stderr)

    captured: dict = {}
    try:
        out_msgs = replay_process_with_name("plannerd", lr, disable_progress=True, captured_output_store=captured)
    finally:
        for proc, streams in captured.items():
            for k, v in (streams or {}).items():
                if v:
                    sys.stderr.write(f"=== captured {proc} {k} ===\n{v}\n")
    print(f"[runner] {args.segment}: replay produced {len(out_msgs)} messages", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for m in out_msgs:
            if m.which() != "longitudinalPlan":
                continue
            lp = m.longitudinalPlan
            rec = {
                "t": int(m.logMonoTime) * 1e-9,
                "aTarget": float(lp.aTarget),
                "shouldStop": bool(lp.shouldStop),
                "allowBrake": bool(lp.allowBrake),
                "allowThrottle": bool(lp.allowThrottle),
                "accels0": float(lp.accels[0]) if len(lp.accels) > 0 else None,
                "hasLead": bool(lp.hasLead),
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[runner] {args.segment}: wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
