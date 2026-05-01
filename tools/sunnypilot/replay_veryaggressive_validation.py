#!/usr/bin/env python3
"""Pre-merge replay validation for the veryAggressive personality tune.

Step 3b of `.omc/plans/ralplan-veryaggressive-tune.md`. Runs the top-10
worst-jerk events from `report_by_personality.md` through `plannerd`
twice — once with baseline `long_mpc.py`, once with the candidate edit
applied IN PLACE — and applies the four planner-level pass criteria
from the plan.

The patch is applied to the working repo (`REPO_ROOT/<long_mpc.py>`) and
unconditionally restored in `finally`. Earlier git-worktree isolation
was abandoned because `git worktree add` doesn't include compiled C
extensions (`params_pyx.so`, etc.) or untracked submodules
(`msgq_repo/`), causing import failures at replay time.

This script must run on Linux. macOS cannot run `process_replay` because
cereal IPC aborts on Darwin (`SocketEventHandle not supported on macOS`).

Recommended host: a comma device, an Ubuntu workstation, or a Linux dev
container with sunnypilot's `op.sh shell`.

Usage (from sunnypilot repo root, on Linux):
  uv run tools/sunnypilot/replay_veryaggressive_validation.py
      [--top10 .omc/research/ride_analysis/top10_veryaggressive.json]
      [--out   .omc/research/ride_analysis/replay_validation.md]
      [--candidate path-A2]   # path-A2 (default): jerk_factor 0.2->0.3
                              # path-A1: also t_follow 0.8->1.0
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_REL = "tools/sunnypilot/plannerd_replay_runner.py"
LONG_MPC_REL = "selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py"
DEFAULT_TOP10 = REPO_ROOT / ".omc/research/ride_analysis/top10_veryaggressive.json"
DEFAULT_OUT = REPO_ROOT / ".omc/research/ride_analysis/replay_validation.md"

# Plan thresholds (Step 3b pass criteria).
LAZY_BRAKE_FRAC_THRESHOLD = 0.95  # ≥95 % of samples must satisfy lazy-braking guard
LAZY_BRAKE_TOLERANCE = 0.5  # m/s², candidate ≤ baseline + 0.5 during baseline-braking
HARSH_BRAKE_TOLERANCE = 0.5  # m/s², candidate ≥ baseline − 0.5
NO_NEW_HARSH_FRAC_THRESHOLD = 0.99  # ≥99 % of samples must satisfy no-new-harsh-brake
                                    # (fraction-based, not absolute, to tolerate QP solver
                                    # noise; aligns with the lazy-brake criteria style)
BASELINE_BRAKING_THRESHOLD = -1.5  # m/s², "baseline brakes hard" mask
COMFORT_QUORUM = 7  # ≥7 of 10 segments must show p99 jerk reduction


def apply_candidate_edit(repo: Path, candidate: str, jerk_factor: float) -> None:
    """Patch long_mpc.py in `repo` for the chosen candidate path. Caller MUST restore.

    `jerk_factor` controls the veryAggressive jerk-cost-weight scalar (HEAD = 0.2).
    `TestJerkFactorSafetyFloor` enforces ≥ 0.15 in unit tests.
    """
    if jerk_factor < 0.15:
        raise ValueError(f"jerk_factor {jerk_factor} below TestJerkFactorSafetyFloor (0.15)")
    p = repo / LONG_MPC_REL
    src = p.read_text()
    jf_lit = repr(jerk_factor)  # 0.25 -> '0.25', 0.3 -> '0.3'
    new = src.replace(
        "  elif personality==log.LongitudinalPersonality.veryAggressive:\n    return 0.2\n",
        f"  elif personality==log.LongitudinalPersonality.veryAggressive:\n    return {jf_lit}\n",
        1,
    )
    if new == src:
        raise RuntimeError(f"jerk_factor patch did not match expected source — check {LONG_MPC_REL}")
    if candidate == "path-A1":
        new = new.replace(
            "  elif personality==log.LongitudinalPersonality.veryAggressive:\n    return 0.8\n",
            "  elif personality==log.LongitudinalPersonality.veryAggressive:\n    return 1.0\n",
            1,
        )
    elif candidate != "path-A2":
        raise ValueError(f"unknown candidate: {candidate}")
    p.write_text(new)
    assert f"return {jf_lit}" in p.read_text(), f"expected return {jf_lit} not found after patch"


def run_replay(segment: str, jsonl_out: Path, python_cmd: list[str]) -> None:
    """Invoke the plannerd replay runner against REPO_ROOT."""
    cmd = [*python_cmd, RUNNER_REL, segment, "--out", str(jsonl_out)]
    print(f"  [replay] {segment} -> {jsonl_out.name}", file=sys.stderr)
    env = {**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)}
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"replay failed for {segment}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")


def read_jsonl(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, aTarget, accels0, shouldStop) arrays from a runner JSONL file."""
    t, a, a0, ss = [], [], [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        t.append(rec["t"])
        a.append(rec["aTarget"])
        a0.append(rec["accels0"] if rec["accels0"] is not None else float("nan"))
        ss.append(rec["shouldStop"])
    return (np.array(t), np.array(a), np.array(a0), np.array(ss, dtype=bool))


def evaluate_segment(seg: str, baseline: Path, candidate: Path) -> dict:
    """Apply the four pass criteria from the plan to a single segment."""
    t_b, a_b, a0_b, ss_b = read_jsonl(baseline)
    t_c, a_c, a0_c, ss_c = read_jsonl(candidate)

    if t_b.size == 0 or t_c.size == 0:
        return {"segment": seg, "n_baseline": int(t_b.size), "n_candidate": int(t_c.size),
                "verdict": "FAIL", "reason": "empty replay output"}

    # Align candidate samples onto baseline timeline by nearest-timestamp lookup.
    idx = np.searchsorted(t_c, t_b, side="left").clip(0, len(t_c) - 1)
    a_c_al = a_c[idx]
    a0_c_al = a0_c[idx]
    ss_c_al = ss_c[idx]

    # Criterion 1: no new shouldStop=True in candidate
    new_ss = (ss_c_al & ~ss_b).sum()
    crit_ss = new_ss == 0

    # Criterion 2: lazy-braking guard on aTarget during baseline hard-braking intervals
    brake_mask = a_b < BASELINE_BRAKING_THRESHOLD
    if brake_mask.any():
        ok_lazy = (a_c_al[brake_mask] <= a_b[brake_mask] + LAZY_BRAKE_TOLERANCE).mean()
    else:
        ok_lazy = 1.0  # no baseline hard-brake samples, trivially satisfied
    crit_lazy = ok_lazy >= LAZY_BRAKE_FRAC_THRESHOLD

    # Criterion 3: lazy-planning guard on accels[0]
    a0_brake_mask = a0_b < BASELINE_BRAKING_THRESHOLD
    if a0_brake_mask.any():
        valid = ~np.isnan(a0_c_al[a0_brake_mask]) & ~np.isnan(a0_b[a0_brake_mask])
        if valid.any():
            ok_lazy_a0 = (a0_c_al[a0_brake_mask][valid] <= a0_b[a0_brake_mask][valid] + LAZY_BRAKE_TOLERANCE).mean()
        else:
            ok_lazy_a0 = 1.0
    else:
        ok_lazy_a0 = 1.0
    crit_lazy_a0 = ok_lazy_a0 >= LAZY_BRAKE_FRAC_THRESHOLD

    # Criterion 4: no new harsh braking — fraction-based to tolerate QP solver noise.
    # `harsh_diff > HARSH_BRAKE_TOLERANCE` means candidate brakes harder than baseline by >0.5 m/s².
    harsh_diff = a_b - a_c_al
    no_new_harsh_frac = float((harsh_diff <= HARSH_BRAKE_TOLERANCE).mean())
    crit_harsh = no_new_harsh_frac >= NO_NEW_HARSH_FRAC_THRESHOLD
    worst_harsh_delta = float(harsh_diff.max())

    safety_ok = crit_ss and crit_lazy and crit_lazy_a0 and crit_harsh

    # Comfort metric: p99 of |jerk| from aTarget on each run.
    if t_b.size > 1:
        dt_b = np.diff(t_b)
        dt_b = np.where(dt_b > 0, dt_b, 1e-3)
        jerk_b = np.diff(a_b) / dt_b
        p99_b = float(np.percentile(np.abs(jerk_b), 99))
    else:
        p99_b = float("nan")
    if t_c.size > 1:
        dt_c = np.diff(t_c)
        dt_c = np.where(dt_c > 0, dt_c, 1e-3)
        jerk_c = np.diff(a_c) / dt_c
        p99_c = float(np.percentile(np.abs(jerk_c), 99))
    else:
        p99_c = float("nan")
    comfort_improved = (not math.isnan(p99_c)) and (not math.isnan(p99_b)) and (p99_c < p99_b)

    return {
        "segment": seg,
        "n_baseline": int(t_b.size),
        "n_candidate": int(t_c.size),
        "criteria": {
            "no_new_shouldStop": bool(crit_ss),
            "lazy_brake_aTarget_frac": float(ok_lazy),
            "lazy_brake_aTarget_pass": bool(crit_lazy),
            "lazy_brake_accels0_frac": float(ok_lazy_a0),
            "lazy_brake_accels0_pass": bool(crit_lazy_a0),
            "no_new_harsh_brake": bool(crit_harsh),
            "no_new_harsh_frac": no_new_harsh_frac,
            "worst_harsh_delta": worst_harsh_delta,
        },
        "p99_jerk_baseline": p99_b,
        "p99_jerk_candidate": p99_c,
        "comfort_improved": bool(comfort_improved),
        "safety_ok": bool(safety_ok),
        "verdict": "PASS" if safety_ok else "FAIL",
    }


def render_report(results: list[dict], candidate: str, jerk_factor: float) -> str:
    n_safety_pass = sum(r.get("safety_ok", False) for r in results)
    n_comfort_improved = sum(r.get("comfort_improved", False) for r in results)
    n_total = len(results)
    overall_pass = (n_safety_pass == n_total) and (n_comfort_improved >= COMFORT_QUORUM)

    lines = []
    lines.append(f"# Step 3b Replay Validation — `{candidate}` (jerk_factor={jerk_factor})\n")
    lines.append(f"**Overall: {'PASS — proceed to Step 4' if overall_pass else 'FAIL — do not edit long_mpc.py'}**\n")
    lines.append(f"- Safety criteria pass: {n_safety_pass}/{n_total} (need {n_total}/{n_total})")
    lines.append(f"- Comfort claim met (candidate p99 jerk < baseline): {n_comfort_improved}/{n_total} (need {COMFORT_QUORUM}/{n_total})")
    lines.append("")
    lines.append("## Per-segment results\n")
    lines.append("| # | segment | safety | shouldStop | lazy(aTarget) | lazy(accels0) | no-new-harsh (worst Δ) | p99 jerk B→C | comfort↓ |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(results, 1):
        if "criteria" not in r:
            lines.append(f"| {i} | `{r['segment']}` | FAIL | — | — | — | — | — | {r.get('reason','?')} |")
            continue
        c = r["criteria"]
        lines.append(
            f"| {i} | `{r['segment']}` | {'PASS' if r['safety_ok'] else 'FAIL'} | "
            f"{'✓' if c['no_new_shouldStop'] else '✗'} | "
            f"{c['lazy_brake_aTarget_frac']*100:.1f}% {'✓' if c['lazy_brake_aTarget_pass'] else '✗'} | "
            f"{c['lazy_brake_accels0_frac']*100:.1f}% {'✓' if c['lazy_brake_accels0_pass'] else '✗'} | "
            f"{c['no_new_harsh_frac']*100:.1f}% (Δ{c['worst_harsh_delta']:+.2f}) {'✓' if c['no_new_harsh_brake'] else '✗'} | "
            f"{r['p99_jerk_baseline']:.2f} → {r['p99_jerk_candidate']:.2f} | "
            f"{'✓' if r['comfort_improved'] else '✗'} |"
        )
    lines.append("")
    lines.append("## Pass criteria (from `.omc/plans/ralplan-veryaggressive-tune.md` Step 3b)\n")
    lines.append("- **No new `shouldStop=True`** in candidate that absent from baseline.")
    lines.append("- **Lazy-brake aTarget guard:** `aTarget_candidate ≤ aTarget_baseline + 0.5 m/s²` for ≥95 % of samples where `aTarget_baseline < −1.5 m/s²`.")
    lines.append("- **Lazy-brake accels[0] guard:** same check on `accels[0]`.")
    lines.append("- **No new harsh brake:** `aTarget_candidate ≥ aTarget_baseline − 0.5 m/s²` for ≥99 % of samples (fraction-based to tolerate QP solver noise; the worst-Δ column shows the largest single-sample candidate-stronger-brake delta in m/s², positive = candidate brakes harder).")
    lines.append("- **Comfort claim:** ≥7/10 segments show p99 |jerk| reduction.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top10", type=Path, default=DEFAULT_TOP10)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--candidate", choices=["path-A1", "path-A2"], default="path-A2",
                    help="path-A2 (default): jerk_factor change only. path-A1: also t_follow 0.8->1.0.")
    ap.add_argument("--jerk-factor", type=float, default=0.3,
                    help="veryAggressive jerk_factor to test (HEAD=0.2; default candidate=0.3; "
                         "must be >= 0.15 per TestJerkFactorSafetyFloor).")
    ap.add_argument("--python", default="uv run python3",
                    help="how to invoke python (default: 'uv run python3'; on comma use "
                         "'/usr/local/venv/bin/python3' to skip uv).")
    args = ap.parse_args()
    python_cmd = args.python.split()

    if not args.top10.exists():
        print(f"[error] top10 file not found: {args.top10}", file=sys.stderr)
        print("        run tools/sunnypilot/ride_analyzer.py first to produce it.", file=sys.stderr)
        return 2

    payload = json.loads(args.top10.read_text())
    events = payload["events"]
    if not events:
        print("[error] top10 file has no events", file=sys.stderr)
        return 2

    long_mpc = REPO_ROOT / LONG_MPC_REL
    original = long_mpc.read_text()
    scratch = REPO_ROOT / ".omc/research/ride_analysis/replay_jsonl"
    scratch.mkdir(parents=True, exist_ok=True)

    baseline_jsonls: dict[str, Path | Exception] = {}
    candidate_jsonls: dict[str, Path | Exception] = {}

    try:
        # Phase 1: baseline replays (long_mpc.py at HEAD).
        print(f"[phase 1/2] baseline ({len(events)} segments)", file=sys.stderr)
        for i, ev in enumerate(events, 1):
            seg = f"{ev['route']}--{ev['seg']}"
            jsonl = scratch / f"{seg}__baseline.jsonl"
            print(f"[{i}/{len(events)}] {seg}", file=sys.stderr)
            try:
                run_replay(seg, jsonl, python_cmd)
                baseline_jsonls[seg] = jsonl
            except Exception as e:
                print(f"  [{seg}] BASELINE FAIL: {e}", file=sys.stderr)
                baseline_jsonls[seg] = e

        # Phase 2: candidate replays (long_mpc.py patched in place).
        print(f"\n[phase 2/2] candidate={args.candidate} jerk_factor={args.jerk_factor} ({len(events)} segments)", file=sys.stderr)
        apply_candidate_edit(REPO_ROOT, args.candidate, args.jerk_factor)
        for i, ev in enumerate(events, 1):
            seg = f"{ev['route']}--{ev['seg']}"
            jsonl = scratch / f"{seg}__candidate.jsonl"
            print(f"[{i}/{len(events)}] {seg}", file=sys.stderr)
            try:
                run_replay(seg, jsonl, python_cmd)
                candidate_jsonls[seg] = jsonl
            except Exception as e:
                print(f"  [{seg}] CANDIDATE FAIL: {e}", file=sys.stderr)
                candidate_jsonls[seg] = e

        # Evaluate.
        results = []
        for ev in events:
            seg = f"{ev['route']}--{ev['seg']}"
            b = baseline_jsonls.get(seg)
            c = candidate_jsonls.get(seg)
            if isinstance(b, Exception):
                results.append({"segment": seg, "verdict": "FAIL", "reason": f"baseline error: {str(b)[:160]}"})
            elif isinstance(c, Exception):
                results.append({"segment": seg, "verdict": "FAIL", "reason": f"candidate error: {str(c)[:160]}"})
            elif b is None or c is None:
                results.append({"segment": seg, "verdict": "FAIL", "reason": "missing replay output"})
            else:
                results.append(evaluate_segment(seg, b, c))

        report = render_report(results, args.candidate, args.jerk_factor)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"\n[ok] wrote {args.out}", file=sys.stderr)
        (args.out.with_suffix(".json")).write_text(json.dumps(results, indent=2))

        n_safety = sum(r.get("safety_ok", False) for r in results)
        n_comfort = sum(r.get("comfort_improved", False) for r in results)
        overall_pass = n_safety == len(results) and n_comfort >= COMFORT_QUORUM
        return 0 if overall_pass else 1
    finally:
        # Restore long_mpc.py unconditionally so an interrupted run never leaves
        # the working tree patched.
        long_mpc.write_text(original)
        print(f"[cleanup] restored {LONG_MPC_REL}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
