#!/usr/bin/env python3
"""Rivian ride-quality analyzer.

Walks all routes under ~/sunnypilot-logs, filters to RIVIAN_R1, and
computes per-segment ride-quality metrics from rlog.zst. Emits:

  .omc/research/ride_analysis/segments.csv  (one row per segment)
  .omc/research/ride_analysis/routes.csv    (aggregated per route)
  .omc/research/ride_analysis/report.md     (human-readable summary)
  .omc/research/ride_analysis/plots/*.png   (worst-offender traces)

Run from repo root:  uv run tools/sunnypilot/ride_analyzer.py
"""
from __future__ import annotations

import csv
import os
import re
import sys
import math
import argparse
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np

from openpilot.tools.lib.logreader import LogReader

LOG_ROOT = Path(os.path.expanduser("~/sunnypilot-logs"))
OUT_ROOT = Path(".omc/research/ride_analysis")
PLOT_DIR = OUT_ROOT / "plots"

SEG_RE = re.compile(r"^(?P<route>[0-9a-f]{8}--[0-9a-f]{10})--(?P<seg>\d+)$")
RIVIAN_FPS = {"RIVIAN_R1", "RIVIAN_R1T_GEN1"}

# LongitudinalPersonality enum (cereal/log.capnp:140-145).
PERSONALITY_NAME = {0: "aggressive", 1: "standard", 2: "relaxed", 3: "veryAggressive"}
# Per-personality t_follow setpoint (selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py:78-89).
T_FOLLOW = {0: 1.25, 1: 1.45, 2: 1.75, 3: 0.8}
# Sample-level dominance threshold for the per-segment "personality_dominant" tag.
DOMINANT_THRESHOLD = 0.80

# Metrics we care about — keep keys stable; CSV header is built from this list.
# NOTE: appended-only schema (acceptance criterion 5). Existing consumers see all
# previous columns in the same positions; new lead/personality columns are at the end.
SEG_FIELDS = [
    "route", "seg", "fingerprint", "duration_s",
    "v_mean_mps", "v_max_mps",
    # engagement
    "frac_enabled", "frac_lat_active", "frac_long_active",
    "n_disengage", "n_steer_override", "n_brake_override", "n_gas_override",
    # longitudinal
    "a_rms", "a_p95_decel", "a_p95_accel", "long_jerk_rms", "long_jerk_p99",
    "a_err_rms", "a_err_p95",  # actuators.accel - aEgo (when longActive)
    "n_hard_decel",            # |aEgo| > 3.0
    # lateral
    "lat_a_rms", "lat_a_p95",
    "lat_jerk_rms", "lat_jerk_p99",
    "curv_err_rms", "curv_err_p95",
    "yaw_rate_rms",
    "steer_torque_rms", "steer_torque_rate_rms",
    "angle_offset_mean", "steer_ratio_mean",
    # cumLag
    "cum_lag_p95_ms",
    # NEW (appended) — personality + lead-following per-segment summary
    "personality_dominant", "t_headway_mean", "frac_lead_close",
]


def _safe_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(x))))


def _pctile(x: np.ndarray, p: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, p))


def analyze_segment(seg_path: Path) -> dict | None:
    """Compute ride-quality metrics for a single segment.

    Returns None if the segment is not a Rivian or is unreadable.
    """
    m = SEG_RE.match(seg_path.name)
    if not m:
        return None
    route, seg_idx = m.group("route"), int(m.group("seg"))
    rlog = seg_path / "rlog.zst"
    if not rlog.exists():
        return None

    fp = None
    # Per-message buffers (100 Hz)
    t_cs, v, a_ego, steer_angle, steer_torque, yaw_rate, cum_lag, cur_curv = ([] for _ in range(8))
    enabled, lat_active, long_active = [], [], []
    steer_pressed, brake_pressed, gas_pressed = [], [], []
    desired_curv, actuator_accel = [], []
    # 20 Hz signals
    angle_offset_avg, steer_ratio = [], []
    # selfdriveState — personality enum, latched
    t_sds, sds_personality = [], []
    # radarState.leadOne — for headway / closing-rate metrics
    t_radar, lead_status, lead_dRel, lead_vLead = [], [], [], []

    try:
        for msg in LogReader(str(rlog)):
            w = msg.which()
            if w == "carParams":
                fp = msg.carParams.carFingerprint
                if fp not in RIVIAN_FPS:
                    return None
            elif w == "carState":
                cs = msg.carState
                t_cs.append(msg.logMonoTime * 1e-9)
                v.append(cs.vEgo)
                a_ego.append(cs.aEgo)
                steer_angle.append(cs.steeringAngleDeg)
                steer_torque.append(cs.steeringTorque)
                yaw_rate.append(cs.yawRate)
                cum_lag.append(cs.cumLagMs)
                steer_pressed.append(cs.steeringPressed)
                brake_pressed.append(cs.brakePressed)
                gas_pressed.append(cs.gasPressed)
            elif w == "carControl":
                cc = msg.carControl
                enabled.append(cc.enabled)
                lat_active.append(cc.latActive)
                long_active.append(cc.longActive)
                cur_curv.append(cc.currentCurvature)
                actuator_accel.append(cc.actuators.accel)
            elif w == "controlsState":
                desired_curv.append(msg.controlsState.desiredCurvature)
            elif w == "liveParameters":
                angle_offset_avg.append(msg.liveParameters.angleOffsetAverageDeg)
                steer_ratio.append(msg.liveParameters.steerRatio)
            elif w == "selfdriveState":
                t_sds.append(msg.logMonoTime * 1e-9)
                # personality is a Capnp enum (.value gives the int 0–3); fall back to int().
                p = msg.selfdriveState.personality
                sds_personality.append(int(p) if not hasattr(p, "raw") else p.raw)
            elif w == "radarState":
                t_radar.append(msg.logMonoTime * 1e-9)
                lo = msg.radarState.leadOne
                lead_status.append(lo.status)
                lead_dRel.append(lo.dRel if lo.status else 0.0)
                lead_vLead.append(lo.vLead if lo.status else 0.0)
    except Exception as e:
        sys.stderr.write(f"[skip] {seg_path.name}: {e}\n")
        return None

    if fp not in RIVIAN_FPS or len(t_cs) < 200:
        return None

    t = np.asarray(t_cs)
    v = np.asarray(v)
    a_ego = np.asarray(a_ego)
    steer_angle = np.asarray(steer_angle)
    steer_torque = np.asarray(steer_torque)
    yaw_rate = np.asarray(yaw_rate)
    cum_lag = np.asarray(cum_lag)
    enabled = np.asarray(enabled, dtype=bool)
    lat_active = np.asarray(lat_active, dtype=bool)
    long_active = np.asarray(long_active, dtype=bool)
    steer_pressed = np.asarray(steer_pressed, dtype=bool)
    brake_pressed = np.asarray(brake_pressed, dtype=bool)
    gas_pressed = np.asarray(gas_pressed, dtype=bool)
    cur_curv_arr = np.asarray(cur_curv) if cur_curv else np.array([])
    desired_curv_arr = np.asarray(desired_curv) if desired_curv else np.array([])
    actuator_accel_arr = np.asarray(actuator_accel) if actuator_accel else np.array([])

    # Trim cc-aligned arrays to carState length (they're paired in time but counted
    # independently). Use min length.
    n = min(len(t), len(enabled), len(actuator_accel_arr) or len(t), len(cur_curv_arr) or len(t))
    t, v, a_ego = t[:n], v[:n], a_ego[:n]
    steer_angle, steer_torque, yaw_rate = steer_angle[:n], steer_torque[:n], yaw_rate[:n]
    cum_lag = cum_lag[:n]
    enabled, lat_active, long_active = enabled[:n], lat_active[:n], long_active[:n]
    steer_pressed, brake_pressed, gas_pressed = steer_pressed[:n], brake_pressed[:n], gas_pressed[:n]
    if len(cur_curv_arr): cur_curv_arr = cur_curv_arr[:n]
    if len(actuator_accel_arr): actuator_accel_arr = actuator_accel_arr[:n]

    dt = np.diff(t)
    dt = np.where(dt > 0, dt, 1e-3)
    duration = float(t[-1] - t[0])

    # Helper: compute Δx/dt only where the active mask holds on both endpoints.
    # This avoids counting jerks across engage/disengage transitions.
    def _gated_diff(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if x.size < 2:
            return np.array([])
        d = np.diff(x) / dt
        m = mask[:-1] & mask[1:]
        return d[m]

    # Longitudinal — only while longActive
    long_jerk = _gated_diff(a_ego, long_active)
    a_ego_long = a_ego[long_active]
    a_err = (actuator_accel_arr - a_ego)[long_active] if len(actuator_accel_arr) == len(a_ego) else np.array([])
    decel_mask = a_ego_long < 0
    accel_mask = a_ego_long >= 0
    # Hard decel: rising edge of |aEgo|>3 within longActive
    hard = (a_ego < -3.0) & long_active
    n_hard_decel = int(np.sum(np.diff(hard.astype(int)) == 1))

    # Lateral — only while latActive
    if cur_curv_arr.size:
        lat_a_full = v * v * cur_curv_arr
        lat_a = lat_a_full[lat_active]
        lat_jerk = _gated_diff(lat_a_full, lat_active)
    else:
        lat_a = np.array([])
        lat_jerk = np.array([])

    if cur_curv_arr.size and desired_curv_arr.size:
        ml = min(len(cur_curv_arr), len(desired_curv_arr))
        la = lat_active[:ml]
        curv_err = (cur_curv_arr[:ml] - desired_curv_arr[:ml])[la]
    else:
        curv_err = np.array([])

    steer_torque_rate = _gated_diff(steer_torque, lat_active)
    yaw_rate_active = yaw_rate[lat_active]
    steer_torque_active = steer_torque[lat_active]

    # Sample-align personality (selfdriveState) and lead (radarState) onto carState timeline.
    # Last-known-personality semantics — what the planner sees at each carState tick.
    if t_sds:
        t_sds_arr = np.asarray(t_sds)
        idx = np.searchsorted(t_sds_arr, t, side="right") - 1
        idx = np.clip(idx, 0, len(t_sds_arr) - 1)
        personality_per_sample = np.asarray(sds_personality, dtype=np.int8)[idx]
    else:
        personality_per_sample = np.full(n, -1, dtype=np.int8)

    if t_radar:
        t_radar_arr = np.asarray(t_radar)
        idx = np.searchsorted(t_radar_arr, t, side="right") - 1
        idx = np.clip(idx, 0, len(t_radar_arr) - 1)
        lead_status_arr = np.asarray(lead_status, dtype=bool)[idx]
        lead_dRel_arr = np.asarray(lead_dRel, dtype=float)[idx]
        lead_vLead_arr = np.asarray(lead_vLead, dtype=float)[idx]
    else:
        lead_status_arr = np.zeros(n, dtype=bool)
        lead_dRel_arr = np.zeros(n)
        lead_vLead_arr = np.zeros(n)

    # Per-personality buckets (sample-level masking on personality AND longActive).
    # Each bucket carries raw arrays so the main process can aggregate across segments.
    buckets: dict[int, dict] = {}
    for p in (0, 1, 2, 3):
        mask_p = (personality_per_sample == p) & long_active
        if mask_p.sum() < 50:
            continue
        # Long-jerk samples (gated on the same mask, only across consecutive in-mask pairs).
        jerk_p = _gated_diff(a_ego, mask_p)
        # Lead metrics: where lead is present, vEgo > 1, mask still holds.
        lead_mask_p = mask_p & lead_status_arr & (v > 1.0)
        if lead_mask_p.any():
            t_headway = lead_dRel_arr[lead_mask_p] / v[lead_mask_p]
            closing_rate = v[lead_mask_p] - lead_vLead_arr[lead_mask_p]
            # frac_lead_close: fraction of lead-present time below the personality's t_follow setpoint.
            t_follow_p = T_FOLLOW.get(p, 1.45)
            below = lead_dRel_arr[lead_mask_p] < (t_follow_p * v[lead_mask_p])
            frac_lead_close_p = float(np.mean(below)) if below.size else float("nan")
        else:
            t_headway = np.array([])
            closing_rate = np.array([])
            frac_lead_close_p = float("nan")

        # Peak |jerk| event for this segment+personality (used for top-N event list).
        peak_jerk_abs = float(np.max(np.abs(jerk_p))) if jerk_p.size else 0.0
        # Approximate offset within segment of the peak event:
        if jerk_p.size:
            full_jerk = np.abs(np.diff(a_ego) / dt)
            full_mask = mask_p[:-1] & mask_p[1:]
            full_jerk_masked = np.where(full_mask, full_jerk, -1.0)
            peak_idx = int(np.argmax(full_jerk_masked))
            peak_offset_s = float(t[peak_idx] - t[0])
        else:
            peak_offset_s = float("nan")

        buckets[p] = dict(
            n_samples=int(mask_p.sum()),
            n_lead_samples=int(lead_mask_p.sum()),
            a_ego=a_ego[mask_p],
            jerk=jerk_p,
            t_headway=t_headway,
            closing_rate=closing_rate,
            frac_lead_close=frac_lead_close_p,
            peak_jerk_abs=peak_jerk_abs,
            peak_offset_s=peak_offset_s,
        )

    # Per-segment "personality_dominant" tag — the personality covering ≥80 % of long-active samples.
    long_active_count = int(long_active.sum())
    personality_dominant = "mixed"
    if long_active_count > 0:
        for p, bucket in buckets.items():
            if bucket["n_samples"] / long_active_count >= DOMINANT_THRESHOLD:
                personality_dominant = PERSONALITY_NAME[p]
                break
    elif long_active_count == 0:
        personality_dominant = "n/a"

    # Per-segment lead-following summary (across all longActive samples, regardless of personality).
    lead_long_mask = long_active & lead_status_arr & (v > 1.0)
    if lead_long_mask.any():
        t_headway_seg = lead_dRel_arr[lead_long_mask] / v[lead_long_mask]
        t_headway_mean_seg = float(np.mean(t_headway_seg))
        # Use whichever personality is dominant (or 0.8 as a conservative default if mixed).
        t_follow_seg = T_FOLLOW.get(int(np.median(personality_per_sample[lead_long_mask])), 0.8)
        below = lead_dRel_arr[lead_long_mask] < (t_follow_seg * v[lead_long_mask])
        frac_lead_close_seg = float(np.mean(below)) if below.size else float("nan")
    else:
        t_headway_mean_seg = float("nan")
        frac_lead_close_seg = float("nan")

    # Engagement / overrides — count rising edges
    def _rising(b: np.ndarray) -> int:
        if b.size < 2: return 0
        return int(np.sum(np.diff(b.astype(int)) == 1))

    # disengage = enabled goes 1 -> 0
    n_disengage = int(np.sum(np.diff(enabled.astype(int)) == -1))
    n_steer_ovr = _rising(steer_pressed & enabled)
    n_brake_ovr = _rising(brake_pressed & enabled)
    n_gas_ovr = _rising(gas_pressed & enabled)

    seg_row = dict(
        route=route, seg=seg_idx, fingerprint=fp, duration_s=round(duration, 2),
        v_mean_mps=round(float(np.mean(v)), 3),
        v_max_mps=round(float(np.max(v)), 3),
        frac_enabled=round(float(np.mean(enabled)), 4),
        frac_lat_active=round(float(np.mean(lat_active)), 4),
        frac_long_active=round(float(np.mean(long_active)), 4),
        n_disengage=n_disengage,
        n_steer_override=n_steer_ovr,
        n_brake_override=n_brake_ovr,
        n_gas_override=n_gas_ovr,
        a_rms=round(_safe_rms(a_ego_long), 4) if a_ego_long.size else float("nan"),
        a_p95_decel=round(_pctile(-a_ego_long[decel_mask], 95) if decel_mask.any() else float("nan"), 4),
        a_p95_accel=round(_pctile(a_ego_long[accel_mask], 95) if accel_mask.any() else float("nan"), 4),
        long_jerk_rms=round(_safe_rms(long_jerk), 4) if long_jerk.size else float("nan"),
        long_jerk_p99=round(_pctile(np.abs(long_jerk), 99), 4) if long_jerk.size else float("nan"),
        a_err_rms=round(_safe_rms(a_err), 4) if a_err.size else float("nan"),
        a_err_p95=round(_pctile(np.abs(a_err), 95), 4) if a_err.size else float("nan"),
        n_hard_decel=n_hard_decel,
        lat_a_rms=round(_safe_rms(lat_a), 4) if lat_a.size else float("nan"),
        lat_a_p95=round(_pctile(np.abs(lat_a), 95), 4) if lat_a.size else float("nan"),
        lat_jerk_rms=round(_safe_rms(lat_jerk), 4) if lat_jerk.size else float("nan"),
        lat_jerk_p99=round(_pctile(np.abs(lat_jerk), 99), 4) if lat_jerk.size else float("nan"),
        curv_err_rms=round(_safe_rms(curv_err), 6) if curv_err.size else float("nan"),
        curv_err_p95=round(_pctile(np.abs(curv_err), 95), 6) if curv_err.size else float("nan"),
        yaw_rate_rms=round(_safe_rms(yaw_rate_active), 4) if yaw_rate_active.size else float("nan"),
        steer_torque_rms=round(_safe_rms(steer_torque_active), 4) if steer_torque_active.size else float("nan"),
        steer_torque_rate_rms=round(_safe_rms(steer_torque_rate), 4) if steer_torque_rate.size else float("nan"),
        angle_offset_mean=round(float(np.mean(angle_offset_avg)), 4) if angle_offset_avg else float("nan"),
        steer_ratio_mean=round(float(np.mean(steer_ratio)), 4) if steer_ratio else float("nan"),
        cum_lag_p95_ms=round(_pctile(np.abs(cum_lag), 95), 3),
        # NEW (appended) — personality + lead summary
        personality_dominant=personality_dominant,
        t_headway_mean=round(t_headway_mean_seg, 3) if not math.isnan(t_headway_mean_seg) else float("nan"),
        frac_lead_close=round(frac_lead_close_seg, 4) if not math.isnan(frac_lead_close_seg) else float("nan"),
    )

    return {"seg_row": seg_row, "buckets": buckets}


def _agg(rows: list[dict]) -> dict:
    """Aggregate per-segment metrics into one route summary row."""
    if not rows:
        return {}
    out = {"route": rows[0]["route"], "n_segs": len(rows)}
    # weighted by duration where it makes sense
    durations = np.array([r["duration_s"] for r in rows], dtype=float)
    total_dur = durations.sum()
    out["total_minutes"] = round(total_dur / 60, 2)

    def wmean(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        m = ~np.isnan(vals)
        if not m.any():
            return float("nan")
        return float(np.sum(vals[m] * durations[m]) / np.sum(durations[m]))

    def smax(key):
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        return max(vals) if vals else float("nan")

    def ssum(key):
        return int(sum(r[key] for r in rows))

    for k in ("v_mean_mps", "frac_enabled", "frac_lat_active", "frac_long_active",
             "a_rms", "long_jerk_rms", "a_err_rms", "lat_a_rms", "lat_jerk_rms",
             "curv_err_rms", "yaw_rate_rms", "steer_torque_rms", "steer_torque_rate_rms",
             "angle_offset_mean", "steer_ratio_mean"):
        out[k] = round(wmean(k), 6)
    for k in ("a_p95_decel", "a_p95_accel", "long_jerk_p99", "a_err_p95",
             "lat_a_p95", "lat_jerk_p99", "curv_err_p95", "cum_lag_p95_ms", "v_max_mps"):
        out[k] = round(smax(k), 6)
    for k in ("n_disengage", "n_steer_override", "n_brake_override", "n_gas_override",
             "n_hard_decel"):
        out[k] = ssum(k)
    return out


def _markdown_report(seg_rows: list[dict], route_rows: list[dict]) -> str:
    lines = []
    lines.append("# Rivian Ride-Quality Analysis\n")
    total_min = sum(r["total_minutes"] for r in route_rows)
    lines.append(f"**{len(route_rows)} routes, {len(seg_rows)} segments, {total_min:.1f} min of driving**\n")
    lines.append("\n## Per-route summary (sorted by lateral jerk RMS)\n")
    rs = sorted(route_rows, key=lambda r: -r.get("lat_jerk_rms", 0))
    lines.append("| route | min | v̄ (m/s) | enabled | lat jerk RMS | lat jerk p99 | long jerk RMS | long jerk p99 | curv err p95 | a err RMS | hard decel | disengage |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        lines.append(
            f"| `{r['route']}` | {r['total_minutes']:.1f} | {r['v_mean_mps']:.1f} | "
            f"{r['frac_enabled']*100:.0f}% | {r['lat_jerk_rms']:.3f} | {r['lat_jerk_p99']:.3f} | "
            f"{r['long_jerk_rms']:.3f} | {r['long_jerk_p99']:.3f} | {r['curv_err_p95']:.2e} | "
            f"{r['a_err_rms']:.3f} | {r['n_hard_decel']} | {r['n_disengage']} |"
        )

    lines.append("\n## Worst segments — lateral jerk p99 (top 10)\n")
    top = sorted(seg_rows, key=lambda r: -r.get("lat_jerk_p99", 0))[:10]
    lines.append("| route | seg | dur | v̄ | lat jerk p99 | curv err p95 | steer torque rate RMS |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in top:
        lines.append(
            f"| `{r['route']}` | {r['seg']} | {r['duration_s']:.1f} | {r['v_mean_mps']:.1f} | "
            f"{r['lat_jerk_p99']:.3f} | {r['curv_err_p95']:.2e} | {r['steer_torque_rate_rms']:.3f} |"
        )

    lines.append("\n## Worst segments — longitudinal jerk p99 (top 10)\n")
    top = sorted(seg_rows, key=lambda r: -r.get("long_jerk_p99", 0))[:10]
    lines.append("| route | seg | dur | v̄ | long jerk p99 | a err p95 | hard decel |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in top:
        lines.append(
            f"| `{r['route']}` | {r['seg']} | {r['duration_s']:.1f} | {r['v_mean_mps']:.1f} | "
            f"{r['long_jerk_p99']:.3f} | {r['a_err_p95']:.3f} | {r['n_hard_decel']} |"
        )

    lines.append("\n## Worst segments — accel-tracking error (a_err RMS, top 10)\n")
    top = sorted(seg_rows, key=lambda r: -r.get("a_err_rms", 0) if not math.isnan(r.get("a_err_rms", 0)) else 0)[:10]
    lines.append("| route | seg | dur | v̄ | a err RMS | a err p95 |")
    lines.append("|---|---|---|---|---|---|")
    for r in top:
        lines.append(
            f"| `{r['route']}` | {r['seg']} | {r['duration_s']:.1f} | {r['v_mean_mps']:.1f} | "
            f"{r['a_err_rms']:.3f} | {r['a_err_p95']:.3f} |"
        )

    lines.append("")
    lines.append("## Notes / column glossary\n")
    lines.append("- **lat jerk** (m/s³): time-derivative of v²·κ, where κ is `carControl.currentCurvature`. RMS is overall smoothness; p99 catches transient yanks.\n"
                 "- **curv err**: `currentCurvature − controlsState.desiredCurvature` while `latActive`. High p95 = lateral controller leaving error on the table.\n"
                 "- **long jerk** (m/s³): time-derivative of `aEgo`. p99 captures stop/start judder.\n"
                 "- **a err**: `actuators.accel − aEgo` while `longActive`. RMS = systematic plan-vs-actual drift; p95 = the one-off misses.\n"
                 "- **hard decel**: count of rising edges where `aEgo < −3 m/s²` (≈hard-brake events).\n")
    return "\n".join(lines)


def _aggregate_personality_buckets(segment_results: list[dict]) -> dict:
    """Concatenate per-segment bucket arrays into per-personality aggregates."""
    by_pers: dict[int, dict] = {}
    # For top-N events we need per-segment route/seg references too.
    events: dict[int, list[tuple]] = {p: [] for p in (0, 1, 2, 3)}
    for sr in segment_results:
        seg = sr["seg_row"]
        for p, bucket in sr["buckets"].items():
            agg = by_pers.setdefault(p, {
                "n_samples": 0, "n_lead_samples": 0,
                "a_ego_chunks": [], "jerk_chunks": [],
                "t_headway_chunks": [], "closing_rate_chunks": [],
                "frac_lead_close_weighted_sum": 0.0, "frac_lead_close_weight": 0.0,
            })
            agg["n_samples"] += bucket["n_samples"]
            agg["n_lead_samples"] += bucket["n_lead_samples"]
            agg["a_ego_chunks"].append(bucket["a_ego"])
            agg["jerk_chunks"].append(bucket["jerk"])
            agg["t_headway_chunks"].append(bucket["t_headway"])
            agg["closing_rate_chunks"].append(bucket["closing_rate"])
            if not math.isnan(bucket["frac_lead_close"]) and bucket["n_lead_samples"] > 0:
                agg["frac_lead_close_weighted_sum"] += bucket["frac_lead_close"] * bucket["n_lead_samples"]
                agg["frac_lead_close_weight"] += bucket["n_lead_samples"]
            if bucket["jerk"].size > 0:
                events[p].append((
                    seg["route"], seg["seg"], bucket["peak_offset_s"],
                    bucket["peak_jerk_abs"], seg.get("v_mean_mps", float("nan")),
                ))
    out = {}
    for p, agg in by_pers.items():
        a_ego = np.concatenate(agg["a_ego_chunks"]) if agg["a_ego_chunks"] else np.array([])
        jerk = np.concatenate(agg["jerk_chunks"]) if agg["jerk_chunks"] else np.array([])
        thead = np.concatenate(agg["t_headway_chunks"]) if agg["t_headway_chunks"] else np.array([])
        crate = np.concatenate(agg["closing_rate_chunks"]) if agg["closing_rate_chunks"] else np.array([])
        out[p] = {
            "name": PERSONALITY_NAME[p],
            "t_follow_setpoint": T_FOLLOW[p],
            "n_samples": agg["n_samples"],
            "n_lead_samples": agg["n_lead_samples"],
            "total_minutes": round(agg["n_samples"] / 100.0 / 60.0, 2),
            "a_rms": round(_safe_rms(a_ego), 4),
            "a_p95_decel": round(_pctile(-a_ego[a_ego < 0], 95) if (a_ego < 0).any() else float("nan"), 4),
            "long_jerk_rms": round(_safe_rms(jerk), 4),
            "long_jerk_p99": round(_pctile(np.abs(jerk), 99), 4),
            "n_disengage": int(sum(sr["seg_row"]["n_disengage"] for sr in segment_results
                                   if sr["seg_row"].get("personality_dominant") == PERSONALITY_NAME[p])),
            "t_headway_mean": round(float(np.mean(thead)), 3) if thead.size else float("nan"),
            "t_headway_p10": round(_pctile(thead, 10), 3) if thead.size else float("nan"),
            "closing_rate_p95": round(_pctile(crate[crate > 0], 95) if (crate > 0).any() else float("nan"), 3),
            "frac_lead_close": round(agg["frac_lead_close_weighted_sum"] / agg["frac_lead_close_weight"], 4)
                              if agg["frac_lead_close_weight"] > 0 else float("nan"),
        }
    return {"summary": out, "events": events}


def _markdown_report_by_personality(personality_agg: dict) -> str:
    summary = personality_agg["summary"]
    events = personality_agg["events"]
    lines = []
    lines.append("# Rivian Ride-Quality Analysis by Personality\n")
    lines.append("Sample-level personality attribution: each metric is computed over samples where "
                 "`selfdriveState.personality == X` AND `carControl.longActive == True`. Lead metrics "
                 "additionally require `radarState.leadOne.status == True` and `vEgo > 1 m/s`.\n")

    lines.append("\n## Per-personality summary\n")
    lines.append("| personality | t_follow setpoint | minutes | a RMS | long jerk RMS | long jerk p99 | "
                 "t_headway mean | t_headway p10 | closing_rate p95 | frac_lead_close | n_disengage |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in (3, 0, 1, 2):  # show veryAggressive first
        if p not in summary:
            continue
        s = summary[p]
        def fmt(x, prec=3):
            return "n/a" if (isinstance(x, float) and math.isnan(x)) else f"{x:.{prec}f}"
        lines.append(
            f"| **{s['name']}** | {s['t_follow_setpoint']:.2f}s | {s['total_minutes']:.1f} | "
            f"{fmt(s['a_rms'])} | {fmt(s['long_jerk_rms'])} | {fmt(s['long_jerk_p99'])} | "
            f"{fmt(s['t_headway_mean'])} | {fmt(s['t_headway_p10'])} | {fmt(s['closing_rate_p95'])} | "
            f"{fmt(s['frac_lead_close'], 4)} | {s['n_disengage']} |"
        )

    # Mechanical Step-3 gate evaluation for veryAggressive.
    lines.append("\n## Step-3 gate (veryAggressive)\n")
    if 3 in summary:
        s = summary[3]
        cond_min = s["total_minutes"] >= 10.0
        cond_jerk = (not math.isnan(s["long_jerk_p99"])) and s["long_jerk_p99"] > 12.0
        cond_close = (not math.isnan(s["frac_lead_close"])) and s["frac_lead_close"] >= 0.30
        lines.append(f"- **Sample-size precondition** (`total_minutes >= 10`): {s['total_minutes']:.1f} min → **{'PASS' if cond_min else 'FAIL'}**")
        lines.append(f"- **Jerk precondition** (`long_jerk_p99 > 12 m/s³`): {s['long_jerk_p99']:.2f} → **{'PASS' if cond_jerk else 'FAIL'}**")
        lines.append(f"- **Headway-stress signal** (`frac_lead_close >= 0.30`): {s['frac_lead_close']} → **{'PASS' if cond_close else 'FAIL'}**\n")
        if cond_min and cond_jerk and cond_close:
            verdict = "**PATH A1** — full change (jerk_factor 0.2→0.3, t_follow 0.8s→1.0s)"
        elif cond_min and cond_jerk and not cond_close:
            verdict = "**PATH A2** — jerk-only (jerk_factor 0.2→0.3, t_follow unchanged)"
        else:
            verdict = "**PATH STOP** — preconditions not met; do not edit `long_mpc.py`"
        lines.append(f"### Routing: {verdict}\n")
    else:
        lines.append("- No `veryAggressive` samples in dataset → **PATH STOP**\n")

    # Top-10 worst-jerk events for veryAggressive only.
    lines.append("\n## Top-10 worst-jerk events — veryAggressive bucket\n")
    va_events = sorted(events.get(3, []), key=lambda e: -e[3])[:10]
    if not va_events:
        lines.append("_No veryAggressive events recorded._")
    else:
        lines.append("| # | route | seg | offset (s) | peak |jerk| (m/s³) | v̄ (m/s) |")
        lines.append("|---|---|---|---|---|---|")
        for i, (route, seg, off, peak, vmean) in enumerate(va_events, 1):
            off_s = "n/a" if math.isnan(off) else f"{off:.1f}"
            lines.append(f"| {i} | `{route}` | {seg} | {off_s} | {peak:.2f} | {vmean:.1f} |")

    lines.append("\n## Notes\n")
    lines.append("- **Sample-level attribution**: every metric in the per-personality table is computed over the sample set where the attached personality was active AND longitudinal control was active. Segment-level mode attribution is *not* used.")
    lines.append("- **frac_lead_close** is a proxy: fraction of lead-present time where `dRel < t_follow_setpoint × vEgo`. It indicates that headway stress is plausibly present, not that it caused the jerk.")
    lines.append("- **Top-10 events** are one-per-(route, segment) — peak |jerk| within each veryAggressive bucket of each segment. The `offset (s)` is the wall-clock seconds from segment start.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap segments processed (0 = all)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--filter-route", type=str, default="", help="substring filter on route id")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    seg_paths = sorted(p for p in LOG_ROOT.iterdir() if p.is_dir() and SEG_RE.match(p.name))
    if args.filter_route:
        seg_paths = [p for p in seg_paths if args.filter_route in p.name]
    if args.limit:
        seg_paths = seg_paths[: args.limit]

    print(f"[info] {len(seg_paths)} segments under {LOG_ROOT}, workers={args.workers}", file=sys.stderr)

    seg_results: list[dict] = []  # each is {"seg_row": ..., "buckets": ...}
    if args.workers <= 1:
        for i, p in enumerate(seg_paths, 1):
            r = analyze_segment(p)
            if r:
                seg_results.append(r)
            if i % 10 == 0:
                print(f"  [{i}/{len(seg_paths)}] kept {len(seg_results)}", file=sys.stderr)
    else:
        with mp.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(analyze_segment, seg_paths, chunksize=2), 1):
                if r:
                    seg_results.append(r)
                if i % 10 == 0:
                    print(f"  [{i}/{len(seg_paths)}] kept {len(seg_results)}", file=sys.stderr)

    if not seg_results:
        print("[warn] no Rivian segments analyzed", file=sys.stderr)
        sys.exit(1)

    seg_results.sort(key=lambda r: (r["seg_row"]["route"], r["seg_row"]["seg"]))
    rows = [sr["seg_row"] for sr in seg_results]

    # Per-segment CSV
    seg_csv = OUT_ROOT / "segments.csv"
    with seg_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SEG_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SEG_FIELDS})

    # Per-personality aggregation + report.
    personality_agg = _aggregate_personality_buckets(seg_results)
    (OUT_ROOT / "report_by_personality.md").write_text(_markdown_report_by_personality(personality_agg))

    # Machine-readable top-10 veryAggressive event list, consumed by the Step-3b replay harness.
    import json
    va_events = sorted(personality_agg["events"].get(3, []), key=lambda e: -e[3])[:10]
    top10_payload = {
        "personality": "veryAggressive",
        "events": [
            {"route": e[0], "seg": e[1], "offset_s": (None if math.isnan(e[2]) else e[2]),
             "peak_jerk_abs": e[3], "v_mean_mps": e[4]}
            for e in va_events
        ],
    }
    (OUT_ROOT / "top10_veryaggressive.json").write_text(json.dumps(top10_payload, indent=2))

    # Per-route aggregation
    by_route: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_route[r["route"]].append(r)
    route_rows = [_agg(v) for v in by_route.values()]
    route_rows.sort(key=lambda r: r["route"])

    route_csv = OUT_ROOT / "routes.csv"
    if route_rows:
        with route_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(route_rows[0].keys()))
            w.writeheader()
            w.writerows(route_rows)

    # Markdown summary
    md = _markdown_report(rows, route_rows)
    (OUT_ROOT / "report.md").write_text(md)

    print(f"[ok] wrote {seg_csv}, {route_csv}, {OUT_ROOT/'report.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
