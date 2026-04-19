#!/usr/bin/env python3
"""Analyze Rivian lateral-control tuning from local rlogs.

Reads rlog.zst files, buckets active lat-control frames by speed, and computes
oscillation / error / saturation / override / integrator / contribution stats.
Writes a JSON rollup and per-segment diagnostic PNGs.

Usage: python3 tools/rivian_tune_analysis.py --logs-dir /tmp/rivian_logs \
         --out-dir /tmp/rivian_analysis
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openpilot.tools.lib.logreader import LogReader

SPEED_BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 999)]
BUCKET_LABELS = [f"{lo}-{hi if hi < 999 else '+'}" for lo, hi in SPEED_BUCKETS]
CONTROL_RATE_HZ = 100.0  # controlsState publish rate
DT = 1.0 / CONTROL_RATE_HZ


def bucket_idx(v):
  for i, (lo, hi) in enumerate(SPEED_BUCKETS):
    if lo <= v < hi:
      return i
  return -1


TUNING_PARAM_KEYS = ("KpLowSpeed", "KpMidSpeed", "KpHighSpeed",
                     "KdLowSpeed", "KdMidSpeed", "KdHighSpeed",
                     "LongitudinalPersonality", "GitBranch", "GitCommit")


def parse_segment(path):
  """Extract aligned per-frame arrays keyed on logMonoTime of controlsState."""
  lr = LogReader(path)
  cs_rows, st_rows, cc_rows, lp_rows = [], [], [], []
  tuning_params = {}
  for msg in lr:
    w = msg.which()
    t = msg.logMonoTime
    if w == "initData":
      for e in msg.initData.params.entries:
        if e.key in TUNING_PARAM_KEYS:
          tuning_params[e.key] = bytes(e.value).decode(errors="replace")
      continue
    if w == "controlsState":
      lat = msg.controlsState.lateralControlState
      if lat.which() != "torqueState":
        continue
      ts = lat.torqueState
      cs_rows.append((t, ts.active, ts.saturated, ts.error, ts.p, ts.i, ts.d, ts.f,
                      ts.output, ts.desiredLateralAccel, ts.actualLateralAccel,
                      ts.desiredLateralJerk))
    elif w == "carState":
      cSt = msg.carState
      st_rows.append((t, cSt.vEgo, cSt.steeringAngleDeg, cSt.steeringTorque,
                      cSt.steeringPressed))
    elif w == "carControl":
      cc_rows.append((t, msg.carControl.actuators.torque))
    elif w == "livePose":
      lp_rows.append((t, msg.livePose.angularVelocityDevice.z,
                      msg.livePose.accelerationDevice.y))

  if not cs_rows or not st_rows:
    return None

  def arr(rows, n):
    a = np.empty((len(rows), n))
    for i, r in enumerate(rows):
      a[i] = r
    return a

  cs = arr(cs_rows, 12)
  st = arr(st_rows, 5)
  cc = arr(cc_rows, 2) if cc_rows else np.zeros((1, 2))
  lp = arr(lp_rows, 3) if lp_rows else np.zeros((1, 3))

  # Join onto controlsState timeline by nearest-neighbor in time.
  t_cs = cs[:, 0]
  def align(src, ncols):
    if len(src) < 2:
      return np.zeros((len(t_cs), ncols - 1))
    out = np.empty((len(t_cs), ncols - 1))
    idx = np.searchsorted(src[:, 0], t_cs)
    idx = np.clip(idx, 0, len(src) - 1)
    for k in range(1, ncols):
      out[:, k - 1] = src[idx, k]
    return out

  st_j = align(st, 5)
  cc_j = align(cc, 2)
  lp_j = align(lp, 3)

  return {
    "tuning_params": tuning_params,
    "t": (t_cs - t_cs[0]) * 1e-9,
    "active": cs[:, 1].astype(bool),
    "saturated": cs[:, 2].astype(bool),
    "error_logged": cs[:, 3],  # note: kp_multiplier baked in
    "p": cs[:, 4],
    "i": cs[:, 5],
    "d": cs[:, 6],
    "f": cs[:, 7],
    "output": cs[:, 8],
    "desired_la": cs[:, 9],
    "actual_la": cs[:, 10],
    "desired_jerk": cs[:, 11],
    "vEgo": st_j[:, 0],
    "angleDeg": st_j[:, 1],
    "steerTorque": st_j[:, 2],
    "steerPressed": st_j[:, 3].astype(bool),
    "cmd_torque": cc_j[:, 0],
    "yaw_rate": lp_j[:, 0],
    "lat_accel_pose": lp_j[:, 1],
  }


def welch_psd(x, fs, nperseg=None):
  """Tiny Welch PSD — avoids scipy dep."""
  n = len(x)
  if nperseg is None:
    nperseg = min(1024, max(256, n // 8))
  if n < nperseg:
    return np.array([0.0]), np.array([0.0])
  hop = nperseg // 2
  win = np.hanning(nperseg)
  wnorm = (win * win).sum()
  segs = []
  for start in range(0, n - nperseg + 1, hop):
    seg = x[start:start + nperseg] * win
    spec = np.fft.rfft(seg)
    segs.append(np.abs(spec) ** 2)
  psd = np.mean(segs, axis=0) / (fs * wnorm)
  freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
  return freqs, psd


def band_power(freqs, psd, lo, hi):
  mask = (freqs >= lo) & (freqs < hi)
  if not mask.any():
    return 0.0
  return float(np.trapezoid(psd[mask], freqs[mask]))


def integrator_slope(i_arr, t_arr, window_sec=60.0):
  """Return max absolute slope of |i| over sliding windows."""
  if len(i_arr) < 10:
    return 0.0
  window = int(window_sec * CONTROL_RATE_HZ)
  slopes = []
  for start in range(0, len(i_arr) - window, window // 4):
    seg = np.abs(i_arr[start:start + window])
    tt = t_arr[start:start + window]
    if len(seg) < 10 or tt[-1] - tt[0] < window_sec * 0.5:
      continue
    m, _ = np.polyfit(tt - tt[0], seg, 1)
    slopes.append(abs(float(m)))
  return max(slopes) if slopes else 0.0


def analyze_segment(data, seg_id, out_dir, plot=True):
  active = data["active"] & ~data["steerPressed"]
  bucket_stats = {lbl: defaultdict(float) for lbl in BUCKET_LABELS}

  err = data["desired_la"] - data["actual_la"]
  v = data["vEgo"]

  for bi, lbl in enumerate(BUCKET_LABELS):
    lo, hi = SPEED_BUCKETS[bi]
    mask = active & (v >= lo) & (v < hi)
    n = int(mask.sum())
    bucket_stats[lbl]["frames"] = n
    bucket_stats[lbl]["seconds"] = n / CONTROL_RATE_HZ
    if n < 50:
      continue

    e = err[mask]
    bucket_stats[lbl]["err_mean"] = float(np.mean(e))
    bucket_stats[lbl]["err_p50"] = float(np.median(np.abs(e)))
    bucket_stats[lbl]["err_p95"] = float(np.percentile(np.abs(e), 95))
    bucket_stats[lbl]["err_rms"] = float(np.sqrt(np.mean(e * e)))

    bucket_stats[lbl]["sat_rate"] = float(data["saturated"][mask].mean())

    i_abs = np.abs(data["i"][mask])
    bucket_stats[lbl]["i_p95"] = float(np.percentile(i_abs, 95))

    # contribution shares
    p_abs = np.abs(data["p"][mask])
    d_abs = np.abs(data["d"][mask])
    f_abs = np.abs(data["f"][mask])
    tot = p_abs + i_abs + d_abs + f_abs + 1e-9
    bucket_stats[lbl]["p_share"] = float(np.mean(p_abs / tot))
    bucket_stats[lbl]["i_share"] = float(np.mean(i_abs / tot))
    bucket_stats[lbl]["d_share"] = float(np.mean(d_abs / tot))
    bucket_stats[lbl]["f_share"] = float(np.mean(f_abs / tot))

    # FFT oscillation
    if n >= 1024:
      freqs, psd = welch_psd(e - np.mean(e), CONTROL_RATE_HZ)
      bucket_stats[lbl]["psd_low"]  = band_power(freqs, psd, 0.1, 0.3)
      bucket_stats[lbl]["psd_band"] = band_power(freqs, psd, 0.3, 1.5)
      bucket_stats[lbl]["psd_crosswind"] = band_power(freqs, psd, 0.5, 1.0)
      bucket_stats[lbl]["psd_hunting"] = band_power(freqs, psd, 1.0, 3.0)
    else:
      bucket_stats[lbl]["psd_low"] = 0.0
      bucket_stats[lbl]["psd_band"] = 0.0
      bucket_stats[lbl]["psd_crosswind"] = 0.0
      bucket_stats[lbl]["psd_hunting"] = 0.0

    # Zero-crossing / "hunting" analysis
    # Count sign flips in commanded torque when the driver isn't asking for a hard turn.
    out_arr = data["output"][mask]
    cmd_arr = data["cmd_torque"][mask]
    desired_arr = data["desired_la"][mask]
    # Low-curvature proxy: |desired lat-accel| < 0.8 m/s^2 → driving straight-ish
    straight_mask = np.abs(desired_arr) < 0.8
    n_straight = int(straight_mask.sum())
    bucket_stats[lbl]["straight_sec"] = n_straight / CONTROL_RATE_HZ
    if n_straight > 100:
      out_s = out_arr[straight_mask]
      cmd_s = cmd_arr[straight_mask]
      err_s = e[straight_mask]
      # Flips per second of the output (pid result, torque units)
      flips_out = int(np.sum(np.diff(np.sign(out_s)) != 0))
      flips_cmd = int(np.sum(np.diff(np.sign(cmd_s)) != 0))
      flips_err = int(np.sum(np.diff(np.sign(err_s)) != 0))
      bucket_stats[lbl]["flips_output_per_sec"] = flips_out / (n_straight / CONTROL_RATE_HZ)
      bucket_stats[lbl]["flips_cmd_per_sec"]    = flips_cmd / (n_straight / CONTROL_RATE_HZ)
      bucket_stats[lbl]["flips_err_per_sec"]    = flips_err / (n_straight / CONTROL_RATE_HZ)
      # Amplitude during hunting: std of output in straight segments (excludes genuine turning)
      bucket_stats[lbl]["cmd_std_straight"] = float(np.std(cmd_s))
      bucket_stats[lbl]["output_std_straight"] = float(np.std(out_s))
    else:
      bucket_stats[lbl]["flips_output_per_sec"] = 0.0
      bucket_stats[lbl]["flips_cmd_per_sec"] = 0.0
      bucket_stats[lbl]["flips_err_per_sec"] = 0.0
      bucket_stats[lbl]["cmd_std_straight"] = 0.0
      bucket_stats[lbl]["output_std_straight"] = 0.0

  # override events (rising edges of steerPressed while active)
  pressed = data["steerPressed"] & data["active"]
  edges = np.flatnonzero(np.diff(pressed.astype(np.int8)) == 1)
  dist_km = float(np.sum(v[data["active"]]) * DT / 1000.0)
  override_rate = len(edges) / dist_km if dist_km > 0.1 else 0.0

  # integrator growth (all-active)
  i_slope_any = integrator_slope(data["i"][data["active"]], data["t"][data["active"]])

  out = {
    "segment": seg_id,
    "params": data.get("tuning_params", {}),
    "total_sec": float(data["t"][-1]) if len(data["t"]) else 0.0,
    "active_sec": float(data["active"].sum() / CONTROL_RATE_HZ),
    "dist_active_km": dist_km,
    "override_events": len(edges),
    "override_per_km": override_rate,
    "integrator_slope_max_abs": i_slope_any,
    "buckets": dict(bucket_stats),
  }

  if plot:
    try:
      import matplotlib
      matplotlib.use("Agg")
      import matplotlib.pyplot as plt

      fig, ax = plt.subplots(1, 1, figsize=(8, 4))
      m = data["active"]
      ax.scatter(v[m], err[m], s=1, alpha=0.3)
      ax.set_xlabel("vEgo [m/s]"); ax.set_ylabel("lat-accel error [m/s^2]")
      ax.set_title(f"{seg_id} speed vs error")
      ax.grid(alpha=0.3)
      fig.savefig(os.path.join(out_dir, f"{seg_id}_speed_vs_err.png"), dpi=80)
      plt.close(fig)

      # FFT plot for the highest-speed bucket with enough data
      for bi in range(len(BUCKET_LABELS) - 1, -1, -1):
        lo, hi = SPEED_BUCKETS[bi]
        mask = active & (v >= lo) & (v < hi)
        if mask.sum() < 1024:
          continue
        e = err[mask] - np.mean(err[mask])
        freqs, psd = welch_psd(e, CONTROL_RATE_HZ)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(freqs, psd)
        ax.axvspan(0.3, 1.5, alpha=0.15, color="orange", label="osc band")
        ax.axvspan(0.5, 1.0, alpha=0.25, color="red", label="crosswind band")
        ax.set_xlim(0, 5); ax.set_xlabel("Hz"); ax.set_ylabel("PSD")
        ax.set_title(f"{seg_id} err PSD (bucket {BUCKET_LABELS[bi]})")
        ax.grid(alpha=0.3); ax.legend()
        fig.savefig(os.path.join(out_dir, f"{seg_id}_fft.png"), dpi=80)
        plt.close(fig)
        break

      fig, ax = plt.subplots(figsize=(10, 4))
      tt = data["t"][data["active"]]
      ax.plot(tt, data["cmd_torque"][data["active"]], label="cmd", linewidth=0.7)
      ax2 = ax.twinx()
      ax2.plot(tt, data["steerTorque"][data["active"]], color="C1", label="measured", linewidth=0.7, alpha=0.7)
      ax.set_xlabel("t [s]"); ax.set_ylabel("cmd torque")
      ax2.set_ylabel("measured torque")
      ax.set_title(f"{seg_id} torque cmd vs measured")
      fig.savefig(os.path.join(out_dir, f"{seg_id}_torque_cmd_vs_meas.png"), dpi=80)
      plt.close(fig)

      fig, ax = plt.subplots(figsize=(10, 4))
      ax.plot(data["t"], data["i"], linewidth=0.7)
      ax.set_xlabel("t [s]"); ax.set_ylabel("integrator")
      ax.set_title(f"{seg_id} integrator trajectory")
      ax.grid(alpha=0.3)
      fig.savefig(os.path.join(out_dir, f"{seg_id}_integrator.png"), dpi=80)
      plt.close(fig)
    except ImportError:
      pass

  return out


def rollup(segs):
  """Combine per-segment bucket stats with weighting by frames."""
  combined = {lbl: defaultdict(float) for lbl in BUCKET_LABELS}
  totals = {lbl: 0 for lbl in BUCKET_LABELS}
  for s in segs:
    for lbl, b in s["buckets"].items():
      n = b.get("frames", 0)
      if n == 0:
        continue
      totals[lbl] += n
      for k, v in b.items():
        if k == "frames":
          continue
        combined[lbl][k] += v * n
  for lbl in BUCKET_LABELS:
    n = totals[lbl]
    if n == 0:
      continue
    for k in list(combined[lbl].keys()):
      combined[lbl][k] /= n
    combined[lbl]["frames"] = n
    combined[lbl]["seconds"] = n / CONTROL_RATE_HZ
  return combined


def print_table(roll):
  cols = ["seconds", "straight_sec", "err_p95", "err_rms",
          "flips_cmd_per_sec", "flips_output_per_sec",
          "cmd_std_straight", "psd_crosswind", "psd_hunting",
          "sat_rate", "i_p95", "d_share", "f_share"]
  header = f"{'bucket':<8}" + "".join(f"{c:>13}" for c in cols)
  print(header)
  print("-" * len(header))
  for lbl in BUCKET_LABELS:
    b = roll.get(lbl, {})
    cells = []
    for c in cols:
      v = b.get(c, 0)
      if c == "sat_rate" or c.endswith("_share"):
        cells.append(f"{v*100:>12.1f}%")
      elif c.startswith("psd"):
        cells.append(f"{v:>13.5f}")
      elif c.startswith("flips"):
        cells.append(f"{v:>13.2f}")
      else:
        cells.append(f"{v:>13.3f}")
    print(f"{lbl:<8}" + "".join(cells))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--logs-dir", required=True)
  ap.add_argument("--out-dir", required=True)
  ap.add_argument("--plot", action="store_true")
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  paths = sorted(glob.glob(os.path.join(args.logs_dir, "*/rlog.zst")))
  print(f"Analyzing {len(paths)} segments from {args.logs_dir}")

  segs = []
  for p in paths:
    seg_id = os.path.basename(os.path.dirname(p))
    try:
      data = parse_segment(p)
    except Exception as e:
      print(f"SKIP {seg_id}: {e}")
      continue
    if data is None:
      print(f"SKIP {seg_id}: no controlsState")
      continue
    res = analyze_segment(data, seg_id, args.out_dir, plot=args.plot)
    segs.append(res)
    print(f"  {seg_id}: active={res['active_sec']:.0f}s  km={res['dist_active_km']:.1f}  "
          f"override/km={res['override_per_km']:.2f}  i_slope={res['integrator_slope_max_abs']:.4f}")

  # Group segments by distinct tuning param set
  groups = defaultdict(list)
  for s in segs:
    key = tuple((k, s["params"].get(k, "?")) for k in TUNING_PARAM_KEYS)
    groups[key].append(s)

  print("\n=== Param configurations encountered ===")
  for k, ss in groups.items():
    active = sum(x["active_sec"] for x in ss)
    pdict = dict(k)
    brief = (f"KpL={pdict.get('KpLowSpeed','?')} KpM={pdict.get('KpMidSpeed','?')} "
             f"KpH={pdict.get('KpHighSpeed','?')} "
             f"KdL={pdict.get('KdLowSpeed','-')} KdM={pdict.get('KdMidSpeed','-')} "
             f"KdH={pdict.get('KdHighSpeed','?')}")
    print(f"  {len(ss):>2} segs, active={active:>5.0f}s  branch={pdict.get('GitBranch','?')[:40]}  {brief}")

  roll = rollup(segs)
  print("\n=== Rollup (frame-weighted across all segments) ===")
  print_table(roll)

  summary = {"segments": segs, "rollup": roll}
  out_json = os.path.join(args.out_dir, "summary.json")
  with open(out_json, "w") as f:
    json.dump(summary, f, indent=2, default=float)
  print(f"\nWrote {out_json}")


if __name__ == "__main__":
  main()
