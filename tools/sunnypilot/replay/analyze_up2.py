#!/usr/bin/env python3
"""Post-merge replay validation for UP_2 → openpilot full-disengage.

Verification harness for acceptance criterion #2 of the UP_2 stalk-disengage
plan (`.omc/plans/ralplan-up-2-stalk-disengage-openpilot.md`):

    >=95% of distinct VDM_UserAdasRequest=2 rising edges produce a
    selfdriveState.enabled False edge within 100 ms when openpilot was
    engaged.

For each route segment we extract:
  - raw upstream:    VDM_UserAdasRequest from CAN frames (Bus.pt, addr 354)
  - reaction:        selfdriveState.enabled False edges
  - latency:         (selfdrive disengage time) - (UP_2 raw press time)
  - context:         carState.cruiseState drops (vendor cascade), onroadEvents

A press counts toward the success rate only when openpilot was engaged at
the press tick. Within-window False edges are matched once each. The script
prints a per-route table plus aggregate totals.

Run: uv run tools/sunnypilot/replay/analyze_up2.py [N_ROUTES]
     (default N_ROUTES=8; reads from ~/sunnypilot-logs)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from openpilot.tools.lib.logreader import LogReader
from opendbc.can.parser import CANParser
from opendbc.car import Bus
from opendbc.car.rivian.values import DBC, CAR

LOG_DIR = Path.home() / "sunnypilot-logs"
N_ROUTES = int(sys.argv[1]) if len(sys.argv) > 1 else 8  # matches acceptance criterion #2's stated scope
DBC_NAME = DBC[CAR.RIVIAN_R1][Bus.pt]
ADAS_ADDR = 354  # VDM_AdasSts
DISENGAGE_WINDOW_NS = int(0.100 * 1e9)  # 100 ms per acceptance criterion #2
SUCCESS_THRESHOLD = 0.95

# Pick newest N unique routes (excluding 'boot').
seg_dirs = [d for d in LOG_DIR.iterdir() if d.is_dir() and "--" in d.name]
seg_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
seen: set[str] = set()
routes: list[str] = []
for d in seg_dirs:
  rid = d.name.rsplit("--", 1)[0]
  if rid not in seen:
    seen.add(rid)
    routes.append(rid)
  if len(routes) == N_ROUTES:
    break

print(f"\nAnalyzing {len(routes)} newest routes from {LOG_DIR} (DBC={DBC_NAME})")
print(f"Disengage window: {DISENGAGE_WINDOW_NS / 1e6:.0f} ms; success threshold: {SUCCESS_THRESHOLD:.0%}\n")

agg_engaged_presses = 0
agg_matched = 0

for rid in routes:
  segs = sorted(d for d in LOG_DIR.iterdir() if d.name.startswith(rid + "--"))
  print(f"=== {rid} ({len(segs)} segments) ===")

  # Pass 1: collect UP_2 raw rising-edge timestamps + selfdriveState transitions.
  up2_times: list[int] = []
  selfdrive_drop_times: list[int] = []
  selfdrive_engaged_at: dict[int, bool] = {}  # press_t → was openpilot engaged?
  prev_raw_val = 0
  prev_selfdrive_enabled = False
  selfdrive_seen = False
  prev_car_enabled = False
  car_disengages = 0
  stalk_events: dict[str, int] = defaultdict(int)
  parser = CANParser(DBC_NAME, [("VDM_AdasSts", 0)], 0)

  for seg in segs:
    rlog = seg / "rlog.zst"
    if not rlog.exists():
      continue
    try:
      lr = LogReader(str(rlog))
    except Exception as e:
      print(f"  skip {seg.name}: {e}")
      continue

    for msg in lr:
      t = msg.logMonoTime
      which = msg.which()

      if which == "selfdriveState":
        en = bool(msg.selfdriveState.enabled)
        if selfdrive_seen and prev_selfdrive_enabled and not en:
          selfdrive_drop_times.append(t)
        prev_selfdrive_enabled = en
        selfdrive_seen = True

      elif which == "carState":
        en = bool(msg.carState.cruiseState.enabled)
        if prev_car_enabled and not en:
          car_disengages += 1
        prev_car_enabled = en

      elif which == "onroadEvents":
        for ev in msg.onroadEvents:
          n = str(ev.name)
          if any(k in n.lower() for k in ("cancel", "buttoncancel", "stalk", "disengage")):
            stalk_events[n] += 1

      elif which == "can":
        frames = [(c.address, bytes(c.dat), c.src) for c in msg.can if c.address == ADAS_ADDR]
        if not frames:
          continue
        parser.update([(t, frames)])
        vals = [int(v) for v in parser.vl_all[ADAS_ADDR]["VDM_UserAdasRequest"]]
        for v in vals:
          if v == 2 and prev_raw_val != 2:
            up2_times.append(t)
            selfdrive_engaged_at[t] = prev_selfdrive_enabled
          prev_raw_val = v

  # Pass 2: match each engaged UP_2 press to the nearest later selfdrive drop.
  matched = 0
  unmatched_engaged = 0
  used_drops: set[int] = set()
  for press_t in up2_times:
    if not selfdrive_engaged_at.get(press_t, False):
      continue  # press while openpilot disengaged — not a candidate
    matching = next((dt for dt in selfdrive_drop_times
                     if dt not in used_drops and 0 <= dt - press_t <= DISENGAGE_WINDOW_NS), None)
    if matching is not None:
      matched += 1
      used_drops.add(matching)
    else:
      unmatched_engaged += 1

  engaged_presses = matched + unmatched_engaged
  ratio = (matched / engaged_presses) if engaged_presses else float("nan")

  print(f"  raw UP_2 distinct presses:        {len(up2_times)}")
  print(f"  presses while op engaged:         {engaged_presses}")
  print(f"  presses matched to disengage:     {matched}")
  print(f"  presses unmatched (>100ms or no): {unmatched_engaged}")
  print(f"  carState.cruiseState drops:       {car_disengages}")
  if engaged_presses:
    print(f"  match ratio:                     {ratio:.2%}  (threshold {SUCCESS_THRESHOLD:.0%})")
  if stalk_events:
    print("  cancel/disengage events:")
    for n, c in sorted(stalk_events.items(), key=lambda x: -x[1]):
      print(f"    {n}: {c}")
  print()

  agg_engaged_presses += engaged_presses
  agg_matched += matched

print("=" * 60)
print(f"TOTALS across {len(routes)} routes:")
print(f"  presses while op engaged:  {agg_engaged_presses}")
print(f"  matched to disengage:      {agg_matched}")
if agg_engaged_presses:
  agg_ratio = agg_matched / agg_engaged_presses
  verdict = "PASS" if agg_ratio >= SUCCESS_THRESHOLD else "FAIL"
  print(f"  match ratio:               {agg_ratio:.2%} (threshold {SUCCESS_THRESHOLD:.0%}) — {verdict}")
  sys.exit(0 if agg_ratio >= SUCCESS_THRESHOLD else 1)
else:
  print("  No engaged-state UP_2 presses observed; cannot evaluate criterion.")
  sys.exit(2)
