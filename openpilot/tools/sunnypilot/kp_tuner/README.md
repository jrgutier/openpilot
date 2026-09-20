# kp_tuner — Lateral Torque Kp Multiplier Tuning Loop

Operator workflow for converging the three sunnypilot tuning-menu sliders
(`KpLowSpeed`, `KpMidSpeed`, `KpHighSpeed`) on a real car using mirrored device
rlogs. NN FF off; base Kp / `torque_data` frozen.

## Quick start

```
./tools/sunnypilot/kp_tuner/sync_logs.sh
python -m openpilot.tools.sunnypilot.kp_tuner.report \
  --log-dir /Volumes/home/sunnypilot-logs --joint-per-model --out-html /tmp/kp_tuner.html
# Apply each bucket's Apply triple via the Tuning UI, drive a fresh route, repeat.
```

`--current` is auto-read per (model, kp) bucket from `initData.params`.
`--bucket-filter MODEL` restricts to one model. `--no-cache` forces re-decode;
`--clear-cache` wipes `~/.cache/sunnypilot/kp_tuner/segments/`.

## Theory (one paragraph)

The controller computes `kp_working = np.interp(vEgo, [6.7, 15.6, 33.5],
[low, mid, high]) * KP_INTERP(vEgo)` per cycle. We invert this with a
**weighted-least-squares** solve over the same `np.interp` weights against per-
event tracking ratios `r_i = peak_actual_curvature / peak_desired_curvature`
(r<1 = undershoot, r>1 = oversteer), filtered for `vEgo >= 3 m/s` and
engagement gates. Each iteration is capped at **1.5× per band, both
directions**, then clamped to `[0.1, 5.0]`. Convergence is `r ∈ [0.95, 1.05]`
per band. The optional `--joint-per-model` solver pools events from multiple
kp configs of the same driving model and emits an advisory triple — never
applied to state automatically.

## Revert

If a recommendation feels worse: `python -m openpilot.tools.sunnypilot.kp_tuner.report --revert`
prints the persisted `last_known_safe` triple per bucket. With multiple buckets
you must pass `--bucket-filter MODEL`. Read-only — never mutates state.

## Stop conditions

1. All bands report `CONVERGED` (or `CONVERGED-SLIGHT-OVERSHOOT`) — done.
2. Oscillation guard fires (solver wants any band > 2× current). The underlying
   knob is probably `LAT_ACCEL_FACTOR` / `FRICTION` in
   `opendbc_repo/opendbc/car/torque_data/override.toml`, not Kp. Stop tuning Kp.
3. NN FF abort (exit code 2). The dual-check (`NeuralNetworkLateralControl`
   param on AND a non-mock model bound) fired across any segment; this tool's
   solver assumes NN FF off. Disable NN FF or use a different workflow.

## State + cache

State at `~/.config/sunnypilot/kp_tuner/state.json` (v3 schema; partitioned by
(model, Kp) bucket). Atomic writes via `os.replace`. Per-segment event cache
at `~/.cache/sunnypilot/kp_tuner/segments/` — keyed on absolute rlog path,
size, mtime, and a salt over detector knobs so threshold changes auto-bust.
