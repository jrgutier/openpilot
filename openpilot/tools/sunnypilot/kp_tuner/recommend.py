"""Weighted-least-squares recommender for the lateral Kp multiplier triple.

Plan reference: .omc/plans/rivian-kp-tuning.md, Step 3 + Recommendation Math.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openpilot.tools.sunnypilot.kp_tuner.analyze import (
  BandStats,
  CurveEvent,
  KP_UI_MAX,
  KP_UI_MIN,
  MIN_VEGO_FOR_SOLVER,
  current_kp_working,
  interp_weights,
)

PER_ITERATION_CAP = 1.5  # symmetric: new in [old / 1.5, old * 1.5]
OSCILLATION_THRESHOLD = 2.0  # raw solver > 2x current -> warn + cap at 1.5x
CONVERGENCE_LOW = 0.95
CONVERGENCE_HIGH = 1.05
SLIGHT_OVERSHOOT_HIGH = 1.05
HOLD_DELTA_PCT = 0.01  # if all bands within 1% AND verdicts converged -> hold
MIN_EVENTS_DEFAULT = 5

OSCILLATION_WARNING_TEXT = (
  "Solver wants band X at >2x current; the underlying issue may not be Kp. " +
  "Investigate LAT_ACCEL_FACTOR / FRICTION in " +
  "opendbc_repo/opendbc/car/torque_data/override.toml for RIVIAN_R1 " +
  "before further iterations."
)


# Verdict labels (per-band).
CONVERGED = "CONVERGED"
SLIGHT_OVERSHOOT = "CONVERGED-SLIGHT-OVERSHOOT"
CONTINUE = "CONTINUE"
BACK_OFF = "BACK_OFF"
NO_DATA = "NO_DATA"


@dataclass
class TuningRecommendation:
  triple: tuple[float, float, float]
  per_band_verdict: dict[str, str]
  rationale: str
  oscillation_warning: bool = False
  base_knob_warning_text: str | None = None
  hold: bool = False  # True iff triple == current_triple (no actionable change)


def _classify_band(median_ratio: float) -> str:
  if median_ratio < CONVERGENCE_LOW:
    return CONTINUE
  if median_ratio <= 1.0:
    return CONVERGED
  if median_ratio <= SLIGHT_OVERSHOOT_HIGH:
    return SLIGHT_OVERSHOOT
  return BACK_OFF


def _band_verdicts(band_summaries: BandStats) -> dict[str, str]:
  verdicts: dict[str, str] = {}
  for name in ("low", "mid", "high"):
    stat = getattr(band_summaries, name)
    verdicts[name] = NO_DATA if stat is None else _classify_band(stat.median_ratio)
  return verdicts


def _filter_events_for_solver(events: list[CurveEvent]) -> list[CurveEvent]:
  return [e for e in events if e.vEgo_t0 >= MIN_VEGO_FOR_SOLVER]


def _solve_wls(
  events: list[CurveEvent],
  current_triple: tuple[float, float, float],
) -> tuple[float, float, float]:
  """Solve for the recommended (low_new, mid_new, high_new) via WLS.

  Each row of the design matrix is [w_low(v_i), w_mid(v_i), w_high(v_i)]; the
  target is m_target_i = current_kp_working(v_i) / r_i where r_i is the
  observed undershoot ratio. Sample weights alpha_i = 1 / (1 + |r_i - 1| * 0.5)
  mildly down-weight outliers. We multiply both sides by sqrt(alpha_i) and call
  np.linalg.lstsq.
  """
  rows: list[list[float]] = []
  targets: list[float] = []
  weights: list[float] = []
  for e in events:
    w_low, w_mid, w_high = interp_weights(e.vEgo_t0)
    cur_working = current_kp_working(e.vEgo_t0, current_triple)
    if e.tracking_ratio == 0.0:
      continue  # cannot invert a zero ratio
    target = cur_working / e.tracking_ratio
    # Convert from kp_working target back to multiplier-space target by dividing
    # by the base KP_INTERP at this v_ego — i.e., the (low_new, mid_new, high_new)
    # we solve for are *multipliers*, not raw kp values. (current_kp_working
    # already includes the KP_INTERP factor; cancel it.)
    base = current_kp_working(e.vEgo_t0, (1.0, 1.0, 1.0))  # = KP_INTERP(v_ego)
    if base == 0.0:
      continue
    target_multiplier = target / base
    alpha = 1.0 / (1.0 + abs(e.tracking_ratio - 1.0) * 0.5)
    sqrt_alpha = float(np.sqrt(alpha))
    rows.append([w_low * sqrt_alpha, w_mid * sqrt_alpha, w_high * sqrt_alpha])
    targets.append(target_multiplier * sqrt_alpha)
    weights.append(alpha)

  if not rows:
    return current_triple

  design = np.asarray(rows, dtype=float)
  rhs = np.asarray(targets, dtype=float)
  # `rcond=None` selects numpy's "future" default and keeps regression stable.
  solution, _residuals, _rank, _sv = np.linalg.lstsq(design, rhs, rcond=None)
  return float(solution[0]), float(solution[1]), float(solution[2])


def _apply_caps(
  raw: tuple[float, float, float],
  current: tuple[float, float, float],
  *,
  apply_per_iter_cap: bool,
) -> tuple[tuple[float, float, float], bool]:
  """Apply oscillation guard + 1.5x per-iteration cap + [0.1, 5.0] clamp.

  Returns (capped_triple, oscillation_warning).
  """
  oscillation = False
  out: list[float] = []
  for r, cur in zip(raw, current, strict=False):
    if cur > 0 and abs(r) >= OSCILLATION_THRESHOLD * cur:
      oscillation = True
      capped = cur * PER_ITERATION_CAP if r > cur else cur / PER_ITERATION_CAP
    elif apply_per_iter_cap and cur > 0:
      lo = cur / PER_ITERATION_CAP
      hi = cur * PER_ITERATION_CAP
      capped = max(lo, min(hi, r))
    else:
      capped = r
    capped = max(KP_UI_MIN, min(KP_UI_MAX, capped))
    out.append(capped)
  return (out[0], out[1], out[2]), oscillation


def recommend(
  current_triple: tuple[float, float, float],
  events: list[CurveEvent],
  band_summaries: BandStats,
  *,
  min_events: int = MIN_EVENTS_DEFAULT,
  apply_per_iter_cap: bool = True,
) -> TuningRecommendation:
  """Return a `TuningRecommendation` for the next iteration.

  `apply_per_iter_cap=False` is intended for solver-correctness unit tests
  ONLY — production callers MUST leave it True.
  """
  verdicts = _band_verdicts(band_summaries)
  pool = _filter_events_for_solver(events)

  if len(pool) < min_events:
    return TuningRecommendation(
      triple=current_triple,
      per_band_verdict=verdicts,
      rationale=f"needs more data; drive more curves (have {len(pool)} usable events, need >= {min_events})",
      hold=True,
    )

  raw = _solve_wls(pool, current_triple)
  capped, oscillation = _apply_caps(raw, current_triple, apply_per_iter_cap=apply_per_iter_cap)

  # Hold check: every populated band converged-or-slight AND solver delta < 1%.
  populated = [v for v in verdicts.values() if v != NO_DATA]
  all_converged = bool(populated) and all(v in (CONVERGED, SLIGHT_OVERSHOOT) for v in populated)
  small_delta = all(
    cur > 0 and abs(new - cur) / cur < HOLD_DELTA_PCT
    for new, cur in zip(capped, current_triple, strict=False)
  )
  hold = all_converged and small_delta

  if hold:
    return TuningRecommendation(
      triple=current_triple,
      per_band_verdict=verdicts,
      rationale="all bands converged and solver delta < 1% — hold",
      hold=True,
    )

  rationale_parts = [f"{name}={verdict}" for name, verdict in verdicts.items()]
  rationale = "WLS over interpolation weights; verdicts: " + ", ".join(rationale_parts)
  warning_text = OSCILLATION_WARNING_TEXT if oscillation else None

  return TuningRecommendation(
    triple=capped,
    per_band_verdict=verdicts,
    rationale=rationale,
    oscillation_warning=oscillation,
    base_knob_warning_text=warning_text,
    hold=False,
  )


def _solve_wls_joint(
  events_with_kp: list[tuple[CurveEvent, tuple[float, float, float]]],
) -> tuple[float, float, float] | None:
  """Joint solve: each event uses its own kp_at_event (the kp triple in effect
  during that drive) instead of one global current_triple. Returns None when
  no usable rows survive.
  """
  rows: list[list[float]] = []
  targets: list[float] = []
  for e, kp in events_with_kp:
    if e.tracking_ratio == 0.0:
      continue
    w_low, w_mid, w_high = interp_weights(e.vEgo_t0)
    cur_working = current_kp_working(e.vEgo_t0, kp)
    base = current_kp_working(e.vEgo_t0, (1.0, 1.0, 1.0))
    if base == 0.0:
      continue
    target_multiplier = (cur_working / e.tracking_ratio) / base
    alpha = 1.0 / (1.0 + abs(e.tracking_ratio - 1.0) * 0.5)
    sqrt_alpha = float(np.sqrt(alpha))
    rows.append([w_low * sqrt_alpha, w_mid * sqrt_alpha, w_high * sqrt_alpha])
    targets.append(target_multiplier * sqrt_alpha)

  if not rows:
    return None
  design = np.asarray(rows, dtype=float)
  rhs = np.asarray(targets, dtype=float)
  solution, _r, _k, _s = np.linalg.lstsq(design, rhs, rcond=None)
  return float(solution[0]), float(solution[1]), float(solution[2])


def recommend_joint(
  events_with_kp: list[tuple[CurveEvent, tuple[float, float, float]]],
  *,
  min_events: int = MIN_EVENTS_DEFAULT,
) -> TuningRecommendation:
  """Joint WLS over events from MULTIPLE Kp configurations of the same model.

  Output is ADVISORY (Architect §4 synthesis): never written to state.json.
  Per-bucket records remain the unambiguous revert unit. We still apply
  [KP_UI_MIN, KP_UI_MAX] clamps and an oscillation-style flag (raw at >2x
  the highest kp_at_event seen), but skip the 1.5× per-iteration cap because
  there is no single "current" to cap against.
  """
  pool = [(e, kp) for e, kp in events_with_kp
          if e.vEgo_t0 >= MIN_VEGO_FOR_SOLVER]
  if len(pool) < min_events:
    return TuningRecommendation(
      triple=(1.0, 1.0, 1.0),
      per_band_verdict={"low": NO_DATA, "mid": NO_DATA, "high": NO_DATA},
      rationale=(
        f"joint: needs more data (have {len(pool)} usable events, need >= {min_events})"
      ),
      hold=True,
    )

  raw = _solve_wls_joint(pool)
  if raw is None:
    return TuningRecommendation(
      triple=(1.0, 1.0, 1.0),
      per_band_verdict={"low": NO_DATA, "mid": NO_DATA, "high": NO_DATA},
      rationale="joint: no solvable rows after filtering",
      hold=True,
    )

  # Clamp to UI range; no per-iteration cap since there's no single "current".
  clamped = tuple(max(KP_UI_MIN, min(KP_UI_MAX, x)) for x in raw)

  # Oscillation flag: max kp_at_event seen acts as the "reference" for the warning.
  max_seen = tuple(max(kp[i] for _, kp in pool) for i in range(3))
  osc = any(
    cur > 0 and abs(r) >= OSCILLATION_THRESHOLD * cur
    for r, cur in zip(raw, max_seen, strict=True)
  )

  per_band_verdict: dict[str, str] = {}
  for name, idx in (("low", 0), ("mid", 1), ("high", 2)):
    has_data = any(interp_weights(e.vEgo_t0)[idx] > 0 for e, _ in pool)
    per_band_verdict[name] = (
      _classify_band_for_joint(clamped[idx]) if has_data else NO_DATA
    )

  return TuningRecommendation(
    triple=(clamped[0], clamped[1], clamped[2]),
    per_band_verdict=per_band_verdict,
    rationale=(
      f"joint WLS over {len(pool)} events from {len({tuple(kp) for _, kp in pool})} kp configs; "
      + "advisory output, NOT applied to state"
    ),
    oscillation_warning=osc,
    base_knob_warning_text=OSCILLATION_WARNING_TEXT if osc else None,
    hold=False,
  )


def _classify_band_for_joint(multiplier: float) -> str:
  """Joint output is advisory — describe the recommendation rather than verdict."""
  if multiplier < 0.95:
    return f"reduce to {multiplier:.2f}"
  if multiplier <= 1.05:
    return f"hold near {multiplier:.2f}"
  return f"raise to {multiplier:.2f}"
