"""Tests for the WLS solver + post-solve safety pipeline.

The pinned multi-band test (`test_recommend_wls_multi_band_pinned`) is the
canonical correctness check for the solver — a refactor that silently breaks
the design matrix or weighting will fail it within ±0.01.
"""
from __future__ import annotations

import math

from openpilot.common.test import OpenpilotTestCase

from openpilot.tools.sunnypilot.kp_tuner.analyze import (
  BandStat,
  BandStats,
  CurveEvent,
  KP_UI_MAX,
  KP_UI_MIN,
  current_kp_working,
)
from openpilot.tools.sunnypilot.kp_tuner.recommend import (
  BACK_OFF,
  CONTINUE,
  CONVERGED,
  PER_ITERATION_CAP,
  recommend,
  recommend_joint,
)
from openpilot.tools.sunnypilot.kp_tuner.tests.helpers import approx


def _event(v_ego: float, ratio: float, *, band: str | None = None) -> CurveEvent:
  if band is None:
    band = "low" if v_ego <= 6.7 else "mid" if v_ego <= 15.6 else "high"
  return CurveEvent(
    t0=0.0, t_peak=0.5,
    vEgo_t0=v_ego, vEgo_peak=v_ego,
    peak_desired_curvature=0.01, peak_actual_curvature=0.01 * ratio,
    tracking_ratio=ratio, lag_seconds=0.05, band=band,
  )


def _band_stats(low_r: float | None = None, mid_r: float | None = None, high_r: float | None = None) -> BandStats:
  def _s(r):
    if r is None:
      return None
    return BandStat(count=10, median_ratio=r, median_lag=0.05, p25_ratio=r, p75_ratio=r,
                    vEgo_min=5.0, vEgo_max=20.0)
  return BandStats(low=_s(low_r), mid=_s(mid_r), high=_s(high_r))


def _build_pinned_events(ground_truth: tuple[float, float, float]):
  """For each `v` produce the synthetic ratio that MUST be observed in steady
  state if `ground_truth` is the correct multiplier triple."""
  events = []
  for v in (5.0, 11.0, 20.0, 30.0):
    base = current_kp_working(v, (1.0, 1.0, 1.0))  # KP_INTERP(v)
    truth_kp_working = current_kp_working(v, ground_truth)
    # observed ratio r_i such that current_kp_working(v_i) / r_i == truth_kp_working(v_i).
    # current_kp_working with triple=(1,1,1) is `base`; r_i = base / truth_kp_working.
    r = base / truth_kp_working
    events.append(_event(v, r))
  return events


def _build_joint_events(
  ground_truth: tuple[float, float, float],
  kp_configs: list[tuple[float, float, float]],
  v_egos: tuple[float, ...] = (5.0, 11.0, 20.0, 30.0),
  events_per_v: int = 1,
) -> list[tuple[CurveEvent, tuple[float, float, float]]]:
  """For each (v, kp_config) produce the synthetic ratio that MUST be observed
  in steady state if `ground_truth` is the correct multiplier triple.

      r_i = current_kp_working(v_i, kp_at_event_i) / kp_truth(v_i)
  """
  out = []
  for kp in kp_configs:
    for v in v_egos:
      kp_obs = current_kp_working(v, kp)
      kp_truth_working = current_kp_working(v, ground_truth)
      r = kp_obs / kp_truth_working
      for _ in range(events_per_v):
        out.append((_event(v, r), kp))
  return out


class TestRecommend(OpenpilotTestCase):
  # -------------------------------------------------------------------------
  # Mid-band interior correctness (v1-vs-v2 contrast)
  # -------------------------------------------------------------------------

  def test_recommend_wls_midband_interior(self):
    events = [_event(11.0, 0.7) for _ in range(8)]
    rec = recommend(
      current_triple=(1.0, 1.0, 1.0),
      events=events,
      band_summaries=_band_stats(mid_r=0.7),
    )
    # The achieved kp_working at vEgo=11 with the new triple should be within a
    # cap-allowed delta of the desired m_target. The 1.5x cap means we can move
    # at most to 1.5x current => evaluate that.
    achieved = current_kp_working(11.0, rec.triple)
    baseline = current_kp_working(11.0, (1.0, 1.0, 1.0))
    desired = baseline / 0.7
    # We expect ratio achieved/baseline >= 0.9 * (desired/baseline_capped).
    # With 1.5x cap and ratio=0.7 => the cap binds at 1.5; the solver wants ~1.43.
    ratio_achieved_over_baseline = achieved / baseline
    ratio_desired_over_baseline = desired / baseline
    # achieved should be >= 90% of the cap-allowed move
    cap_allowed = min(ratio_desired_over_baseline, PER_ITERATION_CAP)
    assert ratio_achieved_over_baseline >= 0.9 * cap_allowed

  # -------------------------------------------------------------------------
  # WLS multi-band pinned (BLOCKING — Critic CN2)
  # -------------------------------------------------------------------------

  def test_recommend_wls_multi_band_pinned(self):
    ground_truth = (1.30, 1.45, 1.10)
    events = _build_pinned_events(ground_truth)
    bs = _band_stats(low_r=events[0].tracking_ratio,
                     mid_r=events[1].tracking_ratio,
                     high_r=events[3].tracking_ratio)
    # Disable the 1.5x cap so we isolate solver correctness.
    rec = recommend(
      current_triple=(1.0, 1.0, 1.0),
      events=events,
      band_summaries=bs,
      apply_per_iter_cap=False,
      min_events=4,
    )
    assert rec.triple[0] == approx(ground_truth[0], abs=0.01)
    assert rec.triple[1] == approx(ground_truth[1], abs=0.01)
    assert rec.triple[2] == approx(ground_truth[2], abs=0.01)

  def test_recommend_wls_multi_band_capped_moves_toward_truth(self):
    ground_truth = (1.30, 1.45, 1.10)
    events = _build_pinned_events(ground_truth)
    bs = _band_stats(low_r=events[0].tracking_ratio,
                     mid_r=events[1].tracking_ratio,
                     high_r=events[3].tracking_ratio)
    rec = recommend(
      current_triple=(1.0, 1.0, 1.0),
      events=events,
      band_summaries=bs,
      apply_per_iter_cap=True,  # cap re-enabled
      min_events=4,
    )
    for new, truth in zip(rec.triple, ground_truth, strict=True):
      # correct sign + magnitude <= 0.5 per iteration
      assert new > 1.0, f"{new=} did not move up toward {truth=}"
      assert abs(new - 1.0) <= 0.5

  # -------------------------------------------------------------------------
  # 1.5x caps — both directions
  # -------------------------------------------------------------------------

  def test_recommend_15x_cap_up(self):
    # All ratios at 0.5 (severe undershoot) -> solver wants ~2x, cap binds at 1.5.
    events = [_event(v, 0.5) for v in (5.0, 11.0, 20.0, 30.0) for _ in range(2)]
    bs = _band_stats(low_r=0.5, mid_r=0.5, high_r=0.5)
    rec = recommend((1.0, 1.0, 1.0), events, bs, min_events=4)
    for new in rec.triple:
      # With ratio=0.5 the solver wants 2x => oscillation guard caps at 1.5x.
      # With per-iteration cap also at 1.5x, output should not exceed 1.5.
      assert new <= 1.5 + 1e-9

  def test_recommend_15x_cap_down(self):
    events = [_event(v, 1.5) for v in (5.0, 11.0, 20.0, 30.0) for _ in range(2)]
    bs = _band_stats(low_r=1.5, mid_r=1.5, high_r=1.5)
    rec = recommend((1.0, 1.0, 1.0), events, bs, min_events=4)
    for new in rec.triple:
      assert new >= 1.0 / 1.5 - 1e-9

  # -------------------------------------------------------------------------
  # [0.1, 5.0] clamps
  # -------------------------------------------------------------------------

  def test_recommend_clamp_high(self):
    events = [_event(v, 0.5) for v in (5.0, 11.0, 20.0, 30.0) for _ in range(3)]
    bs = _band_stats(low_r=0.5, mid_r=0.5, high_r=0.5)
    rec = recommend((4.5, 4.5, 4.5), events, bs, min_events=4)
    for new in rec.triple:
      assert new <= KP_UI_MAX + 1e-9

  def test_recommend_clamp_low(self):
    events = [_event(v, 1.5) for v in (5.0, 11.0, 20.0, 30.0) for _ in range(3)]
    bs = _band_stats(low_r=1.5, mid_r=1.5, high_r=1.5)
    rec = recommend((0.15, 0.15, 0.15), events, bs, min_events=4)
    for new in rec.triple:
      assert new >= KP_UI_MIN - 1e-9

  # -------------------------------------------------------------------------
  # Oscillation guard (>2x)
  # -------------------------------------------------------------------------

  def test_recommend_oscillation_guard(self):
    # ratio=0.4 => solver wants ~2.5x => oscillation guard fires.
    events = [_event(v, 0.4) for v in (5.0, 11.0, 20.0, 30.0) for _ in range(2)]
    bs = _band_stats(low_r=0.4, mid_r=0.4, high_r=0.4)
    rec = recommend((1.0, 1.0, 1.0), events, bs, min_events=4)
    assert rec.oscillation_warning is True
    assert rec.base_knob_warning_text is not None
    assert "LAT_ACCEL_FACTOR" in rec.base_knob_warning_text
    assert "FRICTION" in rec.base_knob_warning_text
    for new in rec.triple:
      assert new <= 1.5 + 1e-9

  # -------------------------------------------------------------------------
  # Hold inside band
  # -------------------------------------------------------------------------

  def test_recommend_hold_inside_band(self):
    # Each band has a median ratio in [0.96, 1.04] AND solver delta < 1%.
    events = [_event(v, 1.0) for v in (5.0, 11.0, 20.0, 30.0) for _ in range(2)]
    bs = _band_stats(low_r=0.99, mid_r=1.0, high_r=1.01)
    rec = recommend((1.0, 1.0, 1.0), events, bs, min_events=4)
    assert rec.hold is True
    assert rec.triple == (1.0, 1.0, 1.0)

  # -------------------------------------------------------------------------
  # Back-off on overshoot
  # -------------------------------------------------------------------------

  def test_recommend_back_off_overshoot(self):
    events = [_event(v, 1.10) for v in (11.0, 12.0, 13.0, 14.0) for _ in range(2)]
    bs = _band_stats(mid_r=1.10)
    rec = recommend((1.0, 1.0, 1.0), events, bs, min_events=4)
    assert rec.per_band_verdict["mid"] == BACK_OFF
    # mid component should decrease toward 1/1.10 ≈ 0.91
    assert rec.triple[1] < 1.0

  # -------------------------------------------------------------------------
  # Insufficient samples
  # -------------------------------------------------------------------------

  def test_recommend_insufficient_samples(self):
    events = [_event(11.0, 0.7) for _ in range(2)]
    bs = _band_stats(mid_r=0.7)
    rec = recommend((1.0, 1.0, 1.0), events, bs, min_events=5)
    assert rec.hold is True
    assert "needs more data" in rec.rationale

  # -------------------------------------------------------------------------
  # Low-speed exclusion
  # -------------------------------------------------------------------------

  def test_recommend_low_speed_excluded(self):
    # Mix one very-low-speed event in with five usable ones; verify the result
    # does not change vs the "no low-speed" pool — proves the v_ego>=3 filter ran.
    events_clean = [_event(v, 0.8) for v in (5.0, 11.0, 12.0, 20.0, 30.0)]
    bs = _band_stats(low_r=0.8, mid_r=0.8, high_r=0.8)
    rec_clean = recommend((1.0, 1.0, 1.0), events_clean, bs, min_events=5)

    events_with_lowspeed = [_event(2.0, 0.1)] + events_clean
    rec_filtered = recommend((1.0, 1.0, 1.0), events_with_lowspeed, bs, min_events=5)

    # The two recommendations should be identical: low-speed event was filtered out.
    assert rec_clean.triple == approx(rec_filtered.triple, abs=1e-9)

  # -------------------------------------------------------------------------
  # Verdict classifications
  # -------------------------------------------------------------------------

  def test_verdicts_classification(self):
    events = [_event(v, 0.85) for v in (11.0, 12.0, 13.0, 14.0) for _ in range(2)]
    rec = recommend((1.0, 1.0, 1.0), events, _band_stats(mid_r=0.85), min_events=4)
    assert rec.per_band_verdict["mid"] == CONTINUE

    events = [_event(v, 0.97) for v in (11.0, 12.0, 13.0, 14.0) for _ in range(2)]
    rec = recommend((1.0, 1.0, 1.0), events, _band_stats(mid_r=0.97), min_events=4)
    assert rec.per_band_verdict["mid"] == CONVERGED

  def test_no_nan_in_output(self):
    events = [_event(11.0, 0.85) for _ in range(8)]
    rec = recommend((1.0, 1.0, 1.0), events, _band_stats(mid_r=0.85), min_events=4)
    for v in rec.triple:
      assert not math.isnan(v)
      assert not math.isinf(v)

  # -------------------------------------------------------------------------
  # Joint per-model solver (US-004)
  # -------------------------------------------------------------------------

  def test_joint_two_bucket_recovery(self):
    """Critic C5 BLOCKING: joint solve recovers the ground-truth (1.30, 1.45, 1.10)
    triple within ±0.02 from events captured under TWO different kp_at_event configs.
    """
    ground_truth = (1.30, 1.45, 1.10)
    kp_configs = [(1.0, 1.0, 1.0), (0.7, 0.8, 0.9)]
    events_with_kp = _build_joint_events(ground_truth, kp_configs, events_per_v=2)
    rec = recommend_joint(events_with_kp, min_events=4)
    for got, truth in zip(rec.triple, ground_truth, strict=True):
      assert got == approx(truth, abs=0.02), f"got {got}, truth {truth}"

  def test_joint_single_bucket_matches_recommend(self):
    """When all events come from one kp config, joint solve agrees with the
    per-bucket WLS solver within ±0.005 (numerical equivalence)."""
    ground_truth = (1.20, 1.30, 1.05)
    kp_config = [(1.0, 1.0, 1.0)]
    events_with_kp = _build_joint_events(ground_truth, kp_config, events_per_v=2)
    events = [e for e, _ in events_with_kp]
    bs = _band_stats(low_r=events[0].tracking_ratio,
                     mid_r=events[1].tracking_ratio,
                     high_r=events[3].tracking_ratio)
    per_bucket_rec = recommend(
      current_triple=(1.0, 1.0, 1.0),
      events=events, band_summaries=bs,
      apply_per_iter_cap=False, min_events=4,
    )
    joint_rec = recommend_joint(events_with_kp, min_events=4)
    for joint_v, single_v in zip(joint_rec.triple, per_bucket_rec.triple, strict=True):
      assert joint_v == approx(single_v, abs=0.005), \
        f"joint {joint_v} vs single {single_v}"

  def test_joint_insufficient_events_returns_hold(self):
    rec = recommend_joint([], min_events=4)
    assert rec.hold is True
    assert "needs more data" in rec.rationale

  def test_joint_clamps_to_kp_ui_range(self):
    """Joint output is clamped to [KP_UI_MIN, KP_UI_MAX] like the per-bucket solver."""
    # Ground-truth way outside UI range — solver wants 8.0, must clamp to 5.0.
    ground_truth = (8.0, 8.0, 8.0)
    kp_configs = [(1.0, 1.0, 1.0)]
    events_with_kp = _build_joint_events(ground_truth, kp_configs, events_per_v=2)
    rec = recommend_joint(events_with_kp, min_events=4)
    for v in rec.triple:
      assert 0.1 <= v <= 5.0

  def test_joint_oversteer_bucket_pushes_recommendation_down(self):
    """Sanity check: when observed events show oversteer (r>1) at moderate kp,
    joint output should pull the multipliers below the kp_at_event."""
    events_with_kp = []
    for v in (11.0, 12.0, 13.0, 14.0):
      for _ in range(3):
        events_with_kp.append((_event(v, 1.20), (1.0, 1.0, 1.0)))  # 20% oversteer
    rec = recommend_joint(events_with_kp, min_events=4)
    # Mid band should drop below 1.0 (toward ~0.83 if uncapped).
    assert rec.triple[1] < 1.0
