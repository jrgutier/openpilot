"""Tests for the curve-entry detector + summarizer.

No rlog binaries — every test constructs duck-typed dataclasses inline.
"""
from __future__ import annotations

import math

import pytest

from openpilot.tools.sunnypilot.kp_tuner.analyze import (
  CarStateMsg,
  ControlsStateMsg,
  EXPECTED_TORQUE_STATE_VERSION,
  KP_UI_MAX,
  KP_UI_MIN,
  Message,
  NS_PER_S,
  SelfdriveStateMsg,
  TorqueState,
  band_for_vego,
  current_kp_working,
  detect_nn_ff_active,
  extract_curve_events,
  interp_weights,
  partition_by_band,
  read_kp_triple_from_init_data,
  summarize,
  validate_segment,
)

DT = 0.01  # 100 Hz


def _stream(samples, *, torque_active=True, torque_saturated=False,
            steering_pressed=False, steering_disengage=False, selfdrive_enabled=True):
  """Build a chronologically-ordered Message list from per-cycle tuples.

  samples: list of (t_seconds, vEgo, desired_curvature, actual_curvature)

  Returns a flat stream where each cycle emits selfdriveState, carState,
  controlsState in that order so the cycle is consistent before extract sees it.
  """
  msgs: list[Message] = []
  for t, v, dc, ac in samples:
    t_ns = int(t * NS_PER_S)
    msgs.append(Message(
      which="selfdriveState", logMonoTime=t_ns,
      selfdriveState=SelfdriveStateMsg(
        enabled=selfdrive_enabled,
      ),
    ))
    msgs.append(Message(
      which="carState", logMonoTime=t_ns + 1,
      carState=CarStateMsg(
        vEgo=v,
        steeringPressed=steering_pressed,
        steeringDisengage=steering_disengage,
      ),
    ))
    msgs.append(Message(
      which="controlsState", logMonoTime=t_ns + 2,
      controlsState=ControlsStateMsg(
        desiredCurvature=dc,
        curvature=ac,
        torqueState=TorqueState(
          active=torque_active,
          saturated=torque_saturated,
          version=EXPECTED_TORQUE_STATE_VERSION,
        ),
      ),
    ))
  return msgs


def _ramp_samples(*, v_ego, peak_lat_accel, ramp_seconds=1.0, hold_seconds=0.5,
                  pre_seconds=0.6, post_seconds=0.6, ratio=0.85):
  """Synthesize a curve-entry waveform whose `|curvature * v^2|` linearly ramps
  from 0 to `peak_lat_accel`, holds, then decays. Actual curvature is
  `ratio * desired` so the undershoot ratio is exactly `ratio`.
  """
  peak_curv = peak_lat_accel / max(v_ego * v_ego, 1e-9)
  out = []
  t = 0.0
  # Pre-window flat at zero.
  while t < pre_seconds:
    out.append((t, v_ego, 0.0, 0.0))
    t += DT
  # Linear ramp 0 -> peak.
  t_ramp_start = t
  ramp_end = t + ramp_seconds
  while t < ramp_end:
    frac = (t - t_ramp_start) / ramp_seconds
    dc = peak_curv * frac
    out.append((t, v_ego, dc, ratio * dc))
    t += DT
  # Hold at peak.
  hold_end = t + hold_seconds
  while t < hold_end:
    out.append((t, v_ego, peak_curv, ratio * peak_curv))
    t += DT
  # Decay back to zero.
  decay_end = t + post_seconds
  decay_start = t
  while t < decay_end:
    frac = 1.0 - (t - decay_start) / post_seconds
    dc = peak_curv * frac
    out.append((t, v_ego, dc, ratio * dc))
    t += DT
  return out


# ---------------------------------------------------------------------------
# Helpers / pure functions
# ---------------------------------------------------------------------------

def test_interp_weights_endpoints():
  assert interp_weights(0.0) == (1.0, 0.0, 0.0)
  assert interp_weights(6.7) == (1.0, 0.0, 0.0)
  assert interp_weights(15.6) == pytest.approx((0.0, 1.0, 0.0))
  assert interp_weights(33.5) == pytest.approx((0.0, 0.0, 1.0))
  assert interp_weights(50.0) == (0.0, 0.0, 1.0)


def test_interp_weights_interior_sums_to_one():
  for v in [4.0, 11.0, 20.0, 30.0]:
    w = interp_weights(v)
    assert sum(w) == pytest.approx(1.0)


def test_band_for_vego():
  assert band_for_vego(2.0) == "low"
  assert band_for_vego(6.7) == "low"
  assert band_for_vego(11.0) == "mid"
  assert band_for_vego(15.6) == "mid"
  assert band_for_vego(20.0) == "high"


def test_current_kp_working_uses_interp():
  # Multiplier (1,1,1) reduces to KP_INTERP(v).
  base11 = current_kp_working(11.0, (1.0, 1.0, 1.0))
  assert base11 > 0
  # Doubling all multipliers doubles the kp_working at any v.
  double11 = current_kp_working(11.0, (2.0, 2.0, 2.0))
  assert double11 == pytest.approx(2.0 * base11)


# ---------------------------------------------------------------------------
# extract_curve_events — synthetic happy-path
# ---------------------------------------------------------------------------

def test_extract_synthetic_event():
  msgs = _stream(_ramp_samples(v_ego=11.0, peak_lat_accel=1.5, ratio=0.8))
  events = extract_curve_events(msgs)
  assert len(events) == 1
  e = events[0]
  assert e.band == "mid"
  assert e.tracking_ratio == pytest.approx(0.8, abs=0.02)
  # Lag should be near zero — actual exactly tracks desired in this synth stream.
  assert math.isclose(e.lag_seconds, 0.0, abs_tol=2 * DT) or math.isnan(e.lag_seconds)


# ---------------------------------------------------------------------------
# Engagement gating
# ---------------------------------------------------------------------------

def test_extract_gates_steering_pressed():
  msgs = _stream(_ramp_samples(v_ego=11.0, peak_lat_accel=1.5), steering_pressed=True)
  assert extract_curve_events(msgs) == []


def test_extract_gates_torque_inactive():
  msgs = _stream(_ramp_samples(v_ego=11.0, peak_lat_accel=1.5), torque_active=False)
  assert extract_curve_events(msgs) == []


def test_extract_gates_torque_saturated():
  msgs = _stream(_ramp_samples(v_ego=11.0, peak_lat_accel=1.5), torque_saturated=True)
  assert extract_curve_events(msgs) == []


def test_extract_gates_selfdrive_disabled():
  msgs = _stream(_ramp_samples(v_ego=11.0, peak_lat_accel=1.5), selfdrive_enabled=False)
  assert extract_curve_events(msgs) == []


def test_extract_gates_steering_disengage():
  msgs = _stream(_ramp_samples(v_ego=11.0, peak_lat_accel=1.5), steering_disengage=True)
  assert extract_curve_events(msgs) == []


# ---------------------------------------------------------------------------
# validate_segment
# ---------------------------------------------------------------------------

def test_validate_segment_version_match():
  msgs = _stream(_ramp_samples(v_ego=11.0, peak_lat_accel=1.5))
  assert validate_segment(msgs) is True


def test_validate_segment_version_mismatch():
  msgs = _stream(_ramp_samples(v_ego=11.0, peak_lat_accel=1.5))
  for m in msgs:
    if m.which == "controlsState" and m.controlsState is not None:
      m.controlsState.torqueState.version = EXPECTED_TORQUE_STATE_VERSION + 99
  assert validate_segment(msgs) is False


# ---------------------------------------------------------------------------
# Speed-aware threshold
# ---------------------------------------------------------------------------

def test_curve_entry_threshold_speed_aware():
  # Freeway curve: vEgo=25, curvature=0.003 -> lat_accel ≈ 1.875 m/s^2 -> trigger.
  freeway = _stream(_ramp_samples(v_ego=25.0, peak_lat_accel=1.875))
  assert len(extract_curve_events(freeway)) == 1
  # Parking-lot wiggle: vEgo=4, curvature=0.005 -> lat_accel ≈ 0.08 m/s^2 -> no trigger.
  parking = _stream(_ramp_samples(v_ego=4.0, peak_lat_accel=0.08))
  assert extract_curve_events(parking) == []


# ---------------------------------------------------------------------------
# Threshold edge tests (BLOCKING — Critic CN3 iter-2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("peak,expected_events", [
  (0.29, 0),  # never crosses the 0.3 lower edge -> no event
  (0.31, 0),  # crosses lower edge but stops below 1.0 ceiling -> no event
  (1.0, 0),   # exactly at the upper edge — implementation uses strict >, so no event
  (1.01, 1),  # just above upper edge — exactly one event
])
def test_curve_entry_threshold_edges(peak, expected_events):
  msgs = _stream(_ramp_samples(
    v_ego=11.0,
    peak_lat_accel=peak,
    ramp_seconds=1.5,  # within 2s timeout
    hold_seconds=0.2,
    pre_seconds=0.6,
    post_seconds=0.5,
    ratio=0.9,
  ))
  events = extract_curve_events(msgs)
  assert len(events) == expected_events, (
    f"peak={peak} produced {len(events)} events, expected {expected_events}"
  )


# ---------------------------------------------------------------------------
# Partition / summarize
# ---------------------------------------------------------------------------

def test_partition_and_summarize():
  events_low = []
  events_mid = []
  for v_ego, peak in [(11.0, 1.5), (12.0, 1.6), (13.0, 1.7)]:
    e = extract_curve_events(_stream(_ramp_samples(v_ego=v_ego, peak_lat_accel=peak, ratio=0.9)))
    events_mid.extend(e)
  assert len(events_mid) == 3
  partitioned = partition_by_band(events_mid + events_low)
  bs = summarize(partitioned)
  assert bs.mid is not None and bs.mid.count == 3
  assert bs.low is None
  assert bs.mid.median_ratio == pytest.approx(0.9, abs=0.02)


# ---------------------------------------------------------------------------
# NN-FF detection
# ---------------------------------------------------------------------------

class _InitData:
  def __init__(self, params):
    self.params = params


class _Model:
  def __init__(self, path):
    self.path = path


class _NNLC:
  def __init__(self, path):
    self.model = _Model(path)


class _CPSP:
  def __init__(self, path):
    self.neuralNetworkLateralControl = _NNLC(path)


def test_nn_ff_detection_dual_check():
  init_on = _InitData({"NeuralNetworkLateralControl": b"1"})
  init_off = _InitData({"NeuralNetworkLateralControl": b"0"})
  cpsp_real = _CPSP("/data/openpilot/sunnypilot/neural_network_data/neural_network_lateral_control/RIVIAN.json")
  cpsp_mock = _CPSP("/data/openpilot/sunnypilot/neural_network_data/neural_network_lateral_control/MOCK.json")

  assert detect_nn_ff_active(init_on, cpsp_real) is True
  assert detect_nn_ff_active(init_on, cpsp_mock) is False  # param on but mock model
  assert detect_nn_ff_active(init_off, cpsp_real) is False  # model bound but param off
  assert detect_nn_ff_active(init_off, cpsp_mock) is False
  assert detect_nn_ff_active(None, cpsp_real) is False
  assert detect_nn_ff_active(init_on, None) is False


# ---------------------------------------------------------------------------
# Auto-detect Kp triple from initData.params
# ---------------------------------------------------------------------------

class _Entry:
  def __init__(self, key, value):
    self.key = key
    self.value = value


class _CapnpMap:
  """Mimics the capnp Map struct: exposes `.entries` but no `.get`."""
  def __init__(self, items):
    self.entries = [_Entry(k, v) for k, v in items.items()]


class _InitDataCapnp:
  def __init__(self, items):
    self.params = _CapnpMap(items)


def test_read_kp_triple_dict_path():
  init = _InitData({b"x": b"y", "KpLowSpeed": b"0.7", "KpMidSpeed": b"0.85", "KpHighSpeed": b"0.95"})
  assert read_kp_triple_from_init_data(init) == (0.7, 0.85, 0.95)


def test_read_kp_triple_capnp_entries_path():
  init = _InitDataCapnp({"KpLowSpeed": b"1.2", "KpMidSpeed": b"1.0", "KpHighSpeed": b"0.8"})
  assert read_kp_triple_from_init_data(init) == pytest.approx((1.2, 1.0, 0.8))


def test_read_kp_triple_missing_keys_default_one():
  assert read_kp_triple_from_init_data(_InitData({})) == (1.0, 1.0, 1.0)


def test_read_kp_triple_bad_value_falls_back_to_one():
  init = _InitData({"KpLowSpeed": b"not-a-float", "KpMidSpeed": b"1.5", "KpHighSpeed": b"0.5"})
  assert read_kp_triple_from_init_data(init) == (1.0, 1.5, 0.5)


def test_read_kp_triple_clamps_out_of_range():
  init = _InitData({"KpLowSpeed": b"0.0", "KpMidSpeed": b"999.0", "KpHighSpeed": b"2.5"})
  assert read_kp_triple_from_init_data(init) == (KP_UI_MIN, KP_UI_MAX, 2.5)


def test_read_kp_triple_returns_none_when_no_init_data():
  assert read_kp_triple_from_init_data(None) is None


def test_read_kp_triple_returns_none_when_no_params_attr():
  class _Empty: ...
  assert read_kp_triple_from_init_data(_Empty()) is None


# ---------------------------------------------------------------------------
# Directional reporting (oversteer detection + counts)
# ---------------------------------------------------------------------------

def test_extract_synthetic_oversteer_event():
  """ratio=1.15 (oversteer) → event captured with tracking_ratio≈1.15 and
  band counted as oversteer."""
  from openpilot.tools.sunnypilot.kp_tuner.analyze import partition_by_band, summarize
  msgs = _stream(_ramp_samples(v_ego=20.0, peak_lat_accel=1.5, ratio=1.15))
  events = extract_curve_events(msgs)
  assert len(events) == 1
  assert events[0].tracking_ratio == pytest.approx(1.15, abs=0.02)
  bs = summarize(partition_by_band(events))
  assert bs.high is not None
  assert bs.high.oversteer_count == 1
  assert bs.high.undershoot_count == 0


def test_summarize_mixed_directional_counts():
  """Mixed band: 3 under (0.85, 0.80, 0.92) + 2 over (1.10, 1.15)."""
  from openpilot.tools.sunnypilot.kp_tuner.analyze import (
    CurveEvent, partition_by_band, summarize,
  )
  events = []
  for r in (0.85, 0.80, 0.92, 1.10, 1.15):
    events.append(CurveEvent(
      t0=0.0, t_peak=0.5, vEgo_t0=20.0, vEgo_peak=20.0,
      peak_desired_curvature=0.01, peak_actual_curvature=0.01 * r,
      tracking_ratio=r, lag_seconds=0.05, band="high",
    ))
  bs = summarize(partition_by_band(events))
  assert bs.high is not None
  assert bs.high.count == 5
  assert bs.high.undershoot_count == 3
  assert bs.high.oversteer_count == 2
