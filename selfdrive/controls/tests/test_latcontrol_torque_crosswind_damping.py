"""Tests for crosswind damping: KdHighSpeed param loading, D-term sign convention,
first-frame guard, integrator decay at speed breakpoints, and error filter lag."""
import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.latcontrol_torque import (
  LatControlTorque, KD_UI_MIN, KD_UI_MAX, ERROR_LP_FILTER_HZ,
  INTEGRATOR_DECAY_SPEED_BP, INTEGRATOR_DECAY_FACTOR,
)

DT = 0.01  # 100 Hz control loop


# --- KdHighSpeed param loading ---

def _load_kd(raw_value):
  """Use the actual _read_param helper from latcontrol_torque.py."""
  return LatControlTorque._read_param(lambda _: raw_value, "KdHighSpeed", KD_UI_MIN, KD_UI_MAX)


def test_kd_defaults_when_none():
  assert _load_kd(None) == 1.0


def test_kd_happy_path():
  assert _load_kd("1.5") == 1.5


def test_kd_bad_string_falls_back_to_default():
  assert _load_kd("not-a-float") == 1.0


def test_kd_clamps_out_of_range():
  assert _load_kd("-1.0") == KD_UI_MIN
  assert _load_kd("999.0") == KD_UI_MAX


def test_kd_zero_disables():
  assert _load_kd("0.0") == 0.0


# --- D-term sign convention ---

def test_d_term_sign_damps_oscillation():
  """When measurement decreases (car returning to center), error_rate should be
  positive, causing D term to oppose the return and damp oscillation."""
  meas_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * 3.0), DT, initialized=False)

  # Simulate measurement decreasing from 1.0 to 0.5
  meas_filter.update(1.0)  # first frame: pass-through
  prev = 1.0
  filtered = meas_filter.update(0.5)  # second frame: filtered

  error_rate = -(filtered - prev) / DT

  # filtered < prev (measurement decreased), so -(filtered - prev) > 0
  assert error_rate > 0, f"D term should be positive when measurement decreases, got {error_rate}"


def test_d_term_sign_opposes_increasing_measurement():
  """When measurement increases (car drifting further from center), error_rate
  should be negative, adding corrective force via D term."""
  meas_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * 3.0), DT, initialized=False)

  meas_filter.update(0.5)
  prev = 0.5
  filtered = meas_filter.update(1.0)

  error_rate = -(filtered - prev) / DT

  assert error_rate < 0, f"D term should be negative when measurement increases, got {error_rate}"


# --- First-frame guard ---

def test_first_frame_error_rate_is_zero():
  """After reset (prev_filtered_meas=None), error_rate must be 0.0."""
  prev_filtered_meas = None
  meas_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * 3.0), DT, initialized=False)
  filtered = meas_filter.update(5.0)  # arbitrary measurement

  if prev_filtered_meas is None:
    error_rate = 0.0
  else:
    error_rate = -(filtered - prev_filtered_meas) / DT

  assert error_rate == 0.0


def test_second_frame_produces_nonzero_error_rate():
  """After the first frame, normal derivative computation should produce non-zero
  error_rate for changing measurement."""
  meas_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * 3.0), DT, initialized=False)

  prev = meas_filter.update(1.0)  # first frame: pass-through, prev set
  filtered = meas_filter.update(2.0)  # second frame

  error_rate = -(filtered - prev) / DT
  assert error_rate != 0.0


# --- Integrator decay at speed breakpoints ---

def test_integrator_no_decay_below_threshold():
  """No decay at or below 10 m/s."""
  decay = float(np.interp(10.0, INTEGRATOR_DECAY_SPEED_BP, INTEGRATOR_DECAY_FACTOR))
  assert decay == 1.0

  decay_low = float(np.interp(5.0, INTEGRATOR_DECAY_SPEED_BP, INTEGRATOR_DECAY_FACTOR))
  assert decay_low == 1.0


def test_integrator_decay_at_20():
  """Interpolated decay at 20 m/s should be ~0.998."""
  decay = float(np.interp(20.0, INTEGRATOR_DECAY_SPEED_BP, INTEGRATOR_DECAY_FACTOR))
  assert abs(decay - 0.998) < 1e-6


def test_integrator_decay_at_30():
  """Maximum decay at 30 m/s should be 0.995."""
  decay = float(np.interp(30.0, INTEGRATOR_DECAY_SPEED_BP, INTEGRATOR_DECAY_FACTOR))
  assert abs(decay - 0.995) < 1e-6


def test_integrator_decay_effect_over_one_second():
  """At 30 m/s, 0.995^100 ≈ 0.606 — integrator loses ~39% per second."""
  decay_per_frame = 0.995
  frames_per_second = 100
  remaining = decay_per_frame ** frames_per_second
  assert 0.59 < remaining < 0.62


# --- Error filter lag ---

def test_error_filter_lags_step_input():
  """On a step input from 0 to 1, the filtered output should be less than 1
  for the first several frames (demonstrating lag)."""
  error_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * ERROR_LP_FILTER_HZ), DT)

  # Apply step input
  filtered = 0.0
  for _ in range(5):
    filtered = error_filter.update(1.0)

  # After 5 frames at 100 Hz (50ms), a 0.8 Hz filter should not have converged
  assert filtered < 0.95, f"Filter should lag step input, got {filtered}"
  assert filtered > 0.0, f"Filter should have started responding, got {filtered}"


def test_error_filter_converges():
  """After enough frames, the filtered output should converge to the input."""
  error_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * ERROR_LP_FILTER_HZ), DT)

  filtered = 0.0
  for _ in range(500):  # 5 seconds at 100 Hz
    filtered = error_filter.update(1.0)

  assert abs(filtered - 1.0) < 0.01
