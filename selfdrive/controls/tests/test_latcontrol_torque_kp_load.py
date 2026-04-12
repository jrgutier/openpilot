"""Tests for LatControlTorque._load_kp_multipliers() covering None default,
happy path, bad-string fallback, and out-of-range clamp."""
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque, KP_UI_MIN, KP_UI_MAX


def _load(**params):
  return LatControlTorque._load_kp_multipliers(params.get)


def test_load_kp_multipliers_defaults_when_none():
  assert _load() == [1.0, 1.0, 1.0]


def test_load_kp_multipliers_happy_path():
  assert _load(KpLowSpeed="0.7", KpMidSpeed="0.85", KpHighSpeed="0.95") == [0.7, 0.85, 0.95]


def test_load_kp_multipliers_bad_string_falls_back_to_default():
  assert _load(KpLowSpeed="not-a-float") == [1.0, 1.0, 1.0]


def test_load_kp_multipliers_clamps_out_of_range_values():
  assert _load(KpLowSpeed="0.0", KpMidSpeed="999.0", KpHighSpeed="2.5") == [KP_UI_MIN, KP_UI_MAX, 2.5]
