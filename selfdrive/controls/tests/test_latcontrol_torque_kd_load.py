"""Tests for LatControlTorque Kd param loading (_load_kd_multipliers)
covering None default, happy path, bad-string fallback, out-of-range clamp,
and the partial-params backwards-compat scenario (existing device where only
KdHighSpeed exists in Params and the two new keys are absent)."""
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque, KD_UI_MIN, KD_UI_MAX


def _load(**params):
  return LatControlTorque._load_kd_multipliers(params.get)


def test_load_kd_multipliers_defaults_when_none():
  assert _load() == [1.0, 1.0, 1.0]


def test_load_kd_multipliers_happy_path():
  assert _load(KdLowSpeed="0.3", KdMidSpeed="0.5", KdHighSpeed="1.0") == [0.3, 0.5, 1.0]


def test_load_kd_multipliers_bad_string_falls_back_to_default():
  assert _load(KdLowSpeed="not-a-float") == [1.0, 1.0, 1.0]


def test_load_kd_multipliers_clamps_out_of_range_values():
  assert _load(KdLowSpeed="-0.5", KdMidSpeed="999", KdHighSpeed="1.5") == [KD_UI_MIN, KD_UI_MAX, 1.5]


def test_load_kd_multipliers_partial_params_backwards_compat():
  # Existing device upgrading: only KdHighSpeed exists from the old single-scalar era.
  # KdLowSpeed and KdMidSpeed must default to 1.0 (no behavior change) until the user
  # touches them in the tuning menu.
  assert _load(KdHighSpeed="0.8") == [1.0, 1.0, 0.8]
