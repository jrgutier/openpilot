"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pytest
from unittest.mock import MagicMock

from opendbc.car import Bus, structs
from opendbc.sunnypilot.car.rivian.carstate_ext import CarStateExt
from opendbc.sunnypilot.car.rivian.values import RivianFlagsSP


class MockVL:
  """Minimal CAN message value-lookup mock: returns 0 for unknown keys."""
  def __init__(self, values: dict):
    self._values = values

  def __getitem__(self, key):
    return self._values.get(key, 0)

  def __setitem__(self, key, value):
    self._values[key] = value


class MockCANParser:
  def __init__(self, messages: dict):
    self.vl = {name: MockVL(values) for name, values in messages.items()}


def make_can_parsers(user_adas_req: int = 0) -> dict:
  """Return minimal mock CAN parsers covering the buses CarStateExt reads."""
  return {
    Bus.alt: MockCANParser({
      "WheelButtons_Fwd": {
        "RightButton_Scroll": 233,        # rest position; no scroll event
        "RightButton_RightClick": 0,
        "RightButton_LeftClick": 0,
      },
    }),
    Bus.adas: MockCANParser({
      "Cluster": {"Cluster_Unit": 1, "Cluster_VehicleSpeed": 0.0},  # MPH mode
    }),
    Bus.pt: MockCANParser({
      "VDM_AdasSts": {"VDM_UserAdasRequest": user_adas_req},
    }),
  }


def _make_ext(*, op_long: bool, harness: bool) -> CarStateExt:
  CP = MagicMock()
  CP.openpilotLongitudinalControl = op_long
  CP.enableBsm = False
  CP_SP = MagicMock()
  CP_SP.flags = RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE if harness else 0
  return CarStateExt(CP, CP_SP)


def _feed(ext: CarStateExt, user_adas_req: int) -> tuple[structs.CarState, structs.CarStateSP]:
  """Drive the public CarStateExt.update() path and return its outputs."""
  ret = structs.CarState()
  ret.cruiseState.enabled = False
  ret.vEgoCluster = 0.0
  ret_sp = structs.CarStateSP()
  ext.update(ret, ret_sp, make_can_parsers(user_adas_req))
  return ret, ret_sp


class TestRivianCarStateExtUp2Edge:
  """UP_2 edge / debounce, op-long=True with harness upgrade."""

  @pytest.fixture(autouse=True)
  def setup_method(self):
    self.ext = _make_ext(op_long=True, harness=True)

  def _feed(self, user_adas_req: int) -> structs.CarStateSP:
    return _feed(self.ext, user_adas_req)[1]

  def test_up2_single_frame_does_not_fire(self):
    ret_sp = self._feed(2)
    assert not getattr(ret_sp, "madsDisableRequest", False)
    assert self.ext.up2_counter == 1
    assert self.ext.up2_edge_armed

  def test_up2_two_consecutive_frames_fires(self):
    self._feed(2)
    ret_sp = self._feed(2)
    assert getattr(ret_sp, "madsDisableRequest", False)
    assert not self.ext.up2_edge_armed

  def test_up2_dwell_does_not_refire(self):
    self._feed(2)
    self._feed(2)
    ret_sp = self._feed(2)
    assert not getattr(ret_sp, "madsDisableRequest", False)
    assert not self.ext.up2_edge_armed

  def test_idle_rearms_and_next_pulse_fires(self):
    self._feed(2)
    self._feed(2)
    self._feed(0)
    assert self.ext.up2_counter == 0
    assert self.ext.up2_edge_armed
    self._feed(2)
    ret_sp = self._feed(2)
    assert getattr(ret_sp, "madsDisableRequest", False)

  def test_up1_does_not_trigger_disable(self):
    self._feed(1)
    ret_sp = self._feed(1)
    assert not getattr(ret_sp, "madsDisableRequest", False)
    assert self.ext.up2_counter == 0

  def test_idle_mid_sequence_resets_counter(self):
    self._feed(2)
    self._feed(0)
    ret_sp = self._feed(2)
    assert not getattr(ret_sp, "madsDisableRequest", False)
    assert self.ext.up2_counter == 1


class TestRivianCarStateExtUp2EdgeStockCruiseNoHarness:
  """UP_2 edge with op-long=False AND no LONGITUDINAL_HARNESS_UPGRADE flag.

  This is the case the original commit 26fbf812d2 was designed for ("cruise
  stays engaged") and the case the user reported as broken before this fix.
  """

  @pytest.fixture(autouse=True)
  def setup_method(self):
    self.ext = _make_ext(op_long=False, harness=False)

  def test_up2_two_consecutive_frames_fires_without_op_long_or_harness(self):
    """AC1.1a: UP_2 fires madsDisableRequest even when op-long=False AND no harness flag."""
    _, _ = _feed(self.ext, 2)
    _, ret_sp = _feed(self.ext, 2)
    assert getattr(ret_sp, "madsDisableRequest", False)
    assert not self.ext.up2_edge_armed

  def test_up1_does_not_fire_without_op_long_or_harness(self):
    _, _ = _feed(self.ext, 1)
    _, ret_sp = _feed(self.ext, 1)
    assert not getattr(ret_sp, "madsDisableRequest", False)

  def test_up2_single_frame_does_not_fire_without_op_long_or_harness(self):
    _, ret_sp = _feed(self.ext, 2)
    assert not getattr(ret_sp, "madsDisableRequest", False)
    assert self.ext.up2_counter == 1

  def test_up2_dwell_does_not_refire_without_op_long_or_harness(self):
    _feed(self.ext, 2)
    _feed(self.ext, 2)  # fires
    _, ret_sp = _feed(self.ext, 2)
    assert not getattr(ret_sp, "madsDisableRequest", False)

  def test_set_speed_not_mutated_when_op_long_false(self):
    """AC1.5: with op-long=False, the UP_2 path leaves cruiseState.speed/set_speed untouched."""
    set_speed_before = self.ext.set_speed
    ret, _ = _feed(self.ext, 2)
    _feed(self.ext, 2)  # UP_2 fires
    # set_speed mutation is gated under update_longitudinal_upgrade which only runs with harness;
    # with no harness, set_speed must remain at its constructor default.
    assert self.ext.set_speed == set_speed_before
    # cruiseState.speed must not be set by the UP_2 path either.
    assert ret.cruiseState.speed == 0.0


class TestRivianCarStateExtUp2EdgeStockCruiseWithHarness:
  """UP_2 edge with op-long=False but harness flag set (mixed config)."""

  @pytest.fixture(autouse=True)
  def setup_method(self):
    self.ext = _make_ext(op_long=False, harness=True)

  def test_up2_two_consecutive_frames_fires(self):
    """AC1.1b: UP_2 fires regardless of op-long when harness is set."""
    _feed(self.ext, 2)
    _, ret_sp = _feed(self.ext, 2)
    assert getattr(ret_sp, "madsDisableRequest", False)

  def test_up2_dwell_does_not_refire(self):
    _feed(self.ext, 2)
    _feed(self.ext, 2)
    _, ret_sp = _feed(self.ext, 2)
    assert not getattr(ret_sp, "madsDisableRequest", False)
