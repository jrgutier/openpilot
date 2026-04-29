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
  """Return minimal mock CAN parsers for update_longitudinal_upgrade."""
  return {
    Bus.alt: MockCANParser({
      "WheelButtons_Fwd": {
        "RightButton_Scroll": 255,    # sentinel → no gap-adjust event
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


class TestRivianCarStateExtUp2Edge:
  """Frame-by-frame UP_2 edge / debounce tests (plan section 4, scenario 1)."""

  @pytest.fixture(autouse=True)
  def setup_method(self):
    CP = MagicMock()
    CP.openpilotLongitudinalControl = True
    CP.enableBsm = False

    CP_SP = MagicMock()
    CP_SP.flags = RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE

    self.ext = CarStateExt(CP, CP_SP)

  def _feed(self, user_adas_req: int) -> structs.CarStateSP:
    """Call update_longitudinal_upgrade with the given stalk request value."""
    ret = structs.CarState()
    ret.cruiseState.enabled = False
    ret.vEgoCluster = 0.0
    ret_sp = structs.CarStateSP()
    self.ext.update_longitudinal_upgrade(ret, ret_sp, make_can_parsers(user_adas_req))
    return ret_sp

  # --- scenario 1a: single frame does not fire ---------------------------------

  def test_up2_single_frame_does_not_fire(self):
    """One frame of UP_2 increments counter to 1 but does not emit disable."""
    ret_sp = self._feed(2)
    assert not getattr(ret_sp, "madsDisableRequest", False)
    assert self.ext.up2_counter == 1
    assert self.ext.up2_edge_armed

  # --- scenario 1b: two consecutive frames fire --------------------------------

  def test_up2_two_consecutive_frames_fires(self):
    """Two consecutive UP_2 frames produce a one-frame madsDisableRequest pulse."""
    self._feed(2)           # frame 1: counter=1, armed
    ret_sp = self._feed(2)  # frame 2: counter=2, fires
    assert getattr(ret_sp, "madsDisableRequest", False)
    assert not self.ext.up2_edge_armed

  # --- dwell debounce: holding UP_2 must not re-fire ---------------------------

  def test_up2_dwell_does_not_refire(self):
    """After firing, holding UP_2 down does not emit a second pulse."""
    self._feed(2)
    self._feed(2)           # fires; arm goes False
    ret_sp = self._feed(2)  # still held; counter=3 but armed=False
    assert not getattr(ret_sp, "madsDisableRequest", False)
    assert not self.ext.up2_edge_armed

  # --- IDLE re-arms for next pulse ---------------------------------------------

  def test_idle_rearms_and_next_pulse_fires(self):
    """Returning to IDLE resets counter and re-arms; next 2-frame UP_2 fires again."""
    self._feed(2)
    self._feed(2)           # fires
    self._feed(0)           # IDLE: counter=0, armed=True
    assert self.ext.up2_counter == 0
    assert self.ext.up2_edge_armed

    self._feed(2)
    ret_sp = self._feed(2)  # second pulse
    assert getattr(ret_sp, "madsDisableRequest", False)

  # --- UP_1 must never trigger disable -----------------------------------------

  def test_up1_does_not_trigger_disable(self):
    """UP_1 (value 1) never increments up2_counter and never emits disable."""
    self._feed(1)
    ret_sp = self._feed(1)
    assert not getattr(ret_sp, "madsDisableRequest", False)
    assert self.ext.up2_counter == 0

  # --- IDLE clears counter mid-sequence ----------------------------------------

  def test_idle_mid_sequence_resets_counter(self):
    """One frame of UP_2 followed by IDLE resets counter so next single UP_2 is suppressed."""
    self._feed(2)           # counter=1
    self._feed(0)           # IDLE: counter=0, armed=True
    ret_sp = self._feed(2)  # counter=1 again; should NOT fire
    assert not getattr(ret_sp, "madsDisableRequest", False)
    assert self.ext.up2_counter == 1
