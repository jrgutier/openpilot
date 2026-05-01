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
  """Mock CANParser exposing both `vl` (last sample) and `vl_all` (full tick samples).

  When `vl_all_messages` is omitted, `vl_all` mirrors `messages` with each scalar
  wrapped in a single-element list so existing scalar-feed call sites keep working
  unchanged. Pass `vl_all_messages` explicitly to drive multi-sample sequences.
  """
  def __init__(self, messages: dict, vl_all_messages: dict | None = None):
    self.vl = {name: MockVL(values) for name, values in messages.items()}
    src = vl_all_messages if vl_all_messages is not None else messages
    self.vl_all = {
      name: {field: list(val) if isinstance(val, list) else [val] for field, val in fields.items()}
      for name, fields in src.items()
    }


def make_can_parsers(user_adas_req: int = 0) -> dict:
  """Return minimal mock CAN parsers covering the buses CarStateExt reads.

  Single-value form: `vl["VDM_UserAdasRequest"] == user_adas_req` and
  `vl_all["VDM_UserAdasRequest"] == [user_adas_req]`.
  """
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


def make_can_parsers_multi(user_adas_req_list: list[int]) -> dict:
  """Like `make_can_parsers` but exposes a vl_all sequence for VDM_UserAdasRequest.

  - `vl["VDM_AdasSts"]["VDM_UserAdasRequest"]` = last sample (or 0 if empty), matching real CANParser.
  - `vl_all["VDM_AdasSts"]["VDM_UserAdasRequest"]` = full list provided here.
  An empty list yields an empty vl_all to exercise the producer's empty-fallback branch.
  """
  last_val = user_adas_req_list[-1] if user_adas_req_list else 0
  return {
    Bus.alt: MockCANParser({
      "WheelButtons_Fwd": {
        "RightButton_Scroll": 233,
        "RightButton_RightClick": 0,
        "RightButton_LeftClick": 0,
      },
    }),
    Bus.adas: MockCANParser({
      "Cluster": {"Cluster_Unit": 1, "Cluster_VehicleSpeed": 0.0},
    }),
    Bus.pt: MockCANParser(
      {"VDM_AdasSts": {"VDM_UserAdasRequest": last_val}},
      {"VDM_AdasSts": {"VDM_UserAdasRequest": list(user_adas_req_list)}},
    ),
  }


def _make_ext(*, op_long: bool, harness: bool) -> CarStateExt:
  CP = MagicMock()
  CP.openpilotLongitudinalControl = op_long
  CP.enableBsm = False
  CP_SP = MagicMock()
  CP_SP.flags = RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE if harness else 0
  return CarStateExt(CP, CP_SP)


def _feed(ext: CarStateExt, user_adas_req: int) -> tuple[structs.CarState, structs.CarStateSP]:
  """Drive the public CarStateExt.update() path with a single VDM_UserAdasRequest sample."""
  ret = structs.CarState()
  ret.cruiseState.enabled = False
  ret.vEgoCluster = 0.0
  ret_sp = structs.CarStateSP()
  ext.update(ret, ret_sp, make_can_parsers(user_adas_req))
  return ret, ret_sp


def _feed_multi(ext: CarStateExt, user_adas_req_list: list[int], *,
                cruise_enabled: bool = False, vEgoCluster: float = 0.0
                ) -> tuple[structs.CarState, structs.CarStateSP]:
  """Drive update() with a vl_all sample sequence for VDM_UserAdasRequest."""
  ret = structs.CarState()
  ret.cruiseState.enabled = cruise_enabled
  ret.vEgoCluster = vEgoCluster
  ret_sp = structs.CarStateSP()
  ext.update(ret, ret_sp, make_can_parsers_multi(user_adas_req_list))
  return ret, ret_sp


class TestRivianCarStateExtUp2Edge:
  """UP_2 → madsDisableRequest, op-long=True with harness upgrade."""

  @pytest.fixture(autouse=True)
  def setup_method(self):
    self.ext = _make_ext(op_long=True, harness=True)

  def _feed(self, user_adas_req: int) -> structs.CarStateSP:
    return _feed(self.ext, user_adas_req)[1]

  def test_up2_single_vl_all_value_fires(self):
    ret_sp = self._feed(2)
    assert getattr(ret_sp, "madsDisableRequest", False)
    assert self.ext.prev_user_adas_req == 2

  def test_up2_first_value_fires_immediately(self):
    ret_sp = self._feed(2)
    assert getattr(ret_sp, "madsDisableRequest", False)
    assert self.ext.prev_user_adas_req == 2

  def test_up2_dwell_does_not_refire(self):
    ret_sp_a = self._feed(2)
    ret_sp_b = self._feed(2)
    ret_sp_c = self._feed(2)
    assert getattr(ret_sp_a, "madsDisableRequest", False)
    assert not getattr(ret_sp_b, "madsDisableRequest", False)
    assert not getattr(ret_sp_c, "madsDisableRequest", False)
    assert self.ext.prev_user_adas_req == 2

  def test_idle_rearms_and_next_pulse_fires(self):
    ret_sp_a = self._feed(2)
    ret_sp_b = self._feed(0)
    ret_sp_c = self._feed(2)
    assert getattr(ret_sp_a, "madsDisableRequest", False)
    assert not getattr(ret_sp_b, "madsDisableRequest", False)
    assert getattr(ret_sp_c, "madsDisableRequest", False)
    assert self.ext.prev_user_adas_req == 2

  def test_up1_does_not_trigger_disable(self):
    ret_sp_a = self._feed(1)
    ret_sp_b = self._feed(1)
    assert not getattr(ret_sp_a, "madsDisableRequest", False)
    assert not getattr(ret_sp_b, "madsDisableRequest", False)
    assert self.ext.prev_user_adas_req == 1

  def test_idle_between_up2_rearms(self):
    ret_sp_a = self._feed(2)
    ret_sp_b = self._feed(0)
    ret_sp_c = self._feed(2)
    assert getattr(ret_sp_a, "madsDisableRequest", False)
    assert not getattr(ret_sp_b, "madsDisableRequest", False)
    assert getattr(ret_sp_c, "madsDisableRequest", False)


class TestRivianCarStateExtUp2EdgeStockCruiseNoHarness:
  """UP_2 → madsDisableRequest with op-long=False AND no harness flag.

  Stock-cruise users need MADS to disable while cruise stays engaged, so UP_2
  must fire madsDisableRequest regardless of op-long or the harness gate.
  """

  @pytest.fixture(autouse=True)
  def setup_method(self):
    self.ext = _make_ext(op_long=False, harness=False)

  def test_up2_fires_without_op_long_or_harness(self):
    _, ret_sp = _feed(self.ext, 2)
    assert getattr(ret_sp, "madsDisableRequest", False)
    assert self.ext.prev_user_adas_req == 2

  def test_up1_does_not_fire_without_op_long_or_harness(self):
    _, _ = _feed(self.ext, 1)
    _, ret_sp = _feed(self.ext, 1)
    assert not getattr(ret_sp, "madsDisableRequest", False)
    assert self.ext.prev_user_adas_req == 1

  def test_up2_dwell_does_not_refire_without_op_long_or_harness(self):
    _feed(self.ext, 2)
    _, ret_sp = _feed(self.ext, 2)
    assert not getattr(ret_sp, "madsDisableRequest", False)
    assert self.ext.prev_user_adas_req == 2

  def test_set_speed_not_mutated_when_op_long_false(self):
    set_speed_before = self.ext.set_speed
    ret, _ = _feed(self.ext, 2)
    _feed(self.ext, 0)
    _feed(self.ext, 2)
    assert self.ext.set_speed == set_speed_before
    assert ret.cruiseState.speed == 0.0


class TestRivianCarStateExtUp2EdgeStockCruiseWithHarness:
  """UP_2 edge with op-long=False but harness flag set (mixed config)."""

  @pytest.fixture(autouse=True)
  def setup_method(self):
    self.ext = _make_ext(op_long=False, harness=True)

  def test_up2_fires(self):
    _, ret_sp = _feed(self.ext, 2)
    assert getattr(ret_sp, "madsDisableRequest", False)
    assert self.ext.prev_user_adas_req == 2

  def test_up2_dwell_does_not_refire(self):
    _feed(self.ext, 2)
    _, ret_sp = _feed(self.ext, 2)
    assert not getattr(ret_sp, "madsDisableRequest", False)


class TestRivianCarStateExtVlAllScenarios:
  """vl_all sequence behavior — within-tick bounces, coalesced gestures, fallbacks."""

  @pytest.fixture(autouse=True)
  def setup_method(self):
    self.ext = _make_ext(op_long=True, harness=True)

  def test_up2_within_tick_bounce_single_pulse(self):
    _, ret_sp = _feed_multi(self.ext, [2, 0, 2])
    assert getattr(ret_sp, "madsDisableRequest", False)
    assert self.ext.prev_user_adas_req == 2

  def test_up2_coalesced_release_repress(self):
    # Guards against a sticky-on-fire variant (prev=2 whenever 2 is seen) that
    # would suppress the second press because prev never returns to 0.
    _, ret_sp_a = _feed_multi(self.ext, [0, 2])
    assert getattr(ret_sp_a, "madsDisableRequest", False)
    assert self.ext.prev_user_adas_req == 2

    _, ret_sp_b = _feed_multi(self.ext, [2, 0])
    assert not getattr(ret_sp_b, "madsDisableRequest", False)
    assert self.ext.prev_user_adas_req == 0

    _, ret_sp_c = _feed_multi(self.ext, [0, 2])
    assert getattr(ret_sp_c, "madsDisableRequest", False)
    assert self.ext.prev_user_adas_req == 2

  def test_up2_release_then_repress(self):
    _, ret_sp_a = _feed_multi(self.ext, [0, 2])
    _, ret_sp_b = _feed_multi(self.ext, [0])
    _, ret_sp_c = _feed_multi(self.ext, [0, 2])
    assert getattr(ret_sp_a, "madsDisableRequest", False)
    assert not getattr(ret_sp_b, "madsDisableRequest", False)
    assert getattr(ret_sp_c, "madsDisableRequest", False)

  def test_up2_x1_cosmetic_refire_acceptable(self):
    # [[2,0],[2]] fires twice; the second pulse is consumer-side no-op via the
    # mads enabled-latch, so this known cosmetic behavior is acceptable.
    _, ret_sp_a = _feed_multi(self.ext, [2, 0])
    _, ret_sp_b = _feed_multi(self.ext, [2])
    assert getattr(ret_sp_a, "madsDisableRequest", False)
    assert getattr(ret_sp_b, "madsDisableRequest", False)
    assert self.ext.prev_user_adas_req == 2

  def test_stalk_down_within_tick_fires_set_speed_clamp(self):
    # cruise_enabled=True so the `if not enabled: set_speed = vEgoCluster`
    # branch can't pre-bump; the clamp must come from stalk_down itself.
    _, _ = _feed_multi(self.ext, [0, 3], cruise_enabled=True, vEgoCluster=30.0)
    assert self.ext.set_speed >= 30.0

  def test_vl_all_empty_falls_back_to_prev(self):
    _, ret_sp_a = _feed_multi(self.ext, [])
    assert not getattr(ret_sp_a, "madsDisableRequest", False)
    assert self.ext.prev_user_adas_req == 0

    self.ext.prev_user_adas_req = 2
    _, ret_sp_b = _feed_multi(self.ext, [])
    assert not getattr(ret_sp_b, "madsDisableRequest", False)
    assert self.ext.prev_user_adas_req == 2

  def test_stalk_down_repress_after_release(self):
    # Idle tick between presses is required: stalk_down_counter is presence-
    # based and only resets when no 3/4 sample is seen in a tick. A coalesced
    # [[0,3],[3,0],[0,3]] would only bump on the first tick (counter stays >0).
    set_speed_before = self.ext.set_speed
    _, _ = _feed_multi(self.ext, [0, 3], cruise_enabled=True, vEgoCluster=30.0)
    bump_a = self.ext.set_speed
    assert bump_a >= 30.0

    self.ext.set_speed = set_speed_before
    _, _ = _feed_multi(self.ext, [0], cruise_enabled=True, vEgoCluster=0.0)
    assert self.ext.set_speed < 30.0

    _, _ = _feed_multi(self.ext, [0, 3], cruise_enabled=True, vEgoCluster=30.0)
    bump_b = self.ext.set_speed
    assert bump_b >= 30.0
