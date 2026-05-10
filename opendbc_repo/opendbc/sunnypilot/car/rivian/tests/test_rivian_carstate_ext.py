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

ButtonType = structs.CarState.ButtonEvent.Type


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


def make_can_parsers(user_adas_req: int = 0, right_scroll: int = 233) -> dict:
  """Return minimal mock CAN parsers covering the buses CarStateExt reads.

  - `vl["VDM_UserAdasRequest"] == user_adas_req`, vl_all wraps it in a single-element list.
  - `right_scroll` defaults to 233 (rest position; no scroll event).
  """
  return {
    Bus.alt: MockCANParser({
      "WheelButtons_Fwd": {
        "RightButton_Scroll": right_scroll,
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
  """Like `make_can_parsers` but exposes a vl_all sequence for VDM_UserAdasRequest."""
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


def _feed(ext: CarStateExt, user_adas_req: int, *, right_scroll: int = 233
          ) -> tuple[structs.CarState, structs.CarStateSP]:
  """Drive the public CarStateExt.update() path with a single VDM_UserAdasRequest sample."""
  ret = structs.CarState()
  ret.cruiseState.enabled = False
  ret.vEgoCluster = 0.0
  ret_sp = structs.CarStateSP()
  ext.update(ret, ret_sp, make_can_parsers(user_adas_req, right_scroll=right_scroll))
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


def _has_cancel(ret: structs.CarState) -> bool:
  return any(be.type == ButtonType.cancel and be.pressed for be in ret.buttonEvents)


class TestRivianCarStateExtUp2Edge:
  """UP_2 → ButtonEvent.cancel, op-long=True with harness upgrade."""

  @pytest.fixture(autouse=True)
  def setup_method(self):
    self.ext = _make_ext(op_long=True, harness=True)

  def _feed(self, user_adas_req: int) -> structs.CarState:
    return _feed(self.ext, user_adas_req)[0]

  def test_up2_single_vl_all_value_fires(self):
    ret = self._feed(2)
    assert _has_cancel(ret)
    assert self.ext.prev_user_adas_req == 2

  def test_up2_first_value_fires_immediately(self):
    ret = self._feed(2)
    assert _has_cancel(ret)
    assert self.ext.prev_user_adas_req == 2

  def test_up2_dwell_does_not_refire(self):
    ret_a = self._feed(2)
    ret_b = self._feed(2)
    ret_c = self._feed(2)
    assert _has_cancel(ret_a)
    assert not _has_cancel(ret_b)
    assert not _has_cancel(ret_c)
    assert self.ext.prev_user_adas_req == 2

  def test_idle_rearms_and_next_pulse_fires(self):
    ret_a = self._feed(2)
    ret_b = self._feed(0)
    ret_c = self._feed(2)
    assert _has_cancel(ret_a)
    assert not _has_cancel(ret_b)
    assert _has_cancel(ret_c)
    assert self.ext.prev_user_adas_req == 2

  def test_up1_does_not_trigger_disable(self):
    ret_a = self._feed(1)
    ret_b = self._feed(1)
    assert not _has_cancel(ret_a)
    assert not _has_cancel(ret_b)
    assert self.ext.prev_user_adas_req == 1

  def test_idle_between_up2_rearms(self):
    ret_a = self._feed(2)
    ret_b = self._feed(0)
    ret_c = self._feed(2)
    assert _has_cancel(ret_a)
    assert not _has_cancel(ret_b)
    assert _has_cancel(ret_c)


class TestRivianCarStateExtUp2EdgeStockCruiseNoHarness:
  """UP_2 → ButtonEvent.cancel with op-long=False AND no harness flag.

  Stock-cruise users need full openpilot disengage on UP_2 regardless of op-long
  or the harness gate; VDM_UserAdasRequest is on Bus.pt (always parsed).
  """

  @pytest.fixture(autouse=True)
  def setup_method(self):
    self.ext = _make_ext(op_long=False, harness=False)

  def test_up2_fires_without_op_long_or_harness(self):
    ret, _ = _feed(self.ext, 2)
    assert _has_cancel(ret)
    assert self.ext.prev_user_adas_req == 2

  def test_up1_does_not_fire_without_op_long_or_harness(self):
    _feed(self.ext, 1)
    ret, _ = _feed(self.ext, 1)
    assert not _has_cancel(ret)
    assert self.ext.prev_user_adas_req == 1

  def test_up2_dwell_does_not_refire_without_op_long_or_harness(self):
    _feed(self.ext, 2)
    ret, _ = _feed(self.ext, 2)
    assert not _has_cancel(ret)
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
    ret, _ = _feed(self.ext, 2)
    assert _has_cancel(ret)
    assert self.ext.prev_user_adas_req == 2

  def test_up2_dwell_does_not_refire(self):
    _feed(self.ext, 2)
    ret, _ = _feed(self.ext, 2)
    assert not _has_cancel(ret)


class TestRivianCarStateExtVlAllScenarios:
  """vl_all sequence behavior — within-tick bounces, coalesced gestures, fallbacks."""

  @pytest.fixture(autouse=True)
  def setup_method(self):
    self.ext = _make_ext(op_long=True, harness=True)

  def test_up2_within_tick_bounce_single_pulse(self):
    ret, _ = _feed_multi(self.ext, [2, 0, 2])
    assert _has_cancel(ret)
    assert self.ext.prev_user_adas_req == 2

  def test_up2_coalesced_release_repress(self):
    # Guards against a sticky-on-fire variant (prev=2 whenever 2 is seen) that
    # would suppress the second press because prev never returns to 0.
    ret_a, _ = _feed_multi(self.ext, [0, 2])
    assert _has_cancel(ret_a)
    assert self.ext.prev_user_adas_req == 2

    ret_b, _ = _feed_multi(self.ext, [2, 0])
    assert not _has_cancel(ret_b)
    assert self.ext.prev_user_adas_req == 0

    ret_c, _ = _feed_multi(self.ext, [0, 2])
    assert _has_cancel(ret_c)
    assert self.ext.prev_user_adas_req == 2

  def test_up2_release_then_repress(self):
    ret_a, _ = _feed_multi(self.ext, [0, 2])
    ret_b, _ = _feed_multi(self.ext, [0])
    ret_c, _ = _feed_multi(self.ext, [0, 2])
    assert _has_cancel(ret_a)
    assert not _has_cancel(ret_b)
    assert _has_cancel(ret_c)

  def test_up2_x1_cosmetic_refire_acceptable(self):
    # [[2,0],[2]] fires twice; the second pulse is consumer-side no-op via the
    # selfdrived state machine (already disabled), so this known cosmetic
    # behavior is acceptable.
    ret_a, _ = _feed_multi(self.ext, [2, 0])
    ret_b, _ = _feed_multi(self.ext, [2])
    assert _has_cancel(ret_a)
    assert _has_cancel(ret_b)
    assert self.ext.prev_user_adas_req == 2

  def test_stalk_down_within_tick_fires_set_speed_clamp(self):
    # cruise_enabled=True so the `if not enabled: set_speed = vEgoCluster`
    # branch can't pre-bump; the clamp must come from stalk_down itself.
    _, _ = _feed_multi(self.ext, [0, 3], cruise_enabled=True, vEgoCluster=30.0)
    assert self.ext.set_speed >= 30.0

  def test_vl_all_empty_falls_back_to_prev(self):
    ret_a, _ = _feed_multi(self.ext, [])
    assert not _has_cancel(ret_a)
    assert self.ext.prev_user_adas_req == 0

    self.ext.prev_user_adas_req = 2
    ret_b, _ = _feed_multi(self.ext, [])
    assert not _has_cancel(ret_b)
    assert self.ext.prev_user_adas_req == 2

  def test_stalk_down_repress_after_release(self):
    # Idle tick between presses is required: stalk_down_counter is presence-
    # based and only resets when no 3/4 sample is seen in a tick.
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


class TestRivianCarStateExtConcurrentButtonEvents:
  """Regression guard for the same-tick `buttonEvents`-overwrite bug class.

  Pre-fix, both writers used overwrite-form `ret.buttonEvents = [...]`. A user
  spinning the gap scroll AND pressing UP_2 in the same parser tick would
  silently lose one of the two events.
  """

  @pytest.fixture(autouse=True)
  def setup_method(self):
    self.ext = _make_ext(op_long=True, harness=True)
    # Seed scroll baseline so a +1 delta is detected on the next tick.
    self.ext.distance_button = 233

  def test_up2_concurrent_with_gap_scroll_emits_both_events(self):
    """Same-tick UP_2 rising edge + non-zero RightButton_Scroll delta must
    emit BOTH ButtonType.cancel AND ButtonType.gapAdjustCruise."""
    ret, _ = _feed(self.ext, 2, right_scroll=234)  # delta=+1 → gapAdjust event
    types = [be.type for be in ret.buttonEvents]
    assert ButtonType.cancel in types
    assert ButtonType.gapAdjustCruise in types
    assert len(ret.buttonEvents) == 2

  def test_up2_alone_emits_only_cancel(self):
    """UP_2 with no scroll delta produces exactly one cancel event."""
    ret, _ = _feed(self.ext, 2)  # right_scroll defaults to 233 (rest, no delta)
    types = [be.type for be in ret.buttonEvents]
    assert types == [ButtonType.cancel]

  def test_gap_scroll_alone_emits_only_gap_adjust(self):
    """Gap scroll with no UP_2 produces exactly one gapAdjustCruise event."""
    ret, _ = _feed(self.ext, 0, right_scroll=234)
    types = [be.type for be in ret.buttonEvents]
    assert types == [ButtonType.gapAdjustCruise]
