from dataclasses import dataclass
from unittest.mock import patch

from opendbc.car import Bus, structs
from opendbc.sunnypilot.car.rivian.carstate_ext import CarStateExt, RIVIAN_DIRECTION_SIGN
from opendbc.sunnypilot.car.rivian.values import RivianFlagsSP

ButtonType = structs.CarState.ButtonEvent.Type


@dataclass
class _MockParser:
  vl: dict
  vl_all: dict | None = None

  def __post_init__(self):
    if self.vl_all is None:
      self.vl_all = {
        msg: {f: [v] for f, v in flds.items()}
        for msg, flds in self.vl.items()
      }


def _make_can_parsers(scroll: int = 0, right_click: int = 0, left_click: int = 0,
                      cluster_unit: int = 1, adas_req: int = 0) -> dict:
  return {
    Bus.alt: _MockParser({
      "WheelButtons_Fwd": {
        "RightButton_Scroll": scroll,
        "RightButton_RightClick": right_click,
        "RightButton_LeftClick": left_click,
      },
    }),
    Bus.adas: _MockParser({
      "Cluster": {
        "Cluster_Unit": cluster_unit,
      },
    }),
    Bus.pt: _MockParser({
      "VDM_AdasSts": {
        "VDM_UserAdasRequest": adas_req,
      },
    }),
  }


def _make_ext() -> CarStateExt:
  CP = structs.CarParams()
  CP.openpilotLongitudinalControl = True
  CP.enableBsm = False

  CP_SP = structs.CarParamsSP()
  CP_SP.flags = RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE

  return CarStateExt(CP, CP_SP)


def _call_update(ext: CarStateExt, scroll: int) -> tuple:
  ret = structs.CarState()
  ret_sp = structs.CarStateSP()
  ext.update_longitudinal_upgrade(ret, ret_sp, _make_can_parsers(scroll=scroll))
  return ret, ret_sp


def _expected_pressed_for_delta(delta_sign: int) -> bool:
  """Direction encoding: pressed = (RIVIAN_DIRECTION_SIGN * sign) > 0."""
  return (RIVIAN_DIRECTION_SIGN * delta_sign) > 0


class TestCarStateExtDirectionDetection:
  """Direction-on-pressed encoding (post-bug2-fix).

  Contract:
    - On every R_S change, carstate_ext emits exactly one gapAdjustCruise
      button event with `pressed` carrying direction.
      pressed = (RIVIAN_DIRECTION_SIGN * sign(delta)) > 0
    - No-change frames emit no event.
    - First-frame seed emits no event.
    - R_S=255 is a NORMAL counter value (NOT a sentinel) — confirmed by
      on-vehicle rlog decode 2026-04-29.
  """

  def setup_method(self):
    self.ext = _make_ext()

  def test_first_frame_seed_does_not_emit_event(self):
    assert self.ext.distance_button is None
    ret, _ = _call_update(self.ext, scroll=137)
    assert self.ext.distance_button == 137
    assert len(ret.buttonEvents) == 0

  def test_same_counter_value_emits_nothing(self):
    _call_update(self.ext, scroll=10)
    ret, _ = _call_update(self.ext, scroll=10)
    assert len(ret.buttonEvents) == 0

  def test_counter_increment_emits_pressed_for_positive_delta(self):
    _call_update(self.ext, scroll=10)
    ret, _ = _call_update(self.ext, scroll=11)
    events = list(ret.buttonEvents)
    assert len(events) == 1
    assert events[0].type == ButtonType.gapAdjustCruise
    assert events[0].pressed is _expected_pressed_for_delta(+1)

  def test_counter_decrement_emits_pressed_for_negative_delta(self):
    _call_update(self.ext, scroll=10)
    ret, _ = _call_update(self.ext, scroll=9)
    events = list(ret.buttonEvents)
    assert len(events) == 1
    assert events[0].type == ButtonType.gapAdjustCruise
    assert events[0].pressed is _expected_pressed_for_delta(-1)

  def test_wrap_forward_254_to_0_is_positive_delta(self):
    # delta = ((0 - 254 + 128) % 256) - 128 = +2  (positive)
    _call_update(self.ext, scroll=254)
    ret, _ = _call_update(self.ext, scroll=0)
    events = list(ret.buttonEvents)
    assert len(events) == 1
    assert events[0].pressed is _expected_pressed_for_delta(+1)

  def test_wrap_backward_0_to_254_is_negative_delta(self):
    # delta = ((254 - 0 + 128) % 256) - 128 = -2  (negative)
    _call_update(self.ext, scroll=0)
    ret, _ = _call_update(self.ext, scroll=254)
    events = list(ret.buttonEvents)
    assert len(events) == 1
    assert events[0].pressed is _expected_pressed_for_delta(-1)

  def test_value_255_is_normal_counter_not_sentinel(self):
    """R_S=255 must be tracked as a normal counter value (Bug A regression guard)."""
    _call_update(self.ext, scroll=254)
    _, _ = _call_update(self.ext, scroll=255)
    assert self.ext.distance_button == 255  # NOT cleared to None

  def test_consecutive_value_255_to_254_emits_one_event(self):
    """255→254 must emit a -1 delta event, not be silently dropped."""
    _call_update(self.ext, scroll=255)  # seed
    ret, _ = _call_update(self.ext, scroll=254)
    events = list(ret.buttonEvents)
    assert len(events) == 1
    assert events[0].pressed is _expected_pressed_for_delta(-1)

  def test_consecutive_value_254_to_255_emits_one_event(self):
    """254→255 must emit a +1 delta event, not be silently dropped."""
    _call_update(self.ext, scroll=254)
    ret, _ = _call_update(self.ext, scroll=255)
    events = list(ret.buttonEvents)
    assert len(events) == 1
    assert events[0].pressed is _expected_pressed_for_delta(+1)

  def test_flipping_rivian_direction_sign_inverts_pressed(self):
    _call_update(self.ext, scroll=10)
    with patch("opendbc.sunnypilot.car.rivian.carstate_ext.RIVIAN_DIRECTION_SIGN", 1):
      ret, _ = _call_update(self.ext, scroll=11)
    events = list(ret.buttonEvents)
    assert len(events) == 1
    # With RIVIAN_DIRECTION_SIGN=+1, delta=+1 → (1*1)>0 → True.
    assert events[0].pressed is True

  def test_scroll_change_emits_exactly_one_gap_adjust_cruise_event(self):
    _call_update(self.ext, scroll=10)
    ret, _ = _call_update(self.ext, scroll=11)
    events = list(ret.buttonEvents)
    assert len(events) == 1
    assert events[0].type == ButtonType.gapAdjustCruise
