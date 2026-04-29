from dataclasses import dataclass
from unittest.mock import patch

from opendbc.car import Bus, structs
from opendbc.sunnypilot.car.rivian.carstate_ext import CarStateExt, RIVIAN_DIRECTION_SIGN
from opendbc.sunnypilot.car.rivian.values import RivianFlagsSP

ButtonType = structs.CarState.ButtonEvent.Type


@dataclass
class _MockParser:
  vl: dict


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


class TestCarStateExtDirectionDetection:

  def setup_method(self):
    self.ext = _make_ext()

  def test_first_frame_seed_does_not_emit_event(self):
    assert self.ext.distance_button is None
    ret, ret_sp = _call_update(self.ext, scroll=137)
    assert self.ext.distance_button == 137
    assert ret_sp.personalityDirection == 0
    assert len(ret.buttonEvents) == 0

  def test_same_counter_value_emits_nothing(self):
    _call_update(self.ext, scroll=10)
    ret, ret_sp = _call_update(self.ext, scroll=10)
    assert ret_sp.personalityDirection == 0
    assert len(ret.buttonEvents) == 0

  def test_counter_increment_produces_rivian_sign_times_positive(self):
    _call_update(self.ext, scroll=10)
    _, ret_sp = _call_update(self.ext, scroll=11)
    assert ret_sp.personalityDirection == RIVIAN_DIRECTION_SIGN * 1

  def test_counter_decrement_produces_rivian_sign_times_negative(self):
    _call_update(self.ext, scroll=10)
    _, ret_sp = _call_update(self.ext, scroll=9)
    assert ret_sp.personalityDirection == RIVIAN_DIRECTION_SIGN * -1

  def test_wrap_forward_254_to_0_is_positive_delta(self):
    # delta = ((0 - 254 + 128) % 256) - 128 = +2  (positive)
    _call_update(self.ext, scroll=254)
    _, ret_sp = _call_update(self.ext, scroll=0)
    assert ret_sp.personalityDirection == RIVIAN_DIRECTION_SIGN * 1

  def test_wrap_backward_0_to_254_is_negative_delta(self):
    # delta = ((254 - 0 + 128) % 256) - 128 = -2  (negative)
    _call_update(self.ext, scroll=0)
    _, ret_sp = _call_update(self.ext, scroll=254)
    assert ret_sp.personalityDirection == RIVIAN_DIRECTION_SIGN * -1

  def test_sentinel_255_clears_distance_button_to_none(self):
    _call_update(self.ext, scroll=10)
    ret, ret_sp = _call_update(self.ext, scroll=255)
    assert self.ext.distance_button is None
    assert ret_sp.personalityDirection == 0
    assert len(ret.buttonEvents) == 0

  def test_resume_after_sentinel_reseeds_without_event(self):
    _call_update(self.ext, scroll=10)
    _call_update(self.ext, scroll=255)
    ret, ret_sp = _call_update(self.ext, scroll=20)
    assert self.ext.distance_button == 20
    assert ret_sp.personalityDirection == 0
    assert len(ret.buttonEvents) == 0

  def test_flipping_rivian_direction_sign_inverts_published_direction(self):
    _call_update(self.ext, scroll=10)
    with patch("opendbc.sunnypilot.car.rivian.carstate_ext.RIVIAN_DIRECTION_SIGN", 1):
      _, ret_sp = _call_update(self.ext, scroll=11)
    assert ret_sp.personalityDirection == 1

  def test_scroll_change_emits_exactly_one_pressed_false_gap_adjust_cruise_event(self):
    _call_update(self.ext, scroll=10)
    ret, _ = _call_update(self.ext, scroll=11)
    events = list(ret.buttonEvents)
    assert len(events) == 1
    assert events[0].pressed is False
    assert events[0].type == ButtonType.gapAdjustCruise
