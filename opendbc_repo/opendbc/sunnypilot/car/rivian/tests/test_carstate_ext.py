from dataclasses import dataclass
from unittest.mock import patch

from opendbc.car import Bus, structs
from opendbc.sunnypilot.car.rivian.carstate_ext import CarStateExt, RIVIAN_DIRECTION_SIGN
from opendbc.sunnypilot.car.rivian.values import RivianFlagsSP
from openpilot.common.test import OpenpilotTestCase

ButtonType = structs.CarState.ButtonEvent.Type


@dataclass
class _MockParser:
  vl: dict
  vl_all: dict | None = None

  def __post_init__(self):
    if self.vl_all is None:
      self.vl_all = {msg: {f: [v] for f, v in flds.items()} for msg, flds in self.vl.items()}


def _make_can_parsers(scroll: int) -> dict:
  return {
    Bus.alt: _MockParser({
      "WheelButtons_Fwd": {
        "RightButton_Scroll": scroll,
        "RightButton_RightClick": 0,
        "RightButton_LeftClick": 0,
      },
    }),
    Bus.adas: _MockParser({"Cluster": {"Cluster_Unit": 1}}),
    Bus.pt: _MockParser({"VDM_AdasSts": {"VDM_UserAdasRequest": 0}}),
  }


class TestRivianScrollWheelDirection(OpenpilotTestCase):
  """RightButton_Scroll is a mod-256 rotary counter, not a press/release button.

  Two properties are load-bearing for the personality toggle and both are asserted here:
    - Every counter change emits exactly one gapAdjustCruise event. No value is a sentinel;
      in particular 255 is an ordinary mid-stream value, and the old `!= 255` guard dropped
      one real scroll detent per wrap of the counter.
    - The scroll direction is carried on ButtonEvent.pressed, so selfdrived can step the
      personality up or down instead of running a one-way cycle.
  """

  def setUp(self):
    super().setUp()
    CP = structs.CarParams()
    CP.openpilotLongitudinalControl = True
    CP.enableBsm = False
    CP_SP = structs.CarParamsSP()
    CP_SP.flags = RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE
    self.ext = CarStateExt(CP, CP_SP)

  def _scroll(self, value: int) -> list:
    """Feed one counter value and return the gapAdjustCruise events it produced."""
    events = self.ext.update_longitudinal_upgrade(structs.CarState(), _make_can_parsers(value))
    return [be for be in events if be.type == ButtonType.gapAdjustCruise]

  @staticmethod
  def _pressed_for(delta_sign: int) -> bool:
    return (RIVIAN_DIRECTION_SIGN * delta_sign) > 0

  # --- event emission ----------------------------------------------------

  def test_first_value_seeds_without_emitting(self):
    self.assertIsNone(self.ext.distance_button)
    self.assertEqual(self._scroll(137), [])
    self.assertEqual(self.ext.distance_button, 137)

  def test_unchanged_counter_emits_nothing(self):
    self._scroll(10)
    self.assertEqual(self._scroll(10), [])

  def test_counter_change_emits_exactly_one_event(self):
    self._scroll(10)
    self.assertEqual(len(self._scroll(11)), 1)

  # --- direction encoding ------------------------------------------------

  def test_increment_and_decrement_encode_opposite_directions(self):
    self._scroll(10)
    up = self._scroll(11)[0].pressed
    down = self._scroll(10)[0].pressed
    self.assertEqual(up, self._pressed_for(+1))
    self.assertEqual(down, self._pressed_for(-1))
    self.assertNotEqual(up, down)

  def test_direction_sign_constant_inverts_encoding(self):
    """The physical polarity lives in one constant; flipping it must flip `pressed`."""
    self._scroll(10)
    with patch("opendbc.sunnypilot.car.rivian.carstate_ext.RIVIAN_DIRECTION_SIGN", 1):
      self.assertTrue(self._scroll(11)[0].pressed)
    with patch("opendbc.sunnypilot.car.rivian.carstate_ext.RIVIAN_DIRECTION_SIGN", -1):
      self.assertFalse(self._scroll(12)[0].pressed)

  # --- mod-256 wraparound (the bug this port exists to fix) --------------

  def test_255_is_a_normal_counter_value_not_a_sentinel(self):
    """Regression guard for the `if right_scroll != 255` guard.

    With that guard, 254 -> 255 emitted nothing and left distance_button at 254, so the
    driver's detent was silently swallowed. Both assertions fail if it is reintroduced.
    """
    self._scroll(254)
    events = self._scroll(255)
    self.assertEqual(len(events), 1)
    self.assertEqual(events[0].pressed, self._pressed_for(+1))
    self.assertEqual(self.ext.distance_button, 255)

  def test_scrolling_off_255_emits_a_single_step_not_a_jump(self):
    """255 -> 254 is one detent backwards; the `!= 255` guard dropped it entirely."""
    self._scroll(255)
    events = self._scroll(254)
    self.assertEqual(len(events), 1)
    self.assertEqual(events[0].pressed, self._pressed_for(-1))

  def test_wrap_255_to_0_is_one_step_forward(self):
    """delta = ((0 - 255 + 128) % 256) - 128 = +1, not -255."""
    self._scroll(255)
    events = self._scroll(0)
    self.assertEqual(len(events), 1)
    self.assertEqual(events[0].pressed, self._pressed_for(+1))

  def test_wrap_0_to_255_is_one_step_backward(self):
    """delta = ((255 - 0 + 128) % 256) - 128 = -1, not +255."""
    self._scroll(0)
    events = self._scroll(255)
    self.assertEqual(len(events), 1)
    self.assertEqual(events[0].pressed, self._pressed_for(-1))

  def test_continuous_scroll_across_the_wrap_keeps_one_direction(self):
    """A real scroll burst crosses 255 -> 0. Every detent must report the same direction."""
    self._scroll(252)
    directions = set()
    for value in (253, 254, 255, 0, 1, 2):
      events = self._scroll(value)
      self.assertEqual(len(events), 1, f"no event emitted at counter value {value}")
      directions.add(events[0].pressed)
    self.assertEqual(directions, {self._pressed_for(+1)})

  def test_continuous_backward_scroll_across_the_wrap_keeps_one_direction(self):
    self._scroll(2)
    directions = set()
    for value in (1, 0, 255, 254, 253):
      events = self._scroll(value)
      self.assertEqual(len(events), 1, f"no event emitted at counter value {value}")
      directions.add(events[0].pressed)
    self.assertEqual(directions, {self._pressed_for(-1)})

  def test_every_counter_value_emits_an_event(self):
    """No value in 0..255 may be treated as "no event"."""
    self._scroll(0)
    for value in range(1, 256):
      self.assertEqual(len(self._scroll(value)), 1, f"counter value {value} was dropped")
