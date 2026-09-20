from opendbc.car import structs
from openpilot.cereal import log
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.selfdrived.selfdrived import (
  PERSONALITY_RANK_ORDER,
  PERSONALITY_TO_RANK,
  SelfdriveD,
  _step_personality_ranked,
)

ButtonType = structs.CarState.ButtonEvent.Type

AGGRESSIVE = log.LongitudinalPersonality.schema.enumerants["aggressive"]
STANDARD = log.LongitudinalPersonality.schema.enumerants["standard"]
RELAXED = log.LongitudinalPersonality.schema.enumerants["relaxed"]


def gap(pressed: bool):
  return structs.CarState.ButtonEvent(pressed=pressed, type=ButtonType.gapAdjustCruise)


class _Consumer:
  """The real SelfdriveD personality methods over a minimal stand-in state.

  SelfdriveD.__init__ needs the whole onroad service graph, so the two methods under test
  are borrowed onto this object instead of mocked out -- a mirror implementation here could
  not fail when the production branch regresses.
  """

  _set_personality = SelfdriveD._set_personality
  _update_personality = SelfdriveD._update_personality

  def __init__(self, brand: str, personality: int = STANDARD, openpilot_long: bool = True):
    self.CP = structs.CarParams()
    self.CP.brand = brand
    self.CP.openpilotLongitudinalControl = openpilot_long
    self.personality = personality
    self.experimental_mode_switched = False
    self.written = []
    self.events_added = []
    self.params = self
    self.events = self

  def put(self, key, value):
    self.written.append((key, value))

  def add(self, event):
    self.events_added.append(event)

  def scroll(self, *events):
    self._update_personality(structs.CarState(buttonEvents=list(events)))
    return self.personality


class TestPersonalityRankOrder(OpenpilotTestCase):
  """The capnp enum is not ordered by aggressiveness, so the rank table is what makes
  "one notch more aggressive" mean anything. If it drifts out of sync with the enum the
  scroll wheel silently steps to the wrong personality."""

  def test_rank_order_covers_every_personality_exactly_once(self):
    self.assertEqual(sorted(PERSONALITY_RANK_ORDER), sorted(log.LongitudinalPersonality.schema.enumerants.values()))

  def test_rank_order_has_three_steps(self):
    """Very Aggressive is a separate settings toggle on this branch, not a fourth step."""
    self.assertEqual(len(PERSONALITY_RANK_ORDER), 3)

  def test_rank_order_runs_relaxed_to_aggressive(self):
    self.assertEqual(PERSONALITY_RANK_ORDER, [RELAXED, STANDARD, AGGRESSIVE])

  def test_rank_lookup_is_the_exact_inverse_of_rank_order(self):
    for rank, personality in enumerate(PERSONALITY_RANK_ORDER):
      self.assertEqual(PERSONALITY_TO_RANK[personality], rank)


class TestStepPersonalityRanked(OpenpilotTestCase):
  def test_step_up_moves_one_notch_toward_aggressive(self):
    self.assertEqual(_step_personality_ranked(RELAXED, +1), STANDARD)
    self.assertEqual(_step_personality_ranked(STANDARD, +1), AGGRESSIVE)

  def test_step_down_moves_one_notch_toward_relaxed(self):
    self.assertEqual(_step_personality_ranked(AGGRESSIVE, -1), STANDARD)
    self.assertEqual(_step_personality_ranked(STANDARD, -1), RELAXED)

  def test_stepping_clamps_instead_of_wrapping(self):
    """A rotary wheel scrolled past the end must hold, not jump to the opposite extreme:
    wrapping would hand the driver Aggressive when they asked for more space."""
    self.assertEqual(_step_personality_ranked(AGGRESSIVE, +1), AGGRESSIVE)
    self.assertEqual(_step_personality_ranked(RELAXED, -1), RELAXED)

  def test_repeated_stepping_settles_at_the_endpoint(self):
    personality = RELAXED
    for _ in range(10):
      personality = _step_personality_ranked(personality, +1)
    self.assertEqual(personality, AGGRESSIVE)
    for _ in range(10):
      personality = _step_personality_ranked(personality, -1)
    self.assertEqual(personality, RELAXED)


class TestRivianDirectionConsumer(OpenpilotTestCase):
  def test_scroll_burst_in_one_direction_does_not_cycle(self):
    """The reported bug: one physical scroll emits several events, and the legacy
    (p - 1) % 3 cycle walked the driver through every personality regardless of direction."""
    consumer = _Consumer('rivian', STANDARD)
    for _ in range(10):
      consumer.scroll(gap(True))
    self.assertEqual(consumer.personality, AGGRESSIVE)

    for _ in range(10):
      consumer.scroll(gap(False))
    self.assertEqual(consumer.personality, RELAXED)

  def test_reversing_the_scroll_reverses_the_personality(self):
    consumer = _Consumer('rivian', STANDARD)
    self.assertEqual(consumer.scroll(gap(True)), AGGRESSIVE)
    self.assertEqual(consumer.scroll(gap(False)), STANDARD)
    self.assertEqual(consumer.scroll(gap(False)), RELAXED)

  def test_a_batch_of_events_steps_once_per_event(self):
    consumer = _Consumer('rivian', RELAXED)
    self.assertEqual(consumer.scroll(gap(True), gap(True)), AGGRESSIVE)

  def test_other_button_types_are_ignored(self):
    consumer = _Consumer('rivian', STANDARD)
    other = structs.CarState.ButtonEvent(pressed=True, type=ButtonType.accelCruise)
    self.assertEqual(consumer.scroll(other), STANDARD)

  def test_a_real_change_is_persisted_and_alerted_once(self):
    consumer = _Consumer('rivian', STANDARD)
    consumer.scroll(gap(True))
    self.assertEqual(consumer.written, [('LongitudinalPersonality', AGGRESSIVE)])
    self.assertEqual(len(consumer.events_added), 1)

  def test_scrolling_into_the_clamp_does_not_rewrite_the_param(self):
    """Holding the wheel at an endpoint must not spam the param store or the alert."""
    consumer = _Consumer('rivian', AGGRESSIVE)
    for _ in range(5):
      consumer.scroll(gap(True))
    self.assertEqual(consumer.written, [])
    self.assertEqual(consumer.events_added, [])

  def test_stock_longitudinal_ignores_the_wheel(self):
    consumer = _Consumer('rivian', STANDARD, openpilot_long=False)
    self.assertEqual(consumer.scroll(gap(True)), STANDARD)


class TestLegacyConsumerUnchanged(OpenpilotTestCase):
  """Non-Rivian brands have a real press/release button and keep the one-way cycle."""

  def test_release_edge_advances_the_legacy_cycle(self):
    consumer = _Consumer('toyota', AGGRESSIVE)
    self.assertEqual(consumer.scroll(gap(False)), (AGGRESSIVE - 1) % 3)

  def test_press_edge_is_ignored(self):
    consumer = _Consumer('toyota', AGGRESSIVE)
    self.assertEqual(consumer.scroll(gap(True)), AGGRESSIVE)

  def test_legacy_cycle_visits_all_three_personalities(self):
    consumer = _Consumer('toyota', AGGRESSIVE)
    seen = {AGGRESSIVE}
    for _ in range(6):
      seen.add(consumer.scroll(gap(False)))
    self.assertEqual(seen, {AGGRESSIVE, STANDARD, RELAXED})

  def test_experimental_mode_long_press_consumes_one_release(self):
    """The long-press that toggles experimental mode must not also change personality."""
    consumer = _Consumer('toyota', AGGRESSIVE)
    consumer.experimental_mode_switched = True
    self.assertEqual(consumer.scroll(gap(False)), AGGRESSIVE)
    self.assertFalse(consumer.experimental_mode_switched)
    # Next release is a normal one notch of the legacy cycle: aggressive -> (0 - 1) % 3 -> relaxed.
    self.assertEqual(consumer.scroll(gap(False)), RELAXED)
