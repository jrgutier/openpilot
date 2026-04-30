from cereal import log
from opendbc.car import structs
from openpilot.selfdrive.selfdrived.selfdrived import (
  PERSONALITY_RANK_ORDER,
  PERSONALITY_TO_RANK,
  _step_personality_ranked,
)

ButtonType = structs.CarState.ButtonEvent.Type

_AGGRESSIVE = log.LongitudinalPersonality.schema.enumerants["aggressive"]
_STANDARD   = log.LongitudinalPersonality.schema.enumerants["standard"]
_RELAXED    = log.LongitudinalPersonality.schema.enumerants["relaxed"]
_VERY_AGG   = log.LongitudinalPersonality.schema.enumerants["veryAggressive"]


class TestRankTableIntegrity:

  def test_rank_table_covers_all_four_personality_values_exactly_once(self):
    all_vals = sorted(log.LongitudinalPersonality.schema.enumerants.values())
    assert sorted(PERSONALITY_RANK_ORDER) == all_vals

  def test_rank_table_length_is_four(self):
    assert len(PERSONALITY_RANK_ORDER) == 4

  def test_personality_to_rank_is_exact_inverse_of_rank_order(self):
    for rank, p in enumerate(PERSONALITY_RANK_ORDER):
      assert PERSONALITY_TO_RANK[p] == rank


class TestStepPersonalityRanked:

  def test_step_toward_less_aggressive_clamps_at_relaxed(self):
    assert _step_personality_ranked(_RELAXED, direction=-1) == _RELAXED

  def test_step_toward_more_aggressive_clamps_at_very_aggressive(self):
    assert _step_personality_ranked(_VERY_AGG, direction=+1) == _VERY_AGG

  def test_step_through_full_range_toward_less_aggressive(self):
    p = _VERY_AGG
    changes = 0
    for _ in range(4):
      new_p = _step_personality_ranked(p, direction=-1)
      if new_p != p:
        changes += 1
      p = new_p
    assert changes == 3
    assert p == _RELAXED

  def test_step_through_full_range_toward_more_aggressive(self):
    p = _RELAXED
    changes = 0
    for _ in range(4):
      new_p = _step_personality_ranked(p, direction=+1)
      if new_p != p:
        changes += 1
      p = new_p
    assert changes == 3
    assert p == _VERY_AGG

  def test_single_step_toward_more_aggressive_from_standard(self):
    assert _step_personality_ranked(_STANDARD, direction=+1) == _AGGRESSIVE

  def test_single_step_toward_less_aggressive_from_aggressive(self):
    assert _step_personality_ranked(_AGGRESSIVE, direction=-1) == _STANDARD


class TestRivianBrandAwareConsumer:
  """Pure logic test of the brand-aware personality-stepping decision in
  selfdrived.py update_events. Mirrors the production block:
    - Rivian: pressed=True → +1 step, pressed=False → -1 step (direct from event).
    - Other brands: only react to release event (pressed=False); legacy (p-1) % 4 cycle.
  """

  @staticmethod
  def _step_rivian(personality: int, button_events: list) -> int:
    """Replicates the Rivian branch from selfdrived.py update_events."""
    for be in button_events:
      if be.type != ButtonType.gapAdjustCruise:
        continue
      direction = +1 if be.pressed else -1
      personality = _step_personality_ranked(personality, direction)
    return personality

  @staticmethod
  def _step_legacy(personality: int, button_events: list) -> int:
    """Replicates the non-Rivian branch from selfdrived.py update_events."""
    if any(not be.pressed and be.type == ButtonType.gapAdjustCruise for be in button_events):
      return (personality - 1) % 4
    return personality

  @staticmethod
  def _gap(pressed: bool) -> structs.CarState.ButtonEvent:
    return structs.CarState.ButtonEvent(pressed=pressed, type=ButtonType.gapAdjustCruise)

  # --- Rivian branch -----------------------------------------------------

  def test_rivian_pressed_true_steps_toward_more_aggressive(self):
    assert self._step_rivian(_STANDARD, [self._gap(True)]) == _AGGRESSIVE

  def test_rivian_pressed_false_steps_toward_more_relaxed(self):
    assert self._step_rivian(_AGGRESSIVE, [self._gap(False)]) == _STANDARD

  def test_rivian_does_not_fall_back_to_legacy_on_pressed_true(self):
    """A burst of pressed=True events must NOT cycle through all 4 personalities
    in legacy order (the bug the user reported)."""
    p = _STANDARD
    for _ in range(10):
      p = self._step_rivian(p, [self._gap(True)])
    assert p == _VERY_AGG  # clamps, doesn't cycle past

  def test_rivian_does_not_fall_back_to_legacy_on_pressed_false(self):
    p = _AGGRESSIVE
    for _ in range(10):
      p = self._step_rivian(p, [self._gap(False)])
    assert p == _RELAXED  # clamps, doesn't cycle past

  def test_rivian_alternating_pressed_oscillates_one_step(self):
    p = _STANDARD
    p = self._step_rivian(p, [self._gap(True)])   # → aggressive
    p = self._step_rivian(p, [self._gap(False)])  # → standard
    p = self._step_rivian(p, [self._gap(True)])   # → aggressive
    assert p == _AGGRESSIVE

  # --- Non-Rivian regression guard ---------------------------------------

  def test_non_rivian_legacy_cycle_unchanged_on_release(self):
    assert self._step_legacy(_AGGRESSIVE, [self._gap(False)]) == (_AGGRESSIVE - 1) % 4

  def test_non_rivian_legacy_filters_out_pressed_true(self):
    """Non-Rivian brands must NOT react to pressed=True (preserves Rivian-only
    direction encoding without affecting other brands)."""
    assert self._step_legacy(_AGGRESSIVE, [self._gap(True)]) == _AGGRESSIVE

  def test_non_rivian_legacy_full_cycle_through_all_four(self):
    p = 0
    seen = {p}
    for _ in range(8):
      p = self._step_legacy(p, [self._gap(False)])
      seen.add(p)
    assert seen == {0, 1, 2, 3}
