from cereal import log
from openpilot.selfdrive.selfdrived.selfdrived import (
  PERSONALITY_RANK_ORDER,
  PERSONALITY_TO_RANK,
  _step_personality_ranked,
)

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

  def test_direction_gated_on_signal_not_on_car_name(self):
    assert _step_personality_ranked(_RELAXED, direction=+1) == _STANDARD


class TestLatchMechanism:

  @staticmethod
  def _data_sample_latch(prior_latch: int, direction: int) -> int:
    if direction != 0:
      return direction
    return prior_latch

  @staticmethod
  def _gap_button_consume(latch: int) -> tuple:
    return latch, 0

  def test_non_zero_direction_is_latched(self):
    latch = self._data_sample_latch(prior_latch=0, direction=-1)
    assert latch == -1

  def test_zero_direction_preserves_existing_latch(self):
    latch = self._data_sample_latch(prior_latch=-1, direction=0)
    assert latch == -1

  def test_latch_survives_one_frame_skew(self):
    latch = self._data_sample_latch(prior_latch=0, direction=-1)
    latch = self._data_sample_latch(prior_latch=latch, direction=0)
    assert latch == -1
    consumed, latch = self._gap_button_consume(latch)
    assert consumed == -1

  def test_latch_is_cleared_after_consume(self):
    latch = self._data_sample_latch(prior_latch=0, direction=-1)
    _, latch = self._gap_button_consume(latch)
    assert latch == 0

  def test_second_button_press_without_new_direction_falls_back_to_legacy_cycle(self):
    latch = self._data_sample_latch(prior_latch=0, direction=-1)
    _, latch = self._gap_button_consume(latch)
    assert latch == 0
    latch = self._data_sample_latch(prior_latch=latch, direction=0)
    consumed, _ = self._gap_button_consume(latch)
    assert consumed == 0
