"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Very Aggressive is a REPLACEMENT for the Aggressive personality, not a modifier layered
over whichever personality happens to be selected.

This distinction is the whole content of the gate, and it is safety-relevant in one
direction specifically: without it, a driver who cycles the distance stalk to Relaxed --
explicitly asking for MORE space -- still gets a 0.8s follow, the tightest the fork can
command, while the UI, the personalityChanged alert and the logged `personality` all
continue to read Relaxed. Displayed state and actual behaviour disagree, which is the
exact failure the fork field was introduced to avoid.

These are pure-function tests on purpose. The replay corpus cannot defend this gate: the
harness migrates legacy Very Aggressive frames to personality=aggressive, so it only ever
exercises the branch where the gate PASSES. test_following_distance.py cannot defend it
either -- it never sets the param, so all its cases run with the feature off. Delete the
gate and both of those still go green. This file is what actually fails.
"""
import itertools

from openpilot.cereal import log
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import (
  LongitudinalPlannerSP,
  VERY_AGGRESSIVE_JERK_FACTOR,
  VERY_AGGRESSIVE_T_FOLLOW,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_jerk_factor, get_T_FOLLOW

Personality = log.LongitudinalPersonality
NON_AGGRESSIVE = [Personality.standard, Personality.relaxed]
ALL_PERSONALITIES = [Personality.aggressive, *NON_AGGRESSIVE]


def _planner(toggle_on: bool) -> LongitudinalPlannerSP:
  """A planner with only the attributes the Very Aggressive gate touches."""
  planner = object.__new__(LongitudinalPlannerSP)
  planner.very_aggressive = toggle_on
  planner.very_aggressive_active = False
  return planner


def _resolve(toggle_on: bool, personality) -> tuple:
  """(published_flag, effective_jerk_factor, effective_t_follow) for one frame."""
  planner = _planner(toggle_on)
  planner.update_very_aggressive_active(personality)
  jerk_override, t_follow_override = planner.very_aggressive_overrides()
  return (planner.very_aggressive_active,
          get_jerk_factor(personality, jerk_override),
          get_T_FOLLOW(personality, t_follow_override))


class TestVeryAggressiveIsAReplacementNotAModifier(OpenpilotTestCase):
  @parameterized.expand(NON_AGGRESSIVE, names=["personality"])
  def test_toggle_on_does_not_touch_other_personalities(self, personality):
    """The regression this gate exists for: Relaxed must stay Relaxed."""
    active, jerk, t_follow = _resolve(True, personality)
    _, stock_jerk, stock_t_follow = _resolve(False, personality)
    assert (jerk, t_follow) == (stock_jerk, stock_t_follow)
    assert not active

  def test_toggle_on_replaces_aggressive(self):
    active, jerk, t_follow = _resolve(True, Personality.aggressive)
    assert (jerk, t_follow) == (VERY_AGGRESSIVE_JERK_FACTOR, VERY_AGGRESSIVE_T_FOLLOW)
    assert active

  @parameterized.expand(ALL_PERSONALITIES, names=["personality"])
  def test_toggle_off_is_a_no_op(self, personality):
    planner = _planner(False)
    planner.update_very_aggressive_active(personality)
    assert planner.very_aggressive_overrides() == (None, None)
    assert not planner.very_aggressive_active

  @parameterized.expand(NON_AGGRESSIVE, names=["personality"])
  def test_toggle_on_never_shortens_follow_distance(self, personality):
    """Direction check: gating may only ever lengthen the follow, never shorten it."""
    _, _, gated = _resolve(True, personality)
    assert gated > VERY_AGGRESSIVE_T_FOLLOW


class TestPublishedFlagMatchesAppliedTuning(OpenpilotTestCase):
  @parameterized.expand(itertools.product([True, False], ALL_PERSONALITIES), names=["toggle_on", "personality"])
  def test_flag_is_true_iff_tuning_applied(self, toggle_on, personality):
    """longitudinalPlanSP.veryAggressive must mean 'this frame used the VA tuning'.

    Replay reads this field as the oracle for round-trip criterion (c); if it could report
    True while the MPC ran stock tuning, that criterion would be circular.
    """
    active, jerk, t_follow = _resolve(toggle_on, personality)
    applied = (jerk, t_follow) == (VERY_AGGRESSIVE_JERK_FACTOR, VERY_AGGRESSIVE_T_FOLLOW)
    assert active == applied

  def test_flag_is_not_the_raw_toggle(self):
    """Guards the specific defect: publishing self.very_aggressive instead of the applied state."""
    active, _, _ = _resolve(True, Personality.relaxed)
    assert not active
