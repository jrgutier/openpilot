import numpy as np
import pytest
import itertools
from openpilot.common.parameterized import parameterized_class

from cereal import log

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_safe_obstacle_distance, get_stopped_equivalence_factor, get_T_FOLLOW, \
  JERK_FACTOR_VA_BP, JERK_FACTOR_VA_V
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver


def desired_follow_distance(v_ego, v_lead, t_follow=None):
  if t_follow is None:
    t_follow = get_T_FOLLOW()
  return get_safe_obstacle_distance(v_ego, t_follow) - get_stopped_equivalence_factor(v_lead)

def run_following_distance_simulation(v_lead, t_end=100.0, e2e=False, personality=0):
  man = Maneuver(
    '',
    duration=t_end,
    initial_speed=float(v_lead),
    lead_relevancy=True,
    initial_distance_lead=100,
    speed_lead_values=[v_lead],
    breakpoints=[0.],
    e2e=e2e,
    personality=personality,
  )
  valid, output = man.evaluate()
  assert valid
  return output[-1,2] - output[-1,1]


@parameterized_class(("e2e", "personality", "speed"), itertools.product(
                      [True, False], # e2e
                      [log.LongitudinalPersonality.relaxed, # personality
                       log.LongitudinalPersonality.standard,
                       log.LongitudinalPersonality.aggressive,
                       log.LongitudinalPersonality.veryAggressive],
                      [0,10,35])) # speed
class TestFollowingDistance:
  def test_following_distance(self):
    v_lead = float(self.speed)
    simulation_steady_state = run_following_distance_simulation(v_lead, e2e=self.e2e, personality=self.personality)
    # For veryAggressive, t_follow is speed-dependent. At steady-state all MPC horizon
    # velocities converge to v_lead, so the per-timestep t_follow collapses to a scalar.
    correct_steady_state = desired_follow_distance(v_lead, v_lead, get_T_FOLLOW(self.personality, v_ego=v_lead))
    err_ratio = 0.2 if self.e2e else 0.1
    abs_err_margin = 0.5 if v_lead > 0.0 else 1.15
    assert simulation_steady_state == pytest.approx(correct_steady_state, abs=err_ratio * correct_steady_state + abs_err_margin)


class TestVeryAggressiveJerkFactor:
  def test_jerk_factor_increases_with_speed(self):
    """Speed-dependent jerk factor must increase with speed: responsive launch, smooth approach."""
    jf_stop = float(np.interp(0.0, JERK_FACTOR_VA_BP, JERK_FACTOR_VA_V))
    jf_mid = float(np.interp(10.0, JERK_FACTOR_VA_BP, JERK_FACTOR_VA_V))
    jf_highway = float(np.interp(30.0, JERK_FACTOR_VA_BP, JERK_FACTOR_VA_V))

    assert jf_stop < jf_mid < jf_highway
    assert jf_stop < 0.5, "Low-speed jerk factor must be below aggressive (0.5) for responsive launch"
    assert jf_highway >= 0.5, "High-speed jerk factor must be at or above aggressive (0.5) for smooth approach"
