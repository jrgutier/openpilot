import math
import numpy as np
from collections import deque

from cereal import log
from opendbc.car.lateral import FRICTION_THRESHOLD, get_friction
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.pid import PIDController

from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext import LatControlTorqueExt

# At higher speeds (25+mph) we can assume:
# Lateral acceleration achieved by a specific car correlates to
# torque applied to the steering rack. It does not correlate to
# wheel slip, or to speed.

# This controller applies torque to achieve desired lateral
# accelerations. To compensate for the low speed effects the
# proportional gain is increased at low speeds by the PID controller.
# Additionally, there is friction in the steering wheel that needs
# to be overcome to move it at all, this is compensated for too.

KP = 0.8
KI = 0.15

INTERP_SPEEDS = [1, 1.5, 2.0, 3.0, 5, 7.5, 10, 15, 30]
KP_INTERP = [250, 120, 65, 30, 11.5, 5.5, 3.5, 2.0, KP]

LP_FILTER_CUTOFF_HZ = 1.2
JERK_LOOKAHEAD_SECONDS = 0.19
JERK_GAIN = 0.3
LAT_ACCEL_REQUEST_BUFFER_SECONDS = 1.0
VERSION = 1

# Crosswind damping: error LP filter, derivative-on-measurement, integrator decay
KD = 0.1
KD_INTERP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.04, 0.07, KD]
ERROR_LP_FILTER_HZ = 0.8
MEASUREMENT_LP_FILTER_HZ = 3.0
INTEGRATOR_DECAY_SPEED_BP = [10.0, 20.0, 30.0]
INTEGRATOR_DECAY_FACTOR = [1.0, 0.998, 0.995]

# UI-tunable Kp/Kd multipliers layered on top of the PID's internal KP_INTERP / KD_INTERP
# schedules to tame oscillation from the stock gains without replacing them.
KP_UI_PARAMS = ("KpLowSpeed", "KpMidSpeed", "KpHighSpeed")
KD_UI_PARAMS = ("KdLowSpeed", "KdMidSpeed", "KdHighSpeed")
UI_SPEED_BREAKPOINTS = (6.7, 15.6, 33.5)  # m/s, ~15/35/75 mph — shared by Kp and Kd
KP_UI_MIN, KP_UI_MAX = 0.1, 5.0  # matches Tuning menu slider range
KD_UI_MIN, KD_UI_MAX = 0.0, 3.0
PARAM_REFRESH_FRAMES = 300  # 3 s at 100 Hz

class LatControlTorque(LatControl):
  def __init__(self, CP, CP_SP, CI, dt):
    super().__init__(CP, CP_SP, CI, dt)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    self.pid = PIDController([INTERP_SPEEDS, KP_INTERP], KI, k_d=[INTERP_SPEEDS, KD_INTERP], rate=1/self.dt)
    self.update_limits()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    self.lat_accel_request_buffer_len = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / self.dt)
    self.lat_accel_request_buffer = deque([0.] * self.lat_accel_request_buffer_len , maxlen=self.lat_accel_request_buffer_len)
    self.lookahead_frames = int(JERK_LOOKAHEAD_SECONDS / self.dt)
    self.jerk_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), self.dt)

    # Crosswind damping filters
    self.error_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * ERROR_LP_FILTER_HZ), self.dt)
    self.measurement_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * MEASUREMENT_LP_FILTER_HZ), self.dt, initialized=False)
    self.prev_filtered_meas = None

    self._params = Params()
    self.kp_multipliers = self._load_kp_multipliers(self._params.get)
    self.kd_multipliers = self._load_kd_multipliers(self._params.get)
    self._param_update_frame = 0

    self.extension = LatControlTorqueExt(self, CP, CP_SP, CI)

  @staticmethod
  def _read_param(param_getter, key: str, lo: float, hi: float, default: float = 1.0) -> float:
    raw = param_getter(key)
    try:
      val = float(raw) if raw is not None else default
    except (TypeError, ValueError):
      val = default
    return max(lo, min(hi, val))

  @staticmethod
  def _load_multipliers(param_getter, params, lo: float, hi: float) -> list[float]:
    return [LatControlTorque._read_param(param_getter, k, lo, hi) for k in params]

  @staticmethod
  def _load_kp_multipliers(param_getter) -> list[float]:
    return LatControlTorque._load_multipliers(param_getter, KP_UI_PARAMS, KP_UI_MIN, KP_UI_MAX)

  @staticmethod
  def _load_kd_multipliers(param_getter) -> list[float]:
    return LatControlTorque._load_multipliers(param_getter, KD_UI_PARAMS, KD_UI_MIN, KD_UI_MAX)

  def reset(self):
    super().reset()
    self.error_filter.x = 0.0
    self.error_filter.initialized = False
    self.measurement_filter.x = 0.0
    self.measurement_filter.initialized = False
    self.prev_filtered_meas = None

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction
    self.update_limits()

  def update_limits(self):
    self.pid.set_limits(self.lateral_accel_from_torque(self.steer_max, self.torque_params),
                        self.lateral_accel_from_torque(-self.steer_max, self.torque_params))

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited, lat_delay):
    # Override torque params from extension
    if self.extension.update_override_torque_params(self.torque_params):
      self.update_limits()

    # Re-read Tuning menu params periodically (~6 s at 50 Hz)
    self._param_update_frame += 1
    if self._param_update_frame % PARAM_REFRESH_FRAMES == 0:
      self.kp_multipliers = self._load_kp_multipliers(self._params.get)
      self.kd_multipliers = self._load_kd_multipliers(self._params.get)

    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION
    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    measurement = measured_curvature * CS.vEgo ** 2
    future_desired_lateral_accel = desired_curvature * CS.vEgo ** 2
    self.lat_accel_request_buffer.append(future_desired_lateral_accel)

    roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
    curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
    lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

    delay_frames = int(np.clip(lat_delay / self.dt + 1, 1, self.lat_accel_request_buffer_len))
    expected_lateral_accel = self.lat_accel_request_buffer[-delay_frames]
    setpoint = expected_lateral_accel
    error = setpoint - measurement

    lookahead_idx = int(np.clip(-delay_frames + self.lookahead_frames, -self.lat_accel_request_buffer_len+1, -2))
    raw_lateral_jerk = (self.lat_accel_request_buffer[lookahead_idx+1] - self.lat_accel_request_buffer[lookahead_idx-1]) / (2 * self.dt)
    desired_lateral_jerk = self.jerk_filter.update(raw_lateral_jerk)
    gravity_adjusted_future_lateral_accel = future_desired_lateral_accel - roll_compensation
    ff = gravity_adjusted_future_lateral_accel
    # latAccelOffset corrects roll compensation bias from device roll misalignment relative to car roll
    ff -= self.torque_params.latAccelOffset
    ff += get_friction(error + JERK_GAIN * desired_lateral_jerk, lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)

    if not active:
      output_torque = 0.0
      pid_log.active = False
    else:
      filtered_error = self.error_filter.update(error)

      # Derivative-on-measurement avoids derivative kick on setpoint changes (e.g. lane change)
      filtered_meas = self.measurement_filter.update(measurement)
      if self.prev_filtered_meas is None:
        error_rate = 0.0
      else:
        error_rate = -(filtered_meas - self.prev_filtered_meas) / self.dt
      self.prev_filtered_meas = filtered_meas

      kp_working = np.interp(CS.vEgo, UI_SPEED_BREAKPOINTS, self.kp_multipliers)
      kd_working = np.interp(CS.vEgo, UI_SPEED_BREAKPOINTS, self.kd_multipliers)
      pid_log.error = float(filtered_error * kp_working)

      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 2
      output_lataccel = self.pid.update(pid_log.error, error_rate=error_rate * kd_working,
                                        speed=CS.vEgo, feedforward=ff, freeze_integrator=freeze_integrator)
      output_torque = self.torque_from_lateral_accel(output_lataccel, self.torque_params)

      if not freeze_integrator:
        decay = float(np.interp(CS.vEgo, INTEGRATOR_DECAY_SPEED_BP, INTEGRATOR_DECAY_FACTOR))
        self.pid.i *= decay

      # Lateral acceleration torque controller extension updates
      # Overrides pid_log.error and output_torque
      pid_log, output_torque = self.extension.update(CS, VM, self.pid, params, ff, pid_log, setpoint, measurement, calibrated_pose, roll_compensation,
                                                     future_desired_lateral_accel, measurement, lateral_accel_deadzone, gravity_adjusted_future_lateral_accel,
                                                     desired_curvature, measured_curvature, steer_limited_by_safety, output_torque)

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.d = float(self.pid.d)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(-output_torque) # TODO: log lat accel?
      pid_log.actualLateralAccel = float(measurement)
      pid_log.desiredLateralAccel = float(setpoint)
      pid_log.desiredLateralJerk = float(desired_lateral_jerk)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    # TODO left is positive in this convention
    return -output_torque, 0.0, pid_log
