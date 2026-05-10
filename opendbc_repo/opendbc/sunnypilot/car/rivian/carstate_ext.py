"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
from enum import StrEnum

from opendbc.car import Bus, structs
from opendbc.can.parser import CANParser
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.rivian.values import DBC
from opendbc.sunnypilot.car.rivian.values import RivianFlagsSP

ButtonType = structs.CarState.ButtonEvent.Type

MAX_SET_SPEED = 85 * CV.MPH_TO_MS
MIN_SET_SPEED = 20 * CV.MPH_TO_MS

# Counter-increment direction asserted from logs. Flip if on-vehicle test shows wrong direction.
RIVIAN_DIRECTION_SIGN: int = -1


def _append_button_event(ret: structs.CarState, *, pressed: bool, button_type: structs.CarState.ButtonEvent.Type) -> None:
  """Append a ButtonEvent to ret.buttonEvents without losing existing entries.

  ret.buttonEvents is a capnp builder list; iterating it yields readers tied to
  the current allocation. Reassigning the list invalidates those readers, so a
  naive `list(ret.buttonEvents) + [new]` corrupts the existing entries to
  default values. Copy each existing entry through a fresh detached
  ButtonEvent builder so the values survive the reassignment.
  """
  events = [structs.CarState.ButtonEvent(pressed=be.pressed, type=be.type) for be in ret.buttonEvents]
  events.append(structs.CarState.ButtonEvent(pressed=pressed, type=button_type))
  ret.buttonEvents = events


class CarStateExt:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP):
    self.CP = CP
    self.CP_SP = CP_SP

    self.set_speed = 10
    self.increase_button = False
    self.decrease_button = False
    self.distance_button: int | None = None  # None until first valid scroll seen
    self.increase_counter = 0
    self.decrease_counter = 0
    self.stalk_down_counter = 0
    self.prev_user_adas_req: int = 0

  def update_longitudinal_upgrade(self, ret: structs.CarState, ret_sp: structs.CarStateSP, can_parsers: dict[StrEnum, CANParser]) -> None:
    cp_park = can_parsers[Bus.alt]
    cp_adas = can_parsers[Bus.adas]
    cp = can_parsers[Bus.pt]

    prev_increase_button = self.increase_button
    prev_decrease_button = self.decrease_button

    if self.CP.openpilotLongitudinalControl:
      # RightButton_Scroll is a true bidirectional 8-bit rotary encoder counter
      # (mod 256). Confirmed by on-vehicle rlog decode 2026-04-29: value 255 is
      # a NORMAL mid-stream counter value, NOT an unconnected sentinel.
      right_scroll = int(cp_park.vl["WheelButtons_Fwd"]["RightButton_Scroll"])
      if self.distance_button is None:
        self.distance_button = right_scroll
      elif self.distance_button != right_scroll:
        # Signed delta with mod-256 wrap; result in [-128, 127].
        delta = ((right_scroll - self.distance_button + 128) % 256) - 128
        if delta != 0:
          sign = 1 if delta > 0 else -1
          # Encode direction inline on ButtonEvent.pressed so the consumer
          # (selfdrived.py, Rivian branch) reads it directly without a
          # cross-socket latch race against carStateSP.
          #   pressed=True  → +1 step in PERSONALITY_RANK_ORDER (more aggressive)
          #   pressed=False → -1 step (more relaxed)
          pressed = (RIVIAN_DIRECTION_SIGN * sign) > 0
          _append_button_event(ret, pressed=pressed, button_type=ButtonType.gapAdjustCruise)
        self.distance_button = right_scroll

      # button logic for set-speed
      self.increase_button = cp_park.vl["WheelButtons_Fwd"]["RightButton_RightClick"] == 2
      self.decrease_button = cp_park.vl["WheelButtons_Fwd"]["RightButton_LeftClick"] == 2

      self.increase_counter = self.increase_counter + 1 if self.increase_button else 0
      self.decrease_counter = self.decrease_counter + 1 if self.decrease_button else 0

      metric = cp_adas.vl["Cluster"]["Cluster_Unit"] == 0
      conversion = CV.KPH_TO_MS if metric else CV.MPH_TO_MS
      long_press_step = 10.0 if metric else 5.0
      set_speed_converted = self.set_speed * (CV.MS_TO_KPH if metric else CV.MS_TO_MPH)

      if self.increase_button:
        if self.increase_counter % 66 == 0:
          self.set_speed = (int(math.ceil((set_speed_converted + 1) / long_press_step)) * long_press_step) * conversion
        elif not prev_increase_button:
          self.set_speed += conversion

      if self.decrease_button:
        if self.decrease_counter % 66 == 0:
          self.set_speed = (int(math.floor((set_speed_converted - 1) / long_press_step)) * long_press_step) * conversion
        elif not prev_decrease_button:
          self.set_speed -= conversion

      if not ret.cruiseState.enabled:
        self.set_speed = ret.vEgoCluster

      # VDM_UserAdasRequest: 0=IDLE, 1=UP_1, 2=UP_2, 3=DOWN_1, 4=DOWN_2.
      # UP_2 → ButtonEvent.cancel is handled in update() unconditionally; here
      # we only consume DOWN_1/DOWN_2 for set-speed-on-first-stalk-down.
      adas_vals = list(cp.vl_all["VDM_AdasSts"]["VDM_UserAdasRequest"]) or [self.prev_user_adas_req]
      stalk_down = any(v in (3, 4) for v in adas_vals)
      self.stalk_down_counter = self.stalk_down_counter + 1 if stalk_down else 0
      if self.stalk_down_counter == 1:
        self.set_speed = max(self.set_speed, ret.vEgoCluster)

      self.set_speed = max(MIN_SET_SPEED, min(self.set_speed, MAX_SET_SPEED))
      ret.cruiseState.speed = self.set_speed

    if self.CP.enableBsm:
      ret.leftBlindspot = cp_park.vl["BSM_BlindSpotIndicator_Fwd"]["BSM_BlindSpotIndicator_Left"] != 0
      ret.rightBlindspot = cp_park.vl["BSM_BlindSpotIndicator_Fwd"]["BSM_BlindSpotIndicator_Right"] != 0

  def update(self, ret: structs.CarState, ret_sp: structs.CarStateSP, can_parsers: dict[StrEnum, CANParser]) -> None:
    # UP_2 → ButtonEvent.cancel runs unconditionally for all Rivians; stock-cruise
    # (op-long=False) users need to disengage too. VDM_UserAdasRequest is on
    # Bus.pt (always parsed). Do NOT re-gate on LONGITUDINAL_HARNESS_UPGRADE.
    cp = can_parsers[Bus.pt]
    # vl_all (not vl): UP_2 fires for one CAN frame at 50Hz; vl drops it when the
    # tick's last sample is IDLE. Empty-list fallback to prev (not cp.vl) avoids
    # spurious edges from stale vl on parser ticks with no new frames.
    vals = list(cp.vl_all["VDM_AdasSts"]["VDM_UserAdasRequest"]) or [self.prev_user_adas_req]
    if self.prev_user_adas_req != 2 and 2 in vals:
      _append_button_event(ret, pressed=True, button_type=ButtonType.cancel)
    self.prev_user_adas_req = int(vals[-1])

    # update_longitudinal_upgrade runs second and may also append to
    # ret.buttonEvents (gapAdjustCruise on scroll-wheel delta).
    if self.CP_SP.flags & RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE:
      self.update_longitudinal_upgrade(ret, ret_sp, can_parsers)

  @staticmethod
  def get_parser(CP, CP_SP) -> dict[StrEnum, CANParser]:
    messages = {}

    if CP_SP.flags & RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE:
      messages[Bus.alt] = CANParser(DBC[CP.carFingerprint][Bus.alt], [], 1)

    return messages
