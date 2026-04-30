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


class CarStateExt:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP):
    self.CP = CP
    self.CP_SP = CP_SP

    self.set_speed = 10
    self.increase_button = False
    self.decrease_button = False
    self.distance_button: int | None = None  # None until first valid scroll seen, also reset on 255 sentinel
    self.increase_counter = 0
    self.decrease_counter = 0
    self.stalk_down_counter = 0
    self.up2_counter: int = 0
    self.up2_edge_armed: bool = True

  def update_longitudinal_upgrade(self, ret: structs.CarState, ret_sp: structs.CarStateSP, can_parsers: dict[StrEnum, CANParser]) -> None:
    cp_park = can_parsers[Bus.alt]
    cp_adas = can_parsers[Bus.adas]
    cp = can_parsers[Bus.pt]

    prev_increase_button = self.increase_button
    prev_decrease_button = self.decrease_button

    if self.CP.openpilotLongitudinalControl:
      personality_direction = 0
      right_scroll = int(cp_park.vl["WheelButtons_Fwd"]["RightButton_Scroll"])
      if right_scroll == 255:
        self.distance_button = None
      elif self.distance_button is None:
        self.distance_button = right_scroll
      elif self.distance_button != right_scroll:
        # Signed delta with mod-256 wrap; result in [-128, 127].
        delta = ((right_scroll - self.distance_button + 128) % 256) - 128
        if delta != 0:
          sign = 1 if delta > 0 else -1
          personality_direction = RIVIAN_DIRECTION_SIGN * sign
          ret.buttonEvents = [structs.CarState.ButtonEvent(pressed=False, type=ButtonType.gapAdjustCruise)]
        self.distance_button = right_scroll

      # Publish every frame for replay correctness (default 0).
      ret_sp.personalityDirection = personality_direction

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
      # UP_2 → madsDisableRequest is handled in update() unconditionally; here
      # we only consume DOWN_1/DOWN_2 for set-speed-on-first-stalk-down.
      user_adas_req = int(cp.vl["VDM_AdasSts"]["VDM_UserAdasRequest"])
      stalk_down = user_adas_req in (3, 4)
      self.stalk_down_counter = self.stalk_down_counter + 1 if stalk_down else 0
      if self.stalk_down_counter == 1:
        self.set_speed = max(self.set_speed, ret.vEgoCluster)

      self.set_speed = max(MIN_SET_SPEED, min(self.set_speed, MAX_SET_SPEED))
      ret.cruiseState.speed = self.set_speed

    if self.CP.enableBsm:
      ret.leftBlindspot = cp_park.vl["BSM_BlindSpotIndicator_Fwd"]["BSM_BlindSpotIndicator_Left"] != 0
      ret.rightBlindspot = cp_park.vl["BSM_BlindSpotIndicator_Fwd"]["BSM_BlindSpotIndicator_Right"] != 0

  def update(self, ret: structs.CarState, ret_sp: structs.CarStateSP, can_parsers: dict[StrEnum, CANParser]) -> None:
    # UP_2 → madsDisableRequest. Runs unconditionally for all Rivians:
    # - VDM_UserAdasRequest is parsed off Bus.pt by base CarState, always available.
    # - LONGITUDINAL_HARNESS_UPGRADE is a hardware-tap flag (interface.py, set on
    #   fingerprint 0x131a), not a feature flag — no semantic relationship to a
    #   stalk-position read on Bus.pt.
    # - Commit 26fbf812d2: "UP_2 stalk position disables MADS while cruise stays
    #   engaged" — by design this fires for stock-cruise (op-long=False) users.
    # Do NOT re-gate behind openpilotLongitudinalControl or LONGITUDINAL_HARNESS_UPGRADE.
    cp = can_parsers[Bus.pt]
    user_adas_req = int(cp.vl["VDM_AdasSts"]["VDM_UserAdasRequest"])
    if user_adas_req == 2:
      self.up2_counter += 1
      if self.up2_counter >= 2 and self.up2_edge_armed:
        ret_sp.madsDisableRequest = True
        self.up2_edge_armed = False
    else:
      self.up2_counter = 0
      self.up2_edge_armed = True

    if self.CP_SP.flags & RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE:
      self.update_longitudinal_upgrade(ret, ret_sp, can_parsers)

  @staticmethod
  def get_parser(CP, CP_SP) -> dict[StrEnum, CANParser]:
    messages = {}

    if CP_SP.flags & RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE:
      messages[Bus.alt] = CANParser(DBC[CP.carFingerprint][Bus.alt], [], 1)

    return messages
