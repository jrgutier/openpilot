"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.sunnypilot.lib.styles import style
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, multiple_button_item_sp, option_item_sp


class CruiseLayout(Widget):
  def __init__(self):
    super().__init__()

    self._params = Params()
    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    items = [
      # Dynamic Experimental Control
      toggle_item_sp(
        lambda: tr("Dynamic Experimental Control"),
        lambda: tr("Automatically switch between ACC and E2E longitudinal mode based on driving conditions."),
        param="DynamicExperimentalControl"
      ),

      # Intelligent Cruise Button Management
      toggle_item_sp(
        lambda: tr("Intelligent Cruise Button Management"),
        lambda: tr("Allow sunnypilot to manage cruise button presses for smoother speed control."),
        param="IntelligentCruiseButtonManagement"
      ),

      # Speed Limit Assist Mode
      multiple_button_item_sp(
        lambda: tr("Speed Limit Assist Mode"),
        lambda: tr("Off: Disabled. Information: Display only. Warning: Alert when exceeding. "
                   "Assist: Automatically adjust speed to match limit."),
        buttons=[lambda: tr("Off"), lambda: tr("Info"), lambda: tr("Warn"), lambda: tr("Assist")],
        button_width=200,
        param="SpeedLimitMode",
        inline=False
      ),

      # Speed Limit Source
      multiple_button_item_sp(
        lambda: tr("Speed Limit Source"),
        lambda: tr("Select the source for speed limit data: Car State (from vehicle), Map Data (from OSM), "
                   "or priority combinations."),
        buttons=[lambda: tr("Car"), lambda: tr("Map"), lambda: tr("Car 1st"), lambda: tr("Map 1st"), lambda: tr("Combined")],
        button_width=160,
        param="SpeedLimitPolicy",
        inline=False
      ),

      # Speed Limit Offset Type
      multiple_button_item_sp(
        lambda: tr("Speed Limit Offset Type"),
        lambda: tr("How to apply the speed limit offset: Off (no offset), Fixed (constant offset), "
                   "or Percentage (proportional offset)."),
        buttons=[lambda: tr("Off"), lambda: tr("Fixed"), lambda: tr("Percent")],
        button_width=200,
        param="SpeedLimitOffsetType",
        inline=False
      ),

      # Speed Limit Offset Value
      option_item_sp(
        lambda: tr("Speed Limit Offset Value"),
        "SpeedLimitValueOffset",
        -30, 30,
        lambda: tr("Offset to add or subtract from the speed limit. Positive values increase, negative decrease."),
        value_change_step=1,
        label_width=style.BUTTON_ACTION_WIDTH,
        label_callback=lambda v: f"{'+' if v > 0 else ''}{v} {'km/h' if ui_state.is_metric else 'mph'}",
        inline=True
      ),

      # Smart Cruise Control - Map
      toggle_item_sp(
        lambda: tr("Smart Cruise Control: Map"),
        lambda: tr("Use map data to anticipate curves and slow down proactively."),
        param="SmartCruiseControlMap"
      ),

      # Smart Cruise Control - Vision
      toggle_item_sp(
        lambda: tr("Smart Cruise Control: Vision"),
        lambda: tr("Use vision model to anticipate curves and slow down proactively."),
        param="SmartCruiseControlVision"
      ),

      # Custom ACC Increments
      toggle_item_sp(
        lambda: tr("Custom ACC Increments"),
        lambda: tr("Customize the speed change increments for cruise control button presses."),
        param="CustomAccIncrementsEnabled"
      ),

      # Custom ACC Short Press Increment
      option_item_sp(
        lambda: tr("Short Press Increment"),
        "CustomAccShortPressIncrement",
        1, 10,
        lambda: tr("Speed change for a short press of the cruise control button."),
        value_change_step=1,
        label_width=style.BUTTON_ACTION_WIDTH,
        label_callback=lambda v: f"{v} {'km/h' if ui_state.is_metric else 'mph'}",
        inline=True
      ),

      # Custom ACC Long Press Increment
      option_item_sp(
        lambda: tr("Long Press Increment"),
        "CustomAccLongPressIncrement",
        1, 10,
        lambda: tr("Speed change for a long press of the cruise control button."),
        value_change_step=1,
        label_width=style.BUTTON_ACTION_WIDTH,
        label_callback=lambda v: f"{v} {'km/h' if ui_state.is_metric else 'mph'}",
        inline=True
      ),
    ]
    return items

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    self._scroller.show_event()
