"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.common.params import Params
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.sunnypilot.lib.styles import style
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, multiple_button_item_sp, option_item_sp


class VisualsLayout(Widget):
  def __init__(self):
    super().__init__()

    self._params = Params()
    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    items = [
      # Developer UI Info
      multiple_button_item_sp(
        lambda: tr("Developer UI"),
        lambda: tr("Show developer information on the driving screen. Higher levels show more data."),
        buttons=[lambda: tr("Off"), "1", "2"],
        button_width=200,
        param="DevUIInfo",
        inline=False
      ),

      # Chevron Info
      option_item_sp(
        lambda: tr("Lead Car Chevron Info"),
        "ChevronInfo",
        0, 5,
        lambda: tr("Select what information to display on the lead car chevron. "
                   "0=Off, 1-5 show different data combinations."),
        value_change_step=1,
        label_width=style.BUTTON_ACTION_WIDTH,
        label_callback=lambda v: tr("Off") if v == 0 else str(v),
        inline=True
      ),

      # Rainbow Mode
      toggle_item_sp(
        lambda: tr("Rainbow Path"),
        lambda: tr("Display a rainbow-colored path instead of the standard path visualization."),
        param="RainbowMode"
      ),

      # Show Turn Signals
      toggle_item_sp(
        lambda: tr("Show Turn Signals"),
        lambda: tr("Display turn signal indicators on the driving screen."),
        param="ShowTurnSignals"
      ),

      # Standstill Timer
      toggle_item_sp(
        lambda: tr("Standstill Timer"),
        lambda: tr("Display a timer showing how long the vehicle has been stopped."),
        param="StandstillTimer"
      ),

      # Blind Spot
      toggle_item_sp(
        lambda: tr("Blind Spot Visualization"),
        lambda: tr("Highlight blind spot areas on the driving visualization when a vehicle is detected."),
        param="BlindSpot"
      ),

      # Green Light Alert
      toggle_item_sp(
        lambda: tr("Green Light Alert"),
        lambda: tr("Get an alert when a red light changes to green while stopped."),
        param="GreenLightAlert"
      ),

      # Lead Depart Alert
      toggle_item_sp(
        lambda: tr("Lead Departure Alert"),
        lambda: tr("Get an alert when the lead vehicle starts moving after being stopped."),
        param="LeadDepartAlert"
      ),

      # Hide vEgo UI
      toggle_item_sp(
        lambda: tr("Hide Speed Display"),
        lambda: tr("Hide the current speed display on the driving screen."),
        param="HideVEgoUI"
      ),

      # True vEgo UI
      toggle_item_sp(
        lambda: tr("Show True Speed"),
        lambda: tr("Display the true vehicle speed from wheel sensors instead of GPS speed."),
        param="TrueVEgoUI"
      ),
    ]
    return items

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    self._scroller.show_event()
