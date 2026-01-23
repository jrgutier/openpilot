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
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, option_item_sp


class DisplayLayout(Widget):
  def __init__(self):
    super().__init__()

    self._params = Params()
    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    items = [
      # Screen Brightness
      option_item_sp(
        lambda: tr("Screen Brightness"),
        "Brightness",
        0, 100,
        lambda: tr("Adjust the screen brightness. 0 is automatic brightness."),
        value_change_step=5,
        label_width=style.BUTTON_ACTION_WIDTH,
        label_callback=lambda v: tr("Auto") if v == 0 else f"{v}%",
        inline=True
      ),

      # Onroad Screen Off Control
      toggle_item_sp(
        lambda: tr("Reduce Onroad Brightness"),
        lambda: tr("Reduce screen brightness or turn off the display after driving starts to save power."),
        param="OnroadScreenOffControl"
      ),

      # Onroad Screen Off Brightness
      option_item_sp(
        lambda: tr("Onroad Brightness"),
        "OnroadScreenOffBrightness",
        0, 100,
        lambda: tr("Screen brightness level when driving. 0 turns the screen off completely."),
        value_change_step=5,
        label_width=style.BUTTON_ACTION_WIDTH,
        label_callback=lambda v: tr("Off") if v == 0 else f"{v}%",
        inline=True
      ),

      # Onroad Screen Off Timer
      option_item_sp(
        lambda: tr("Brightness Reduction Delay"),
        "OnroadScreenOffTimer",
        0, 60,
        lambda: tr("Time to wait after driving starts before reducing screen brightness."),
        value_change_step=5,
        label_width=style.BUTTON_ACTION_WIDTH,
        label_callback=lambda v: f"{v}s",
        inline=True
      ),

      # Interactivity Timeout
      option_item_sp(
        lambda: tr("Screen Timeout"),
        "InteractivityTimeout",
        0, 120,
        lambda: tr("Time before the screen returns to reduced brightness after touch interaction. "
                   "0 disables the timeout."),
        value_change_step=5,
        label_width=style.BUTTON_ACTION_WIDTH,
        label_callback=lambda v: tr("Off") if v == 0 else f"{v}s",
        inline=True
      ),
    ]
    return items

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    self._scroller.show_event()
