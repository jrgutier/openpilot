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


class SteeringLayout(Widget):
  def __init__(self):
    super().__init__()

    self._params = Params()
    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    items = [
      # Neural Network Lateral Control
      toggle_item_sp(
        lambda: tr("Neural Network Lateral Control"),
        lambda: tr("Use a neural network model for lateral control instead of the stock torque controller."),
        param="NeuralNetworkLateralControl"
      ),

      # Auto Lane Change Timer
      multiple_button_item_sp(
        lambda: tr("Auto Lane Change Timer"),
        lambda: tr("Set the delay for automatic lane changes. Nudge requires steering input, "
                   "Nudgeless changes immediately, and timed options add a delay."),
        buttons=[lambda: tr("Nudge"), lambda: tr("Nudgeless"), "0.5s", "1s", "2s", "3s"],
        button_width=170,
        param="AutoLaneChangeTimer",
        inline=False
      ),

      # Auto Lane Change BSM Delay
      toggle_item_sp(
        lambda: tr("Auto Lane Change: BSM Delay"),
        lambda: tr("Delay the lane change if the Blind Spot Monitor detects a vehicle in the adjacent lane."),
        param="AutoLaneChangeBsmDelay"
      ),

      # Blinker Pause Lateral Control
      toggle_item_sp(
        lambda: tr("Pause Steering on Blinker"),
        lambda: tr("Pause lateral control when the turn signal is activated above the minimum speed."),
        param="BlinkerPauseLateralControl"
      ),

      # Blinker Min Lateral Control Speed
      option_item_sp(
        lambda: tr("Blinker Pause: Min Speed"),
        "BlinkerMinLateralControlSpeed",
        0, 60,
        lambda: tr("Minimum speed at which lateral control will be paused when the turn signal is activated."),
        value_change_step=5,
        label_width=style.BUTTON_ACTION_WIDTH,
        label_callback=lambda v: f"{v} {'km/h' if ui_state.is_metric else 'mph'}",
        inline=True
      ),

      # Enforce Torque Control
      toggle_item_sp(
        lambda: tr("Enforce Torque Control"),
        lambda: tr("Force sunnypilot to use Torque lateral control regardless of car defaults."),
        param="EnforceTorqueControl"
      ),

      # Live Torque Toggle
      toggle_item_sp(
        lambda: tr("Live Torque"),
        lambda: tr("Continuously adapt torque parameters to your car's steering characteristics in real-time."),
        param="LiveTorqueParamsToggle"
      ),

      # Live Torque Relaxed Toggle
      toggle_item_sp(
        lambda: tr("Live Torque Relaxed"),
        lambda: tr("Use a more relaxed live torque tuning that allows larger variations."),
        param="LiveTorqueParamsRelaxedToggle"
      ),

      # Custom Torque Params
      toggle_item_sp(
        lambda: tr("Custom Torque Tuning"),
        lambda: tr("Enable custom tuning for Torque lateral control parameters."),
        param="CustomTorqueParams"
      ),

      # Torque Override Enabled
      toggle_item_sp(
        lambda: tr("Manual Real-Time Tuning"),
        lambda: tr("Manually override torque friction and lateral acceleration factor in real-time."),
        param="TorqueParamsOverrideEnabled"
      ),

      # Torque Friction
      option_item_sp(
        lambda: tr("Manual Tune: Friction"),
        "TorqueParamsOverrideFriction",
        0, 100,
        lambda: tr("Friction coefficient for manual torque tuning. Higher values add more resistance."),
        value_change_step=1,
        label_width=style.BUTTON_ACTION_WIDTH,
        use_float_scaling=True,
        label_callback=lambda v: f"{v / 100:.2f}",
        inline=True
      ),

      # Torque Lat Accel Factor
      option_item_sp(
        lambda: tr("Manual Tune: Lat Accel Factor"),
        "TorqueParamsOverrideLatAccelFactor",
        10, 500,
        lambda: tr("Lateral acceleration factor for manual torque tuning. Higher values increase steering response."),
        value_change_step=10,
        label_width=style.BUTTON_ACTION_WIDTH,
        use_float_scaling=True,
        label_callback=lambda v: f"{v / 100:.1f}",
        inline=True
      ),
    ]
    return items

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    self._scroller.show_event()
