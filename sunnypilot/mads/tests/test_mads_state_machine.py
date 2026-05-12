"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import pytest
from pytest_mock import MockerFixture

from cereal import custom, log
from opendbc.car import structs
from opendbc.car.hyundai.values import HyundaiFlags
from openpilot.common.realtime import DT_CTRL
from openpilot.sunnypilot.mads.mads import ModularAssistiveDrivingSystem
from openpilot.sunnypilot.mads.state import StateMachine, SOFT_DISABLE_TIME
from openpilot.selfdrive.selfdrived.events import ET, NormalPermanentAlert, Events
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP, EVENTS_SP

State = custom.ModularAssistiveDrivingSystem.ModularAssistiveDrivingSystemState
EventName = log.OnroadEvent.EventName
EventNameSP = custom.OnroadEventSP.EventName

# The event types that maintain the current state
MAINTAIN_STATES = {State.enabled: (None,), State.disabled: (None,), State.softDisabling: (ET.SOFT_DISABLE,),
                   State.paused: (None,), State.overriding: (ET.OVERRIDE_LATERAL,)}
ALL_STATES = (State.schema.enumerants.values())
# The event types checked in DISABLED section of state machine
ENABLE_EVENT_TYPES = (ET.ENABLE, ET.OVERRIDE_LATERAL)


def make_event(event_types):
  event = {}
  for ev in event_types:
    event[ev] = NormalPermanentAlert("alert")
  EVENTS_SP[0] = event
  return 0


class MockMADS:
  def __init__(self, mocker: MockerFixture):
    self.selfdrive = mocker.MagicMock()
    self.selfdrive.state_machine = mocker.MagicMock()
    self.selfdrive.events = Events()
    self.selfdrive.events_sp = EventsSP()


class TestMADSStateMachine:
  @pytest.fixture(autouse=True)
  def setup_method(self, mocker: MockerFixture):
    self.mads = MockMADS(mocker)
    self.state_machine = StateMachine(self.mads)
    self.events = self.mads.selfdrive.events
    self.events_sp = self.mads.selfdrive.events_sp
    self.mads.selfdrive.state_machine.soft_disable_timer = int(SOFT_DISABLE_TIME / DT_CTRL)

  def clear_events(self):
    self.events.clear()
    self.events_sp.clear()

  def test_immediate_disable(self):
    for state in ALL_STATES:
      for et in MAINTAIN_STATES[state]:
        self.events_sp.add(make_event([et, ET.IMMEDIATE_DISABLE]))
        self.state_machine.state = state
        self.state_machine.update()
        assert State.disabled == self.state_machine.state
        self.clear_events()

  def test_user_disable(self):
    for state in ALL_STATES:
      for et in MAINTAIN_STATES[state]:
        self.events_sp.add(make_event([et, ET.USER_DISABLE]))
        self.state_machine.state = state
        self.state_machine.update()
        assert State.disabled == self.state_machine.state
        self.clear_events()

  def test_user_disable_to_paused(self):
    paused_events = (EventNameSP.silentLkasDisable, )
    for state in ALL_STATES:
      for et in MAINTAIN_STATES[state]:
        self.events_sp.add(make_event([et, ET.USER_DISABLE]))
        for en in paused_events:
          self.events_sp.add(en)
          self.state_machine.state = state
          self.state_machine.update()
          final_state = State.paused if self.events_sp.has(en) and state != State.disabled else State.disabled
          assert self.state_machine.state == final_state
          self.clear_events()

  def test_soft_disable(self):
    for state in ALL_STATES:
      for et in MAINTAIN_STATES[state]:
        self.events_sp.add(make_event([et, ET.SOFT_DISABLE]))
        self.state_machine.state = state
        self.state_machine.update()
        assert self.state_machine.state == State.disabled if state == State.disabled else State.softDisabling
        self.clear_events()

  def test_soft_disable_timer(self):
    self.state_machine.state = State.enabled
    self.events_sp.add(make_event([ET.SOFT_DISABLE]))
    self.state_machine.update()
    for _ in range(int(SOFT_DISABLE_TIME / DT_CTRL)):
      assert self.state_machine.state == State.softDisabling
      self.mads.selfdrive.state_machine.soft_disable_timer -= 1
      self.state_machine.update()

    assert self.state_machine.state == State.disabled
    self.clear_events()

  def test_no_entry(self):
    for et in ENABLE_EVENT_TYPES:
      self.events_sp.add(make_event([ET.NO_ENTRY, et]))
      self.state_machine.update()
      assert self.state_machine.state == State.disabled
      self.clear_events()

  def test_no_entry_paused(self):
    self.state_machine.state = State.paused
    self.events_sp.add(make_event([ET.NO_ENTRY]))
    self.state_machine.update()
    assert self.state_machine.state == State.paused
    self.clear_events()

  def test_override_lateral(self):
    self.state_machine.state = State.enabled
    self.events_sp.add(make_event([ET.OVERRIDE_LATERAL]))
    self.state_machine.update()
    assert self.state_machine.state == State.overriding
    self.clear_events()

  def test_paused_to_enabled(self):
    self.state_machine.state = State.paused
    self.events_sp.add(make_event([ET.ENABLE]))
    self.state_machine.update()
    assert self.state_machine.state == State.enabled
    self.clear_events()

  def test_maintain_states(self):
    for state in ALL_STATES:
      for et in MAINTAIN_STATES[state]:
        self.state_machine.state = state
        self.events_sp.add(make_event([et]))
        self.state_machine.update()
        assert self.state_machine.state == state
        self.clear_events()


class TestMADSBrandLkasButton:
  """Brand-specific LKAS button regressions (Tesla + Hyundai LDA)."""

  @pytest.fixture(autouse=True)
  def setup_method(self, mocker: MockerFixture):
    # Patch Params so no filesystem access is needed in ModularAssistiveDrivingSystem.__init__
    mock_params_cls = mocker.patch("openpilot.sunnypilot.mads.mads.Params")
    self.mock_params = mock_params_cls.return_value
    self.mock_params.get_bool.side_effect = lambda key: {
      "DisengageOnAccelerator": False,
      "Mads": True,
      "MadsMainCruiseAllowed": False,
      "MadsUnifiedEngagementMode": True,
    }.get(key, False)
    self.mock_params.get.return_value = None

    # Build a minimal mock selfdrive that satisfies ModularAssistiveDrivingSystem.__init__
    self.sd = mocker.MagicMock()
    self.sd.CP.brand = "rivian"
    self.sd.CP.passive = False
    self.sd.CP.flags = 0
    self.sd.CP_SP.flags = 0
    self.sd.params = self.mock_params
    self.sd.enabled = False
    self.sd.enabled_prev = False
    self.sd.initialized = True
    self.sd.events = Events()
    self.sd.events_sp = EventsSP()
    self.sd.CS_prev.cruiseState.enabled = False
    self.sd.CS_prev.cruiseState.available = False
    self.sd.CS_prev.gasPressed = False
    self.sd.state_machine.soft_disable_timer = int(SOFT_DISABLE_TIME / DT_CTRL)

  def _make_cs(self, cruise_enabled=False, cruise_available=True, button_events=None):
    cs = structs.CarState()
    cs.cruiseState.enabled = cruise_enabled
    cs.cruiseState.available = cruise_available
    if button_events:
      cs.buttonEvents = button_events
    return cs

  def _make_cs_sp(self):
    return structs.CarStateSP()

  def test_tesla_lkas_button_unchanged(self):
    """Tesla: lkas button while enabled + selfdrive.enabled → manualSteeringRequired (regression)."""
    self.sd.CP.brand = "tesla"
    self.sd.enabled = True
    self.sd.enabled_prev = True
    mads = ModularAssistiveDrivingSystem(self.sd)
    mads.enabled = True

    be = structs.CarState.ButtonEvent(pressed=True, type=structs.CarState.ButtonEvent.Type.lkas)
    cs = self._make_cs(cruise_available=True, button_events=[be])
    cs_sp = self._make_cs_sp()
    mads.update(cs, cs_sp)

    assert self.sd.events_sp.has(EventNameSP.manualSteeringRequired)

  def test_hyundai_lda_button_unchanged(self):
    """Hyundai LDA: lkas button while MADS disabled → lkasEnable (regression)."""
    self.sd.CP.brand = "hyundai"
    self.sd.CP.flags = HyundaiFlags.HAS_LDA_BUTTON
    mads = ModularAssistiveDrivingSystem(self.sd)
    assert mads.allow_always
    mads.enabled = False

    be = structs.CarState.ButtonEvent(pressed=True, type=structs.CarState.ButtonEvent.Type.lkas)
    # allow_always=True means cruiseState.available is not required
    cs = self._make_cs(cruise_available=False, button_events=[be])
    cs_sp = self._make_cs_sp()
    mads.update(cs, cs_sp)

    assert self.sd.events_sp.has(EventNameSP.lkasEnable)

  def test_rivian_button_cancel_disengages_mads(self):
    """Rivian UP_2 → ButtonEvent.cancel must transition MADS state machine to disabled."""
    self.sd.CP.brand = "rivian"
    mads = ModularAssistiveDrivingSystem(self.sd)
    assert mads.cancel_disengages_mads
    mads.enabled = True
    mads.state_machine.state = State.enabled

    self.sd.events.add(EventName.buttonCancel)
    be = structs.CarState.ButtonEvent(pressed=True, type=structs.CarState.ButtonEvent.Type.cancel)
    cs = self._make_cs(button_events=[be])
    cs_sp = self._make_cs_sp()
    mads.update(cs, cs_sp)

    assert mads.state_machine.state == State.disabled
    assert not mads.enabled
    assert not mads.active

  def test_hyundai_button_cancel_does_not_disengage_mads(self):
    """Non-Rivian: ButtonEvent.cancel must NOT disengage MADS lateral (stock-cruise semantics)."""
    self.sd.CP.brand = "hyundai"
    mads = ModularAssistiveDrivingSystem(self.sd)
    assert not mads.cancel_disengages_mads
    mads.enabled = True
    mads.state_machine.state = State.enabled

    self.sd.events.add(EventName.buttonCancel)
    be = structs.CarState.ButtonEvent(pressed=True, type=structs.CarState.ButtonEvent.Type.cancel)
    cs = self._make_cs(button_events=[be])
    cs_sp = self._make_cs_sp()
    mads.update(cs, cs_sp)

    assert mads.state_machine.state == State.enabled
