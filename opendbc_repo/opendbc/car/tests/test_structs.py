"""Regression tests for opendbc.car.structs dataclass invariants."""
import unittest

from opendbc.car import structs


class TestCarStateSPSlots(unittest.TestCase):
  def test_carstatesp_rejects_undeclared_attributes(self):
    """CarStateSP must reject dynamic attribute writes.

    Without slots, writing an undeclared attribute such as ret_sp.madsDisableRequest
    silently succeeded as an ordinary instance attribute, but convert_to_capnp's
    asdictref() only iterates __dataclass_fields__, so the value never reached the
    wire. The write looked fine and the signal was simply missing downstream.

    With slots=True the bad write raises immediately, turning a silent data-loss
    bug into a loud one.
    """
    cs_sp = structs.CarStateSP()
    with self.assertRaises(AttributeError):
      cs_sp.madsDisableRequest = True  # type: ignore[attr-defined]

  def test_carstatesp_declared_fields_still_writable(self):
    """The slots change must not break legitimate writes -- speedLimit is populated
    by the Rivian carstate and consumed by the speed limit resolver."""
    cs_sp = structs.CarStateSP()
    cs_sp.speedLimit = 27.0
    self.assertEqual(cs_sp.speedLimit, 27.0)
