"""Regression tests for opendbc.car.structs dataclass invariants."""
import pytest

from opendbc.car import structs


def test_carstatesp_slotted():
  """CarStateSP must reject dynamic attribute writes.

  Pre-fix, writing ret_sp.madsDisableRequest = True silently dropped the value
  because the attribute was not declared on the dataclass — the dataclass
  accepted the write as a regular instance attribute, but `convert_to_capnp`'s
  `asdictref()` only iterates `__dataclass_fields__`, so the value never
  reached the wire. With slots=True, the bad write becomes an immediate
  AttributeError, surfacing the bug class instead of silently dropping it.
  """
  cs_sp = structs.CarStateSP()
  with pytest.raises(AttributeError):
    cs_sp.madsDisableRequest = True  # type: ignore[attr-defined]


def test_carstatesp_speedlimit_assignable():
  """Sanity check: declared fields are still writable under slots."""
  cs_sp = structs.CarStateSP()
  cs_sp.speedLimit = 30.0
  assert cs_sp.speedLimit == 30.0
