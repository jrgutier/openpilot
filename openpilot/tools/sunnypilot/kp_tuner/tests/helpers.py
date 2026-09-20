"""Shared pytest-fixture-equivalents for kp_tuner's OpenpilotTestCase-based tests.

pytest is not available on this branch (tools/test_runner.py replaces it). These
generator functions plug into OpenpilotTestCase._fixture the same way a module-level
pytest fixture would: a test method requests one by naming it as a parameter, and a
test module makes one available by binding it at module scope, e.g. `tmp_path =
helpers.tmp_path`.
"""
from __future__ import annotations

import contextlib
import io
import logging
import sys
import tempfile
from pathlib import Path


def tmp_path():
  """Equivalent of pytest's `tmp_path`: a fresh, auto-cleaned directory per test."""
  with tempfile.TemporaryDirectory() as d:
    yield Path(d)


class _CaughtIO:
  def __init__(self, out, err):
    self.out = out
    self.err = err


class CapSys:
  """Equivalent of pytest's `capsys`: captures sys.stdout/sys.stderr writes."""

  def __init__(self):
    self._out = io.StringIO()
    self._err = io.StringIO()

  def readouterr(self):
    out, err = self._out.getvalue(), self._err.getvalue()
    self._out.seek(0)
    self._out.truncate()
    self._err.seek(0)
    self._err.truncate()
    return _CaughtIO(out, err)


def capsys():
  cap = CapSys()
  old_out, old_err = sys.stdout, sys.stderr
  sys.stdout, sys.stderr = cap._out, cap._err
  try:
    yield cap
  finally:
    sys.stdout, sys.stderr = old_out, old_err


class CapLog:
  """Equivalent of pytest's `caplog`: records log messages within an `at_level` block."""

  class _Handler(logging.Handler):
    def __init__(self):
      super().__init__()
      self.records = []

    def emit(self, record):
      self.records.append(record)

  def __init__(self):
    self._handler = self._Handler()

  @property
  def records(self):
    return self._handler.records

  @property
  def messages(self):
    return [r.getMessage() for r in self._handler.records]

  @contextlib.contextmanager
  def at_level(self, level, logger=None):
    target = logging.getLogger(logger)
    old_level = target.level
    target.addHandler(self._handler)
    target.setLevel(level)
    try:
      yield
    finally:
      target.removeHandler(self._handler)
      target.setLevel(old_level)


def caplog():
  yield CapLog()


class Approx:
  """Minimal stand-in for pytest.approx: scalar or same-length sequence comparison."""

  def __init__(self, expected, rel=1e-6, abs=None):  # noqa: A002 - mirrors pytest.approx's kwarg name
    self._expected = expected
    self._rel = rel
    self._abs = abs

  def _scalar_eq(self, actual, expected):
    import math
    if self._abs is not None:
      return math.isclose(actual, expected, rel_tol=0.0, abs_tol=self._abs)
    return math.isclose(actual, expected, rel_tol=self._rel, abs_tol=1e-12)

  def __eq__(self, actual):
    if isinstance(self._expected, (list, tuple)):
      if not isinstance(actual, (list, tuple)) or len(actual) != len(self._expected):
        return False
      return all(self._scalar_eq(a, e) for a, e in zip(actual, self._expected, strict=True))
    return self._scalar_eq(actual, self._expected)

  def __repr__(self):
    return f"approx({self._expected!r})"


def approx(expected, rel=1e-6, abs=None):  # noqa: A002 - mirrors pytest.approx's kwarg name
  return Approx(expected, rel=rel, abs=abs)
