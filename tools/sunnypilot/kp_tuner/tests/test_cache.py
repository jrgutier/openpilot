"""Tests for the per-segment cache layer (`_cache` module).

The cache wraps `extract_curve_events` so subsequent runs of the kp_tuner CLI
on an unchanged rlog mirror skip re-decoding. Tests cover hit/miss, schema +
extractor + salt invalidation, mtime/size drift, corrupt JSON, and the
write-failure-preserves-analysis contract.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from openpilot.tools.sunnypilot.kp_tuner import _cache
from openpilot.tools.sunnypilot.kp_tuner.analyze import CurveEvent


def _fake_event(ratio: float = 0.9) -> CurveEvent:
  return CurveEvent(
    t0=0.0, t_peak=0.5, vEgo_t0=20.0, vEgo_peak=20.0,
    peak_desired_curvature=0.01, peak_actual_curvature=0.01 * ratio,
    tracking_ratio=ratio, lag_seconds=0.05, band="high",
  )


def _make_rlog(tmp_path: Path, content: bytes = b"fake-rlog-bytes") -> Path:
  p = tmp_path / "fake.rlog.zst"
  p.write_bytes(content)
  return p


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path: Path, monkeypatch):
  """Each test gets its own cache directory under tmp_path."""
  monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
  _cache.CACHE_STATS.reset()
  yield


def test_cache_miss_then_hit(tmp_path: Path):
  rlog = _make_rlog(tmp_path)
  assert _cache.read_cached(rlog) is None
  evt = _fake_event(0.85)
  _cache.write_cached_atomic(rlog, "OPM10V3", (1.0, 1.0, 1.0), True, [evt])
  cached = _cache.read_cached(rlog)
  assert cached is not None
  assert cached.model == "OPM10V3"
  assert cached.kp_triple == (1.0, 1.0, 1.0)
  assert cached.validate_pass is True
  assert len(cached.events) == 1
  assert cached.events[0].tracking_ratio == pytest.approx(0.85)


def test_cache_miss_no_file(tmp_path: Path):
  rlog = tmp_path / "does-not-exist.rlog.zst"
  assert _cache.read_cached(rlog) is None


def test_schema_version_mismatch_silent_miss(tmp_path: Path):
  rlog = _make_rlog(tmp_path)
  _cache.write_cached_atomic(rlog, None, (1.0, 1.0, 1.0), True, [])
  # Manually corrupt schema_version in the cached file.
  cache_file = _cache._cache_file(rlog)
  data = json.loads(cache_file.read_text())
  data["schema_version"] = 999
  cache_file.write_text(json.dumps(data))
  assert _cache.read_cached(rlog) is None


def test_extractor_version_mismatch_silent_miss(tmp_path: Path):
  rlog = _make_rlog(tmp_path)
  _cache.write_cached_atomic(rlog, None, (1.0, 1.0, 1.0), True, [])
  cache_file = _cache._cache_file(rlog)
  data = json.loads(cache_file.read_text())
  data["extractor_version"] = 999
  cache_file.write_text(json.dumps(data))
  assert _cache.read_cached(rlog) is None


def test_salt_mismatch_silent_miss(tmp_path: Path, monkeypatch):
  rlog = _make_rlog(tmp_path)
  _cache.write_cached_atomic(rlog, None, (1.0, 1.0, 1.0), True, [])
  # Monkeypatch the salt builder so subsequent reads compute a different salt.
  monkeypatch.setattr(_cache, "_build_detector_salt", lambda: "deadbeef")
  assert _cache.read_cached(rlog) is None


def test_mtime_drift_silent_miss(tmp_path: Path):
  rlog = _make_rlog(tmp_path)
  _cache.write_cached_atomic(rlog, None, (1.0, 1.0, 1.0), True, [_fake_event()])
  assert _cache.read_cached(rlog) is not None
  # Bump mtime by a clearly observable delta.
  os.utime(rlog, (0, 1.0e8))
  assert _cache.read_cached(rlog) is None


def test_size_drift_silent_miss(tmp_path: Path):
  rlog = _make_rlog(tmp_path, content=b"abc")
  _cache.write_cached_atomic(rlog, None, (1.0, 1.0, 1.0), True, [])
  assert _cache.read_cached(rlog) is not None
  rlog.write_bytes(b"abcdef")  # different size
  assert _cache.read_cached(rlog) is None


def test_corrupt_json_silent_miss(tmp_path: Path):
  rlog = _make_rlog(tmp_path)
  _cache.write_cached_atomic(rlog, None, (1.0, 1.0, 1.0), True, [])
  cache_file = _cache._cache_file(rlog)
  cache_file.write_text("{not valid json")
  assert _cache.read_cached(rlog) is None


def test_write_failure_preserves_analysis(tmp_path: Path, monkeypatch, caplog):
  """OSError on write must NOT raise — analysis must continue."""
  rlog = _make_rlog(tmp_path)

  def _raise(*_args, **_kw):
    raise OSError("disk full")

  monkeypatch.setattr(_cache.os, "replace", _raise)
  with caplog.at_level("WARNING", logger="kp_tuner.cache"):
    _cache.write_cached_atomic(rlog, None, (1.0, 1.0, 1.0), True, [_fake_event()])
  assert _cache.CACHE_STATS.writes == 0
  assert any("cache write failed" in m for m in caplog.messages)


def test_clear_cache_removes_files(tmp_path: Path):
  (tmp_path / "a").mkdir(parents=True, exist_ok=True)
  (tmp_path / "b").mkdir(parents=True, exist_ok=True)
  rlog1 = tmp_path / "a" / "fake.rlog.zst"
  rlog1.write_bytes(b"a")
  rlog2 = tmp_path / "b" / "fake.rlog.zst"
  rlog2.write_bytes(b"b")
  _cache.write_cached_atomic(rlog1, None, (1.0, 1.0, 1.0), True, [])
  _cache.write_cached_atomic(rlog2, None, (1.0, 1.0, 1.0), True, [])
  removed = _cache.clear_cache()
  assert removed == 2
  assert _cache.read_cached(rlog1) is None
  assert _cache.read_cached(rlog2) is None


def test_monkeypatched_salt_busts_cache(tmp_path: Path, monkeypatch):
  """Critic C2: invalidation by monkeypatching _build_detector_salt forces a re-decode."""
  rlog = _make_rlog(tmp_path)
  _cache.write_cached_atomic(rlog, None, (1.0, 1.0, 1.0), True, [_fake_event()])
  initial_writes = _cache.CACHE_STATS.writes
  assert _cache.read_cached(rlog) is not None  # baseline hit

  # Force a different salt.
  monkeypatch.setattr(_cache, "_build_detector_salt", lambda: "00112233")
  assert _cache.read_cached(rlog) is None  # silent miss after salt rotation

  # Re-write under the new salt and confirm read works.
  _cache.write_cached_atomic(rlog, None, (1.0, 1.0, 1.0), True, [_fake_event()])
  assert _cache.CACHE_STATS.writes == initial_writes + 1
  refetched = _cache.read_cached(rlog)
  assert refetched is not None
