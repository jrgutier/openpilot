"""Per-segment cache for kp_tuner.

Skips re-decoding rlogs on subsequent runs by keying on (md5 of absolute path,
detector salt, schema/extractor versions, mtime, size). Cache files live under
`$XDG_CACHE_HOME/sunnypilot/kp_tuner/segments/` (default `~/.cache/...`).

Mirrors the md5+local-dir convention used by `tools/lib/cache.py`. Atomic
writes mirror the tmp+fsync+os.replace pattern used by `report.save_state_atomic`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from openpilot.tools.sunnypilot.kp_tuner.analyze import (
  CurveEvent,
  EXPECTED_TORQUE_STATE_VERSION,
  EVENT_PADDING_S,
  KP_UI_SPEED_BREAKPOINTS,
  LAT_ACCEL_HIGH,
  LAT_ACCEL_LOW,
  MIN_VEGO_FOR_SOLVER,
  RAMP_TIMEOUT_S,
)

log = logging.getLogger("kp_tuner.cache")

CACHE_SCHEMA_VERSION = 1
# Bump whenever extract_curve_events / _gates_pass / _half_amplitude_crossing /
# _DetectorState transitions / _build_event / read_segment_metadata / the
# CurveEvent dataclass shape change in a way that affects extracted events.
CACHE_EXTRACTOR_VERSION = 1


def _build_detector_salt() -> str:
  """8-char md5 over the detector knobs whose change MUST invalidate cache.

  Built as a function (not import-time global) so tests can monkeypatch this
  to force re-decode without mutating the underlying constants.
  """
  payload = (
    f"{LAT_ACCEL_LOW}:{LAT_ACCEL_HIGH}:{RAMP_TIMEOUT_S}:"
    + f"{EVENT_PADDING_S}:{MIN_VEGO_FOR_SOLVER}:"
    + f"{tuple(KP_UI_SPEED_BREAKPOINTS)}:"
    + f"{EXPECTED_TORQUE_STATE_VERSION}:"
    + f"{CACHE_EXTRACTOR_VERSION}"
  )
  return hashlib.md5(payload.encode()).hexdigest()[:8]


def _cache_dir() -> Path:
  base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
  return Path(base) / "sunnypilot" / "kp_tuner" / "segments"


@dataclass
class CachedSegment:
  model: str | None
  kp_triple: tuple[float, float, float]
  validate_pass: bool
  events: list[CurveEvent]


@dataclass
class CacheStats:
  hits: int = 0
  misses: int = 0
  writes: int = 0

  def reset(self) -> None:
    self.hits = 0
    self.misses = 0
    self.writes = 0


CACHE_STATS = CacheStats()


def _key_for(rlog_path: Path) -> str:
  return hashlib.md5(str(rlog_path.resolve()).encode()).hexdigest()[:16]


def _cache_file(rlog_path: Path) -> Path:
  return _cache_dir() / f"{_key_for(rlog_path)}.json"


def read_cached(rlog_path: Path) -> CachedSegment | None:
  """Return a cached entry or None on any failure mode (silent miss).

  Misses on: schema/extractor/salt mismatch, mtime drift, size drift, corrupt
  JSON, missing file, OSError. Never raises.
  """
  cache_file = _cache_file(rlog_path)
  try:
    if not cache_file.exists():
      return None
    rlog_stat = rlog_path.stat()
    raw = cache_file.read_text(encoding="utf-8")
    data = json.loads(raw)
    if data.get("schema_version") != CACHE_SCHEMA_VERSION:
      return None
    if data.get("extractor_version") != CACHE_EXTRACTOR_VERSION:
      return None
    if data.get("salt") != _build_detector_salt():
      return None
    if int(data.get("size", -1)) != rlog_stat.st_size:
      return None
    cached_mtime = float(data.get("mtime", -1))
    if abs(cached_mtime - rlog_stat.st_mtime) > 1e-6:
      return None
    events = [CurveEvent(**e) for e in data.get("events", [])]
    kp_raw = data.get("kp_triple", [1.0, 1.0, 1.0])
    kp = (float(kp_raw[0]), float(kp_raw[1]), float(kp_raw[2]))
    return CachedSegment(
      model=data.get("model"),
      kp_triple=kp,
      validate_pass=bool(data.get("validate_pass", False)),
      events=events,
    )
  except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
    return None


def write_cached_atomic(
  rlog_path: Path,
  model: str | None,
  kp_triple: tuple[float, float, float],
  validate_pass: bool,
  events: list[CurveEvent],
) -> None:
  """Write a cache entry atomically. On failure, log warning and continue."""
  cache_dir = _cache_dir()
  cache_file = cache_dir / f"{_key_for(rlog_path)}.json"
  try:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rlog_stat = rlog_path.stat()
    payload = {
      "schema_version": CACHE_SCHEMA_VERSION,
      "extractor_version": CACHE_EXTRACTOR_VERSION,
      "salt": _build_detector_salt(),
      "segment_path": str(rlog_path.resolve()),
      "size": rlog_stat.st_size,
      "mtime": rlog_stat.st_mtime,
      "model": model,
      "kp_triple": list(kp_triple),
      "validate_pass": validate_pass,
      "events": [dataclasses.asdict(e) for e in events],
    }
    tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
      json.dump(payload, f)
      f.flush()
      os.fsync(f.fileno())
    os.replace(tmp, cache_file)
    CACHE_STATS.writes += 1
  except OSError as e:
    log.warning("kp_tuner: cache write failed for %s (%s); analysis continues", rlog_path, e)


def clear_cache() -> int:
  """Remove all cached segment files. Returns the count removed."""
  cache_dir = _cache_dir()
  if not cache_dir.exists():
    return 0
  removed = 0
  for f in cache_dir.glob("*.json"):
    try:
      f.unlink()
      removed += 1
    except OSError:
      pass
  return removed
