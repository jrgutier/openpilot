"""Tests for report.py state JSON I/O (v3 schema), --revert, NN-FF abort, and
directional reporting (HTML column + console split).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from openpilot.tools.sunnypilot.kp_tuner import report
from openpilot.tools.sunnypilot.kp_tuner.analyze import BandStat, BandStats
from openpilot.tools.sunnypilot.kp_tuner.recommend import TuningRecommendation


def _bs(median_ratio: float = 0.9) -> BandStats:
  return BandStats(
    low=None,
    mid=BandStat(count=8, median_ratio=median_ratio, median_lag=0.05,
                 p25_ratio=median_ratio, p75_ratio=median_ratio,
                 vEgo_min=10.0, vEgo_max=14.0),
    high=None,
  )


def _rec(triple=(1.1, 1.2, 1.05), hold=False) -> TuningRecommendation:
  return TuningRecommendation(
    triple=triple,
    per_band_verdict={"low": "NO_DATA", "mid": "CONTINUE", "high": "NO_DATA"},
    rationale="test",
    oscillation_warning=False,
    hold=hold,
  )


def _bucket_lks(state: dict, bucket_id: str) -> list | None:
  return state["buckets"][bucket_id].get("last_known_safe")


def _bucket_iters(state: dict, bucket_id: str) -> list:
  return state["buckets"][bucket_id]["iterations"]


# ---------------------------------------------------------------------------
# State roundtrip (v3 schema)
# ---------------------------------------------------------------------------

def test_state_roundtrip(tmp_path: Path):
  sp = tmp_path / "state.json"
  state = report._empty_state()
  bid = report.bucket_id_for("OPM10V3", (1.0, 1.0, 1.0))
  report.append_iteration(state, (1.0, 1.0, 1.0), _bs(), _rec(),
                          bucket_id=bid, model="OPM10V3")
  report.save_state_atomic(state, sp)
  loaded = report.load_state(sp)
  assert loaded["schema_version"] == report.SCHEMA_VERSION
  assert bid in loaded["buckets"]
  assert len(_bucket_iters(loaded, bid)) == 1
  assert _bucket_iters(loaded, bid)[0]["applied_triple"] == [1.0, 1.0, 1.0]
  assert _bucket_lks(loaded, bid) == [1.0, 1.0, 1.0]


def test_state_atomic_write_no_tmp_left(tmp_path: Path):
  sp = tmp_path / "state.json"
  state = report._empty_state()
  report.save_state_atomic(state, sp)
  assert sp.exists()
  assert not sp.with_suffix(sp.suffix + ".tmp").exists()


# ---------------------------------------------------------------------------
# v2 → v3 migration
# ---------------------------------------------------------------------------

def test_v2_to_v3_migration(tmp_path: Path):
  """A v2 state.json with iterations and last_known_safe migrates into a
  single 'legacy' bucket with all iterations preserved."""
  sp = tmp_path / "state.json"
  v2 = {
    "schema_version": 2,
    "iterations": [
      {"ts": "2026-04-30T10:00:00Z", "applied_triple": [1.0, 1.0, 1.0],
       "per_band_stats": {}, "recommendation": {"hold": False}},
      {"ts": "2026-04-30T11:00:00Z", "applied_triple": [1.1, 1.2, 1.05],
       "per_band_stats": {}, "recommendation": {"hold": True}},
      {"ts": "2026-04-30T12:00:00Z", "applied_triple": [1.1, 1.2, 1.05],
       "per_band_stats": {}, "recommendation": {"hold": True}},
    ],
    "last_known_safe": [1.0, 1.0, 1.0],
  }
  sp.write_text(json.dumps(v2))
  loaded = report.load_state(sp)
  assert loaded["schema_version"] == 3
  assert len(loaded["buckets"]) == 1
  bid, bucket = next(iter(loaded["buckets"].items()))
  assert bucket["model"] == report.LEGACY_BUCKET_MODEL
  assert len(bucket["iterations"]) == 3
  assert bucket["last_known_safe"] == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# last_known_safe semantics (per-bucket)
# ---------------------------------------------------------------------------

def test_last_known_safe_persists(tmp_path: Path):
  sp = tmp_path / "state.json"
  state = report._empty_state()
  bid = report.bucket_id_for("M", (1.0, 1.0, 1.0))
  # Iteration 1: actionable recommendation produces (1.1, 1.2, 1.05) for next drive.
  report.append_iteration(state, (1.0, 1.0, 1.0), _bs(),
                          _rec(triple=(1.1, 1.2, 1.05), hold=False),
                          bucket_id=bid, model="M")
  # Iteration 2: bucket's currently-applied is now (1.1, 1.2, 1.05); previously-
  # applied (1.0, 1.0, 1.0) becomes lks because previous rec was actionable.
  # Append continuation in the SAME bucket id — lks promotion only works
  # within one bucket; staying in `bid` exercises that contract.
  report.append_iteration(state, (1.1, 1.2, 1.05), _bs(0.95), _rec(hold=True),
                          bucket_id=bid, model="M")
  assert _bucket_lks(state, bid) == [1.0, 1.0, 1.0]
  report.save_state_atomic(state, sp)
  reloaded = report.load_state(sp)
  assert _bucket_lks(reloaded, bid) == [1.0, 1.0, 1.0]


def test_last_known_safe_does_not_advance_on_hold(tmp_path: Path):
  state = report._empty_state()
  bid = report.bucket_id_for("M", (1.0, 1.0, 1.0))
  report.append_iteration(state, (1.0, 1.0, 1.0), _bs(), _rec(hold=True),
                          bucket_id=bid, model="M")
  assert _bucket_lks(state, bid) == [1.0, 1.0, 1.0]
  report.append_iteration(state, (1.0, 1.0, 1.0), _bs(), _rec(hold=True),
                          bucket_id=bid, model="M")
  assert _bucket_lks(state, bid) == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# --revert flag
# ---------------------------------------------------------------------------

def test_revert_flag_returns_lks_and_does_not_mutate(tmp_path: Path, capsys):
  sp = tmp_path / "state.json"
  state = report._empty_state()
  bid = report.bucket_id_for("M", (1.0, 1.0, 1.0))
  report.append_iteration(state, (1.0, 1.0, 1.0), _bs(),
                          _rec(triple=(1.1, 1.2, 1.05), hold=False),
                          bucket_id=bid, model="M")
  report.append_iteration(state, (1.1, 1.2, 1.05), _bs(0.97), _rec(hold=True),
                          bucket_id=bid, model="M")
  report.save_state_atomic(state, sp)
  before = sp.read_text()

  rc = report.main(["--revert", "--state-path", str(sp)])
  assert rc == report.EXIT_OK
  out = capsys.readouterr().out
  assert "Set sliders to" in out
  assert "1.0" in out
  assert sp.read_text() == before


def test_revert_flag_no_state(tmp_path: Path, capsys):
  sp = tmp_path / "state.json"  # never created
  rc = report.main(["--revert", "--state-path", str(sp)])
  assert rc == report.EXIT_NO_STATE
  err = capsys.readouterr().err
  assert "no last_known_safe" in err


def test_revert_ambiguity_multiple_buckets(tmp_path: Path, capsys):
  """When state has multiple buckets and no --bucket-filter, --revert must
  exit with usage error and list available buckets."""
  sp = tmp_path / "state.json"
  state = report._empty_state()
  for model, kp in (("A", (1.0, 1.0, 1.0)), ("B", (0.7, 0.8, 0.9))):
    bid = report.bucket_id_for(model, kp)
    report.append_iteration(state, kp, _bs(), _rec(hold=False), bucket_id=bid, model=model)
  report.save_state_atomic(state, sp)

  rc = report.main(["--revert", "--state-path", str(sp)])
  assert rc == report.EXIT_USAGE
  err = capsys.readouterr().err
  assert "multiple buckets" in err
  assert "--bucket-filter" in err


def test_revert_with_filter_resolves_one_bucket(tmp_path: Path, capsys):
  sp = tmp_path / "state.json"
  state = report._empty_state()
  for model, kp in (("A", (1.0, 1.0, 1.0)), ("B", (0.7, 0.8, 0.9))):
    bid = report.bucket_id_for(model, kp)
    report.append_iteration(state, kp, _bs(), _rec(hold=False), bucket_id=bid, model=model)
  report.save_state_atomic(state, sp)

  rc = report.main(["--revert", "--state-path", str(sp), "--bucket-filter", "A"])
  assert rc == report.EXIT_OK
  out = capsys.readouterr().out
  assert "Set sliders to" in out


def test_revert_with_filter_zero_matches(tmp_path: Path, capsys):
  sp = tmp_path / "state.json"
  state = report._empty_state()
  bid = report.bucket_id_for("A", (1.0, 1.0, 1.0))
  report.append_iteration(state, (1.0, 1.0, 1.0), _bs(), _rec(hold=False), bucket_id=bid, model="A")
  report.save_state_atomic(state, sp)

  rc = report.main(["--revert", "--state-path", str(sp), "--bucket-filter", "Z"])
  assert rc == report.EXIT_USAGE
  err = capsys.readouterr().err
  assert "matched zero" in err


# ---------------------------------------------------------------------------
# NN-FF detection abort (now scans ALL segment metadata)
# ---------------------------------------------------------------------------

class _Model:
  def __init__(self, path):
    self.path = path


class _NNLC:
  def __init__(self, path):
    self.model = _Model(path)


class _CPSP:
  def __init__(self, path):
    self.neuralNetworkLateralControl = _NNLC(path)


class _InitData:
  def __init__(self, params):
    self.params = params


def _stub_partition(events_by_bucket, init_datas, car_params_sps):
  return events_by_bucket, init_datas, car_params_sps


def test_nn_ff_abort_when_param_on_and_real_model(tmp_path: Path, capsys, monkeypatch):
  log_dir = tmp_path / "logs"
  log_dir.mkdir()
  init_on = _InitData({"NeuralNetworkLateralControl": b"1"})
  cpsp_real = _CPSP("/sunnypilot/models/RIVIAN.json")

  def _fake_partition(_log_dir, *, use_cache):
    # 2 buckets, one of which has NN-FF active.
    return ({(None, (1.0, 1.0, 1.0)): []}, [init_on], [cpsp_real])

  monkeypatch.setattr(report, "_partition_segments_by_bucket", _fake_partition)

  rc = report.main([
    "--log-dir", str(log_dir),
    "--current", "1.0,1.0,1.0",
    "--state-path", str(tmp_path / "state.json"),
  ])
  assert rc == report.EXIT_NN_FF_ACTIVE
  err = capsys.readouterr().err
  assert "NN feedforward is active" in err


def test_nn_ff_proceeds_when_param_on_but_mock(tmp_path: Path, monkeypatch):
  log_dir = tmp_path / "logs"
  log_dir.mkdir()
  init_on = _InitData({"NeuralNetworkLateralControl": b"1"})
  cpsp_mock = _CPSP("/sunnypilot/models/MOCK.json")

  monkeypatch.setattr(
    report, "_partition_segments_by_bucket",
    lambda _d, *, use_cache: ({(None, (1.0, 1.0, 1.0)): []}, [init_on], [cpsp_mock]),
  )

  rc = report.main([
    "--log-dir", str(log_dir),
    "--current", "1.0,1.0,1.0",
    "--state-path", str(tmp_path / "state.json"),
  ])
  assert rc == report.EXIT_OK


# ---------------------------------------------------------------------------
# --bucket-filter on full-run
# ---------------------------------------------------------------------------

def test_bucket_filter_zero_matches_full_run(tmp_path: Path, monkeypatch, capsys):
  log_dir = tmp_path / "logs"
  log_dir.mkdir()
  monkeypatch.setattr(
    report, "_partition_segments_by_bucket",
    lambda _d, *, use_cache: ({("A", (1.0, 1.0, 1.0)): []}, [], []),
  )
  rc = report.main([
    "--log-dir", str(log_dir),
    "--state-path", str(tmp_path / "state.json"),
    "--bucket-filter", "Z",
  ])
  assert rc == report.EXIT_USAGE
  err = capsys.readouterr().err
  assert "matched zero" in err


# ---------------------------------------------------------------------------
# Triple parsing
# ---------------------------------------------------------------------------

def test_parse_triple_valid():
  assert report._parse_triple("1.0,1.5,2.0") == (1.0, 1.5, 2.0)


def test_parse_triple_wrong_arity():
  with pytest.raises(argparse.ArgumentTypeError):
    report._parse_triple("1.0,2.0")


def test_parse_triple_non_numeric():
  with pytest.raises(argparse.ArgumentTypeError):
    report._parse_triple("a,b,c")


# ---------------------------------------------------------------------------
# Help / parser smoke
# ---------------------------------------------------------------------------

def test_parser_help_includes_revert():
  parser = report._build_parser()
  text = parser.format_help()
  assert "--revert" in text
  assert "--current" in text
  assert "--log-dir" in text
  assert "--bucket-filter" in text
  assert "--no-cache" in text


# ---------------------------------------------------------------------------
# Full e2e smoke — empty bucket dict produces usage exit
# ---------------------------------------------------------------------------

def test_e2e_no_buckets_exits_usage(tmp_path: Path, monkeypatch, capsys):
  """When `_partition_segments_by_bucket` returns an empty bucket dict, the
  CLI must NOT create empty buckets in state.json — instead exit usage."""
  log_dir = tmp_path / "logs"
  log_dir.mkdir()
  monkeypatch.setattr(
    report, "_partition_segments_by_bucket",
    lambda _d, *, use_cache: ({}, [], []),
  )
  rc = report.main([
    "--log-dir", str(log_dir),
    "--state-path", str(tmp_path / "state.json"),
  ])
  assert rc == report.EXIT_USAGE
  err = capsys.readouterr().err
  assert "no usable segments" in err


# ---------------------------------------------------------------------------
# Directional reporting (HTML column + console split)
# ---------------------------------------------------------------------------

def test_render_html_under_over_column():
  bs = BandStats(
    low=None,
    mid=None,
    high=BandStat(
      count=6, median_ratio=0.95, median_lag=0.05,
      p25_ratio=0.9, p75_ratio=1.05,
      vEgo_min=20.0, vEgo_max=30.0,
      undershoot_count=4, oversteer_count=2,
    ),
  )
  rec = _rec(triple=(1.0, 1.0, 1.0))
  state = report._empty_state()
  html_out = report.render_html(
    current_triple=(1.0, 1.0, 1.0),
    recommendation=rec, band_stats=bs, state=state, excluded_low_speed=0,
  )
  assert "<th>under/over</th>" in html_out
  assert ">4/2<" in html_out


def test_console_directional_text(tmp_path: Path, capsys, monkeypatch):
  """`_do_full_run` prints a directional events line per band."""
  from openpilot.tools.sunnypilot.kp_tuner.analyze import CurveEvent
  events = [
    CurveEvent(t0=0.0, t_peak=0.5, vEgo_t0=20.0, vEgo_peak=20.0,
               peak_desired_curvature=0.01, peak_actual_curvature=0.01 * r,
               tracking_ratio=r, lag_seconds=0.05, band="high")
    for r in (0.85, 0.92, 1.10)
  ]
  monkeypatch.setattr(
    report, "_partition_segments_by_bucket",
    lambda _d, *, use_cache: ({(None, (1.0, 1.0, 1.0)): events}, [], []),
  )

  log_dir = tmp_path / "logs"
  log_dir.mkdir()
  ns = argparse.Namespace(
    log_dir=log_dir, current=(1.0, 1.0, 1.0),
    out_html=None, min_events_per_band=2,
    state_path=tmp_path / "state.json", verbose=False,
    no_cache=True, clear_cache=False, bucket_filter=None,
  )
  report._do_full_run(ns)
  out = capsys.readouterr().out
  import re
  assert re.search(r"events: total=\d+ \(low=\d+ under/\d+ over", out), out
