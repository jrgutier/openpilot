"""Synthetic-fixture tests for per-(model, Kp) bucketing in `report._do_full_run`.

These tests are the CI gate (Critic C4): real-data 4-bucket discovery is
advisory smoke and not in CI. Synthetic fixtures pin behavior across machines
and don't depend on the user's `~/sunnypilot-logs/` mirror.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpilot.common.test import OpenpilotTestCase

from openpilot.tools.sunnypilot.kp_tuner import report
from openpilot.tools.sunnypilot.kp_tuner.analyze import CurveEvent
from openpilot.tools.sunnypilot.kp_tuner.tests import helpers as _fixtures

tmp_path = _fixtures.tmp_path


def _evt(v: float, r: float, band: str = "high") -> CurveEvent:
  return CurveEvent(
    t0=0.0, t_peak=0.5, vEgo_t0=v, vEgo_peak=v,
    peak_desired_curvature=0.01, peak_actual_curvature=0.01 * r,
    tracking_ratio=r, lag_seconds=0.05, band=band,
  )


def _ns(tmp_path: Path, *, out_html: Path | None = None) -> argparse.Namespace:
  log_dir = tmp_path / "logs"
  log_dir.mkdir(exist_ok=True)
  return argparse.Namespace(
    log_dir=log_dir, current=None,
    out_html=out_html, min_events_per_band=2,
    state_path=tmp_path / "state.json", verbose=False,
    no_cache=True, clear_cache=False, bucket_filter=None,
  )


class TestPerModel(OpenpilotTestCase):
  def test_synthetic_2_bucket_walk_discovers_2_buckets(self, tmp_path: Path, monkeypatch):
    """Two distinct (model, kp) combinations → exactly 2 buckets in state.json."""
    events_a = [_evt(20.0, 0.85), _evt(22.0, 0.90)]
    events_b = [_evt(20.0, 1.05), _evt(22.0, 1.08)]
    monkeypatch.setattr(
      report, "_partition_segments_by_bucket",
      lambda _d, *, use_cache: (
        {("A", (1.0, 1.0, 1.0)): events_a, ("B", (0.7, 0.8, 0.9)): events_b},
        [], [],
      ),
    )
    rc = report._do_full_run(_ns(tmp_path))
    assert rc == report.EXIT_OK
    state = json.loads((tmp_path / "state.json").read_text())
    assert len(state["buckets"]) == 2
    assert any(b.get("model") == "A" for b in state["buckets"].values())
    assert any(b.get("model") == "B" for b in state["buckets"].values())

  def test_synthetic_4_bucket_walk_discovers_4_buckets(self, tmp_path: Path, monkeypatch):
    events = [_evt(20.0, 0.9), _evt(22.0, 0.95)]
    monkeypatch.setattr(
      report, "_partition_segments_by_bucket",
      lambda _d, *, use_cache: (
        {
          ("PMV2", (0.7, 0.8, 0.9)): events,
          ("PMV2", (0.6, 0.75, 0.9)): events,
          ("OPM10V3", (1.0, 1.0, 1.0)): events,
          ("OPM10V3", (0.7, 0.8, 0.9)): events,
        },
        [], [],
      ),
    )
    rc = report._do_full_run(_ns(tmp_path))
    assert rc == report.EXIT_OK
    state = json.loads((tmp_path / "state.json").read_text())
    assert len(state["buckets"]) == 4

  def test_combined_html_has_per_bucket_sections(self, tmp_path: Path, monkeypatch):
    events = [_evt(20.0, 0.9), _evt(22.0, 0.95)]
    monkeypatch.setattr(
      report, "_partition_segments_by_bucket",
      lambda _d, *, use_cache: (
        {("A", (1.0, 1.0, 1.0)): events, ("B", (0.7, 0.8, 0.9)): events},
        [], [],
      ),
    )
    out_html = tmp_path / "report.html"
    rc = report._do_full_run(_ns(tmp_path, out_html=out_html))
    assert rc == report.EXIT_OK
    text = out_html.read_text()
    assert text.count("<h2>") == 2  # one section per bucket
    assert "MODEL: A" in text
    assert "MODEL: B" in text

  def test_bucket_filter_restricts_to_one_model(self, tmp_path: Path, monkeypatch):
    events = [_evt(20.0, 0.9), _evt(22.0, 0.95)]
    monkeypatch.setattr(
      report, "_partition_segments_by_bucket",
      lambda _d, *, use_cache: (
        {("A", (1.0, 1.0, 1.0)): events, ("B", (0.7, 0.8, 0.9)): events},
        [], [],
      ),
    )
    rc = report._do_full_run(_ns(tmp_path), bucket_filter=["A"])
    assert rc == report.EXIT_OK
    state = json.loads((tmp_path / "state.json").read_text())
    assert len(state["buckets"]) == 1
    assert next(iter(state["buckets"].values()))["model"] == "A"

  def test_bucket_id_is_deterministic_and_human_readable(self):
    bid = report.bucket_id_for("OPM10V3", (1.0, 1.0, 1.0))
    assert bid == "OPM10V3__1.0000_1.0000_1.0000"
    bid_unknown = report.bucket_id_for(None, (0.7, 0.8, 0.9))
    assert bid_unknown == "unknown__0.7000_0.8000_0.9000"

  # ---------------------------------------------------------------------------
  # Joint per-model solver — HTML rendering + state immutability (US-004)
  # ---------------------------------------------------------------------------

  def _two_bucket_pipeline(self, tmp_path: Path, monkeypatch, *, joint: bool, out_html: Path):
    events = [_evt(v, 0.9) for v in (5.0, 11.0, 20.0, 30.0)] * 3  # 12 events
    monkeypatch.setattr(
      report, "_partition_segments_by_bucket",
      lambda _d, *, use_cache: (
        {("A", (1.0, 1.0, 1.0)): events, ("A", (0.7, 0.8, 0.9)): events},
        [], [],
      ),
    )
    argv = [
      "--log-dir", str(tmp_path / "logs"),
      "--state-path", str(tmp_path / "state.json"),
      "--out-html", str(out_html),
    ]
    (tmp_path / "logs").mkdir(exist_ok=True)
    if joint:
      argv.append("--joint-per-model")
    return report.main(argv)

  def test_joint_html_section_present_when_flag_set(self, tmp_path: Path, monkeypatch):
    out_html = tmp_path / "report.html"
    rc = self._two_bucket_pipeline(tmp_path, monkeypatch, joint=True, out_html=out_html)
    assert rc == report.EXIT_OK
    text = out_html.read_text()
    assert "JOINT:" in text
    assert "Joint per-model recommendations" in text

  def test_joint_html_section_absent_when_flag_unset(self, tmp_path: Path, monkeypatch):
    out_html = tmp_path / "report.html"
    rc = self._two_bucket_pipeline(tmp_path, monkeypatch, joint=False, out_html=out_html)
    assert rc == report.EXIT_OK
    text = out_html.read_text()
    assert "JOINT:" not in text
    assert "Joint per-model recommendations" not in text

  def test_joint_never_writes_to_state(self, tmp_path: Path, monkeypatch):
    """After --joint-per-model, state.json must contain ONLY per-bucket iterations.
    Joint advisory output must not leak into any iteration record's applied_triple.
    """
    out_html = tmp_path / "report.html"
    rc = self._two_bucket_pipeline(tmp_path, monkeypatch, joint=True, out_html=out_html)
    assert rc == report.EXIT_OK
    state = json.loads((tmp_path / "state.json").read_text())
    for bucket in state["buckets"].values():
      last_iter = bucket["iterations"][-1]
      applied = tuple(last_iter["applied_triple"])
      # Each bucket's last iteration must have applied_triple matching that bucket's
      # kp_triple_initial (the per-bucket recommend doesn't run multiple iterations
      # in this test). Joint output would NOT match either bucket's initial triple.
      assert applied == tuple(bucket["kp_triple_initial"]), \
        f"applied {applied} != initial {bucket['kp_triple_initial']} — joint leaked into state"
