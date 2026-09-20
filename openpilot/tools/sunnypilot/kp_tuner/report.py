"""kp_tuner CLI — read logs, compute recommendation, persist state, render HTML.

Plan reference: .omc/plans/rivian-kp-tuning.md, Step 4.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import html
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

from openpilot.tools.sunnypilot.kp_tuner import _cache
from openpilot.tools.sunnypilot.kp_tuner.analyze import (
  BandStats,
  CurveEvent,
  MIN_VEGO_FOR_SOLVER,
  detect_nn_ff_active,
  extract_curve_events,
  iter_segments,
  load_messages,
  partition_by_band,
  read_segment_metadata_from_messages,
  summarize,
  validate_segment,
)
from openpilot.tools.sunnypilot.kp_tuner.recommend import (
  MIN_EVENTS_DEFAULT,
  TuningRecommendation,
  recommend,
  recommend_joint,
)

STATE_DIR = Path.home() / ".config" / "sunnypilot" / "kp_tuner"
STATE_PATH = STATE_DIR / "state.json"
SCHEMA_VERSION = 3
LEGACY_BUCKET_MODEL = "legacy"  # marker for v2-migrated single-bucket data

EXIT_OK = 0
EXIT_USAGE = 64
EXIT_NN_FF_ACTIVE = 2
EXIT_NO_STATE = 3

log = logging.getLogger("kp_tuner.report")


# -----------------------------------------------------------------------------
# State JSON I/O
# -----------------------------------------------------------------------------

def _empty_state() -> dict[str, Any]:
  return {"schema_version": SCHEMA_VERSION, "buckets": {}}


def _empty_bucket(model: str | None, kp_initial: tuple[float, float, float]) -> dict[str, Any]:
  return {
    "model": model,
    "kp_triple_initial": list(kp_initial),
    "iterations": [],
    "last_known_safe": None,
  }


def bucket_id_for(model: str | None, kp: tuple[float, float, float]) -> str:
  m = model or "unknown"
  return f"{m}__{kp[0]:.4f}_{kp[1]:.4f}_{kp[2]:.4f}"


def _migrate_v2_to_v3(v2: dict[str, Any]) -> dict[str, Any]:
  """Map a v2 state to a v3 single-bucket state under model='legacy'.

  v2 had top-level `iterations` + `last_known_safe`; preserve every iteration.
  bucket kp_triple_initial is taken from the first iteration's applied_triple
  (or last_known_safe when iterations are empty).
  """
  state = _empty_state()
  iterations = list(v2.get("iterations") or [])
  lks = v2.get("last_known_safe")
  if not iterations and lks is None:
    return state
  kp_initial = tuple(iterations[0]["applied_triple"]) if iterations else tuple(lks or (1.0, 1.0, 1.0))
  bid = bucket_id_for(LEGACY_BUCKET_MODEL, kp_initial)  # type: ignore[arg-type]
  bucket = _empty_bucket(LEGACY_BUCKET_MODEL, kp_initial)  # type: ignore[arg-type]
  bucket["iterations"] = iterations
  bucket["last_known_safe"] = list(lks) if lks is not None else None
  state["buckets"][bid] = bucket
  return state


def load_state(state_path: Path = STATE_PATH) -> dict[str, Any]:
  if not state_path.exists():
    return _empty_state()
  try:
    with state_path.open("r", encoding="utf-8") as f:
      data = json.load(f)
  except (OSError, json.JSONDecodeError) as e:
    log.warning("kp_tuner: failed to read %s (%s); starting fresh", state_path, e)
    return _empty_state()
  ver = data.get("schema_version")
  if ver == SCHEMA_VERSION:
    return data
  if ver == 2:
    log.info("kp_tuner: migrating state %s from v2 to v%d", state_path, SCHEMA_VERSION)
    return _migrate_v2_to_v3(data)
  log.warning("kp_tuner: schema_version=%s in %s; starting fresh", ver, state_path)
  return _empty_state()


def save_state_atomic(state: dict[str, Any], state_path: Path = STATE_PATH) -> None:
  """Atomic write: serialize to a sibling temp file, fsync, os.replace.

  Plan §Atomic write: a `--revert` mid-iteration (Ctrl-C while analyzer is
  writing) must observe a CONSISTENT state file even if we crashed. `os.replace`
  is atomic on POSIX (single inode swap) and on Windows NTFS (`MoveFileEx`).
  """
  state_path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
  with tmp_path.open("w", encoding="utf-8") as f:
    json.dump(state, f, indent=2, sort_keys=True)
    f.flush()
    os.fsync(f.fileno())
  os.replace(tmp_path, state_path)


# -----------------------------------------------------------------------------
# Pipeline helpers
# -----------------------------------------------------------------------------

def _parse_triple(s: str) -> tuple[float, float, float]:
  parts = [p.strip() for p in s.split(",")]
  if len(parts) != 3:
    raise argparse.ArgumentTypeError(f"expected LOW,MID,HIGH triple, got {s!r}")
  try:
    return float(parts[0]), float(parts[1]), float(parts[2])
  except ValueError as e:
    raise argparse.ArgumentTypeError(f"non-numeric component in {s!r}") from e


def _segment_record(seg: Path, *, use_cache: bool = True) -> tuple[_cache.CachedSegment, bool]:
  """Single source of truth for events + metadata per segment.

  Cache hit: return (cached, cached.validate_pass). Cache miss: walk messages
  ONCE, derive metadata + events from the same walk, write cache, return.
  """
  cached = _cache.read_cached(seg) if use_cache else None
  if cached is not None:
    _cache.CACHE_STATS.hits += 1
    return cached, cached.validate_pass
  _cache.CACHE_STATS.misses += 1
  msgs = list(load_messages(seg))
  validate_pass = validate_segment(msgs)
  meta = read_segment_metadata_from_messages(msgs)
  events = extract_curve_events(msgs) if validate_pass else []
  model = meta.model if meta else None
  kp = meta.kp if meta else (1.0, 1.0, 1.0)
  if use_cache:
    _cache.write_cached_atomic(seg, model, kp, validate_pass, events)
  return _cache.CachedSegment(model=model, kp_triple=kp,
                               validate_pass=validate_pass, events=events), validate_pass


# -----------------------------------------------------------------------------
# Iteration record + state mutation
# -----------------------------------------------------------------------------

def _stat_to_dict(stat: Any) -> dict[str, Any] | None:
  if stat is None:
    return None
  return dataclasses.asdict(stat)


def _band_stats_to_dict(b: BandStats) -> dict[str, Any]:
  return {
    "low": _stat_to_dict(b.low),
    "mid": _stat_to_dict(b.mid),
    "high": _stat_to_dict(b.high),
    "spread_warning": b.spread_warning,
    "excluded_low_speed_count": b.excluded_low_speed_count,
  }


def _recommendation_to_dict(rec: TuningRecommendation) -> dict[str, Any]:
  return {
    "triple": list(rec.triple),
    "per_band_verdict": dict(rec.per_band_verdict),
    "rationale": rec.rationale,
    "oscillation_warning": rec.oscillation_warning,
    "base_knob_warning_text": rec.base_knob_warning_text,
    "hold": rec.hold,
  }


def append_iteration(
  state: dict[str, Any],
  applied_triple: tuple[float, float, float],
  band_stats: BandStats,
  recommendation: TuningRecommendation,
  *,
  ts: str | None = None,
  bucket_id: str | None = None,
  model: str | None = None,
) -> dict[str, Any]:
  """Append a new iteration record into the bucket identified by `bucket_id`.

  When `bucket_id` is None, derives it from `(model, applied_triple)`. Promotes
  the previously-applied triple to `last_known_safe` for THIS bucket when the
  previous iteration produced an actionable recommendation.
  """
  ts = ts or dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
  if bucket_id is None:
    bucket_id = bucket_id_for(model, applied_triple)

  buckets = state.setdefault("buckets", {})
  bucket = buckets.setdefault(bucket_id, _empty_bucket(model, applied_triple))

  iterations = bucket["iterations"]
  if iterations:
    prev = iterations[-1]
    prev_rec = prev.get("recommendation", {})
    if not prev_rec.get("hold", True):
      bucket["last_known_safe"] = list(prev["applied_triple"])
  elif bucket.get("last_known_safe") is None:
    # First iteration in this bucket — seed last_known_safe to the applied triple.
    bucket["last_known_safe"] = list(applied_triple)

  iterations.append({
    "ts": ts,
    "applied_triple": list(applied_triple),
    "per_band_stats": _band_stats_to_dict(band_stats),
    "recommendation": _recommendation_to_dict(recommendation),
  })
  state["schema_version"] = SCHEMA_VERSION
  return state


# -----------------------------------------------------------------------------
# HTML report
# -----------------------------------------------------------------------------

def _fmt(value: Any) -> str:
  if value is None:
    return "n/a"
  if isinstance(value, float):
    return "n/a" if math.isnan(value) else f"{value:.3f}"
  return str(value)


_HTML_HEAD = "".join([
  "<!doctype html>\n<html><head><meta charset='utf-8'><title>kp_tuner report</title>",
  "<style>",
  "body{font-family:system-ui,sans-serif;max-width:880px;margin:2em auto;padding:0 1em}",
  "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:4px 8px}",
  ".callout{background:#f0f4ff;border:1px solid #ccd;padding:1em;margin:1em 0;font-size:1.05em}",
  ".banner{padding:0.6em 1em;margin:0.6em 0;border-radius:4px}",
  ".banner.warn{background:#fff5f0;border:1px solid #f4a583}",
  ".banner.info{background:#f0f7fc;border:1px solid #bcd5e6}",
  "</style></head><body>",
  "<h1>kp_tuner report</h1>",
])


def _render_bucket_section(
  current_triple: tuple[float, float, float],
  recommendation: TuningRecommendation,
  band_stats: BandStats,
  excluded_low_speed: int,
  prev_triple: tuple[float, float, float],
  *,
  heading: str | None = None,
) -> str:
  """Render ONE bucket's section (apply line + banners + per-band table)."""
  new_triple = recommendation.triple
  delta = tuple(n - c for n, c in zip(new_triple, current_triple, strict=False))

  def _row(name: str, stat: Any, verdict: str) -> str:
    if stat is None:
      return f"<tr><td>{name}</td><td colspan=7>no data</td><td>{verdict}</td></tr>"
    parts = [
      f"<tr><td>{name}</td>",
      f"<td>{stat['count']}</td>",
      f"<td>{stat.get('undershoot_count', 0)}/{stat.get('oversteer_count', 0)}</td>",
      f"<td>{_fmt(stat['median_ratio'])}</td>",
      f"<td>{_fmt(stat['median_lag'])}</td>",
      f"<td>{_fmt(stat['p25_ratio'])}</td>",
      f"<td>{_fmt(stat['p75_ratio'])}</td>",
      f"<td>{_fmt(stat['vEgo_min'])}–{_fmt(stat['vEgo_max'])}</td>",
      f"<td>{verdict}</td></tr>",
    ]
    return "".join(parts)

  bs = _band_stats_to_dict(band_stats)
  rows = "\n".join([
    _row("low", bs["low"], html.escape(recommendation.per_band_verdict.get("low", "—"))),
    _row("mid", bs["mid"], html.escape(recommendation.per_band_verdict.get("mid", "—"))),
    _row("high", bs["high"], html.escape(recommendation.per_band_verdict.get("high", "—"))),
  ])

  banners: list[str] = []
  if recommendation.oscillation_warning:
    warn_text = html.escape(recommendation.base_knob_warning_text or "")
    banners.append(
      f'<div class="banner warn"><b>Oscillation guard fired.</b> {warn_text}</div>'
    )
  if excluded_low_speed:
    banners.append(
      f'<div class="banner info">Excluded {excluded_low_speed} events with vEgo &lt; '
      + f'{MIN_VEGO_FOR_SOLVER} m/s from solver (base KP dominant).</div>'
    )
  if band_stats.spread_warning:
    banners.append(
      '<div class="banner info">Low-band events span a wide vEgo range; '
      + 'multiplier response may be base-KP-limited at the lower end.</div>'
    )

  apply_line = (
    f"<b>Apply:</b> low={new_triple[0]:.2f} (Δ{delta[0]:+.2f}), "
    + f"mid={new_triple[1]:.2f} (Δ{delta[1]:+.2f}), "
    + f"high={new_triple[2]:.2f} (Δ{delta[2]:+.2f}) | "
    + f"<b>Previous:</b> {prev_triple[0]:.2f}, {prev_triple[1]:.2f}, {prev_triple[2]:.2f} — "
    + "to revert: rerun with --revert"
  )

  pieces = []
  if heading:
    pieces.append(f"<h2>{html.escape(heading)}</h2>")
  pieces.extend([
    f"<div class='callout'>{apply_line}</div>",
    "\n".join(banners),
    "<table><thead><tr><th>band</th><th>count</th><th>under/over</th><th>median r</th><th>median lag s</th>",
    "<th>p25 r</th><th>p75 r</th><th>vEgo range</th><th>verdict</th></tr></thead><tbody>",
    rows,
    "</tbody></table>",
    f"<p><b>Rationale:</b> {html.escape(recommendation.rationale)}</p>",
  ])
  return "".join(pieces)


def render_html(
  current_triple: tuple[float, float, float],
  recommendation: TuningRecommendation,
  band_stats: BandStats,
  state: dict[str, Any],  # legacy single-bucket signature; v3 buckets ignored here
  excluded_low_speed: int,
) -> str:
  """Single-bucket HTML render — kept for back-compat with existing tests."""
  prev_triple = current_triple
  buckets = state.get("buckets") or {}
  if buckets:
    bucket = next(iter(buckets.values()))
    iterations = bucket.get("iterations") or []
    if len(iterations) >= 2:
      prev_triple = tuple(iterations[-2]["applied_triple"])  # type: ignore[assignment]
  return (
    _HTML_HEAD
    + _render_bucket_section(
      current_triple, recommendation, band_stats, excluded_low_speed, prev_triple,
    )
    + "</body></html>"
  )


def render_combined_html(
  bucket_results: list[dict[str, Any]],
  state: dict[str, Any],
  *,
  joint_results: list[dict[str, Any]] | None = None,
) -> str:
  """Combined HTML with one section per bucket plus optional per-model joint
  sections. Bucket results come from `_do_full_run`'s per-bucket loop. Joint
  results (when `--joint-per-model` is set) are appended after but NEVER
  written to state.json — they're advisory only.
  """
  buckets = state.get("buckets") or {}
  sections: list[str] = []
  for r in bucket_results:
    model = r["model"] or "unknown"
    kp = r["kp"]
    iterations = (buckets.get(r["bucket_id"]) or {}).get("iterations") or []
    prev_triple = (
      tuple(iterations[-2]["applied_triple"])  # type: ignore[arg-type]
      if len(iterations) >= 2 else kp
    )
    heading = f"MODEL: {model}  Kp=({kp[0]:.4f}, {kp[1]:.4f}, {kp[2]:.4f})  ({r['events_total']} events)"
    sections.append(
      _render_bucket_section(kp, r["rec"], r["bs"], r["excluded"], prev_triple, heading=heading)
    )

  if joint_results:
    sections.append("<hr><h2>Joint per-model recommendations (advisory only — not applied)</h2>")
    for j in joint_results:
      model = j["model"] or "unknown"
      rec = j["rec"]
      verdict = ", ".join(f"{b}: {v}" for b, v in rec.per_band_verdict.items())
      sections.append(
        f"<h2>JOINT: {html.escape(model)}</h2>"
        + "<div class='callout'>Advisory triple: "
        + f"low={rec.triple[0]:.4f}, mid={rec.triple[1]:.4f}, high={rec.triple[2]:.4f} "
        + f"&middot; {j['n_events']} events</div>"
        + f"<p><b>Verdict:</b> {html.escape(verdict)}</p>"
        + f"<p><b>Rationale:</b> {html.escape(rec.rationale)}</p>"
      )
  return _HTML_HEAD + "\n".join(sections) + "</body></html>"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    prog="python -m openpilot.tools.sunnypilot.kp_tuner.report",
    description="Recommend lateral-Kp multiplier triple updates from sunnypilot rlogs.",
  )
  p.add_argument("--log-dir", type=Path, help="root directory of mirrored device logs")
  p.add_argument("--current", type=_parse_triple,
                 help="currently-applied triple LOW,MID,HIGH (auto-detected from initData.params if omitted)")
  p.add_argument("--out-html", type=Path, help="path to write the HTML report")
  p.add_argument("--revert", action="store_true",
                 help="print last_known_safe triple and exit without mutating state")
  p.add_argument("--min-events-per-band", type=int, default=MIN_EVENTS_DEFAULT,
                 help="solver requires this many post-filter events or returns hold")
  p.add_argument("--no-cache", action="store_true",
                 help="skip per-segment cache reads/writes; force full re-decode")
  p.add_argument("--clear-cache", action="store_true",
                 help="wipe the segment cache directory and exit 0")
  p.add_argument("--bucket-filter",
                 help="comma-separated model names to restrict per-bucket analysis / --revert")
  p.add_argument("--joint-per-model", action="store_true",
                 help="advisory: emit a per-model joint recommendation across all of that model's buckets")
  p.add_argument("--state-path", type=Path, default=STATE_PATH,
                 help=argparse.SUPPRESS)
  p.add_argument("-v", "--verbose", action="store_true")
  return p


def _do_revert(state_path: Path, bucket_filter: list[str] | None = None) -> int:
  state = load_state(state_path)
  buckets: dict[str, Any] = state.get("buckets") or {}
  if not buckets:
    msg = (
      "kp_tuner: no last_known_safe persisted in state; cannot revert. "
      + f"State path: {state_path}"
    )
    print(msg, file=sys.stderr)
    return EXIT_NO_STATE

  if bucket_filter:
    matching = {bid: b for bid, b in buckets.items() if (b.get("model") or "unknown") in bucket_filter}
    if not matching:
      print(
        f"kp_tuner: --bucket-filter {bucket_filter!r} matched zero buckets in {state_path}",
        file=sys.stderr,
      )
      return EXIT_USAGE
    buckets = matching

  if len(buckets) > 1:
    print("kp_tuner: state contains multiple buckets; pass --bucket-filter MODEL to disambiguate.",
          file=sys.stderr)
    print("Available buckets:", file=sys.stderr)
    for bid, b in sorted(buckets.items()):
      print(f"  {bid} (model={b.get('model')!r})", file=sys.stderr)
    return EXIT_USAGE

  bid, bucket = next(iter(buckets.items()))
  lks = bucket.get("last_known_safe")
  if lks is None:
    print(
      f"kp_tuner: no last_known_safe persisted in bucket {bid}; cannot revert.",
      file=sys.stderr,
    )
    return EXIT_NO_STATE
  iter_n = len(bucket.get("iterations") or [])
  msg = (
    f"Set sliders to: low={lks[0]:.4f}, mid={lks[1]:.4f}, high={lks[2]:.4f} "
    + f"(last_known_safe from bucket {bid}, iteration {iter_n})"
  )
  print(msg)
  return EXIT_OK


def _band_dir(band: Any) -> str:
  return f"{band.undershoot_count} under/{band.oversteer_count} over" if band else "0 under/0 over"


def _partition_segments_by_bucket(
  log_dir: Path,
  *,
  use_cache: bool,
) -> tuple[dict[tuple[str | None, tuple[float, float, float]], list[CurveEvent]],
           list[Any], list[Any]]:
  """Walk segments via the cache. Returns (events_by_bucket, init_datas,
  car_params_sps). The init_datas / car_params_sps lists are scanned across
  ALL segments (not just first) so NN-FF detection covers every bucket.

  When `use_cache` is True, cached segments contribute their events directly;
  init_data/carParamsSP come from a single full-walk per cache miss path. For
  cache HITS we don't have init_data objects, so we record per-bucket Kp/model
  and rely on a separate light walk (load_messages once) for NN-FF validation
  ONLY on the first segment of each unique (model, kp) bucket — not all 285.
  """
  events_by_bucket: dict[tuple[str | None, tuple[float, float, float]], list[CurveEvent]] = {}
  init_datas: list[Any] = []
  car_params_sps: list[Any] = []
  seen_buckets_for_nnff: set[tuple[str | None, tuple[float, float, float]]] = set()

  for seg in iter_segments(log_dir):
    cached, valid = _segment_record(seg, use_cache=use_cache)
    bucket_key = (cached.model, cached.kp_triple)
    events_by_bucket.setdefault(bucket_key, [])
    if valid:
      events_by_bucket[bucket_key].extend(cached.events)
    # NN-FF check: probe at least one segment per bucket. Cache hits don't carry
    # init_data, so re-walk just that one segment to extract it (cheap relative
    # to extract_curve_events which is the expensive part).
    if bucket_key not in seen_buckets_for_nnff:
      seen_buckets_for_nnff.add(bucket_key)
      try:
        msgs = list(load_messages(seg))
        meta = read_segment_metadata_from_messages(msgs)
      except Exception:
        meta = None
      if meta is not None:
        init_datas.append(meta.init_data)
        car_params_sps.append(meta.car_params_sp)
  return events_by_bucket, init_datas, car_params_sps


def _do_full_run(args: argparse.Namespace, *, bucket_filter: list[str] | None = None) -> int:
  if args.log_dir is None:
    print("kp_tuner: --log-dir is required (unless --revert)", file=sys.stderr)
    return EXIT_USAGE
  if not args.log_dir.exists():
    print(f"kp_tuner: log dir does not exist: {args.log_dir}", file=sys.stderr)
    return EXIT_USAGE

  _cache.CACHE_STATS.reset()
  events_by_bucket, init_datas, car_params_sps = _partition_segments_by_bucket(
    args.log_dir, use_cache=not args.no_cache,
  )
  log.info(
    "kp_tuner: cache stats — hits=%d misses=%d writes=%d",
    _cache.CACHE_STATS.hits, _cache.CACHE_STATS.misses, _cache.CACHE_STATS.writes,
  )

  if not events_by_bucket:
    print("kp_tuner: no usable segments found in log dir", file=sys.stderr)
    return EXIT_USAGE

  # NN-FF: scan ALL segment metadata; abort if ANY segment has NN-FF active.
  for init_data, car_params_sp in zip(init_datas, car_params_sps, strict=False):
    if detect_nn_ff_active(init_data, car_params_sp):
      msg = (
        "kp_tuner: NN feedforward is active in these logs "
        + "(NeuralNetworkLateralControl param=on AND a non-mock model is bound); "
        + "this tool's scope explicitly assumes NN FF off. Refusing to recommend "
        + "Kp changes — multipliers would be trimming a NN residual, not driving "
        + "steady-state torque. Disable NN FF or use a different tuning workflow."
      )
      print(msg, file=sys.stderr)
      return EXIT_NN_FF_ACTIVE
  log.info("kp_tuner: NN FF dual-check passed (proceeding with torque-PID assumption)")

  # Apply --bucket-filter (model name based).
  if bucket_filter:
    matching = {k: v for k, v in events_by_bucket.items()
                if (k[0] or "unknown") in bucket_filter}
    if not matching:
      print(
        f"kp_tuner: --bucket-filter {bucket_filter!r} matched zero buckets; "
        + f"available models: {sorted({k[0] or 'unknown' for k in events_by_bucket})}",
        file=sys.stderr,
      )
      return EXIT_USAGE
    events_by_bucket = matching

  state = load_state(args.state_path)
  bucket_results: list[dict[str, Any]] = []
  for bucket_key, events in sorted(events_by_bucket.items(),
                                    key=lambda x: (x[0][0] or "", x[0][1])):
    model, kp = bucket_key
    excluded = sum(1 for e in events if e.vEgo_t0 < MIN_VEGO_FOR_SOLVER)
    bs = summarize(partition_by_band(events))
    bs.excluded_low_speed_count = excluded

    rec = recommend(kp, events, bs, min_events=args.min_events_per_band)
    bid = bucket_id_for(model, kp)
    append_iteration(state, kp, bs, rec, bucket_id=bid, model=model)
    bucket_results.append({
      "bucket_id": bid,
      "model": model,
      "kp": kp,
      "bs": bs,
      "rec": rec,
      "events_total": len(events),
      "excluded": excluded,
    })

    print("=" * 78)
    print(f"MODEL: {model or 'unknown'}  Kp=({kp[0]:.4f}, {kp[1]:.4f}, {kp[2]:.4f})")
    print(f"events: total={len(events)} (low={_band_dir(bs.low)}, "
          + f"mid={_band_dir(bs.mid)}, high={_band_dir(bs.high)})")
    print(f"Apply: low={rec.triple[0]:.4f}, mid={rec.triple[1]:.4f}, high={rec.triple[2]:.4f}")
    print(f"Rationale: {rec.rationale}")
    if rec.oscillation_warning:
      print(f"WARNING: {rec.base_knob_warning_text}")

  save_state_atomic(state, args.state_path)

  # Joint per-model solver — advisory output, never written to state.json.
  joint_results: list[dict[str, Any]] = []
  if getattr(args, "joint_per_model", False):
    by_model: dict[str | None, list[tuple[CurveEvent, tuple[float, float, float]]]] = {}
    for (model, kp), events in events_by_bucket.items():
      by_model.setdefault(model, []).extend((e, kp) for e in events)
    for model, events_with_kp in sorted(by_model.items(), key=lambda x: x[0] or ""):
      joint_rec = recommend_joint(events_with_kp, min_events=args.min_events_per_band)
      joint_results.append({"model": model, "rec": joint_rec, "n_events": len(events_with_kp)})
      print("=" * 78)
      print(f"JOINT: {model or 'unknown'}  ({len(events_with_kp)} events across all kp configs)")
      print(f"Advisory triple: low={joint_rec.triple[0]:.4f}, "
            + f"mid={joint_rec.triple[1]:.4f}, high={joint_rec.triple[2]:.4f}")
      print(f"Rationale: {joint_rec.rationale}")

  if args.out_html is not None:
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(
      render_combined_html(bucket_results, state, joint_results=joint_results),
      encoding="utf-8",
    )
    print(f"kp_tuner: wrote {args.out_html}")

  return EXIT_OK


def main(argv: list[str] | None = None) -> int:
  parser = _build_parser()
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                      format="%(message)s")
  if args.clear_cache:
    n = _cache.clear_cache()
    print(f"kp_tuner: cleared {n} cached segment(s)")
    return EXIT_OK
  bucket_filter = (
    [m.strip() for m in args.bucket_filter.split(",") if m.strip()]
    if args.bucket_filter else None
  )
  if args.revert:
    return _do_revert(args.state_path, bucket_filter=bucket_filter)
  return _do_full_run(args, bucket_filter=bucket_filter)


if __name__ == "__main__":
  raise SystemExit(main())
