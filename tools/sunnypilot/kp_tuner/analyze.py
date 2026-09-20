"""Curve-entry event extraction from sunnypilot rlogs.

All public functions accept duck-typed dataclasses (`Message` / etc. defined
below) so unit tests can construct synthetic streams without rlog binaries.
The capnp-decoding helpers (`iter_segments`, `load_messages`) live in this
module too but are intentionally separated from `extract_curve_events` so the
solver-side tests remain capnp-free.

Plan reference: .omc/plans/rivian-kp-tuning.md, Step 2.
"""
from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# NOTE: KP base schedule and breakpoints duplicated from
# selfdrive/controls/lib/latcontrol_torque.py:26-42. We deliberately do NOT
# import from selfdrive/* — this tool must not couple to the runtime package
# tree. If the controller's base schedule changes, update both places.
KP = 0.8
KI = 0.15
INTERP_SPEEDS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 30.0)
KP_INTERP_VALUES: tuple[float, ...] = (250.0, 120.0, 65.0, 30.0, 11.5, 5.5, 3.5, 2.0, KP)
KP_UI_SPEED_BREAKPOINTS: tuple[float, float, float] = (6.7, 15.6, 33.5)
KP_UI_MIN, KP_UI_MAX = 0.1, 5.0
# Mirrors selfdrive/controls/lib/latcontrol_torque.py:40 — duplicated for tool
# decoupling. Keep ordered low/mid/high.
KP_UI_PARAMS: tuple[str, str, str] = ("KpLowSpeed", "KpMidSpeed", "KpHighSpeed")

# Curve-entry detector thresholds (lateral acceleration, m/s^2).
LAT_ACCEL_LOW = 0.3
LAT_ACCEL_HIGH = 1.0
RAMP_TIMEOUT_S = 2.0  # ramp must reach high within this window
EVENT_PADDING_S = 0.5  # +/- around the trigger
MIN_VEGO_FOR_SOLVER = 3.0  # base KP saturates below this
LOW_BAND_SPREAD_WARN_M_S = 4.0
EXPECTED_TORQUE_STATE_VERSION = 1

# Single source of truth for log timestamp scaling — rlogs use nanoseconds.
NS_PER_S = 1_000_000_000

log = logging.getLogger(__name__)


def kp_interp_base(v_ego: float) -> float:
  """Mirror of `np.interp(v_ego, INTERP_SPEEDS, KP_INTERP_VALUES)`."""
  return float(np.interp(v_ego, INTERP_SPEEDS, KP_INTERP_VALUES))


def current_kp_working(v_ego: float, triple: tuple[float, float, float]) -> float:
  """Mirror of `latcontrol_torque.py:125`: `np.interp(vEgo, anchors, triple) * KP_INTERP(vEgo)`.

  Used by the recommender to back out target multipliers from observed undershoot.
  """
  multiplier = float(np.interp(v_ego, KP_UI_SPEED_BREAKPOINTS, triple))
  return multiplier * kp_interp_base(v_ego)


def interp_weights(v_ego: float) -> tuple[float, float, float]:
  """np.interp linear-interpolation weights for the three multiplier anchors at v_ego.

  Returns (w_low, w_mid, w_high) summing to 1.0. Clamped at the endpoints.
  Plan formulas at lines 136-139.
  """
  low_anchor, mid_anchor, high_anchor = KP_UI_SPEED_BREAKPOINTS
  if v_ego <= low_anchor:
    return (1.0, 0.0, 0.0)
  if v_ego <= mid_anchor:
    w_low = (mid_anchor - v_ego) / (mid_anchor - low_anchor)
    return (w_low, 1.0 - w_low, 0.0)
  if v_ego <= high_anchor:
    w_mid = (high_anchor - v_ego) / (high_anchor - mid_anchor)
    return (0.0, w_mid, 1.0 - w_mid)
  return (0.0, 0.0, 1.0)


# Duck-typed message dataclasses. Production capnp -> dataclass adaption lives
# in `load_messages`; tests construct these inline.

@dataclass
class TorqueState:
  active: bool = True
  saturated: bool = False
  version: int = EXPECTED_TORQUE_STATE_VERSION
  p: float = 0.0
  error: float = 0.0
  errorRate: float = 0.0


@dataclass
class ControlsStateMsg:
  desiredCurvature: float = 0.0
  curvature: float = 0.0
  torqueState: TorqueState = field(default_factory=TorqueState)


@dataclass
class CarStateMsg:
  vEgo: float = 0.0
  steeringPressed: bool = False
  steeringDisengage: bool = False


@dataclass
class CarControlMsg:
  actualCurvature: float = 0.0  # placeholder, not used by extractor; lives for symmetry
  curvature: float = 0.0


@dataclass
class SelfdriveStateMsg:
  enabled: bool = True


@dataclass
class Message:
  """Duck-typed cereal event. `which` selects the union arm; only one of the
  payload fields is populated per message, mirroring capnp semantics."""
  which: str
  logMonoTime: int  # nanoseconds, monotonic
  controlsState: ControlsStateMsg | None = None
  carState: CarStateMsg | None = None
  carControl: CarControlMsg | None = None
  selfdriveState: SelfdriveStateMsg | None = None
  initData: Any = None
  carParamsSP: Any = None


# Helper for tests: factory that builds a Message in one of the canonical shapes.
def make_msg(which: str, t_seconds: float, **kwargs: Any) -> Message:
  t_ns = int(t_seconds * NS_PER_S)
  payloads: dict[str, Any] = {}
  if which == "controlsState":
    payloads["controlsState"] = kwargs.pop("payload", None) or ControlsStateMsg(**kwargs)
  elif which == "carState":
    payloads["carState"] = kwargs.pop("payload", None) or CarStateMsg(**kwargs)
  elif which == "carControl":
    payloads["carControl"] = kwargs.pop("payload", None) or CarControlMsg(**kwargs)
  elif which == "selfdriveState":
    payloads["selfdriveState"] = kwargs.pop("payload", None) or SelfdriveStateMsg(**kwargs)
  else:
    raise ValueError(f"unknown which={which}")
  return Message(which=which, logMonoTime=t_ns, **payloads)


@dataclass
class CurveEvent:
  t0: float  # seconds (event window start)
  t_peak: float  # seconds
  vEgo_t0: float
  vEgo_peak: float
  peak_desired_curvature: float
  peak_actual_curvature: float
  tracking_ratio: float  # peak_actual / peak_desired; r<1 = undershoot, r>1 = oversteer
  lag_seconds: float
  band: str  # "low" | "mid" | "high"
  # Diagnostics — useful for HTML report.
  trace_p: list[float] = field(default_factory=list)
  trace_error: list[float] = field(default_factory=list)
  trace_error_rate: list[float] = field(default_factory=list)


@dataclass
class BandStat:
  count: int
  median_ratio: float
  median_lag: float
  p25_ratio: float
  p75_ratio: float
  vEgo_min: float
  vEgo_max: float
  undershoot_count: int = 0  # ratio < 1.0
  oversteer_count: int = 0   # ratio > 1.0


@dataclass
class BandStats:
  low: BandStat | None
  mid: BandStat | None
  high: BandStat | None
  spread_warning: bool = False
  excluded_low_speed_count: int = 0


# -----------------------------------------------------------------------------
# Banding
# -----------------------------------------------------------------------------

def band_for_vego(v_ego: float) -> str:
  low_anchor, mid_anchor, _ = KP_UI_SPEED_BREAKPOINTS
  if v_ego <= low_anchor:
    return "low"
  if v_ego <= mid_anchor:
    return "mid"
  return "high"


# -----------------------------------------------------------------------------
# NN-FF detection (dual-check, mirrors nnlc.py:36-38)
# -----------------------------------------------------------------------------

# Mirror of MOCK_MODEL_PATH placeholder used at runtime. We compare the BASENAME
# rather than the full path, because rlogs may have come from a device with a
# different BASEDIR. See sunnypilot/selfdrive/controls/lib/nnlc/helpers.py:16.
MOCK_MODEL_PATH_BASENAME = "MOCK.json"


def _nn_param_on(init_data: Any) -> bool:
  """`Params().get_bool("NeuralNetworkLateralControl")` semantics — on iff bytes literal `b"1"`.

  Mirrors `Params.get_bool` (utils): truthy only when the stored bytes are exactly `b"1"`.
  """
  flat = _params_dict(init_data)
  if flat is None:
    return False
  raw = flat.get("NeuralNetworkLateralControl")
  if raw is None:
    return False
  if isinstance(raw, str):
    raw = raw.encode()
  return raw == b"1"


def _nn_model_bound(car_params_sp: Any) -> bool:
  """True iff `CP_SP.neuralNetworkLateralControl.model.path` points to a real model.

  Mirrors the runtime check `path != MOCK_MODEL_PATH` at nnlc.py:39.
  """
  if car_params_sp is None:
    return False
  nnlc = getattr(car_params_sp, "neuralNetworkLateralControl", None)
  if nnlc is None:
    return False
  model = getattr(nnlc, "model", None)
  if model is None:
    return False
  path = getattr(model, "path", "") or ""
  if not path:
    return False
  # Compare basename so cross-machine paths still detect the mock placeholder.
  return Path(path).name != MOCK_MODEL_PATH_BASENAME


def _params_dict(init_data: Any) -> dict[str, bytes] | None:
  """Return `init_data.params` flattened to a `{str: bytes}` dict, or None.

  Capnp emits the field as a Map struct whose `.entries` is a list of
  `{key, value}` records; accessing `.get` directly raises. Tests pass an
  already-flattened dict via duck typing — handle both.
  """
  if init_data is None:
    return None
  params = getattr(init_data, "params", None)
  if params is None:
    return None
  if hasattr(params, "get") and not hasattr(params, "entries"):
    return params  # already a dict (tests)
  entries = getattr(params, "entries", None)
  if entries is None:
    return None
  out: dict[str, bytes] = {}
  for e in entries:
    out[e.key] = e.value
  return out


def read_kp_triple_from_init_data(init_data: Any) -> tuple[float, float, float] | None:
  """Mirror of `latcontrol_torque._load_kp_multipliers`: parse + clamp, default 1.0
  on missing/non-numeric, clamp to [KP_UI_MIN, KP_UI_MAX]. Returns None when
  `init_data.params` is absent so the CLI can distinguish "no log metadata" from
  "Kp params not yet written" (the latter is the all-1.0 default at first boot).
  """
  flat = _params_dict(init_data)
  if flat is None:
    return None

  def _read(key: str) -> float:
    raw = flat.get(key) if hasattr(flat, "get") else None
    if isinstance(raw, (bytes, bytearray)):
      try:
        raw = raw.decode("utf-8")
      except UnicodeDecodeError:
        return 1.0
    try:
      val = float(raw) if raw is not None else 1.0
    except (TypeError, ValueError):
      val = 1.0
    return max(KP_UI_MIN, min(KP_UI_MAX, val))

  return _read(KP_UI_PARAMS[0]), _read(KP_UI_PARAMS[1]), _read(KP_UI_PARAMS[2])


@dataclass
class SegmentMetadata:
  """Per-segment metadata snapshot from initData.params and carParamsSP."""
  model: str | None        # internalName from ModelManager_ActiveBundle, e.g. "OPM10V3"
  generation: int | None
  kp: tuple[float, float, float]  # always populated; defaults to (1.0, 1.0, 1.0) if absent
  init_data: Any
  car_params_sp: Any


def read_segment_metadata_from_messages(messages: Iterable[Message]) -> SegmentMetadata | None:
  """Extract (model, generation, kp) from already-loaded messages. Returns None
  on Exception (corrupt log) or when no initData is present.
  """
  try:
    init_data: Any = None
    car_params_sp: Any = None
    for m in messages:
      if m.which == "initData" and init_data is None:
        init_data = m.initData
      elif m.which == "carParamsSP" and car_params_sp is None:
        car_params_sp = m.carParamsSP
      if init_data is not None and car_params_sp is not None:
        break
    if init_data is None:
      return None
    kp = read_kp_triple_from_init_data(init_data) or (1.0, 1.0, 1.0)
    flat = _params_dict(init_data) or {}
    bundle_raw = flat.get("ModelManager_ActiveBundle") if hasattr(flat, "get") else None
    model: str | None = None
    generation: int | None = None
    if isinstance(bundle_raw, (bytes, bytearray)):
      try:
        bundle = json.loads(bundle_raw.decode("utf-8"))
        model = bundle.get("internalName") or bundle.get("displayName")
        gen = bundle.get("generation")
        generation = int(gen) if gen is not None else None
      except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return SegmentMetadata(model=model, generation=generation, kp=kp,
                           init_data=init_data, car_params_sp=car_params_sp)
  except Exception:
    return None


def read_segment_metadata(rlog_path: Path) -> SegmentMetadata | None:
  """Path-based wrapper that walks the rlog and delegates to
  `read_segment_metadata_from_messages`. Returns None on any read failure.
  """
  try:
    return read_segment_metadata_from_messages(load_messages(rlog_path))
  except Exception:
    return None


def detect_nn_ff_active(init_data: Any, car_params_sp: Any) -> bool:
  """NN feedforward is active iff BOTH the param is on AND a non-mock model is bound.

  Mirrors `sunnypilot/selfdrive/controls/lib/nnlc/nnlc.py:36-38`:
    self.enabled       = Params().get_bool("NeuralNetworkLateralControl")
    self.has_nn_model  = CP_SP.neuralNetworkLateralControl.model.path != MOCK_MODEL_PATH
    self._nnlc_enabled = self.enabled and self.has_nn_model

  Returning True here means the report.py CLI must abort with exit 2 — multipliers
  would be trimming a NN residual rather than driving steady-state torque.
  """
  return _nn_param_on(init_data) and _nn_model_bound(car_params_sp)


# -----------------------------------------------------------------------------
# Segment file walking (capnp lives here, kept out of extract_curve_events)
# -----------------------------------------------------------------------------

def iter_segments(log_dir: Path) -> Iterator[Path]:
  """Yield rlog files under `log_dir` in deterministic order.

  Recognized extensions match `tools/lib/logreader.py`: `.zst`, `.bz2`, `.zstd`,
  bare `rlog`. We yield Path objects; the caller decides how to decode them.
  """
  if not log_dir.exists():
    return
  exts = {".bz2", ".zst", ".zstd"}
  for p in sorted(log_dir.rglob("rlog*")):
    if p.is_file() and (p.suffix in exts or p.name in {"rlog", "rlog.bz2", "rlog.zst"}):
      yield p


def load_messages(rlog_path: Path) -> Iterator[Message]:  # pragma: no cover - I/O wrapper
  """Decode an rlog into the duck-typed `Message` stream consumed by `extract_curve_events`.

  Imports `tools.lib.logreader` lazily so unit tests can avoid the capnp/cereal
  import chain entirely. Raises `RuntimeError` if `LogReader` is unavailable.
  """
  try:
    from openpilot.tools.lib.logreader import LogReader  # type: ignore[import-not-found]
  except Exception as e:
    raise RuntimeError(
      f"kp_tuner.load_messages: openpilot.tools.lib.logreader unavailable ({e}); " +
      "ensure the openpilot venv is active."
    ) from e

  for evt in LogReader(str(rlog_path)):
    which = evt.which()
    t_ns = int(evt.logMonoTime)
    if which == "controlsState":
      cs = evt.controlsState
      ts = cs.lateralControlState.torqueState if cs.lateralControlState.which() == "torqueState" else None
      yield Message(
        which="controlsState",
        logMonoTime=t_ns,
        controlsState=ControlsStateMsg(
          desiredCurvature=float(cs.desiredCurvature),
          curvature=float(cs.curvature),
          torqueState=TorqueState(
            active=bool(ts.active) if ts is not None else False,
            saturated=bool(ts.saturated) if ts is not None else False,
            version=int(ts.version) if ts is not None else 0,
            p=float(ts.p) if ts is not None else 0.0,
            error=float(ts.error) if ts is not None else 0.0,
            errorRate=float(ts.errorRate) if ts is not None else 0.0,
          ),
        ),
      )
    elif which == "carState":
      yield Message(
        which="carState",
        logMonoTime=t_ns,
        carState=CarStateMsg(
          vEgo=float(evt.carState.vEgo),
          steeringPressed=bool(evt.carState.steeringPressed),
          steeringDisengage=bool(evt.carState.steeringDisengage),
        ),
      )
    elif which == "selfdriveState":
      yield Message(
        which="selfdriveState",
        logMonoTime=t_ns,
        selfdriveState=SelfdriveStateMsg(enabled=bool(evt.selfdriveState.enabled)),
      )
    elif which == "initData":
      yield Message(which="initData", logMonoTime=t_ns, initData=evt.initData)
    elif which == "carParamsSP":
      yield Message(which="carParamsSP", logMonoTime=t_ns, carParamsSP=evt.carParamsSP)


# -----------------------------------------------------------------------------
# Segment-version validation (BLOCKING amendment 12)
# -----------------------------------------------------------------------------

def validate_segment(messages: Iterable[Message]) -> bool:
  """True if the first observed `torqueState.version` matches our expectation.

  Walks the iterator only until the first controlsState message — callers should
  pre-buffer or pass a list when they want to consume the messages afterwards.
  """
  for msg in messages:
    if msg.which == "controlsState" and msg.controlsState is not None:
      version = msg.controlsState.torqueState.version
      if version != EXPECTED_TORQUE_STATE_VERSION:
        log.warning(
          "kp_tuner: skipping segment — torqueState.version=%d incompatible with expected=%d",
          version, EXPECTED_TORQUE_STATE_VERSION,
        )
        return False
      return True
  # No controlsState observed — treat as valid-but-empty; downstream filters drop it.
  return True


# -----------------------------------------------------------------------------
# Curve-entry detector
# -----------------------------------------------------------------------------

class _DetectorState:
  """Three-state machine: BELOW_LOW -> RAMPING -> ABOVE_HIGH."""
  BELOW_LOW = 0
  RAMPING = 1
  ABOVE_HIGH = 2


@dataclass
class _Sample:
  t: float  # seconds
  v_ego: float
  desired_curvature: float
  actual_curvature: float
  selfdrive_enabled: bool
  torque_active: bool
  torque_saturated: bool
  steering_pressed: bool
  steering_disengage: bool
  p: float
  err: float
  err_rate: float


def _zip_samples(messages: Iterable[Message]) -> list[_Sample]:
  """Group messages by time using zero-order-hold on the most-recent of each type."""
  cs: ControlsStateMsg | None = None
  carstate: CarStateMsg | None = None
  sdrv: SelfdriveStateMsg | None = None

  samples: list[_Sample] = []
  for m in messages:
    if m.which == "controlsState" and m.controlsState is not None:
      cs = m.controlsState
    elif m.which == "carState" and m.carState is not None:
      carstate = m.carState
    elif m.which == "selfdriveState" and m.selfdriveState is not None:
      sdrv = m.selfdriveState
    else:
      continue
    if cs is None or carstate is None:
      continue  # need both before we can emit
    enabled = sdrv.enabled if sdrv is not None else False
    samples.append(_Sample(
      t=m.logMonoTime / NS_PER_S,
      v_ego=carstate.vEgo,
      desired_curvature=cs.desiredCurvature,
      actual_curvature=cs.curvature,
      selfdrive_enabled=enabled,
      torque_active=cs.torqueState.active,
      torque_saturated=cs.torqueState.saturated,
      steering_pressed=carstate.steeringPressed,
      steering_disengage=carstate.steeringDisengage,
      p=cs.torqueState.p,
      err=cs.torqueState.error,
      err_rate=cs.torqueState.errorRate,
    ))
  return samples


def _gates_pass(s: _Sample) -> bool:
  return (
    s.selfdrive_enabled
    and s.torque_active
    and not s.steering_pressed
    and not s.steering_disengage
    and not s.torque_saturated
  )


def _half_amplitude_crossing(times: list[float], values: list[float], peak_value: float) -> float | None:
  """Return time of first |value| crossing 0.5 * |peak_value|, else None."""
  if peak_value == 0.0:
    return None
  threshold = 0.5 * abs(peak_value)
  for t, v in zip(times, values, strict=False):
    if abs(v) >= threshold:
      return t
  return None


def extract_curve_events(messages: Iterable[Message]) -> list[CurveEvent]:
  """Identify curve-entry events in the message stream.

  A curve-entry event triggers when the magnitude of the lateral-acceleration
  signal `|desiredCurvature * vEgo^2|` rises from `< LAT_ACCEL_LOW` (0.3 m/s^2)
  to `> LAT_ACCEL_HIGH` (1.0 m/s^2) within `RAMP_TIMEOUT_S` seconds. The event
  window spans `[t_start - 0.5s, t_peak + 0.5s]` and every cycle in that window
  must pass the engagement gates.
  """
  samples = _zip_samples(messages)
  if not samples:
    return []

  events: list[CurveEvent] = []
  state = _DetectorState.BELOW_LOW
  ramp_start_idx: int | None = None

  for i, s in enumerate(samples):
    lat_accel = abs(s.desired_curvature * s.v_ego * s.v_ego)

    if state == _DetectorState.BELOW_LOW:
      if lat_accel >= LAT_ACCEL_LOW:
        # Rising past the lower edge — start RAMPING.
        state = _DetectorState.RAMPING
        ramp_start_idx = i
    elif state == _DetectorState.RAMPING:
      if ramp_start_idx is not None and (s.t - samples[ramp_start_idx].t) > RAMP_TIMEOUT_S:
        # Ramp did not complete in time — abort and re-arm.
        state = _DetectorState.BELOW_LOW
        ramp_start_idx = None
      elif lat_accel < LAT_ACCEL_LOW:
        # Fell back below low edge before ever crossing high — abort.
        state = _DetectorState.BELOW_LOW
        ramp_start_idx = None
      elif lat_accel > LAT_ACCEL_HIGH:
        if ramp_start_idx is None:
          state = _DetectorState.BELOW_LOW
          continue
        evt = _build_event(samples, ramp_start_idx, i)
        if evt is not None:
          events.append(evt)
        # Move to ABOVE_HIGH; we re-arm only when lat_accel falls back below
        # LAT_ACCEL_LOW, preventing double-emission during the hold/decay tail.
        state = _DetectorState.ABOVE_HIGH
        ramp_start_idx = None
    elif state == _DetectorState.ABOVE_HIGH:
      if lat_accel < LAT_ACCEL_LOW:
        state = _DetectorState.BELOW_LOW

  return events


def _build_event(samples: list[_Sample], ramp_start_idx: int, trigger_idx: int) -> CurveEvent | None:
  """Construct a CurveEvent from the ramp-start..trigger window plus padding.

  Returns None if any cycle inside the event window fails the engagement gates,
  if vEgo at t0 is zero, or if peak_desired_curvature is zero.
  """
  ramp_start_t = samples[ramp_start_idx].t
  trigger_t = samples[trigger_idx].t
  win_t0 = ramp_start_t - EVENT_PADDING_S
  # Tentative window end; refined once we find peak.
  win_t1 = trigger_t + EVENT_PADDING_S

  # Find peak within [ramp_start_t, trigger_t + EVENT_PADDING_S].
  peak_idx = ramp_start_idx
  peak_abs = abs(samples[ramp_start_idx].desired_curvature)
  i = ramp_start_idx
  while i < len(samples) and samples[i].t <= win_t1:
    a = abs(samples[i].desired_curvature)
    if a > peak_abs:
      peak_abs = a
      peak_idx = i
    i += 1
  t_peak = samples[peak_idx].t
  win_t1 = t_peak + EVENT_PADDING_S

  # Validate gates over the window.
  in_window = [s for s in samples if win_t0 <= s.t <= win_t1]
  if not in_window:
    return None
  for s in in_window:
    if not _gates_pass(s):
      return None

  v_ego_t0 = next((s.v_ego for s in in_window if s.t >= ramp_start_t), in_window[0].v_ego)
  if v_ego_t0 <= 0:
    return None
  peak_desired = samples[peak_idx].desired_curvature
  peak_actual = samples[peak_idx].actual_curvature
  if peak_desired == 0.0:
    return None
  ratio = abs(peak_actual) / abs(peak_desired)

  # Lag: time from each signal crossing 50% of ITS OWN peak amplitude.
  desired_50 = _half_amplitude_crossing(
    [s.t for s in in_window],
    [s.desired_curvature for s in in_window],
    peak_desired,
  )
  actual_50 = _half_amplitude_crossing(
    [s.t for s in in_window],
    [s.actual_curvature for s in in_window],
    peak_actual,
  )
  lag = (actual_50 - desired_50) if (desired_50 is not None and actual_50 is not None) else math.nan

  return CurveEvent(
    t0=ramp_start_t,
    t_peak=t_peak,
    vEgo_t0=v_ego_t0,
    vEgo_peak=samples[peak_idx].v_ego,
    peak_desired_curvature=peak_desired,
    peak_actual_curvature=peak_actual,
    tracking_ratio=ratio,
    lag_seconds=lag,
    band=band_for_vego(v_ego_t0),
    trace_p=[s.p for s in in_window],
    trace_error=[s.err for s in in_window],
    trace_error_rate=[s.err_rate for s in in_window],
  )


# -----------------------------------------------------------------------------
# Partition + summarize
# -----------------------------------------------------------------------------

def partition_by_band(events: list[CurveEvent]) -> dict[str, list[CurveEvent]]:
  partitioned: dict[str, list[CurveEvent]] = {"low": [], "mid": [], "high": []}
  for e in events:
    partitioned[e.band].append(e)
  return partitioned


def summarize(events_per_band: dict[str, list[CurveEvent]]) -> BandStats:
  def _stat(events: list[CurveEvent]) -> BandStat | None:
    if not events:
      return None
    ratios = np.array([e.tracking_ratio for e in events], dtype=float)
    lags = np.array([e.lag_seconds for e in events if not math.isnan(e.lag_seconds)], dtype=float)
    median_lag = float(np.median(lags)) if lags.size else math.nan
    vegos = [e.vEgo_t0 for e in events]
    return BandStat(
      count=len(events),
      median_ratio=float(np.median(ratios)),
      median_lag=median_lag,
      p25_ratio=float(np.percentile(ratios, 25)),
      p75_ratio=float(np.percentile(ratios, 75)),
      vEgo_min=min(vegos),
      vEgo_max=max(vegos),
      undershoot_count=int(np.sum(ratios < 1.0)),
      oversteer_count=int(np.sum(ratios > 1.0)),
    )

  low_stat = _stat(events_per_band.get("low", []))
  mid_stat = _stat(events_per_band.get("mid", []))
  high_stat = _stat(events_per_band.get("high", []))

  spread_warning = bool(
    low_stat is not None and (low_stat.vEgo_max - low_stat.vEgo_min) > LOW_BAND_SPREAD_WARN_M_S
  )
  return BandStats(
    low=low_stat,
    mid=mid_stat,
    high=high_stat,
    spread_warning=spread_warning,
    excluded_low_speed_count=0,  # populated by caller (recommend / report)
  )
