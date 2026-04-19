#!/usr/bin/env python3
"""Open local rlog segments in PlotJuggler with a Rivian-tuning layout.

Example:
  python3 tools/rivian_juggle.py /tmp/rivian_logs/00000085--e6c53c4d17--* \
      --layout tools/plotjuggler/layouts/torque-controller.xml
"""
import argparse
import glob
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openpilot.tools.lib.logreader import LogReader, save_log
from openpilot.tools.plotjuggler.juggle import start_juggler
from openpilot.selfdrive.test.process_replay.migration import migrate_all


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("paths", nargs="+",
                  help="rlog.zst files or segment directories (glob-expanded).")
  ap.add_argument("--layout", default="tools/plotjuggler/layouts/torque-controller.xml")
  ap.add_argument("--no-migration", action="store_true")
  args = ap.parse_args()

  rlog_paths = []
  for p in args.paths:
    if os.path.isdir(p):
      c = os.path.join(p, "rlog.zst")
      if os.path.isfile(c):
        rlog_paths.append(c)
    elif p.endswith("rlog.zst") and os.path.isfile(p):
      rlog_paths.append(p)
    else:
      rlog_paths.extend(sorted(glob.glob(p)))
  rlog_paths = [p for p in rlog_paths if p.endswith("rlog.zst")]
  rlog_paths.sort(key=lambda x: int(os.path.basename(os.path.dirname(x)).rsplit("--", 1)[-1])
                  if os.path.basename(os.path.dirname(x)).rsplit("--", 1)[-1].isdigit() else 0)

  if not rlog_paths:
    sys.exit("no rlog.zst paths matched")
  print(f"Loading {len(rlog_paths)} segments:")
  for p in rlog_paths:
    print(f"  {p}")

  all_data = []
  for p in rlog_paths:
    for m in LogReader(p):
      if m.which() in ("can", "sendcan"):
        continue
      all_data.append(m)
  if not args.no_migration:
    all_data = migrate_all(all_data)

  juggle_dir = os.path.join(os.path.dirname(__file__), "plotjuggler")
  with tempfile.NamedTemporaryFile(suffix=".rlog", dir=juggle_dir, delete=False) as tmp:
    save_log(tmp.name, all_data, compress=False)
    del all_data
    print(f"Concatenated rlog: {tmp.name}  -> launching PlotJuggler")
    start_juggler(fn=tmp.name, layout=os.path.abspath(args.layout),
                  route_or_segment_name=os.path.basename(rlog_paths[0]),
                  platform="RIVIAN_R1T_GEN1")


if __name__ == "__main__":
  main()
