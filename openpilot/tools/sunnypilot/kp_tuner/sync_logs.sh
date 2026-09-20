#!/usr/bin/env bash
# Mirror device rlogs incrementally into the NAS archive at /Volumes/home/sunnypilot-logs/realdata/.
#
# Usage: sync_logs.sh [--dry-run] [--host HOST] [--remote-path PATH] [--local-path PATH]
#
# Defaults match the development setup documented in
# .omc/plans/rivian-kp-tuning.md (Step 1).
set -euo pipefail

REMOTE_HOST="comma@192.168.1.115"
REMOTE_PATH="/data/media/0/realdata/"
LOCAL_PATH="/Volumes/home/sunnypilot-logs/realdata/"
DRY_RUN=0

usage() {
  cat <<'EOF'
sync_logs.sh — incrementally mirror device rlogs into /Volumes/home/sunnypilot-logs/realdata/

  --dry-run            run rsync in dry-run mode (no writes)
  --host HOST          override remote host (default: comma@192.168.1.115)
  --remote-path PATH   override remote source path
  --local-path PATH    override local mirror path
  -h, --help           show this help and exit
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --host)
      REMOTE_HOST="$2"
      shift 2
      ;;
    --remote-path)
      REMOTE_PATH="$2"
      shift 2
      ;;
    --local-path)
      LOCAL_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "sync_logs.sh: unknown arg: $1" >&2
      usage
      exit 64
      ;;
  esac
done

# The default lives on an SMB share. If it is not mounted, say so: otherwise mkdir fails with a
# bare "Permission denied" under /Volumes, which points nowhere near the real problem.
if [[ "${LOCAL_PATH}" == /Volumes/* ]]; then
  SHARE_ROOT="/Volumes/$(echo "${LOCAL_PATH#/Volumes/}" | cut -d/ -f1)"
  if ! mount | grep -q " on ${SHARE_ROOT} "; then
    echo "sync_logs.sh: ${SHARE_ROOT} is not mounted -- mount the share or pass --local-path" >&2
    exit 69
  fi
fi

mkdir -p "${LOCAL_PATH}"

# --partial keeps incomplete transfers resumable. Use only flags supported by
# openrsync (the BSD rsync that ships with macOS, identifying as rsync 2.6.9):
# no --append-verify, no --info=progress2. rlogs are immutable post-rotation
# so verify-on-resume is redundant; --progress is good enough.
#
# Filter: descend into every segment dir but only transfer rlog files. The
# realdata/ tree is dominated by camera HEVCs (~60 GB of fcamera/ecamera/
# dcamera/qcamera) that kp_tuner doesn't read; pulling them just wastes time
# and disk. The --include='*/' line is required so rsync recurses; without it
# the --exclude='*' would block directory entries too.
RSYNC_FLAGS=(
  -avz --partial --progress --stats
  --include='*/'
  --include='rlog*'
  --exclude='*'
)
if [ "${DRY_RUN}" -eq 1 ]; then
  RSYNC_FLAGS+=(--dry-run)
fi

# Capture rsync stats so we can print a one-line summary at the end.
TMP_LOG="$(mktemp -t kp_tuner_sync_logs.XXXXXX)"
trap 'rm -f "${TMP_LOG}"' EXIT

set +e
rsync "${RSYNC_FLAGS[@]}" \
  "${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}" \
  | tee "${TMP_LOG}"
RSYNC_RC=${PIPESTATUS[0]}
set -e

if [ "${RSYNC_RC}" -ne 0 ]; then
  echo "sync_logs.sh: rsync exited with status ${RSYNC_RC}" >&2
  exit "${RSYNC_RC}"
fi

# Parse rsync --stats output for a tiny summary; fall back gracefully.
NEW_FILES="$(awk -F': ' '/Number of regular files transferred/ {gsub(",", "", $2); print $2}' "${TMP_LOG}" | head -n1)"
TOTAL_BYTES="$(awk -F': ' '/Total transferred file size/ {gsub(",", "", $2); print $2}' "${TMP_LOG}" | head -n1)"

if [ -n "${TOTAL_BYTES}" ]; then
  TOTAL_MB="$(awk -v b="${TOTAL_BYTES}" 'BEGIN { split(b, a, " "); printf("%.2f", a[1] / 1048576.0) }')"
else
  TOTAL_MB="0.00"
fi

echo
echo "sync_logs.sh: ${NEW_FILES:-0} new segments transferred, ${TOTAL_MB} MB total"
if [ "${DRY_RUN}" -eq 1 ]; then
  echo "sync_logs.sh: dry-run — no files were written"
fi
