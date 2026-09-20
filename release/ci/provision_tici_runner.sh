#!/usr/bin/env bash
# Provision a GitHub Actions self-hosted runner on a comma 3X for build-prebuilt.yaml.
#
# Everything durable lives under /data/github. The rootfs pieces (user, sudoers, systemd unit) are
# wiped by every AGNOS update, so they are isolated in the `rootfs` and `unit-install` subcommands
# and are safe to re-run afterwards.
#
# The unit is deliberately named actions.runner.sunnypilot.<hostname>: that is the name
# system/manager/github_runner.sh looks for, so the device's own manager starts the runner when
# offroad + EnableGithubRunner + unmetered + on desk power (<9V), and stops it otherwise.
#
# NOTE: while EnableGithubRunner is on, github_runner.sh polls every second. Installing the unit
# therefore STARTS THE RUNNER IMMEDIATELY. `unit-install` is the go switch -- run it last.
set -euo pipefail

ROOT=/data/github
RUNNER_DIR=$ROOT/runner
RUNNER_HOME=$ROOT/home
BUILD_ROOT=$ROOT/openpilot
WRAPPER=$ROOT/run-isolated.sh
RUNNER_USER=github-runner
UNIT=actions.runner.sunnypilot.$(uname -n).service
UNIT_PATH=/etc/systemd/system/$UNIT
RUNNER_VERSION=${RUNNER_VERSION:-2.337.0}

rw() { sudo mount -o remount,rw /; }
ro() { sudo mount -o remount,ro / || echo "WARN: could not remount / read-only (busy); it reverts on reboot"; }

cmd_rootfs() {
  rw
  trap ro EXIT
  if ! id "$RUNNER_USER" &>/dev/null; then
    sudo mkdir -p "$ROOT"
    # gpu/video: the build compiles tinygrad models on the QCOM GPU (/dev/kgsl-3d0 is root:gpu 660)
    sudo useradd --system --create-home --home-dir "$RUNNER_HOME" --shell /bin/bash \
      --groups gpu,video "$RUNNER_USER"
  fi
  # The workflow itself calls sudo (mkdir/chown/find under /data, which is comma:comma 755).
  echo "$RUNNER_USER ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/90-github-runner >/dev/null
  sudo chmod 0440 /etc/sudoers.d/90-github-runner
  sudo visudo -cf /etc/sudoers.d/90-github-runner
  echo "rootfs: user + sudoers OK"
}

cmd_dirs() {
  sudo mkdir -p "$RUNNER_DIR" "$BUILD_ROOT" "$RUNNER_HOME"
  sudo chown -R "$RUNNER_USER:$RUNNER_USER" "$RUNNER_DIR" "$BUILD_ROOT" "$RUNNER_HOME"
  echo "dirs OK"
}

cmd_download() {
  local tgz=actions-runner-linux-arm64-$RUNNER_VERSION.tar.gz
  if [ ! -x "$RUNNER_DIR/config.sh" ]; then
    sudo -u "$RUNNER_USER" -H bash -c "cd '$RUNNER_DIR' && \
      curl -fsSL -o '$tgz' 'https://github.com/actions/runner/releases/download/v$RUNNER_VERSION/$tgz' && \
      tar xzf '$tgz' && rm -f '$tgz'"
  fi
  echo "runner $RUNNER_VERSION present"
}

cmd_wrapper() {
  # Runs as root inside `unshare -m`. Fails CLOSED: build-prebuilt.yaml runs
  # `sudo find /data/openpilot -mindepth 1 -delete`, so if this bind mount were ever not in effect
  # the job would wipe the live openpilot install. The sentinel is written through the real path
  # and must be visible through the mount point before the runner is allowed to start.
  sudo tee "$WRAPPER" >/dev/null <<EOF
#!/usr/bin/env bash
set -euo pipefail
BUILD_ROOT=$BUILD_ROOT
LIVE=/data/openpilot
SENTINEL=.runner-build-root

[ -d "\$BUILD_ROOT" ] || { echo "FATAL: \$BUILD_ROOT missing" >&2; exit 1; }
touch "\$BUILD_ROOT/\$SENTINEL"
[ ! -e "\$LIVE/\$SENTINEL" ] || { echo "FATAL: sentinel visible before bind mount -- refusing" >&2; exit 1; }

mount --bind "\$BUILD_ROOT" "\$LIVE"
mountpoint -q "\$LIVE"          || { echo "FATAL: \$LIVE is not a mount point" >&2; exit 1; }
[ -f "\$LIVE/\$SENTINEL" ]       || { echo "FATAL: bind mount not effective" >&2; exit 1; }

exec runuser -u $RUNNER_USER -- env HOME=$RUNNER_HOME "$RUNNER_DIR/run.sh"
EOF
  sudo chown root:root "$WRAPPER"
  sudo chmod 0755 "$WRAPPER"
  echo "wrapper OK"
}

cmd_register() {
  : "${RUNNER_TOKEN:?set RUNNER_TOKEN}"
  : "${RUNNER_URL:?set RUNNER_URL}"
  sudo -u "$RUNNER_USER" -H bash -c "cd '$RUNNER_DIR' && ./config.sh --unattended --replace \
    --url '$RUNNER_URL' --token '$RUNNER_TOKEN' --name '$(uname -n)' --labels tici --work _work"
}

cmd_unit_install() {
  rw
  trap ro EXIT
  # No [Install] section and never `systemctl enable`: the device's manager owns start/stop.
  # KillMode=control-group so going onroad kills an in-flight build along with its namespace.
  sudo tee "$UNIT_PATH" >/dev/null <<EOF
[Unit]
Description=GitHub Actions Runner (isolated build namespace)
After=network-online.target

[Service]
ExecStart=/usr/bin/unshare -m --propagation private $WRAPPER
KillMode=control-group
KillSignal=SIGTERM
TimeoutStopSec=30
Restart=no
EOF
  sudo systemctl daemon-reload
  echo "unit installed: $UNIT (manager will start it within ~1s if the toggle conditions hold)"
}

cmd_unit_remove() {
  rw
  trap ro EXIT
  sudo systemctl stop "$UNIT" 2>/dev/null || true
  sudo rm -f "$UNIT_PATH"
  sudo systemctl daemon-reload
  echo "unit removed"
}

case "${1:-}" in
  rootfs)       cmd_rootfs ;;
  dirs)         cmd_dirs ;;
  download)     cmd_download ;;
  wrapper)      cmd_wrapper ;;
  register)     cmd_register ;;
  unit-install) cmd_unit_install ;;
  unit-remove)  cmd_unit_remove ;;
  *) echo "usage: $0 {rootfs|dirs|download|wrapper|register|unit-install|unit-remove}"; exit 64 ;;
esac
