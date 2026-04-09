#!/usr/bin/env bash
#
# inference installer bootstrap
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/gregology/inference/refs/heads/main/install.sh | sudo bash
#   curl -fsSL ... | sudo bash -s -- --prune --port 9000
#
set -euo pipefail

REPO_URL="https://github.com/gregology/inference.git"
INSTALL_DIR="/srv/llm/src/inference"

# ── Ensure we're root ────────────────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: This script must be run as root (or via sudo)." >&2
  exit 1
fi

# ── Minimal dependencies for the bootstrap itself ────────────────
echo "Ensuring git and python3 are available ..."
apt-get update -qq
apt-get install -y -qq git python3 >/dev/null

# ── Clone or update the repo ────────────────────────────────────
if [ -d "${INSTALL_DIR}/.git" ]; then
  echo "Updating ${INSTALL_DIR} ..."
  git -C "${INSTALL_DIR}" pull --ff-only
else
  echo "Cloning ${REPO_URL} → ${INSTALL_DIR} ..."
  mkdir -p "$(dirname "${INSTALL_DIR}")"
  git clone "${REPO_URL}" "${INSTALL_DIR}"
fi

# ── Hand off to Python installer ─────────────────────────────────
cd "${INSTALL_DIR}"
echo ""
exec python3 -m installer "$@"
