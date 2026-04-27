#!/usr/bin/env bash
# setup_data.sh — Recreate the spatial_gpat data directory
#
# Run from inside the data/ directory or from anywhere; paths are resolved
# relative to the directory that contains this script.
#
# Prerequisites: curl, tar, unzip
# Required free disk space: 200 GB
#
# Before running this script, manually download audioset_bundle.tar and place
# it in the same directory as this script. See README.md for the download link.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AUDIOSET_TAR="${SCRIPT_DIR}/audioset_bundle.tar"
AUDIOSET_DIR="${SCRIPT_DIR}/audioset-qiuqiangkong"
STARSS23_DIR="${SCRIPT_DIR}/STARSS23"
TAU_DIR="${SCRIPT_DIR}/TAU-NIGENS2021"

STARSS23_URL="https://zenodo.org/api/records/7880637/files-archive"
TAU_URL="https://zenodo.org/api/records/5476980/files-archive"

REQUIRED_GB=200
REQUIRED_KB=$(( REQUIRED_GB * 1024 * 1024 ))

# ── helpers ───────────────────────────────────────────────────────────────────

log() { echo "[setup] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

# ── prerequisites ─────────────────────────────────────────────────────────────

for cmd in curl tar unzip; do
    command -v "$cmd" &>/dev/null || die "Required tool not found: ${cmd}. Please install it and retry."
done

# ── disk space check ──────────────────────────────────────────────────────────

AVAIL_KB=$(df -k "$SCRIPT_DIR" | awk 'NR==2 {print $4}')
AVAIL_GB=$(( AVAIL_KB / 1024 / 1024 ))
if [[ "$AVAIL_KB" -lt "$REQUIRED_KB" ]]; then
    die "Not enough disk space. Required: ${REQUIRED_GB} GB — Available: ${AVAIL_GB} GB. Free up space and retry."
fi
log "Disk space OK — required ${REQUIRED_GB} GB, available ${AVAIL_GB} GB."

# ── audioset bundle check ─────────────────────────────────────────────────────

if [[ ! -f "$AUDIOSET_TAR" ]]; then
    die "audioset_bundle.tar not found in ${SCRIPT_DIR}.\nDownload it manually from SharePoint (see README.md) and place it here before running this script."
fi

# ── create target directories ─────────────────────────────────────────────────

mkdir -p "$AUDIOSET_DIR" "$STARSS23_DIR" "$TAU_DIR"

# ── step 1: AudioSet ──────────────────────────────────────────────────────────

log "Extracting audioset_bundle.tar → audioset-qiuqiangkong/ ..."
tar -xf "$AUDIOSET_TAR" -C "$AUDIOSET_DIR"
log "Removing audioset_bundle.tar ..."
rm -f "$AUDIOSET_TAR"
log "AudioSet done."

# ── step 2: STARSS23 ──────────────────────────────────────────────────────────

STARSS_ZIP="${SCRIPT_DIR}/starss23_archive.zip"
STARSS_TMP="${SCRIPT_DIR}/.starss23_tmp"

log "Downloading STARSS23 from Zenodo ..."
curl -L --progress-bar -o "$STARSS_ZIP" "$STARSS23_URL"

log "Extracting outer archive ..."
mkdir -p "$STARSS_TMP"
unzip -q "$STARSS_ZIP" -d "$STARSS_TMP"
rm -f "$STARSS_ZIP"

log "Extracting inner archives → STARSS23/ ..."
while IFS= read -r inner; do
    log "  $(basename "$inner")"
    unzip -q -o "$inner" -d "$STARSS23_DIR"
done < <(find "$STARSS_TMP" -maxdepth 1 -name "*.zip" | sort)
rm -rf "$STARSS_TMP"
log "STARSS23 done."

# ── step 3: TAU-NIGENS2021 ────────────────────────────────────────────────────

TAU_ZIP="${SCRIPT_DIR}/tau_nigens_archive.zip"
TAU_TMP="${SCRIPT_DIR}/.tau_nigens_tmp"

log "Downloading TAU-NIGENS2021 from Zenodo ..."
curl -L --progress-bar -o "$TAU_ZIP" "$TAU_URL"

log "Extracting outer archive ..."
mkdir -p "$TAU_TMP"
unzip -q "$TAU_ZIP" -d "$TAU_TMP"
rm -f "$TAU_ZIP"

log "Extracting inner archives → TAU-NIGENS2021/ ..."
while IFS= read -r inner; do
    log "  $(basename "$inner")"
    unzip -q -o "$inner" -d "$TAU_DIR"
done < <(find "$TAU_TMP" -maxdepth 1 -name "*.zip" | sort)
rm -rf "$TAU_TMP"
log "TAU-NIGENS2021 done."

# ── done ──────────────────────────────────────────────────────────────────────

log "Setup complete. Final layout:"
ls "$SCRIPT_DIR"
