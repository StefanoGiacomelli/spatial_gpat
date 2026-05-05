#!/usr/bin/env bash
# data/setup_data.sh — Incremental FOA dataset setup for spatial_gpat
#
# Downloads and extracts each dataset component only if the target
# directory / file is not already present (idempotent, FOA-only for SELD).
# AudioSet is handled separately: the script only extracts the tar bundle
# if it finds audioset_bundle.tar locally; otherwise it warns and skips.
#
# Usage
# -----
#   bash data/setup_data.sh                        # all datasets
#   bash data/setup_data.sh --dataset STARSS23
#   bash data/setup_data.sh --dataset TAU-NIGENS2021
#   bash data/setup_data.sh --dataset TAU2019
#   bash data/setup_data.sh --dataset audioset
#
# Prerequisites: curl, zip, unzip
# Run from anywhere; all paths are resolved relative to this script.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── CLI ───────────────────────────────────────────────────────────────────────

DATASET_FILTER="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET_FILTER="${2:-}"
            [[ -n "$DATASET_FILTER" ]] || { echo "[ERROR] --dataset requires a value" >&2; exit 1; }
            shift 2
            ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            echo "Usage: $0 [--dataset STARSS23|TAU-NIGENS2021|TAU2019|audioset|all]" >&2
            exit 1
            ;;
    esac
done

case "$DATASET_FILTER" in
    STARSS23|TAU-NIGENS2021|TAU2019|audioset|all) ;;
    *)
        echo "[ERROR] --dataset must be one of: STARSS23, TAU-NIGENS2021, TAU2019, audioset, all" >&2
        exit 1
        ;;
esac

# ── Helpers ───────────────────────────────────────────────────────────────────

log()  { echo "[setup] $*"; }
warn() { echo "[WARN]  $*" >&2; }

# True if directory $1 contains at least one .wav file (any depth)
_has_wavs() { [[ -n "$(find "$1" -name '*.wav' 2>/dev/null | head -1)" ]]; }
# True if directory $1 contains at least one .csv file (any depth)
_has_csvs() { [[ -n "$(find "$1" -name '*.csv' 2>/dev/null | head -1)" ]]; }

# Download URL $2 to path $1; skips if file already exists; supports resume.
_fetch() {
    local dest="$1" url="$2"
    if [[ -f "$dest" ]]; then
        log "    $(basename "$dest")  already present — skip"
        return
    fi
    log "    downloading $(basename "$dest") ..."
    curl -L --progress-bar -C - -o "$dest" "$url"
}

# Merge split-zip parts in directory $1 for base name $2, extract to $3.
# Requires all parts (z01, z02, …, zip) to be already present in $1.
# The temporary merged zip is deleted after extraction.
_extract_split() {
    local dir="$1" base="$2" dest="$3"
    local merged="${dir}/${base}_merged.zip"
    log "    merging split archive ${base}.zip ..."
    (cd "$dir" && zip -s 0 "${base}.zip" --out "${base}_merged.zip")
    log "    extracting ${base}_merged.zip ..."
    unzip -q -o "$merged" -d "$dest"
    rm -f "$merged"
}

# Extract regular zip $1 to directory $2.
_extract() {
    log "    extracting $(basename "$1") ..."
    unzip -q -o "$1" -d "$2"
}

# ── Prerequisites ─────────────────────────────────────────────────────────────

_missing_tools=()
for _cmd in curl zip unzip; do
    command -v "$_cmd" &>/dev/null || _missing_tools+=("$_cmd")
done
if [[ ${#_missing_tools[@]} -gt 0 ]]; then
    warn "Missing tools: ${_missing_tools[*]}"
    warn "Please install them before running this script (e.g. brew install ${_missing_tools[*]})."
fi

# ── Dataset roots ─────────────────────────────────────────────────────────────

AUDIOSET_DIR="${SCRIPT_DIR}/audioset-qiuqiangkong"
STARSS23_DIR="${SCRIPT_DIR}/STARSS23"
TAU_DIR="${SCRIPT_DIR}/TAU-NIGENS2021"
TAU2019_DIR="${SCRIPT_DIR}/TAU2019"

# ══════════════════════════════════════════════════════════════════════════════
# AudioSet — manual bundle extraction (not auto-downloaded)
# ══════════════════════════════════════════════════════════════════════════════

_setup_audioset() {
    log "AudioSet ..."

    if [[ -d "$AUDIOSET_DIR" && -n "$(ls -A "$AUDIOSET_DIR" 2>/dev/null)" ]]; then
        log "  audioset-qiuqiangkong/ already present — skipping."
        return
    fi

    local tar="${SCRIPT_DIR}/audioset_bundle.tar"
    if [[ ! -f "$tar" ]]; then
        warn "AudioSet: audioset_bundle.tar not found in ${SCRIPT_DIR}."
        warn "  Download it manually from SharePoint and place it in data/ (see README.md)."
        return
    fi

    mkdir -p "$AUDIOSET_DIR"
    log "  extracting audioset_bundle.tar → audioset-qiuqiangkong/ ..."
    tar -xf "$tar" -C "$AUDIOSET_DIR"
    log "  removing audioset_bundle.tar ..."
    rm -f "$tar"
    log "  AudioSet done."
}

# ══════════════════════════════════════════════════════════════════════════════
# STARSS23 — Zenodo record 7880637
#
# FOA files downloaded:
#   foa_dev.zip      → foa_dev/  (dev-train-sony, dev-train-tau,
#                                  dev-test-sony,  dev-test-tau)
#   foa_eval.zip     → foa_eval/
#   metadata_dev.zip → metadata_dev/
#   LICENSE          (single file)
#   README.md        (single file)
# ══════════════════════════════════════════════════════════════════════════════

_setup_starss23() {
    local D="$STARSS23_DIR"
    local BASE="https://zenodo.org/records/7880637/files"
    mkdir -p "$D"

    log "STARSS23 ..."

    if _has_wavs "$D/foa_dev"; then
        log "  foa_dev/ — OK"
    else
        log "  foa_dev/ — missing, fetching ..."
        _fetch "$D/foa_dev.zip" "${BASE}/foa_dev.zip?download=1"
        _extract "$D/foa_dev.zip" "$D"
    fi

    if _has_wavs "$D/foa_eval"; then
        log "  foa_eval/ — OK"
    else
        log "  foa_eval/ — missing, fetching ..."
        _fetch "$D/foa_eval.zip" "${BASE}/foa_eval.zip?download=1"
        _extract "$D/foa_eval.zip" "$D"
    fi

    if _has_csvs "$D/metadata_dev"; then
        log "  metadata_dev/ — OK"
    else
        log "  metadata_dev/ — missing, fetching ..."
        _fetch "$D/metadata_dev.zip" "${BASE}/metadata_dev.zip?download=1"
        _extract "$D/metadata_dev.zip" "$D"
    fi

    if [[ -f "$D/LICENSE" ]]; then
        log "  LICENSE — OK"
    else
        _fetch "$D/LICENSE" "${BASE}/LICENSE?download=1"
    fi

    if [[ -f "$D/README.md" ]]; then
        log "  README.md — OK"
    else
        _fetch "$D/README.md" "${BASE}/README.md?download=1"
    fi

    log "  STARSS23 done."
}

# ══════════════════════════════════════════════════════════════════════════════
# TAU-NIGENS2021 — Zenodo record 5476980
#
# FOA files downloaded:
#   foa_dev.z01 + foa_dev.zip  (split archive) → foa_dev/
#                                                   (dev-train, dev-val, dev-test)
#   foa_eval.zip               → foa_eval/eval-test/
#   metadata_dev.zip           → metadata_dev/
#   metadata_eval.zip          → metadata_eval/
#   README.md                  (single file)
# ══════════════════════════════════════════════════════════════════════════════

_setup_tau_nigens() {
    local D="$TAU_DIR"
    local BASE="https://zenodo.org/records/5476980/files"
    mkdir -p "$D"

    log "TAU-NIGENS2021 ..."

    # foa_dev is a 2-part split archive (foa_dev.z01 + foa_dev.zip)
    if _has_wavs "$D/foa_dev"; then
        log "  foa_dev/ — OK"
    else
        log "  foa_dev/ — missing, fetching (2-part split) ..."
        _fetch "$D/foa_dev.z01" "${BASE}/foa_dev.z01?download=1"
        _fetch "$D/foa_dev.zip" "${BASE}/foa_dev.zip?download=1"
        _extract_split "$D" "foa_dev" "$D"
    fi

    if _has_wavs "$D/foa_eval"; then
        log "  foa_eval/ — OK"
    else
        log "  foa_eval/ — missing, fetching ..."
        _fetch "$D/foa_eval.zip" "${BASE}/foa_eval.zip?download=1"
        _extract "$D/foa_eval.zip" "$D"
    fi

    if _has_csvs "$D/metadata_dev"; then
        log "  metadata_dev/ — OK"
    else
        log "  metadata_dev/ — missing, fetching ..."
        _fetch "$D/metadata_dev.zip" "${BASE}/metadata_dev.zip?download=1"
        _extract "$D/metadata_dev.zip" "$D"
    fi

    if _has_csvs "$D/metadata_eval"; then
        log "  metadata_eval/ — OK"
    else
        log "  metadata_eval/ — missing, fetching ..."
        _fetch "$D/metadata_eval.zip" "${BASE}/metadata_eval.zip?download=1"
        _extract "$D/metadata_eval.zip" "$D"
    fi

    if [[ -f "$D/README.md" ]]; then
        log "  README.md — OK"
    else
        _fetch "$D/README.md" "${BASE}/README.md?download=1"
    fi

    log "  TAU-NIGENS2021 done."
}

# ══════════════════════════════════════════════════════════════════════════════
# TAU2019 — Zenodo records 2599196 (dev) and 3377088 (eval)
#
# Dev files (record 2599196):
#   foa_dev.z01 + foa_dev.z02 + foa_dev.zip  (3-part split) → dev/foa_dev/
#   metadata_dev.zip                          → dev/metadata_dev/
#   LICENSE, README.md                        → dev/
#
# Eval files (record 3377088):
#   foa_eval.zip     → eval/  (unpacks to eval/proj/asignal/DCASE2019/dataset/foa_eval/)
#   metadata_eval.zip → eval/metadata_eval/
#   short2longname.txt, README.md              → eval/
#
# NOTE: foa_eval lands at eval/proj/asignal/DCASE2019/dataset/foa_eval/ — this
# nested path is expected by TAU2019Dataset._collect_eval() in datasets/tau2019.py.
# ══════════════════════════════════════════════════════════════════════════════

_setup_tau2019() {
    local D="$TAU2019_DIR"
    local BASE_DEV="https://zenodo.org/records/2599196/files"
    local BASE_EVAL="https://zenodo.org/records/3377088/files"
    mkdir -p "$D/dev" "$D/eval"

    log "TAU2019 ..."

    # ── dev/ ─────────────────────────────────────────────────────────────────

    # foa_dev is a 3-part split archive (foa_dev.z01, foa_dev.z02, foa_dev.zip)
    if _has_wavs "$D/dev/foa_dev"; then
        log "  dev/foa_dev/ — OK"
    else
        log "  dev/foa_dev/ — missing, fetching (3-part split) ..."
        _fetch "$D/dev/foa_dev.z01" "${BASE_DEV}/foa_dev.z01?download=1"
        _fetch "$D/dev/foa_dev.z02" "${BASE_DEV}/foa_dev.z02?download=1"
        _fetch "$D/dev/foa_dev.zip" "${BASE_DEV}/foa_dev.zip?download=1"
        _extract_split "$D/dev" "foa_dev" "$D/dev"
    fi

    if _has_csvs "$D/dev/metadata_dev"; then
        log "  dev/metadata_dev/ — OK"
    else
        log "  dev/metadata_dev/ — missing, fetching ..."
        _fetch "$D/dev/metadata_dev.zip" "${BASE_DEV}/metadata_dev.zip?download=1"
        _extract "$D/dev/metadata_dev.zip" "$D/dev"
    fi

    if [[ -f "$D/dev/LICENSE" ]]; then
        log "  dev/LICENSE — OK"
    else
        _fetch "$D/dev/LICENSE" "${BASE_DEV}/LICENSE?download=1"
    fi

    if [[ -f "$D/dev/README.md" ]]; then
        log "  dev/README.md — OK"
    else
        _fetch "$D/dev/README.md" "${BASE_DEV}/README.md?download=1"
    fi

    # ── eval/ ────────────────────────────────────────────────────────────────
    # foa_eval.zip unpacks the nested path:
    #   eval/proj/asignal/DCASE2019/dataset/foa_eval/split0_*.wav

    if _has_wavs "$D/eval/proj"; then
        log "  eval/foa_eval/ — OK"
    else
        log "  eval/foa_eval/ — missing, fetching ..."
        _fetch "$D/eval/foa_eval.zip" "${BASE_EVAL}/foa_eval.zip?download=1"
        _extract "$D/eval/foa_eval.zip" "$D/eval"
    fi

    if _has_csvs "$D/eval/metadata_eval"; then
        log "  eval/metadata_eval/ — OK"
    else
        log "  eval/metadata_eval/ — missing, fetching ..."
        _fetch "$D/eval/metadata_eval.zip" "${BASE_EVAL}/metadata_eval.zip?download=1"
        _extract "$D/eval/metadata_eval.zip" "$D/eval"
    fi

    if [[ -f "$D/eval/short2longname.txt" ]]; then
        log "  eval/short2longname.txt — OK"
    else
        _fetch "$D/eval/short2longname.txt" "${BASE_EVAL}/short2longname.txt?download=1"
    fi

    if [[ -f "$D/eval/README.md" ]]; then
        log "  eval/README.md — OK"
    else
        _fetch "$D/eval/README.md" "${BASE_EVAL}/README.md?download=1"
    fi

    log "  TAU2019 done."
}

# ══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ══════════════════════════════════════════════════════════════════════════════

case "$DATASET_FILTER" in
    all)
        _setup_audioset
        _setup_starss23
        _setup_tau_nigens
        _setup_tau2019
        ;;
    audioset)      _setup_audioset   ;;
    STARSS23)      _setup_starss23   ;;
    TAU-NIGENS2021) _setup_tau_nigens ;;
    TAU2019)       _setup_tau2019    ;;
esac

log "Done."
