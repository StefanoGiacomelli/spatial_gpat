#!/usr/bin/env bash
# fetch_checkpoint.sh — Download and verify the E-PANNs checkpoint
# =================================================================
# Downloads checkpoint_closeto_.44.pt from Zenodo and verifies its
# SHA-256 digest.  Run from the repo root or from semantic_backbone/epanns/.
#
# Usage
# -----
#   bash semantic_backbone/epanns/fetch_checkpoint.sh
#   # or, from inside semantic_backbone/epanns/:
#   bash fetch_checkpoint.sh
#
# Source
# ------
#   Zenodo record : https://zenodo.org/records/7939403
#   Direct URL    : https://zenodo.org/records/7939403/files/checkpoint_closeto_.44.pt?download=1
#
# Reference
# ---------
# A. Singh Boparai, K. Drossos, and T. Heittola,
# "Efficient Pre-trained Audio Neural Networks via Passive Filter Pruning,"
# Inter-Noise 2023.  https://github.com/Arshdeep-Singh-Boparai/E-PANNs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SCRIPT_DIR}/checkpoint_closeto_.44.pt"
URL="https://zenodo.org/records/7939403/files/checkpoint_closeto_.44.pt?download=1"
EXPECTED_SHA256="cdbdff7eb4ef8967fb8bd215664b8d84fb40f3c031d54fc4920728e83361179f"

echo "Target : ${DEST}"

# ── Skip download if file already exists and hash matches ─────────────────────
if [ -f "${DEST}" ]; then
    echo "File already present — verifying SHA-256..."
    if echo "${EXPECTED_SHA256}  ${DEST}" | sha256sum --check --status 2>/dev/null \
    || echo "${EXPECTED_SHA256}  ${DEST}" | shasum -a 256 --check --status 2>/dev/null; then
        echo "✓  Checksum OK — nothing to do."
        exit 0
    else
        echo "! Checksum mismatch — re-downloading..."
    fi
fi

# ── Download ──────────────────────────────────────────────────────────────────
echo "Downloading from Zenodo..."
if command -v curl &>/dev/null; then
    curl -L --progress-bar -o "${DEST}" "${URL}"
elif command -v wget &>/dev/null; then
    wget --show-progress -O "${DEST}" "${URL}"
else
    echo "ERROR: neither curl nor wget found — install one and retry." >&2
    exit 1
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo "Verifying SHA-256..."
if echo "${EXPECTED_SHA256}  ${DEST}" | sha256sum --check --status 2>/dev/null \
|| echo "${EXPECTED_SHA256}  ${DEST}" | shasum -a 256 --check --status 2>/dev/null; then
    echo "✓  ${DEST}"
    echo "    SHA-256 : ${EXPECTED_SHA256}"
else
    echo "ERROR: SHA-256 mismatch after download — file may be corrupt." >&2
    rm -f "${DEST}"
    exit 1
fi
