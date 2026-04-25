#!/usr/bin/env bash
# Retarget a single motion file with the K1/T1 pipeline.
#
# Usage:
#   bash shell/run_file.sh <file> [extra args]
#
# Default format is auto-picked from the extension:
#   .npz / .pkl  → amass_cmu  (covers AMASS _stageii.npz; override with
#                              --input_format smplx for non-AMASS sources)
#   .pt          → gvhmr
#   .bvh         → bvh_lafan1 (experimental)
#
# Override with --input_format <fmt>. Strict zero-penetration is OFF by
# default — pass --strict_zero_pen to enable it (~2x runtime).
#
# Examples:
#   bash shell/run_file.sh ../k1_motion_data/data/AMASS/CMU/02/02_01_stageii.npz
#   bash shell/run_file.sh ../k1_motion_data/data/AMASS/CMU/35/35_01_stageii.npz \
#       --strict_zero_pen --output retargeted/35_01.pkl
#   bash shell/run_file.sh GVHMR/outputs/freekick/hmr4d_results.pt --rate_limit
#   bash shell/run_file.sh dance.bvh --robot booster_t1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FILE="${1:-}"
if [[ -z "$FILE" ]]; then
    echo "Usage: bash shell/run_file.sh <file> [extra args]"
    echo ""
    echo "  .npz/.pkl  -> amass_cmu  (override with --input_format smplx)"
    echo "  .pt        -> gvhmr"
    echo "  .bvh       -> bvh_lafan1 (experimental)"
    exit 1
fi
shift

if [[ "$FILE" != /* ]]; then
    FILE="$REPO_ROOT/$FILE"
fi
if [[ ! -f "$FILE" ]]; then
    echo "File not found: $FILE"
    exit 1
fi

EXT="${FILE##*.}"
case "$EXT" in
    pt)  DEFAULT_FORMAT="gvhmr" ;;
    bvh) DEFAULT_FORMAT="bvh_lafan1" ;;
    *)   DEFAULT_FORMAT="amass_cmu" ;;
esac

echo "=== run_file.sh ==="
echo "  File          : $FILE"
echo "  Default format: $DEFAULT_FORMAT  (override with --input_format)"
echo ""

cd "$REPO_ROOT"

conda run -n gmr python scripts/retarget_no_penetration.py \
    --input        "$FILE" \
    --input_format "$DEFAULT_FORMAT" \
    --ground_mode  qp \
    "$@"
