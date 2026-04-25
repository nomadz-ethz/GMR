#!/usr/bin/env bash
# Retarget every motion file in a directory (batch mode).
#
# Usage:
#   bash shell/run_dir.sh <dir> [--input_format <fmt>] [extra args]
#
# --input_format options (default: amass_cmu):
#   amass_cmu   AMASS _stageii.npz (CMU, HDM05, ACCAD, ...)
#   smplx       generic SMPL-X .npz/.pkl
#   gvhmr       GVHMR hmr4d_results.pt
#   bvh_lafan1  LAFAN1 BVH (experimental)
#
# Strict zero-penetration is OFF by default — pass --strict_zero_pen to enable.
#
# Examples:
#   bash shell/run_dir.sh ../k1_motion_data/data/AMASS/CMU
#   bash shell/run_dir.sh ../k1_motion_data/data/AMASS/CMU/35 --strict_zero_pen
#   bash shell/run_dir.sh /path/to/lafan1 --input_format bvh_lafan1 --robot booster_t1
#   bash shell/run_dir.sh ../k1_motion_data/data/AMASS/CMU --overwrite

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DIR="${1:-}"
if [[ -z "$DIR" ]]; then
    echo "Usage: bash shell/run_dir.sh <dir> [--input_format <fmt>] [extra args]"
    exit 1
fi
shift

if [[ "$DIR" != /* ]]; then
    DIR="$REPO_ROOT/$DIR"
fi
if [[ ! -d "$DIR" ]]; then
    echo "Directory not found: $DIR"
    exit 1
fi

echo "=== run_dir.sh ==="
echo "  Directory     : $DIR"
echo "  Default format: amass_cmu  (override with --input_format)"
echo ""

cd "$REPO_ROOT"

conda run -n gmr python scripts/retarget_no_penetration.py \
    --input        "$DIR" \
    --input_format amass_cmu \
    --ground_mode  qp \
    "$@"
