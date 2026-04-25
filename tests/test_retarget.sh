#!/usr/bin/env bash
# Run scripts/retarget_no_penetration.py on the tracked test motions in
# tests/test_motions/ and produce videos for visual review.
#
# Usage: tests/test_retarget.sh [--amass] [--gvhmr] [--lafan1]
#
# At least one flag is required; pass any combination. Each flag retargets
# every file in tests/test_motions/<flag>/ with:
#     --headless --record_video --ground_mode qp --strict_zero_pen
#     --robot booster_k1
#
# Outputs:
#   tests/results/output/<format>_<stem>.pkl
#   tests/results/videos/booster_k1_<format>_<stem>.mp4
#
# The user reviews the videos to confirm retargeting still works. The input
# motions in tests/test_motions/ are tracked in git so every checkout has the
# same fixtures.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCRIPT="scripts/retarget_no_penetration.py"
ROBOT="booster_k1"
TEST_MOTIONS="tests/test_motions"
OUTPUT_DIR="tests/results/output"
VIDEO_DIR="tests/results/videos"

usage() {
    cat <<EOF
Usage: tests/test_retarget.sh [--amass] [--gvhmr] [--lafan1]

Retargets tracked test motions in $TEST_MOTIONS/ using:
  --headless --record_video --ground_mode qp --strict_zero_pen \\
  --robot $ROBOT

Pass one or more of:
  --amass    SMPL-X .npz files in $TEST_MOTIONS/amass/
  --gvhmr    GVHMR .pt   files in $TEST_MOTIONS/gvhmr/
  --lafan1   BVH .bvh    files in $TEST_MOTIONS/lafan1/

Outputs:
  $OUTPUT_DIR/<format>_<stem>.pkl
  $VIDEO_DIR/${ROBOT}_<format>_<stem>.mp4
EOF
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

DO_AMASS=0
DO_GVHMR=0
DO_LAFAN1=0
for arg in "$@"; do
    case "$arg" in
        --amass)  DO_AMASS=1 ;;
        --gvhmr)  DO_GVHMR=1 ;;
        --lafan1) DO_LAFAN1=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown flag: $arg" >&2; usage; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR" "$VIDEO_DIR"

# retarget_no_penetration.py writes the mp4 to <output_path.parent>/videos/.
# We move it from $OUTPUT_DIR/videos/ up to $VIDEO_DIR/ after each run so the
# final layout is the two flat directories the user asked for.
run_one() {
    local input_path="$1"
    local input_format="$2"
    local label="$3"
    local raw_stem
    raw_stem="$(basename "${input_path%.*}")"
    local stem="${label}_${raw_stem}"
    local output_pkl="$OUTPUT_DIR/${stem}.pkl"

    echo ""
    echo "=== [${label}] ${raw_stem} ==="
    python "$SCRIPT" \
        --input "$input_path" \
        --input_format "$input_format" \
        --robot "$ROBOT" \
        --ground_mode qp \
        --strict_zero_pen \
        --headless \
        --record_video \
        --output "$output_pkl"

    local src_video="$OUTPUT_DIR/videos/${ROBOT}_${stem}.mp4"
    local dst_video="$VIDEO_DIR/${ROBOT}_${stem}.mp4"
    if [[ -f "$src_video" ]]; then
        mv -f "$src_video" "$dst_video"
    else
        echo "warning: expected video not found: $src_video" >&2
    fi
}

run_dir() {
    local subdir="$1"
    local glob="$2"
    local input_format="$3"
    local label="$4"

    shopt -s nullglob
    local files=( "$TEST_MOTIONS/$subdir"/$glob )
    shopt -u nullglob

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "warning: no $glob files found in $TEST_MOTIONS/$subdir/" >&2
        return
    fi
    for f in "${files[@]}"; do
        run_one "$f" "$input_format" "$label"
    done
}

[[ $DO_AMASS  -eq 1 ]] && run_dir amass  "*.npz" smplx      amass
[[ $DO_GVHMR  -eq 1 ]] && run_dir gvhmr  "*.pt"  gvhmr      gvhmr
[[ $DO_LAFAN1 -eq 1 ]] && run_dir lafan1 "*.bvh" bvh_lafan1 lafan1

# The retargeter creates $OUTPUT_DIR/videos/ even after we drain it; remove
# it if empty so the output dir stays clean.
rmdir "$OUTPUT_DIR/videos" 2>/dev/null || true

echo ""
echo "Done."
echo "  pkls:   $OUTPUT_DIR/"
echo "  videos: $VIDEO_DIR/"
