#!/bin/bash
set -e

echo "1. Generating baseline QP retargeting for CMU 35 dataset..."
conda run -n gmr python scripts/retarget_no_penetration.py \
    --input data/AMASS/CMU/35 \
    --input_format amass_cmu \
    --ground_mode qp \
    --output retargeted_35_qp/ \
    --override

echo "2. Generating Smoothed (Two-Pass IK) retargeting for CMU 35 dataset..."
conda run -n gmr python scripts/retarget_no_penetration.py \
    --input data/AMASS/CMU/35 \
    --input_format amass_cmu \
    --ground_mode qp \
    --strict_zero_pen \
    --output retargeted_35_smoothed/ \
    --override

echo "3. Generating comparison videos..."
mkdir -p videos_cmu35

for qp_file in retargeted_35_qp/*.pkl; do
    filename=$(basename "$qp_file")
    stem="${filename%.*}"
    smoothed_file="retargeted_35_smoothed/$filename"
    out_vid="videos_cmu35/${stem}_comparison.mp4"
    
    echo "Rendering $out_vid..."
    # MUJOCO_GL=egl enables headless rendering
    MUJOCO_GL=egl conda run -n gmr python scripts/vis_compare_motions.py \
        --motion_a "$qp_file" \
        --motion_b "$smoothed_file" \
        --label_a "QP" \
        --label_b "Smoothed" \
        --video_path "$out_vid" \
        --show_sole_points
done

echo "All complete! Videos are located in the videos_cmu35/ directory."
