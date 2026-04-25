#!/usr/bin/env bash
# Analyze retargeted PKL files and write a Markdown report.
#
# Usage:
#   bash shell/analyse.sh <pkl_dir> --mode <mode> [--yaml <yaml1> [<yaml2> ...]] [extra args]
#
# <pkl_dir>  — base output directory containing PKL files (e.g. output/).
#              PKL stems are computed relative to this directory.
#
# --mode     — processing mode used when retargeting (e.g. qp_smoothed, qp, vanilla).
#              Required for YAML category mapping so the mode component can be
#              stripped from PKL stems: output/AMASS/CMU/<mode>/02/02_01.pkl
#              → stem AMASS/CMU/02/02_01 → matched against YAML entries.
#
# --yaml     — one or more YAML index files for motion category mapping.
#              If omitted, all clips are grouped under 'unknown'.
#
# Report location (auto-derived when --output is omitted):
#   Single dataset: output/AMASS/CMU/<mode>/REPORT_<mode>.md
#   Multiple datasets: output/REPORT_<mode>.md
#
# Examples:
#
#   # CMU only (qp + two-pass smoothing)
#   bash shell/analyse.sh output --mode qp_smoothed \
#       --yaml ../k1_motion_data/index/amass_cmu_locomotion.yaml
#
#   # CMU + HDM05 together
#   bash shell/analyse.sh output --mode qp_smoothed \
#       --yaml ../k1_motion_data/index/amass_cmu_locomotion.yaml \
#              ../k1_motion_data/index/amass_hdm05_locomotion.yaml
#
#   # No category mapping (just statistics)
#   bash shell/analyse.sh output --mode vanilla
#
#   # Custom output path
#   bash shell/analyse.sh output --mode qp_smoothed \
#       --yaml ../k1_motion_data/index/amass_cmu_locomotion.yaml \
#       --output reports/cmu_report.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PKL_DIR="${1:-}"
if [[ -z "$PKL_DIR" ]]; then
    echo "Usage: bash shell/analyse.sh <pkl_dir> --mode <mode> [--yaml <yaml...>] [extra args]"
    echo ""
    echo "Examples:"
    echo "  bash shell/analyse.sh output --mode qp_smoothed \\"
    echo "      --yaml ../k1_motion_data/index/amass_cmu_locomotion.yaml"
    echo "  bash shell/analyse.sh output --mode qp_smoothed \\"
    echo "      --yaml ../k1_motion_data/index/amass_cmu_locomotion.yaml \\"
    echo "             ../k1_motion_data/index/amass_hdm05_locomotion.yaml"
    exit 1
fi
shift

echo "=== analyse.sh ==="
echo "  PKL dir: $PKL_DIR"
echo ""

cd "$REPO_ROOT"

conda run -n gmr python scripts/analyze_locomotion_dataset.py \
    --pkl_dir "$PKL_DIR" \
    "$@"
