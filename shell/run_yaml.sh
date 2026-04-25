#!/usr/bin/env bash
# Retarget a full AMASS dataset via a YAML index file (--yaml mode).
#
# Usage:
#   bash shell/run_yaml.sh <yaml_or_shortcut> [extra args]
#
# Shortcuts (resolved against ../k1_motion_data/index/):
#   cmu     amass_cmu_locomotion.yaml
#   hdm05   amass_hdm05_locomotion.yaml
# Or pass any *.yaml path directly (absolute or repo-relative).
#
# Defaults: --input_format amass_cmu --ground_mode qp --height_adjust
#           --root_origin_offset
# Strict zero-penetration is OFF by default — pass --strict_zero_pen to enable.
#
# Examples:
#   bash shell/run_yaml.sh cmu
#   bash shell/run_yaml.sh cmu --strict_zero_pen
#   bash shell/run_yaml.sh hdm05 --overwrite
#   bash shell/run_yaml.sh ../k1_motion_data/index/amass_cmu_locomotion.yaml --robot booster_t1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INDEX_DIR="$REPO_ROOT/../k1_motion_data/index"

ARG="${1:-}"
if [[ -z "$ARG" ]]; then
    echo "Usage: bash shell/run_yaml.sh <yaml_or_shortcut> [extra args]"
    echo ""
    echo "Shortcuts (k1_motion_data/index/):"
    echo "  cmu      amass_cmu_locomotion.yaml"
    echo "  hdm05    amass_hdm05_locomotion.yaml"
    echo ""
    echo "Or pass any *.yaml path directly."
    exit 1
fi
shift

case "$ARG" in
    cmu)   YAML="$INDEX_DIR/amass_cmu_locomotion.yaml"   ;;
    hdm05) YAML="$INDEX_DIR/amass_hdm05_locomotion.yaml" ;;
    *)
        if [[ "$ARG" = /* ]]; then YAML="$ARG"; else YAML="$REPO_ROOT/$ARG"; fi
        ;;
esac

if [[ ! -f "$YAML" ]]; then
    echo "YAML file not found: $YAML"
    exit 1
fi

echo "=== run_yaml.sh ==="
echo "  YAML : $YAML"
echo ""

cd "$REPO_ROOT"

conda run -n gmr python scripts/retarget_no_penetration.py \
    --yaml         "$YAML" \
    --input_format amass_cmu \
    --ground_mode  qp \
    --height_adjust \
    --root_origin_offset \
    "$@"
