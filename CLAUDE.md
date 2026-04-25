# CLAUDE.md

Guidance for Claude Code working in this repository (a Booster K1 / T1
motion retargeting fork of [`YanjieZe/GMR`](https://github.com/YanjieZe/GMR)).

## Installation

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
conda install -c conda-forge libstdcxx-ng -y
```

The pipeline reads SMPL-X body models from `assets/body_models/smplx/`.

## What this fork is

A narrowed fork of upstream GMR. Only **Booster K1** and **Booster T1** are
supported. The single retargeting entry point is
`scripts/retarget_no_penetration.py`, which handles:

- `--input_format smplx`     — generic SMPL-X .npz / .pkl (AMASS, OMOMO, …)
- `--input_format amass_cmu` — AMASS CMU `_stageii.npz` with stem cleaning
- `--input_format gvhmr`     — GVHMR `hmr4d_results.pt`
- `--input_format bvh_lafan1` — LAFAN1 BVH (**experimental**)

The pipeline writes a self-describing pkl containing root + dof + (optional)
local body positions + foot-contact flags. Schema: `docs/pipeline.md` §5.7.

For the broader picture of fork additions vs upstream see
`docs/overview.md`. For the pipeline reference: `docs/pipeline.md`.

## Code architecture

### Library: `general_motion_retargeting/`

- `motion_retarget.py` — `GeneralMotionRetargeting` IK class on mink/mujoco
- `kinematics_model.py` — robot kinematics helper
- `params.py` — registry dicts (K1 + T1 only)
- `robot_motion_viewer.py` — MuJoCo viewer
- `ground_constraint.py` — `GroundPlaneLimit` (hard QP), `SoftGroundConstraint`
  / `GMRWithSoftGround` (soft repulsive)
- `sole_points.py` — per-robot sole contact points
- `data_loader.py`, `rot_utils.py`, `torch_utils.py` — utilities
- `retargeting/` — modular pipeline helpers extracted from the CLI:
  `builder.build_retargeter`, `strict_zero_pen.measure_*` /
  `smooth_penetration_envelope`, `foot_contact.detect_smplx_foot_contact`,
  `fk_post.apply_fk_post`, `batch_runner.run_batch` / `Tee` /
  `discover_input_files` / `load_yaml_paths` / `amass_stem`
- `utils/smpl.py`, `utils/lafan1.py`, `utils/lafan_vendor/` — source loaders
- `ik_configs/` — `smplx_to_{k1,t1}.json`, `bvh_lafan1_to_{k1,t1}.json`

### Entry-point scripts: `scripts/`

- `retarget_no_penetration.py` — **the** pipeline CLI (smplx / gvhmr /
  amass_cmu / bvh_lafan1 × k1 / t1 × ground_mode × strict_zero_pen)
- `vis_robot_motion.py` — replay one pkl in the MuJoCo viewer
- `vis_robot_motion_dataset.py` — browse a directory of pkls (`[`/`]`/`x`)
- `vis_robot_motion_debug.py` — viewer with sole markers + penetration colour
- `vis_robot_motion_with_contact.py` — render mp4 with foot-contact overlay
- `vis_compare_motions.py` — side-by-side comparison
- `analyze_locomotion_dataset.py` — per-pkl penetration / jitter stats →
  Markdown report (optional YAML category mapping)
- `compare_penetration_stats.py` — diagnostic CLI
- `add_sole_sites.py` — one-time MJCF prep injecting sole `<site>`s

### Shell wrappers: `shell/`

- `run_file.sh`, `run_dir.sh`, `run_yaml.sh` — thin wrappers around
  `retarget_no_penetration.py` for a single file / directory / YAML index
- `analyse.sh` — wraps `analyze_locomotion_dataset.py`
- `run_batch_comparison.sh` — qp vs strict_zero_pen comparison runner

### Docs: `docs/`

- `overview.md` — what this fork adds vs upstream GMR
- `pipeline.md` — canonical pipeline reference
- `foot_contact.md` — foot-contact PKL schema and consumer examples
- `bvh.md` — BVH (LAFAN1) integration, experimental
- `ik_config.md` — IK config field reference
- `test_motions.md` — tricky-motion checklist
- `archive/` — `UPSTREAM_README.md`, `penetration.md`,
  `CHANGELOG_ZERO_PENETRATION.md` (historical)

## Common commands

```bash
# Single AMASS CMU file, K1, two-pass smoothing, headless
python scripts/retarget_no_penetration.py \
    --input /data/AMASS/CMU/35/35_01_stageii.npz --input_format amass_cmu \
    --robot booster_k1 --ground_mode qp --strict_zero_pen \
    --headless --output retargeted/35_01.pkl

# GVHMR with viewer
python scripts/retarget_no_penetration.py \
    --input GVHMR/outputs/freekick/hmr4d_results.pt --input_format gvhmr \
    --robot booster_k1 --ground_mode qp --strict_zero_pen --rate_limit

# Replay
python scripts/vis_robot_motion.py \
    --robot booster_k1 --robot_motion_path retargeted/35_01.pkl
```

## Key technical details

- **IK Solver**: `mink` with `daqp`, damping 5e-1.
- **Ground constraint**: hard QP via `GroundPlaneLimit`; soft via
  `GMRWithSoftGround`. Defaults tuned for K1 / T1.
- **Strict zero-penetration**: two-pass IK with smoothed envelope; ~2× the
  single-pass runtime.
- **Foot-ground contact**: detected on raw SMPL-X toe kinematics, so labels
  are stable across `--ground_mode` and robot choices.
- **Body model dependencies**: SMPL-X in `assets/body_models/smplx/`.

## Regression check

`tests/check_baseline.py` re-runs canonical motions through the CLI and
asserts numeric arrays (and contact flags) are unchanged vs pinned baseline
pkls in `tests/baseline/` (gitignored). Use it after touching anything in
`general_motion_retargeting/retargeting/`, `params.py`, or
`scripts/retarget_no_penetration.py`.
