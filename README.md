# GMR — Booster K1 / T1 retargeting

A focused fork of [`YanjieZe/GMR`](https://github.com/YanjieZe/GMR) (via
[`nomadz-ethz/GMR`](https://github.com/nomadz-ethz/GMR)) narrowed to two
robots — **Booster K1** and **Booster T1** — with one ground-aware retargeting
pipeline (`scripts/retarget_no_penetration.py`).

What this fork adds on top of upstream GMR:

- **Ground-penetration prevention** (hard QP / soft constraint) plus a
  **strict zero-penetration two-pass IK** that produces 0 mm penetration
  with no foot-slide.
- **Foot-ground contact labels** derived from raw SMPL-X toe kinematics,
  written into every output pkl.
- **Unified pipeline CLI** for SMPL-X (AMASS / OMOMO), AMASS CMU
  `_stageii.npz`, GVHMR `.pt`, and BVH LAFAN1 (experimental).
- **Modular `general_motion_retargeting.retargeting/` subpackage** so the
  pipeline pieces are testable and reusable.
- **Self-describing pkl schema** (records `robot`, `input_format`,
  `source_file`, ground mode, contact-detection params, …).

For the full breakdown, read [`docs/overview.md`](docs/overview.md). For the
pipeline reference, [`docs/pipeline.md`](docs/pipeline.md).

## Install

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
conda install -c conda-forge libstdcxx-ng -y
```

The pipeline depends on SMPL-X body models in `assets/body_models/smplx/`
(`SMPLX_F.npz`, `SMPLX_M.npz`, `SMPLX_N.npz`).

## Quickstart

```bash
# AMASS CMU walk on K1, two-pass smoothing, headless
python scripts/retarget_no_penetration.py \
    --input /data/AMASS/CMU/35/35_01_stageii.npz \
    --input_format amass_cmu \
    --robot booster_k1 \
    --ground_mode qp --strict_zero_pen \
    --headless --output retargeted/35_01.pkl

# GVHMR motion with the live MuJoCo viewer
python scripts/retarget_no_penetration.py \
    --input GVHMR/outputs/freekick/hmr4d_results.pt \
    --input_format gvhmr \
    --robot booster_k1 \
    --ground_mode qp --strict_zero_pen --rate_limit

# LAFAN1 BVH (experimental — see docs/bvh.md)
python scripts/retarget_no_penetration.py \
    --input dance.bvh --input_format bvh_lafan1 \
    --robot booster_t1 --ground_mode qp \
    --output retargeted/dance.pkl

# Replay a saved pkl
python scripts/vis_robot_motion.py \
    --robot booster_k1 --robot_motion_path retargeted/35_01.pkl

# Replay with foot-contact overlay
python scripts/vis_robot_motion_with_contact.py \
    --robot booster_k1 --robot_motion_path retargeted/35_01.pkl
```

Batch and YAML-driven runs are wrapped in `shell/run_file.sh`,
`shell/run_dir.sh`, and `shell/run_yaml.sh`.

## Documentation

- [`docs/overview.md`](docs/overview.md) — what this fork adds vs upstream
- [`docs/pipeline.md`](docs/pipeline.md) — full pipeline reference (CLI,
  algorithms, pkl schema, empirical results, known limitations)
- [`docs/foot_contact.md`](docs/foot_contact.md) — contact-flag schema and
  consumer-side examples (Python / PyTorch / RL)
- [`docs/bvh.md`](docs/bvh.md) — BVH (LAFAN1) integration, **experimental**
- [`docs/ik_config.md`](docs/ik_config.md) — IK config field reference
- [`docs/test_motions.md`](docs/test_motions.md) — tricky motions for
  validation
- [`docs/archive/UPSTREAM_README.md`](docs/archive/UPSTREAM_README.md) —
  upstream `YanjieZe/GMR` README, kept for historical reference

## Acknowledgements

- Upstream IK retargeter: [YanjieZe/GMR](https://github.com/YanjieZe/GMR)
  ([arXiv 2510.02252](https://arxiv.org/abs/2510.02252))
- IK solver: [`mink`](https://github.com/kevinzakka/mink) on top of
  [MuJoCo](https://github.com/google-deepmind/mujoco)
- Robot models: Booster Robotics K1 / T1
- BVH skeleton fork via [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)

Licensed under the [MIT License](LICENSE).
