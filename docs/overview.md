# Fork Overview — What's Different from Upstream GMR

This fork of [`YanjieZe/GMR`](https://github.com/YanjieZe/GMR) (via
[`nomadz-ethz/GMR`](https://github.com/nomadz-ethz/GMR)) is narrowed and
hardened around two robots and one pipeline so that retargeted motions can be
fed directly into RL training and physical deployment on Booster K1 / T1.

## 1. What upstream GMR provides

A general-purpose IK retargeter built on `mink`/`mujoco`. The `GeneralMotionRetargeting`
class maps per-frame human body poses (SMPL-X / BVH / FBX / OptiTrack) to a
robot's joint state by minimising weighted task residuals defined in
`ik_configs/<src>_to_<robot>.json`. Upstream supports nine robots out of the
box and a real-time MuJoCo viewer.

The IK core is unchanged in this fork. We built **on top of it**, not around
it.

## 2. What this fork adds

| Addition | Where | Why it matters |
|---|---|---|
| **Ground-penetration prevention** | `general_motion_retargeting/ground_constraint.py` | Upstream lets feet sink into the floor. We add a hard QP inequality (`GroundPlaneLimit`) and a soft repulsive variant (`SoftGroundConstraint`/`GMRWithSoftGround`) wired into `mink`. See `docs/pipeline.md` §2. |
| **Strict zero-penetration two-pass IK** | `general_motion_retargeting/retargeting/strict_zero_pen.py` | Pass 1 surveys per-frame penetration, dilation + Gaussian smoothing produces a soft envelope, Pass 2 re-runs IK with the entire target skeleton lifted by the envelope. Result: 0.0 mm penetration with no foot-slide. ~2× runtime. See `docs/pipeline.md` §2.5. |
| **Foot-ground contact labels** | `general_motion_retargeting/retargeting/foot_contact.py` | Per-frame `(N, 2)` bool flags derived from raw SMPL-X toe kinematics — stable across `--ground_mode` and robot choices. Written into the output pkl as `foot_ground_contact_flags` + a self-describing `foot_contact_meta` dict. See `docs/foot_contact.md`. |
| **Unified pipeline CLI** | `scripts/retarget_no_penetration.py` | One entry point for SMPL-X (AMASS / OMOMO), AMASS CMU `_stageii.npz` with stem cleaning, GVHMR `.pt`, and BVH LAFAN1 (experimental). Replaces the per-format converters. |
| **Modular pipeline subpackage** | `general_motion_retargeting/retargeting/` | The CLI is glue; the heavy lifting is in `builder.py`, `strict_zero_pen.py`, `foot_contact.py`, `fk_post.py`, `batch_runner.py` so each piece is testable and reusable. |
| **Self-describing pkl schema** | `scripts/retarget_no_penetration.py` | Outputs record `robot`, `input_format`, `source_file`, `strict_zero_pen`, `ground_mode`, `actual_human_height`, plus the foot-contact metadata, so a downstream consumer doesn't need CLI args to interpret the result. See `docs/pipeline.md` §5.7. |
| **Booster K1 + T1 focus** | `assets/`, `general_motion_retargeting/sole_points.py`, `params.py`, `ik_configs/` | Sole geometry, ground-constraint defaults, and validation are tuned for K1 + T1. Other upstream robots have been removed to keep the surface area small and the documentation honest. |
| **Dataset analysis** | `scripts/analyze_locomotion_dataset.py` | Walks a tree of pkls, computes per-file penetration / jitter via MuJoCo FK, optionally maps stems to motion categories from a YAML index, emits a Markdown report. Useful for triaging which retargeted clips are good enough to train on. |
| **Foot-contact viz overlay** | `scripts/vis_robot_motion_with_contact.py` | Renders a pkl to mp4 and re-encodes with per-frame `L FOOT` / `R FOOT` text drawn while each foot is flagged in contact — quick visual sanity check on the contact labels. |

## 3. Where things go in this repo

```
GMR/
├── README.md                 # quickstart, points here for context
├── docs/
│   ├── overview.md           # this file
│   ├── pipeline.md           # canonical pipeline reference
│   ├── foot_contact.md       # contact flag PKL schema + consumer examples
│   ├── bvh.md                # BVH (LAFAN1) integration, experimental
│   ├── ik_config.md          # IK config field reference
│   ├── test_motions.md       # tricky motions to run when validating
│   └── archive/              # historical predecessors of pipeline.md
├── general_motion_retargeting/
│   ├── motion_retarget.py, kinematics_model.py    # upstream IK core
│   ├── ground_constraint.py, sole_points.py       # NEW: ground-aware IK
│   ├── retargeting/                               # NEW: pipeline subpackage
│   │   ├── builder.py            # GMR + ground constraint + sole offset
│   │   ├── strict_zero_pen.py    # Pass-1 surveyor + smoothed envelope
│   │   ├── foot_contact.py       # SMPL-X toe contact detection
│   │   ├── fk_post.py            # height/origin shifts + local_body_pos
│   │   └── batch_runner.py       # directory & YAML batch driver
│   ├── ik_configs/                                 # K1 + T1 only
│   └── utils/                                      # SMPL/BVH loaders
├── scripts/
│   ├── retarget_no_penetration.py                  # THE pipeline CLI
│   ├── vis_robot_motion*.py                        # MuJoCo viewers
│   ├── vis_robot_motion_with_contact.py            # contact overlay
│   ├── analyze_locomotion_dataset.py               # dataset stats
│   ├── compare_penetration_stats.py                # diagnostic
│   └── add_sole_sites.py                           # one-time MJCF prep
├── shell/                                          # thin wrappers
└── assets/booster_k1, booster_t1, body_models/
```

## 4. Getting started

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
conda install -c conda-forge libstdcxx-ng -y

# AMASS CMU walk on K1, two-pass smoothing, headless
python scripts/retarget_no_penetration.py \
    --input /data/AMASS/CMU/35/35_01_stageii.npz \
    --input_format amass_cmu \
    --robot booster_k1 \
    --ground_mode qp --strict_zero_pen \
    --headless --output retargeted/35_01.pkl
```

For the full pipeline reference (every flag, the two-pass algorithm, schema
details, empirical results, known limitations), read **`docs/pipeline.md`**.
For consuming the output pkl in downstream RL code, read
**`docs/foot_contact.md`**.
