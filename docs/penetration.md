# Ground Penetration Prevention for GMR (archived)

> **Archived.** This document predates the unified pipeline doc. The current
> design reference is [`docs/pipeline.md`](pipeline.md), which covers
> everything below plus the two-pass IK, dynamic CBF gain, T1 robot, and
> foot-ground contact flags. This file is kept for git-blame continuity.

This document covers the ground penetration prevention system added to the General Motion
Retargeting (GMR) pipeline for the Booster K1 robot. It explains the problem, the
implementation, how to use it, empirical results, and open problems.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Architecture Overview](#2-architecture-overview)
3. [Files Changed / Added](#3-files-changed--added)
4. [Implementation Details](#4-implementation-details)
   - [Sole Contact Points](#41-sole-contact-points)
   - [GroundPlaneLimit (QP hard constraint)](#42-groundplanelimit-qp-hard-constraint)
   - [RootZLimit (per-step root cap)](#43-rootzlimit-per-step-root-cap)
   - [SoftGroundConstraint & GMRWithSoftGround](#44-softgroundconstraint--gmrwithsoftground)
   - [Sole Compensation (ground_offset)](#45-sole-compensation-ground_offset)
   - [height_adjust incompatibility fix](#46-height_adjust-incompatibility-fix)
   - [ik_limits2 separation in motion_retarget.py](#47-ik_limits2-separation-in-motion_retargetpy)
5. [How to Run](#5-how-to-run)
   - [Single-file retargeting](#51-single-file-retargeting)
   - [AMASS CMU batch via YAML](#52-amass-cmu-batch-via-yaml)
   - [Visualisation and comparison](#53-visualisation-and-comparison)
   - [Preparing soft-mode MJCF](#54-preparing-soft-mode-mjcf)
   - [Generated data files](#55-generated-data-files)
6. [Empirical Results](#6-empirical-results)
7. [Pros and Cons of Each Mode](#7-pros-and-cons-of-each-mode)
8. [Known Limitations and Open Problems](#8-known-limitations-and-open-problems)
9. [What Can Be Further Improved](#9-what-can-be-further-improved)

---

## 1. The Problem

Vanilla GMR retargets human motion by solving an inverse kinematics (IK) problem at each
frame independently. The IK minimises the error between robot body positions/orientations
and scaled human body positions/orientations — it has no notion of the ground plane. As a
result, foot links frequently penetrate below z = 0:

- **Walking motions**: 78–102% of frames have at least one sole corner below the ground,
  with penetration depths up to 80 mm.
- **Running motions**: 86–95% of frames penetrate, depths up to 77 mm.
- **Kick motions**: Moderate penetration at peak-swing and landing frames.

This is problematic for RL policy training data, physics simulation, and visual plausibility.

The goal: keep all four corners of both foot soles at or above `z = clearance` (default 3 mm)
throughout the motion, with minimal distortion of the original retargeted pose.

---

## 2. Architecture Overview

The mink IK solver formulates each solve step as a **Quadratic Program (QP)**:

```
minimise   ½ ‖J·Δq − e‖²  (task residuals)
subject to  G·Δq ≤ h       (inequality limits)
            Aq = b         (equality limits, e.g. joint limits)
```

The QP variable is **Δq** (joint displacement), not velocity. `solve_ik` returns
`v = Δq / dt`; `integrate_inplace(v, dt)` applies `q += v·dt = Δq`.

Ground penetration prevention is implemented as **additional rows in G·Δq ≤ h**:

- **Hard QP constraint (GroundPlaneLimit)**: For each sole corner near the ground, adds one
  row `-J_z · Δq ≤ gain · margin` that prevents the corner from descending further.
- **Soft repulsive task (SoftGroundConstraint)**: Adds high-weight FrameTask instances that
  push sole sites upward when they approach the ground. Less rigorous but does not
  modify the constraint matrix directly.

---

## 3. Files Changed / Added

| File | Status | What changed |
|------|--------|--------------|
| `general_motion_retargeting/ground_constraint.py` | **NEW** | `GroundPlaneLimit`, `RootZLimit`, `SoftGroundConstraint`, `GMRWithSoftGround` |
| `general_motion_retargeting/sole_points.py` | **NEW** | K1 sole corner coordinates, site name generation |
| `general_motion_retargeting/motion_retarget.py` | **MODIFIED** | Added `ik_limits2 = []`; all `solve_ik` calls use `limits=` keyword; tasks-2 solve uses `limits2 = ik_limits + ik_limits2` |
| `scripts/retarget_no_penetration.py` | **NEW** | Unified entry point for vanilla/qp/soft retargeting, single/batch/YAML modes, AMASS CMU format support, FK post-processing |
| `scripts/vis_compare_motions.py` | **NEW** | Side-by-side comparison video renderer |
| `scripts/compare_penetration_stats.py` | **NEW** | Penetration and root-jitter statistics across PKL files |
| `scripts/add_sole_sites.py` | **NEW** | One-time MJCF prep: injects `<site>` elements for soft mode |

### Change to `motion_retarget.py`

The original code passed `ik_limits` as a **positional argument** to `solve_ik`, which
silently mapped it to the `safety_break` parameter instead of `limits`. This was a critical
bug: no joint limits were enforced. Fixed by using the `limits=` keyword explicitly.

The `ik_limits2` list was added to allow constraints that should apply only to the tasks-2
solve (position+rotation matching) — specifically, to avoid infeasibility in tasks-1
(pure-rotation matching) which can require large root translations during initialisation.

---

## 4. Implementation Details

### 4.1 Sole Contact Points

`general_motion_retargeting/sole_points.py` defines the four bottom corners of each foot's
collision box in the **foot link's local frame**. For K1:

```
foot box geom: size=[0.08, 0.035, 0.016] (half-extents), pos=[0.014, 0.0, -0.008]
sole surface z = -0.008 - 0.016 = -0.024 m below the foot link origin
```

Four corners (fl, fr, rl, rr) span x ∈ [-0.066, 0.094], y ∈ [-0.035, 0.035], z = -0.024.

To add support for another robot, add an entry to `SOLE_CONTACT_POINTS` in `sole_points.py`.
Use `detect_sole_points_from_mjcf()` as a starting point (it parses the MJCF box geoms
automatically).

### 4.2 GroundPlaneLimit (QP hard constraint)

**File**: `ground_constraint.py`, class `GroundPlaneLimit(Limit)`

For each sole corner with world z-coordinate `z_p` near the ground, linearise the constraint
`z_p(q + Δq) ≥ ground_height + clearance`:

```
z_p + J_z · Δq ≥ ground_height + clearance
→  -J_z · Δq ≤ gain · (z_p - ground_height - clearance)   [margin = z_p - gh - cl]
```

This is one row of the QP inequality `G·Δq ≤ h`.

- When `margin > 0` (sole above clearance): h > 0, constraint is loose — the sole can move
  down a little, but only by `gain · margin` per step (CBF-style).
- When `margin < 0` (sole penetrating): h < 0, constraint forces upward correction of at
  least `gain · |margin|` per step.
- `gain = 0.5` (default): half-correction per step; converges in ~3 iterations.
- `activation_distance = 0.02` (default): constraint is skipped entirely for soles more than
  20 mm above `ground_height + clearance` — avoids unnecessary QP rows during swing phase.

**Key detail**: The Jacobian is computed at the **world position of the corner point**, not
the body origin. `mj_jac(model, data, jacp, jacr, world_pt, body_id)` gives the full
translational Jacobian including contributions from all ancestor joints including the free
joint. This correctly captures how root translation and joint rotations jointly affect the
corner's world z-coordinate.

### 4.3 RootZLimit (per-step root cap)

**File**: `ground_constraint.py`, class `RootZLimit(Limit)`

**Design intent**: Cap the upward root-z displacement per IK step when any sole is near the
ground, to prevent the "bang-bang" root snap seen at heel-strike in locomotion.

**How it works**: When any sole corner is within `activation_distance` of `ground_height +
clearance`, adds one QP row `e_z · Δq ≤ max_dz`, where `e_z` is the z-unit vector for the
free-joint z DOF (index 2).

**Empirical finding**: This class has no measurable effect in practice for locomotion.

The root cause of heel-strike root jitter was diagnosed as follows: at each frame, **tasks-1**
(pure-rotation solve) moves the root upward by **19–47 mm** as a side-effect of matching body
orientations. Then **tasks-2** partially corrects this downward (−6 to −23 mm). The net
frame-to-frame jump is 10–24 mm. Since `RootZLimit` is placed in `ik_limits2` (tasks-2
only), it never fires during the phase that causes the damage.

Moving `RootZLimit` to shared `ik_limits` (affecting tasks-1 too) causes QP **infeasibility**:
`GroundPlaneLimit` requires root to go UP to keep soles above ground, while `RootZLimit`
prevents upward movement beyond `max_dz`. These two constraints directly conflict.

The `--max_root_dz` CLI flag is retained but has no practical effect.

### 4.4 SoftGroundConstraint & GMRWithSoftGround

**File**: `ground_constraint.py`

An alternative to hard constraints: add `mink.FrameTask` instances for sole *sites* with
dynamically weighted position cost that ramps up as the site descends toward the ground.

```
weight = max_weight * (1 - (z - target_z) / activation_distance)   when z < target_z + activation_distance
       = 0                                                            when z ≥ target_z + activation_distance
```

The task pushes each site's z up to `ground_height + clearance`, competing with the motion
tracking tasks.

**Preparation required**: The K1 MJCF does not have `<site>` elements by default. Run
`python scripts/add_sole_sites.py` once to produce `assets/booster_k1/K1_serial_with_sites.xml`.

`GMRWithSoftGround` is a wrapper class that injects `soft_ground.update()` before every
`solve_ik` call to refresh task weights and targets.

**Empirical finding**: Soft mode is less effective than QP — 62% of walk frames still
penetrate vs 6% for QP. The soft tasks compete with but do not dominate the high-weight
foot-tracking tasks. It is not recommended for walk or run data generation.

### 4.5 Sole Compensation (ground_offset)

A critical correction absent from vanilla retargeting: the IK config maps the human *ankle*
to the robot *ankle* at z = 0. But the sole surface is **24 mm below** the ankle origin.
Without compensation, the robot "stands" with its ankle at ground level and its sole 24 mm
underground.

Fix: call `retarget.set_ground_offset(sole_compensation)` where:

```python
min_sole_z = min(pt[2] for pts in sole_config.values() for pt in pts)  # = -0.024 m
sole_compensation = min_sole_z - clearance  # = -0.024 - 0.003 = -0.027 m
retarget.set_ground_offset(-0.027)
```

`apply_ground_offset()` subtracts `ground_offset` from all body z targets, effectively
raising all targets by 27 mm so the robot stands with its sole at `z = clearance = 3 mm`.

This is applied automatically when `--ground_mode` is not `none`.

### 4.6 height_adjust incompatibility fix

The AMASS CMU pipeline supports `--height_adjust`, which uses FK to find the lowest body
link position and shifts the root so the robot is "grounded". However, the FK uses link
**origins** (ankle-level), not sole corners. The ankle is ~27 mm above the sole, so
`height_adjust` would shift the root down by 27 mm — exactly undoing the sole compensation
and leaving soles 24 mm underground.

Fix: when `--ground_mode != none`, `height_adjust` is silently skipped with a warning
message. The ground constraint already handles grounding correctly.

### 4.7 ik_limits2 separation in motion_retarget.py

`GeneralMotionRetargeting` now has two limit lists:

- `ik_limits`: shared between tasks-1 and tasks-2. Contains `ConfigurationLimit` (always)
  and `GroundPlaneLimit` (when using QP mode).
- `ik_limits2`: applied to tasks-2 only. Currently used for `RootZLimit` (though
  ineffective) and designed for future use.

Tasks-2 solve uses `limits2 = ik_limits + ik_limits2`.

---

## 5. How to Run

### 5.1 Single-file retargeting

```bash
AMASS=/path/to/AMASS   # root of AMASS dataset (parent of CMU/)

# Vanilla (no constraint)
python scripts/retarget_no_penetration.py \
    --input $AMASS/CMU/35/35_14_stageii.npz \
    --input_format amass_cmu \
    --ground_mode none \
    --no_viz \
    --output data/locomotion_amass_cmu/35_14_vanilla.pkl

# QP hard constraint (recommended for walk/run)
python scripts/retarget_no_penetration.py \
    --input $AMASS/CMU/35/35_14_stageii.npz \
    --input_format amass_cmu \
    --ground_mode qp \
    --no_viz \
    --output data/locomotion_amass_cmu/35_14_qp.pkl

# Soft repulsive tasks (requires K1_serial_with_sites.xml — see 5.4)
python scripts/retarget_no_penetration.py \
    --input $AMASS/CMU/35/35_14_stageii.npz \
    --input_format amass_cmu \
    --ground_mode soft \
    --no_viz \
    --output data/locomotion_amass_cmu/35_14_soft.pkl

# GVHMR input (e.g. kick motion)
python scripts/retarget_no_penetration.py \
    --input data/kick_gvhmr/kick_soogon_2.pt \
    --input_format gvhmr \
    --ground_mode qp \
    --no_viz \
    --output data/kick_gvhmr/kick_soogon_2_qp.pkl
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--ground_mode` | `none` | `none` = vanilla, `qp` = hard constraint, `soft` = repulsive tasks |
| `--clearance` | `0.003` | Minimum sole z above ground (metres) |
| `--gain` | `0.5` | QP CBF gain: fraction of violation corrected per IK step |
| `--activation_distance` | `0.02` | Distance above clearance at which constraint activates (metres) |
| `--max_weight` | `500.0` | Soft mode: maximum repulsive task weight |
| `--height_adjust` | off | Shift root z so lowest FK body point is at z=0 (skipped if ground_mode≠none) |
| `--root_origin_offset` | off | Translate XY so frame 0 root is at origin |
| `--no_viz` | off | Disable viewer (always set this for batch/headless) |

### 5.2 AMASS CMU batch via YAML

```bash
python scripts/retarget_no_penetration.py \
    --yaml /path/to/amass_cmu_locomotion_list.yaml \
    --amass_root /path/to/AMASS \
    --input_format amass_cmu \
    --ground_mode qp \
    --output retargeted_cmu/ \
    --override
```

The YAML format (e.g. `AMASS/CMU/amass_cmu_locomotion_list.yaml`) is:

```yaml
walk:
  - description: "walk"
    path: "CMU/02/02_01_stageii.npz"
  - description: "walk"
    path: "CMU/02/02_02_stageii.npz"
run:
  - description: "run"
    path: "CMU/09/09_01_stageii.npz"
```

`--amass_root` is prepended to each `path`. The `_stageii` suffix is stripped from output
filenames. Output structure mirrors the input: `retargeted_cmu/CMU/02/02_01.pkl`.

### 5.3 Visualisation and comparison

```bash
# Compare two PKL files side by side
python scripts/vis_compare_motions.py \
    --motion_a data/35_14_vanilla.pkl \
    --motion_b data/35_14_qp.pkl \
    --label_a "Vanilla" \
    --label_b "QP" \
    --video_path videos/35_14_comparison.mp4 \
    --show_sole_points   # coloured spheres: red = penetrating, green = ok

# Penetration and jitter statistics
python scripts/compare_penetration_stats.py \
    data/35_14_vanilla.pkl \
    data/35_14_qp.pkl \
    data/35_14_soft.pkl

# Headless rendering (no display server)
MUJOCO_GL=egl python scripts/vis_compare_motions.py ...
```

**Stats columns:**
- `pen/N`: frames with any sole corner below `ground_height + clearance`
- `max_pen`: maximum penetration depth (mm)
- `max_up`: maximum frame-to-frame upward root-z jump (mm) — proxy for jitter
- `mean_up`: mean upward jump (mm)
- `rz_range`: total range of root z over the motion (mm)

### 5.4 Preparing soft-mode MJCF

```bash
python scripts/add_sole_sites.py
# Output: assets/booster_k1/K1_serial_with_sites.xml
```

This is a one-time operation. The script reads `sole_points.py` and injects `<site>` elements
at each sole corner into the K1 MJCF. Soft mode uses this XML automatically.

---

## 5.5 Generated data files

PKL files are not tracked in git. The following files were generated for evaluation and are
stored locally. Re-generate with the commands in §5.1–5.2.

**`data/locomotion_amass_cmu/`** — AMASS CMU locomotion motions:

| File | Source | Motion type | Frames | FPS |
|------|--------|-------------|-------:|----:|
| `35_14_vanilla.pkl` | CMU/35/35_14_stageii.npz | walk | 102 | 29.8 |
| `35_14_qp.pkl` | CMU/35/35_14_stageii.npz | walk | 102 | 29.8 |
| `35_14_soft.pkl` | CMU/35/35_14_stageii.npz | walk | 102 | 29.8 |
| `02_01_vanilla.pkl` | CMU/02/02_01_stageii.npz | walk | 85 | 29.7 |
| `02_01_qp.pkl` | CMU/02/02_01_stageii.npz | walk | 85 | 29.7 |
| `02_01_soft.pkl` | CMU/02/02_01_stageii.npz | walk | 85 | 29.7 |
| `07_01_vanilla.pkl` | CMU/07/07_01_stageii.npz | walk | 79 | 30.0 |
| `07_01_qp.pkl` | CMU/07/07_01_stageii.npz | walk | 79 | 30.0 |
| `07_01_soft.pkl` | CMU/07/07_01_stageii.npz | walk | 79 | 30.0 |
| `02_03_vanilla.pkl` | CMU/02/02_03_stageii.npz | run/jog | 43 | 29.8 |
| `02_03_qp.pkl` | CMU/02/02_03_stageii.npz | run/jog | 43 | 29.8 |
| `02_03_soft.pkl` | CMU/02/02_03_stageii.npz | run/jog | 43 | 29.8 |
| `09_01_vanilla.pkl` | CMU/09/09_01_stageii.npz | run | 37 | 30.0 |
| `09_01_qp.pkl` | CMU/09/09_01_stageii.npz | run | 37 | 30.0 |
| `09_01_soft.pkl` | CMU/09/09_01_stageii.npz | run | 37 | 30.0 |

**`data/kick_gvhmr/`** — GVHMR kick motion (input `.pt` file is tracked):

| File | Motion type | Frames | FPS |
|------|-------------|-------:|----:|
| `kick_soogon_2_vanilla.pkl` | kick | 134 | 30.0 |
| `kick_soogon_2_qp.pkl` | kick | 134 | 30.0 |
| `kick_soogon_2_soft.pkl` | kick | 134 | 30.0 |

---

## 6. Empirical Results

All results use Booster K1, 30 fps, clearance = 3 mm, gain = 0.5, activation_distance = 20 mm.
Stats computed by `scripts/compare_penetration_stats.py`.

Column definitions:
- **pen/N**: frames with any sole corner below `ground_height + clearance`
- **max_pen**: worst-case penetration depth (mm)
- **max_up**: maximum frame-to-frame upward root-z jump (mm) — proxy for jitter
- **mean_up**: mean upward jump across all positive jumps (mm)
- **rz_range**: total root-z range over the motion (mm)

### Walk motions (AMASS CMU)

| Motion | Mode | frames | pen/N | max_pen | max_up | mean_up | rz_range |
|--------|------|-------:|-------|--------:|-------:|--------:|---------:|
| CMU 35_14 | vanilla | 102 | 102/102 | 63.5 mm | 14.6 mm | 4.6 mm | 45.7 mm |
| CMU 35_14 | **qp** | 102 | **6/102** | **9.3 mm** | **10.5 mm** | **3.6 mm** | 37.4 mm |
| CMU 35_14 | soft | 102 | 62/102 | 36.5 mm | 14.6 mm | 4.6 mm | 45.7 mm |
| CMU 02_01 | vanilla | 85 | 85/85 | 80.6 mm | 10.4 mm | 4.1 mm | 60.9 mm |
| CMU 02_01 | **qp** | 85 | **0/85** | **0.0 mm** | **7.5 mm** | **2.5 mm** | 32.3 mm |
| CMU 02_01 | soft | 85 | 44/85 | 53.6 mm | 12.0 mm | 4.1 mm | 60.9 mm |
| CMU 07_01 | vanilla | 79 | 78/79 | 81.3 mm | 11.2 mm | 5.1 mm | 70.0 mm |
| CMU 07_01 | **qp** | 79 | **2/79** | **8.5 mm** | **9.6 mm** | **3.6 mm** | 39.7 mm |
| CMU 07_01 | soft | 79 | 45/79 | 54.3 mm | 14.5 mm | 5.1 mm | 70.0 mm |

### Run motions (AMASS CMU)

| Motion | Mode | frames | pen/N | max_pen | max_up | mean_up | rz_range |
|--------|------|-------:|-------|--------:|-------:|--------:|---------:|
| CMU 02_03 (jog) | vanilla | 43 | 41/43 | 77.2 mm | 21.5 mm | 8.7 mm | 79.0 mm |
| CMU 02_03 (jog) | **qp** | 43 | **0/43** | **0.0 mm** | 23.3 mm | 9.2 mm | 61.6 mm |
| CMU 02_03 (jog) | soft | 43 | 23/43 | 50.0 mm | 21.5 mm | 8.7 mm | 78.9 mm |
| CMU 09_01 (run) | vanilla | 37 | 32/37 | 52.3 mm | 13.3 mm | 5.9 mm | 45.4 mm |
| CMU 09_01 (run) | **qp** | 37 | **0/37** | **0.0 mm** | 23.8 mm | 7.2 mm | 42.0 mm |
| CMU 09_01 (run) | soft | 37 | 18/37 | 25.6 mm | 11.8 mm | 4.5 mm | 39.1 mm |

### Kick motion (GVHMR)

| Motion | Mode | frames | pen/N | max_pen | max_up | mean_up | rz_range |
|--------|------|-------:|-------|--------:|-------:|--------:|---------:|
| kick_soogon_2 | vanilla | 134 | 134/134 | 48.7 mm | 7.8 mm | 1.0 mm | 45.9 mm |
| kick_soogon_2 | **qp** | 134 | **0/134** | **0.0 mm** | **6.3 mm** | **0.8 mm** | 33.1 mm |
| kick_soogon_2 | soft | 134 | 24/134 | 21.7 mm | 7.8 mm | 1.0 mm | 45.9 mm |

### Key observations

1. **QP eliminates penetration completely for run and kick motions** (0/43, 0/37, 0/134)
   and reduces walk penetration by 94–100%.
2. **QP reduces root jitter for walk and kick motions** — max_up drops from 14.6→10.5,
   10.4→7.5, 11.2→9.6 mm (walk) and 7.8→6.3 mm (kick).
3. **QP increases root jitter for run motions** (13.3→23.8 mm for 09_01, 21.5→23.3 mm
   for 02_03). This is a fundamental tradeoff — the constraint prevents natural heel-strike
   compression and causes root overshoot at landing (see §8.2).
4. **Soft mode performs poorly across all motion types** — 44–62% of walk frames still
   penetrate, 53–60% of run frames, 18% of kick frames. The repulsive tasks are
   outcompeted by the high-weight foot-tracking tasks and are not recommended for
   production data generation.
5. **Soft mode is surprisingly competitive on run motions** relative to QP on jitter
   (09_01: soft max_up = 11.8 mm vs QP = 23.8 mm), but at the cost of 18/37 frames
   still penetrating — an unacceptable tradeoff for training data.

---

## 7. Pros and Cons of Each Mode

### Mode: `none` (vanilla GMR)

**Pros:**
- No additional constraints → lowest computational overhead
- Preserves the original retargeted pose exactly
- Root trajectory is smooth (natural human gait dynamics)

**Cons:**
- Massive ground penetration in virtually all locomotion frames (up to 80 mm)
- Unusable for physics simulation without post-processing
- Visually implausible

**Use when:** Only visualising retargeting quality; not for training data.

---

### Mode: `qp` (GroundPlaneLimit)

**Pros:**
- Eliminates penetration almost entirely for walk motions (0–6% residual frames)
- Eliminates penetration completely for run motions (0%)
- Hard constraint: guaranteed not to worsen within a single solve step (CBF property)
- Reduces root jitter for walk motions as a side-effect
- No special MJCF preparation required
- Computationally cheap: adds at most 8 QP rows (4 corners × 2 feet) per iteration

**Cons:**
- Small residual penetrations remain in walk (a few frames, ≤9 mm) due to:
  (a) finite CBF gain (partial correction per step), and
  (b) multi-step iterative convergence stopping at a local minimum
- Increases root jitter for run motions (13→24 mm). The constraint prevents natural
  heel-strike compression and causes root overshoot
- Sole compensation (ground_offset) must be set correctly; wrong values → robot floats
  or penetrates
- Only implemented for Booster K1; adding other robots requires calibrating
  `SOLE_CONTACT_POINTS` in `sole_points.py`

**Use when:** Generating locomotion (walk/run) training data. Walk quality is excellent;
run quality is good (no penetrations) but with slightly increased root jitter.

---

### Mode: `soft` (SoftGroundConstraint)

**Pros:**
- Softer intervention: the repulsive force fades out smoothly above the activation zone
- No risk of QP infeasibility (it's just a task, not a hard constraint)
- Theoretically more natural foot-ground interaction

**Cons:**
- Poor empirical performance: 62% of walk frames still penetrate at default settings
- The soft weight (default 500) is far outcompeted by foot-tracking tasks (weight 100–50
  combined with both position and rotation matching)
- Requires a modified MJCF (`K1_serial_with_sites.xml`) with injected `<site>` elements
- Tuning `max_weight` to fix penetration causes other tracking artifacts ("flying" robot)
- Only applicable to robots with `<site>` elements defined

**Use when:** Experimental / comparative study only. Not recommended for production data.

---

## 8. Known Limitations and Open Problems

### 8.1 Residual penetrations in walk mode

A few frames per motion (typically 2–6%) still have slight penetration (≤9 mm). These
occur at frames where:
- The human foot target is well below ground (the IK can't fully satisfy both foot tracking
  and ground constraint in the convergence budget)
- CBF gain = 0.5 deliberately allows partial correction per step to avoid over-constraining
  the IK; a single step can only reduce violation by 50%

### 8.2 Run motion root jitter

For running, the QP constraint causes root overshoot at landing frames. The sole enters the
activation zone during the foot's descent (20 mm above ground). The constraint fires and
forces root upward before the foot naturally lands. The IK converges to a pose where the
root is ~24 mm higher than vanilla — and on the next frame, root snaps back down.

This is a **fundamental architectural problem**: the frame-independent IK has no continuity
between frames, so it cannot anticipate or smooth the landing.

### 8.3 RootZLimit is ineffective

The `RootZLimit` class was designed to cap upward root displacement per IK step. Investigation
showed that:
1. The root jump is mainly caused by **tasks-1** (rotation-only solve), which runs before
   tasks-2 where `RootZLimit` lives.
2. Moving `RootZLimit` to shared `ik_limits` causes QP infeasibility because
   `GroundPlaneLimit` (ground correction) and `RootZLimit` (cap upward movement) directly
   conflict when a sole is penetrating and root must go up to fix it.

### 8.4 Only Booster K1 is supported

The constraint infrastructure requires calibrated sole contact points in `sole_points.py`.
Only K1 has these defined. Other robots (G1, T1, etc.) would need:
1. Sole corner coordinates measured from their MJCF
2. New entries in `SOLE_CONTACT_POINTS` and optionally `FOOT_LINK_NAMES`

### 8.5 Frame-independent IK

The IK solves each frame independently. There is no temporal regularisation. This means:
- Discontinuities between frames are possible (especially at state transitions)
- The solver may converge to different local minima for adjacent frames
- Jitter is measured statistically (max_up) but not corrected at the IK level

---

## 9. What Can Be Further Improved

### 9.1 Temporal smoothing (post-processing)

The simplest improvement: apply a low-pass filter to `root_pos[:, 2]` after retargeting.

```python
from scipy.ndimage import uniform_filter1d
root_pos[:, 2] = uniform_filter1d(root_pos[:, 2], size=3)
```

This would directly reduce the max_up jump (measured as frame-to-frame delta) with no
changes to the IK. Risk: if the smoothing lowers root-z into a frame where soles are
constrained, sole position would look wrong. Could be combined with a post-processing
penetration fix. Would benefit both walk (10.5→~7 mm) and run (23.8→~16 mm) modes.

### 9.2 Warm-starting from the previous frame

Currently, the IK starts each frame from where the previous frame left off (the
`configuration` object carries over between frames — this is intentional). But no
**target temporal continuity** is enforced. Adding a small soft task that keeps the
configuration close to the previous frame's solution would damp oscillations:

```python
task = mink.PostureTask(model, cost=0.1)
task.set_target(previous_qpos)
```

This would act as a temporal regulariser without hard constraints.

### 9.3 Velocity limits

`GeneralMotionRetargeting.__init__` already supports `use_velocity_limit=True`, which adds
`mink.VelocityLimit` to `ik_limits`. This is separate from the ground constraint and could
be combined with QP mode. It would limit joint velocities (not root velocity) per step.

### 9.4 Contact-aware foot placement

Rather than simply preventing foot corners from going below z = 0, a contact model could:
1. Detect heel/toe contact from human motion (heel-strike, toe-off events)
2. Enforce exact foot contact during stance phase (foot stays at ground level)
3. Allow full foot freedom during swing phase

This would produce more physically plausible motions but requires detecting the gait cycle.

### 9.5 Extending to other robots

Adding a new robot requires:
1. Calibrate sole corner coordinates from the MJCF (use `detect_sole_points_from_mjcf()`)
2. Add to `SOLE_CONTACT_POINTS` and `FOOT_LINK_NAMES` in `sole_points.py`
3. Run `python scripts/add_sole_sites.py` (once updated for the new robot) if using soft mode
4. Change `ROBOT_TYPE = "booster_k1"` in the scripts to the new robot

The `retarget_no_penetration.py` script hardcodes `ROBOT_TYPE = "booster_k1"`. Adding a
`--robot` flag would make the pipeline robot-agnostic.

### 9.6 Better CBF gain scheduling

The current CBF gain is fixed at 0.5. A variable gain that:
- Increases when penetration is large (faster correction)
- Decreases when near the threshold (smoother approach)

would improve both convergence speed and oscillation. Example: `gain = min(1.0, 0.5 + 2.0 * |margin|)`.

### 9.7 Global trajectory optimisation

The most principled fix: instead of frame-by-frame IK, solve a **trajectory optimisation**
over the full motion that simultaneously matches human poses and respects ground contact.
This is computationally expensive but would eliminate frame-independence artifacts entirely.
Libraries like `pinocchio` + `crocoddyl` or `mink`'s planned trajectory mode could be used.

---

*Generated from empirical testing on AMASS CMU locomotion motions and GVHMR kick motion,
Booster K1 robot, March 2026.*
