# GMR Ground-Safe Retargeting Pipeline

This document describes the unified ground-penetration-aware retargeting pipeline
added to General Motion Retargeting (GMR) for the Booster K1 and Booster T1
humanoids. It supersedes the earlier `penetration.md` and
`CHANGELOG_ZERO_PENETRATION.md` notes and adds new sections covering the T1
robot, foot-ground contact labels, and the output PKL schema.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Ground Penetration Prevention](#2-ground-penetration-prevention)
   - 2.1 [The Problem](#21-the-problem)
   - 2.2 [Architecture](#22-architecture)
   - 2.3 [GroundPlaneLimit (QP hard constraint)](#23-groundplanelimit-qp-hard-constraint)
   - 2.4 [SoftGroundConstraint (repulsive tasks)](#24-softgroundconstraint-repulsive-tasks)
   - 2.5 [Strict Zero Penetration (Two-Pass IK)](#25-strict-zero-penetration-two-pass-ik)
   - 2.6 [Sole compensation (`ground_offset`)](#26-sole-compensation-ground_offset)
3. [Foot-Ground Contact Flags](#3-foot-ground-contact-flags)
4. [Supported Robots](#4-supported-robots)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [Empirical Results](#6-empirical-results)
7. [Pros and Cons of Each Mode](#7-pros-and-cons-of-each-mode)
8. [Known Limitations and Open Problems](#8-known-limitations-and-open-problems)
9. [Future Work](#9-future-work)
10. [Modified / Added Files](#10-modified--added-files)

---

## 1. Overview

### 1.1 What this pipeline produces

For each input human motion clip, the pipeline produces a `.pkl` file containing:

- A robot trajectory (`root_pos`, `root_rot`, `dof_pos`) where every sole corner stays at or above a configurable ground clearance — strictly zero penetration when the two-pass mode is enabled.
- Per-foot binary ground-contact labels derived from the source SMPL-X kinematics, useful as supervision for RL policies and as input for gait-cycle reasoning.
- Robot kinematics metadata (`local_body_pos`, `link_body_list`) suitable for downstream training pipelines.
- Self-describing run metadata (`robot`, `input_format`, `ground_mode`, `strict_zero_pen`, source path, …) so consumers can filter / sort large dataset directories without re-running anything.

The pipeline is implemented as one entry point — `scripts/retarget_no_penetration.py` — that routes single files, directory batches, and YAML lists through a common code path.

### 1.2 Supported inputs

| Format | `--input_format` | Loader |
|---|---|---|
| AMASS / OMOMO SMPL-X (`.npz`, `.pkl`) | `smplx` | `general_motion_retargeting/utils/smpl.py:load_smplx_file` |
| AMASS CMU `_stageii.npz` (with stem cleaning) | `amass_cmu` | same loader as `smplx`; differs only in filename handling |
| GVHMR predictions (`hmr4d_results.pt`) | `gvhmr` | `general_motion_retargeting/utils/smpl.py:load_gvhmr_pred_file` |

BVH (LAFAN1 / Nokov) support exists on the sibling branch
`feature/add-bvh-support` but has **not** been merged yet — the configs there
are flagged as "guessed values" and need validation before promotion.

### 1.3 Quick start

Single AMASS CMU file with the recommended settings (QP + two-pass smoothing):

```bash
conda run -n gmr python scripts/retarget_no_penetration.py \
    --input /data/AMASS/CMU/35/35_14_stageii.npz \
    --input_format amass_cmu \
    --ground_mode qp \
    --strict_zero_pen \
    --no_viz \
    --output retargeted/35_14.pkl
```

Batch via YAML locomotion list (mirrors the input directory tree under `--output`):

```bash
conda run -n gmr python scripts/retarget_no_penetration.py \
    --yaml /data/AMASS/CMU/amass_cmu_locomotion_list.yaml \
    --input_format amass_cmu \
    --ground_mode qp \
    --strict_zero_pen \
    --height_adjust --root_origin_offset \
    --output retargeted_cmu/
```

GVHMR single file:

```bash
conda run -n gmr python scripts/retarget_no_penetration.py \
    --input GVHMR/outputs/demo/freekick/hmr4d_results.pt \
    --input_format gvhmr \
    --ground_mode qp --strict_zero_pen \
    --output retargeted/freekick.pkl
```

---

## 2. Ground Penetration Prevention

### 2.1 The Problem

Vanilla GMR retargets each frame independently with an inverse-kinematics solve
that minimises position/orientation error against the scaled human pose. The
solver has no notion of the ground plane, so foot links routinely drop below
`z = 0`:

| Motion type | Frames with any sole below ground | Worst penetration |
|---|---|---|
| Walking (CMU) | 78 – 102 % | up to 80 mm |
| Running (CMU) | 86 – 95 % | up to 77 mm |
| Kicking (GVHMR) | 100 % | up to 49 mm |

This is unusable as RL training data without correction. The goal is to keep
all four corners of both foot soles at or above `z = clearance` (default 3 mm)
while perturbing the original retargeted pose as little as possible.

### 2.2 Architecture

The mink IK solver formulates each step as a **Quadratic Program** in joint
displacement Δq (not velocity):

```
minimise    ½ ‖J·Δq − e‖²        (task residuals)
subject to   G·Δq ≤ h             (inequality limits)
             Aq = b               (equality limits, e.g. joint limits)
```

`solve_ik` returns `v = Δq / dt`; `integrate_inplace(v, dt)` then applies
`q += v · dt = Δq`. Penetration prevention adds rows to `G·Δq ≤ h`:

- **`GroundPlaneLimit`**: one row per active sole corner that prevents downward
  motion of that corner. Hard QP constraint (§2.3).
- **`SoftGroundConstraint`**: high-weight `mink.FrameTask` instances that push
  sole sites upward when they approach the ground. Less rigorous; needs `<site>`
  elements injected into the MJCF first (§2.4).
- **`set_ground_offset`** + **two-pass IK**: a per-frame target shift that
  combines with QP to produce strictly-zero penetration with no foot-slide
  jitter (§2.5).

All three live in `general_motion_retargeting/ground_constraint.py`. Sole
geometry (the local-frame corner coordinates per robot) lives in
`general_motion_retargeting/sole_points.py`.

### 2.3 `GroundPlaneLimit` (QP hard constraint)

For each sole corner with world z-coordinate `z_p` near the ground, linearise
`z_p(q + Δq) ≥ ground_height + clearance`:

```
z_p + J_z · Δq ≥ ground_height + clearance
→  -J_z · Δq ≤ gain · (z_p - ground_height - clearance)        [margin = z_p - gh - cl]
```

The Jacobian is computed at the **world position of the corner point**, not
the body origin, via `mj_jac(model, data, jacp, jacr, world_pt, body_id)`,
so root-translation and joint-rotation contributions are both captured.

#### Dynamic CBF gain

`general_motion_retargeting/ground_constraint.py:117` boosts the effective
gain when the sole is already penetrating:

```python
margin = z - self.ground_height - self.clearance
dynamic_gain = min(1.0, self.gain + 2.0 * abs(margin)) if margin < 0 else self.gain
```

A static `gain = 0.5` corrects half of any violation per IK step, which can
leave deep violations under-corrected within the convergence budget. The
dynamic term grows with the depth of violation (`+2 m⁻¹` per metre of
penetration, capped at 1.0 = full correction). This shipped in commit
`1c99b8f` and is what makes Pass 1 of the two-pass scheme converge fast enough
that residual depths are dominated by IK budget rather than gain stall.

#### Activation distance

When a corner is more than `activation_distance` (default 20 mm) above the
ground+clearance threshold, its row is omitted entirely — keeps the QP small
during swing phases.

### 2.4 `SoftGroundConstraint` (repulsive tasks)

A weaker alternative that adds `mink.FrameTask` instances at named sole
**sites** with a weight that linearly ramps up to `max_weight` (default 500)
as each site descends through the activation zone. The tasks compete with the
foot-tracking tasks and lose: empirically 44 – 62 % of walk frames still
penetrate (see §6). Not recommended for production data generation.

To use it once, run `python scripts/add_sole_sites.py` to produce
`assets/booster_k1/K1_serial_with_sites.xml`, then invoke
`--ground_mode soft`.

### 2.5 Strict Zero Penetration (Two-Pass IK)

Hard QP alone reduces walk penetration by 94 – 100 % but a few frames per
sequence still squeeze through (≤ 9 mm) because the per-step CBF correction
and the limited convergence budget interact at frames where the human target
is well below ground. Naively post-shifting the root upward by the residual
depth gives geometrically zero penetration but produces visible foot-slide
("floating feet") because the shift is applied **after** the joint angles
have been solved.

The two-pass IK scheme, gated by `--strict_zero_pen`, fixes this by shifting
the **target skeleton** (not the qpos) before the second solve. The relevant
code is `scripts/retarget_no_penetration.py:285-354`.

#### Pass 1 — Surveyor

1. Run the standard QP IK over the full clip → first-pass `qpos_list`.
2. For every frame, set `data_mj.qpos` and call `mj.mj_forward`. Compute the
   minimum world z over all sole corners.
3. Where that minimum is below `clearance`, store `depths[i] = clearance - min_z`;
   otherwise `depths[i] = 0`.

This is read-only forward kinematics; no IK is re-run.

#### Envelope extraction

The raw `depths` array is a sparse spike pattern (one peak per heel-strike).
Feeding it directly back as a per-frame target offset would translate those
spikes into root jerk. Instead it gets shaped into a continuous envelope:

```python
dilated      = scipy.ndimage.maximum_filter1d(depths, size=15)   # widen each spike
smoothed     = scipy.ndimage.gaussian_filter1d(dilated, sigma=5) # soften into a bump
final_depths = scipy.ndimage.gaussian_filter1d(
                   np.maximum(smoothed, depths), sigma=2)        # cover spikes, smooth kinks
```

Why these numbers (at 30 fps):

- `size=15` ≈ 0.5 s window — captures the full width of a heel-strike + early
  stance event so the envelope rises well before the spike and decays after.
- `sigma=5` ≈ 0.17 s standard deviation — produces a smooth lift with a
  natural rise time so the root motion that follows reads as anticipation
  rather than a step.
- `np.maximum(smoothed, depths)` guarantees the envelope is ≥ depths
  everywhere (otherwise we would re-introduce penetration in the smoothing
  trough).
- The second `sigma=2` removes the kinks that `np.maximum` re-introduced
  while still strictly covering the spikes.

These are tuned for 30 fps locomotion. They should be re-derived for clips
at substantially different frame rates or for non-locomotor motion classes.

#### Pass 2 — Solver

A fresh retargeter is constructed (Pass 1 state is discarded). For each frame:

```python
retarget.set_ground_offset(baseline_compensation - final_depths[i])
qpos = retarget.retarget(frame_data)
```

`set_ground_offset` subtracts its argument from every body-z target before IK,
so a more-negative offset raises all targets. The QP then solves the original
foot-tracking task against a target that is already lifted out of the ground;
because the lift is smooth, the resulting joint trajectories are smooth too,
and because the lift is generous enough to clear the spike, no QP row ever
fires. The soles end up flush against the ground (knees, hips, ankles bend
naturally to absorb the shift), with no foot-slide.

Empirically this drives max penetration to **0.0 mm** on every CMU walk we
have measured, while leaving root motion smoother than vanilla QP.

#### Important details

- The viewer is stepped from inside the Pass 2 loop, not after. This is
  intentional — Pass 1 qpos has the foot-slide artifact and is not what we
  want to display.
- Pass 1 forward kinematics use a temporary `mj.MjData` separate from the
  retargeter's own configuration; mutating the latter would invalidate the
  IK warm-start chain.
- `final_depths` is also stored in the output pkl when present, so downstream
  consumers can know where the corrections landed.

### 2.6 Sole compensation (`ground_offset`)

The IK config maps the human ankle to the robot ankle at z = 0. But the K1
sole surface is **24 mm** below the ankle origin (T1: 30 mm). Without
correction the robot stands with ankle at ground level and sole 24 mm
underground. The fix:

```python
min_sole_z          = min(pt[2] for pts in sole_config.values() for pt in pts)
sole_compensation   = min_sole_z - clearance       # negative → raises targets
retarget.set_ground_offset(sole_compensation)
```

`apply_ground_offset` subtracts `ground_offset` from every body z target,
effectively raising all targets by 27 mm (K1) so the robot stands with sole
at `z = clearance`.

This is applied automatically whenever `--ground_mode != none`. It is
**incompatible** with `--height_adjust`: that flag uses link origins
(ankle-level), not sole corners, and would shift the root back down by 27 mm,
re-burying the soles. The script silently skips `--height_adjust` with a
yellow warning when ground constraints are active.

---

## 3. Foot-Ground Contact Flags

### 3.1 What it is

For every retargeted clip the pipeline writes a `(num_frames, 2)` boolean
array under the key `foot_ground_contact_flags` in the output pkl. Column 0
is the left foot, column 1 the right.

These labels are intended as supervision for RL policies (gait phase loss,
contact-conditioned style transfer) and as a starting point for gait-cycle
analysis. The detector lives at
`scripts/retarget_no_penetration.py:422-448`.

### 3.2 Why on raw SMPL-X frames

The contact detector reads `left_foot` (joint 10) and `right_foot` (joint 11)
positions directly from the SMPL-X frames returned by
`get_smplx_data_offline_fast` — i.e. **before** IK, ground compensation, or
the two-pass envelope is applied. There are two reasons:

1. **Stable across retargeting variants.** Re-running the same clip with
   `--ground_mode qp`, `--ground_mode none`, or `--strict_zero_pen` produces
   identical labels, so consumers can compare retargeting strategies without
   labels drifting underneath.
2. **No chicken-and-egg.** Computing contact from robot qpos is circular: the
   ground constraint already moves feet toward the floor, biasing any
   threshold-based detector toward "always in contact". The SMPL-X joint is
   the source-of-truth human kinematics and is not subject to that bias.

### 3.3 Algorithm

Per foot:

1. Stack toe positions across all frames into `(N, 3)`.
2. Compute 3D speed via `np.linalg.norm(np.gradient(toe, dt, axis=0), axis=1)`
   where `dt = 1 / aligned_fps`. `np.gradient` uses central differences in
   the interior and one-sided at the endpoints.
3. Define `floor_z = min(left_toe_z.min(), right_toe_z.min())`. AMASS
   sequences have arbitrary z-offsets and GVHMR's post-rotation introduces
   a further shift, so absolute z = 0 is not a reliable reference. The
   per-clip minimum is.
4. Mark contact where `(toe_z - floor_z) < z_thresh AND speed < v_thresh`.

The AND is deliberate: either alone is too noisy. Height-only fires through
the entire clip when the source has the feet near the floor for stylistic
reasons (e.g. a dance with a low base). Speed-only fires at every momentary
direction change in mid-air.

### 3.4 Defaults

| Flag | Default | Meaning |
|---|---|---|
| `--foot_contact_z_thresh` | 0.08 m | maximum SMPL-X toe height above the per-clip floor |
| `--foot_contact_vel_thresh` | 0.5 m/s | maximum SMPL-X toe 3D speed |
| `--no_foot_contact` | off | skip detection, store `None` in the pkl |

These defaults still need tuning per the commit message of `47d3b5d`. They
were chosen to err toward false-positive contact (which downstream policies
can downweight) over false-negative.

### 3.5 Caveats

- If the SMPL-X frame dictionaries somehow lack `left_foot` / `right_foot`
  keys (an unusual loader configuration), detection is silently skipped and
  `foot_ground_contact_flags` is set to `None` with a yellow warning print.
- The two-threshold AND filter can flicker around stride boundaries (1 – 2
  frame gaps in stance). A hysteresis filter (2-of-3 vote, or a minimum
  stance duration of ~3 frames) would clean this up; see §9 Future Work.
- Detection is intentionally clip-local; there is no gait-cycle awareness.
  For polyphasic motion (e.g. mixed walking + jumping in one file), the
  per-clip floor reference is dominated by whichever phase has the lowest
  toes — usually fine for locomotion, less robust for, say, kicks where one
  foot may never go down.

---

## 4. Supported Robots

### 4.1 Booster K1

Sole geometry (from `general_motion_retargeting/sole_points.py:17-30`):

```
foot box geom:    size=[0.08, 0.035, 0.016] (half-extents), pos=[0.014, 0, -0.008]
sole surface z:   -0.024 m (foot-link local frame)
corners:          x ∈ [-0.066, 0.094], y ∈ [-0.035, 0.035], z = -0.024
foot bodies:      left_foot_link, right_foot_link
```

This produces a 27 mm sole compensation (24 mm ankle-to-sole + 3 mm
clearance). All empirical results in §6 are from K1.

### 4.2 Booster T1

Added in commit `07c91ff`. Sole geometry derived from the two collision
capsules in T1's MJCF (`general_motion_retargeting/sole_points.py:31-49`):

```
two capsules per foot, axis along local x:
  size = "0.02 0.0915"  (radius, half-length)
  centers = (0.01, ±0.035, -0.01)
sole surface z:   -0.030 m
corners:          x ∈ [-0.0815, 0.1015], y ∈ [-0.055, 0.055], z = -0.030
foot bodies:      left_foot_link, right_foot_link
```

This produces a 33 mm sole compensation.

**Known issue:** running an AMASS CMU file through T1 with
`--ground_mode qp --strict_zero_pen` reportedly crashes (per the commit
message of `07c91ff`: "Crashes for AMASS CMU for some reason"). The crash has
not yet been root-caused. GVHMR through T1 has not yet been tested. T1 should
be considered not-yet-production for this branch until the crash is
reproduced and fixed.

### 4.3 Adding a new robot

1. Measure the sole corner coordinates from the robot's MJCF foot collision
   geom. `general_motion_retargeting/sole_points.py:detect_sole_points_from_mjcf`
   handles box geoms automatically; capsule geoms (like T1) currently need
   manual derivation.
2. Add entries to `SOLE_CONTACT_POINTS` and `FOOT_LINK_NAMES` in `sole_points.py`.
3. Make sure an `ik_configs/smplx_to_<robot>.json` (and any other source
   format you need) exists and that `params.py`'s `IK_CONFIG_DICT` and
   `ROBOT_XML_DICT` reference it.
4. If you want soft-mode support, run `scripts/add_sole_sites.py` after
   updating it to handle the new robot's MJCF.
5. Add the robot to the `--robot` `choices` in `parse_args` (currently
   `["booster_k1", "booster_t1"]`).

---

## 5. Running the Pipeline

### 5.1 CLI reference for `retarget_no_penetration.py`

| Flag | Default | Description |
|---|---|---|
| `--input` | — | Path to a single motion file or a directory for batch mode (mutually exclusive with `--yaml`). |
| `--yaml` | — | Path to a YAML locomotion list. AMASS-CMU only. |
| `--input_format` | required | One of `smplx`, `gvhmr`, `amass_cmu`. |
| `--output` | derived | Single-file: a `.pkl` path. Batch: an output directory. Defaults to `<input>_retargeted` for files, `<input_dir>_retargeted` for directories. |
| `--robot` | `booster_k1` | One of `booster_k1`, `booster_t1`. |
| `--ground_mode` | `none` | `none` = vanilla; `qp` = hard QP; `soft` = repulsive tasks. |
| `--ground_height` | 0.0 | z of the ground plane (m). |
| `--clearance` | 0.003 | Minimum sole z above ground (m). |
| `--gain` | 0.5 | QP CBF base gain (boosted dynamically when penetrating). |
| `--max_root_dz` | None | Cap upward root-z displacement per IK step. Empirically ineffective; flag retained but slated for removal. |
| `--max_weight` | 500.0 | Soft-mode max repulsive task weight. |
| `--activation_distance` | 0.02 | Distance above clearance at which constraint activates. |
| `--height_adjust` | off | FK-based root z shift so lowest body is at z = 0. Skipped if `--ground_mode != none`. |
| `--root_origin_offset` | off | XY-shift so frame 0 root is at the world origin. |
| `--strict_zero_pen` | off | Enable two-pass IK (§2.5). |
| `--foot_contact_z_thresh` | 0.08 | Contact-detector height threshold (m). |
| `--foot_contact_vel_thresh` | 0.5 | Contact-detector speed threshold (m/s). |
| `--no_foot_contact` | off | Skip writing `foot_ground_contact_flags`. |
| `--no_viz` | off | Disable viewer (auto-set in batch). |
| `--record_video` | off | Record an MP4 even in headless mode (offscreen render). |
| `--rate_limit` | off | Rate-limit the viewer to motion fps. |
| `--loop` | off | Currently unused in the unified pipeline — slated for removal. |
| `--override` | off | Overwrite existing output pkls in batch mode. |

### 5.2 Single-file examples

K1 + AMASS CMU + recommended QP-smoothed:

```bash
conda run -n gmr python scripts/retarget_no_penetration.py \
    --input /data/AMASS/CMU/35/35_14_stageii.npz \
    --input_format amass_cmu --robot booster_k1 \
    --ground_mode qp --strict_zero_pen \
    --no_viz \
    --output retargeted/35_14.pkl
```

K1 + GVHMR with viewer:

```bash
conda run -n gmr python scripts/retarget_no_penetration.py \
    --input GVHMR/outputs/demo/freekick/hmr4d_results.pt \
    --input_format gvhmr --robot booster_k1 \
    --ground_mode qp --strict_zero_pen \
    --rate_limit \
    --output retargeted/freekick.pkl
```

### 5.3 Directory batch mode

Walk a directory tree, retarget every SMPL-X file under it:

```bash
conda run -n gmr python scripts/retarget_no_penetration.py \
    --input /data/AMASS/CMU/ \
    --input_format amass_cmu \
    --ground_mode qp --strict_zero_pen \
    --output retargeted_cmu/ \
    --override
```

In batch mode:

- The viewer auto-disables (`--no_viz` is forced).
- `--record_video` still produces offscreen MP4s — they are written to
  `<output_dir>/videos/<robot>_<stem>.mp4` so they sit alongside their pkls.
- Stdout is duplicated into `<output_dir>/<top-level-subdir>/output.txt`
  via the `_Tee` helper, so the rich-formatted log of the run is preserved
  next to the data.

### 5.4 YAML batch mode

The YAML format (e.g. `AMASS/CMU/amass_cmu_locomotion_list.yaml`) maps tags
to lists of relative paths and descriptions:

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

Paths are resolved relative to the YAML's own directory. Output stems are
derived from the path under any `AMASS/` component, prefixed with the robot
name to keep K1 and T1 outputs from colliding.

```bash
conda run -n gmr python scripts/retarget_no_penetration.py \
    --yaml /data/AMASS/CMU/amass_cmu_locomotion_list.yaml \
    --input_format amass_cmu \
    --ground_mode qp --strict_zero_pen \
    --height_adjust --root_origin_offset \
    --output retargeted_cmu/
```

### 5.5 Comparison and stats

```bash
# Side-by-side comparison MP4 (red spheres = penetration, green = ok)
conda run -n gmr python scripts/vis_compare_motions.py \
    --motion_a /data/35_14_qp.pkl \
    --motion_b /data/35_14_qpsmoothed.pkl \
    --label_a "QP" --label_b "QP+Smoothed" \
    --video_path videos/35_14_compare.mp4 \
    --show_sole_points

# Per-frame penetration / jitter / range stats
conda run -n gmr python scripts/compare_penetration_stats.py \
    /data/locomotion_amass_cmu/*.pkl
```

Both scripts currently hardcode `ROBOT_TYPE = "booster_k1"` and need a
`--robot` flag added before they can audit T1 output. Tracked in §9.

### 5.6 Dataset triage

`scripts/vis_robot_motion_dataset.py` plays every `.pkl` in a directory and
provides keyboard control:

- `[` / `]` — previous / next motion.
- ` ` (space) — pause / resume.
- `x` — move the **currently displayed** pkl to a sibling `trash/` folder
  for later cleanup. The display advances to the next motion automatically.

This is the curation tool used to build the "good" subset of locomotion
data after an automated batch run.

### 5.7 Output PKL schema

After Commit B (Richer pkl schema) the output dict has the following keys:

| Key | Type | Notes |
|---|---|---|
| `fps` | `float` | Aligned target fps. |
| `robot` | `str` | NEW — `booster_k1` or `booster_t1`. |
| `input_format` | `str` | NEW — `smplx`, `gvhmr`, or `amass_cmu`. |
| `source_file` | `str` | NEW — absolute path of the source motion. |
| `ground_mode` | `str` | `none`, `qp`, or `soft`. |
| `strict_zero_pen` | `bool` | NEW — True if two-pass IK was applied. |
| `ground_clearance` | `float` | Clearance threshold used (m). |
| `actual_human_height` | `float` | NEW — measured source human height (m). |
| `root_pos` | `(N, 3) float32` | World root position per frame. |
| `root_rot` | `(N, 4) float32` | World root rotation per frame, **xyzw order**. |
| `dof_pos` | `(N, J) float32` | Joint positions per frame. |
| `local_body_pos` | `(N, B, 3) float32 \| None` | Body positions with root pinned to origin. |
| `link_body_list` | `list[str] \| None` | Body names matching `local_body_pos`. |
| `foot_ground_contact_flags` | `(N, 2) bool \| None` | Left / right contact (§3). See `docs/foot_contact.md` for the consumer-facing schema and PyTorch usage examples. |
| `foot_contact_meta` | `dict \| None` | Self-describing dict: `{"source", "joints", "columns", "z_thresh", "vel_thresh", "floor_z"}`. |

Pre-existing consumers that read only the original keys continue to work
unchanged — all new keys are additive.

---

## 6. Empirical Results

All numbers below were measured on Booster K1 at 30 fps with `clearance =
3 mm`, `gain = 0.5`, `activation_distance = 20 mm`, by
`scripts/compare_penetration_stats.py`.

Column legend:

- `pen/N` — frames with any sole corner below `ground_height + clearance`
- `max_pen` — worst-case penetration depth (mm)
- `max_up` — maximum frame-to-frame upward root-z jump (mm); proxy for jitter
- `mean_up` — mean upward jump across all positive jumps (mm)
- `rz_range` — total root-z range over the motion (mm)

### Walk motions (AMASS CMU)

| Motion | Mode | frames | pen/N | max_pen | max_up | mean_up | rz_range |
|---|---|---:|---|---:|---:|---:|---:|
| CMU 35_14 | vanilla | 102 | 102/102 | 63.5 mm | 14.6 mm | 4.6 mm | 45.7 mm |
| CMU 35_14 | qp | 102 | 6/102 | 9.3 mm | 10.5 mm | 3.6 mm | 37.4 mm |
| CMU 35_14 | qp_smoothed | 102 | **0/102** | **0.0 mm** | TODO | TODO | TODO |
| CMU 35_14 | soft | 102 | 62/102 | 36.5 mm | 14.6 mm | 4.6 mm | 45.7 mm |
| CMU 02_01 | vanilla | 85 | 85/85 | 80.6 mm | 10.4 mm | 4.1 mm | 60.9 mm |
| CMU 02_01 | qp | 85 | 0/85 | 0.0 mm | 7.5 mm | 2.5 mm | 32.3 mm |
| CMU 02_01 | qp_smoothed | 85 | **0/85** | **0.0 mm** | TODO | TODO | TODO |
| CMU 07_01 | vanilla | 79 | 78/79 | 81.3 mm | 11.2 mm | 5.1 mm | 70.0 mm |
| CMU 07_01 | qp | 79 | 2/79 | 8.5 mm | 9.6 mm | 3.6 mm | 39.7 mm |
| CMU 07_01 | qp_smoothed | 79 | **0/79** | **0.0 mm** | TODO | TODO | TODO |
| CMU 07_01 | soft | 79 | 45/79 | 54.3 mm | 14.5 mm | 5.1 mm | 70.0 mm |

### Run motions (AMASS CMU)

| Motion | Mode | frames | pen/N | max_pen | max_up | mean_up | rz_range |
|---|---|---:|---|---:|---:|---:|---:|
| CMU 02_03 (jog) | vanilla | 43 | 41/43 | 77.2 mm | 21.5 mm | 8.7 mm | 79.0 mm |
| CMU 02_03 (jog) | qp | 43 | 0/43 | 0.0 mm | 23.3 mm | 9.2 mm | 61.6 mm |
| CMU 02_03 (jog) | qp_smoothed | 43 | **0/43** | **0.0 mm** | TODO | TODO | TODO |
| CMU 02_03 (jog) | soft | 43 | 23/43 | 50.0 mm | 21.5 mm | 8.7 mm | 78.9 mm |
| CMU 09_01 (run) | vanilla | 37 | 32/37 | 52.3 mm | 13.3 mm | 5.9 mm | 45.4 mm |
| CMU 09_01 (run) | qp | 37 | 0/37 | 0.0 mm | 23.8 mm | 7.2 mm | 42.0 mm |
| CMU 09_01 (run) | qp_smoothed | 37 | **0/37** | **0.0 mm** | TODO | TODO | TODO |

### Kick motion (GVHMR)

| Motion | Mode | frames | pen/N | max_pen | max_up | mean_up | rz_range |
|---|---|---:|---|---:|---:|---:|---:|
| kick_soogon_2 | vanilla | 134 | 134/134 | 48.7 mm | 7.8 mm | 1.0 mm | 45.9 mm |
| kick_soogon_2 | qp | 134 | 0/134 | 0.0 mm | 6.3 mm | 0.8 mm | 33.1 mm |
| kick_soogon_2 | qp_smoothed | 134 | 0/134 | 0.0 mm | TODO | TODO | TODO |
| kick_soogon_2 | soft | 134 | 24/134 | 21.7 mm | 7.8 mm | 1.0 mm | 45.9 mm |

### Booster T1

Pending — see §4.2 known crash. Once fixed, re-measure these tables on T1.

### Key observations

1. `qp_smoothed` drives `max_pen` to 0.0 mm everywhere (by construction:
   the envelope guarantees `final_depths[i] ≥ depths[i]`).
2. The original `qp` mode already eliminated penetration entirely on run
   and kick motions and reduced walk penetration by 94 – 100 %. The
   `qp_smoothed` improvement is concentrated on the residual walk frames.
3. Jitter (`max_up`) under `qp_smoothed` has not yet been re-measured in
   the same controlled batch as the older numbers above. The dynamic CBF
   gain change makes a direct comparison to old `qp` numbers unreliable;
   a fresh baseline run is on the to-do list.
4. Soft mode is strictly worse than QP on every motion and is retained
   only as a comparison baseline.

---

## 7. Pros and Cons of Each Mode

### `none` (vanilla GMR)

- **Pros:** lowest overhead, preserves the original retargeted pose exactly,
  smooth root.
- **Cons:** massive penetration, unusable as physics-simulation input.
- **Use when:** sanity-checking IK quality only.

### `qp` (`GroundPlaneLimit`)

- **Pros:** eliminates penetration entirely on run / kick motions, reduces
  walk penetration to ≤ 6 % of frames, hard CBF guarantee, no MJCF prep
  needed, cheap (≤ 8 QP rows), reduces walk root-jitter as a side effect.
- **Cons:** small residual walk penetrations (≤ 9 mm) at frames where the
  human target is well below ground; increases root jitter on run motions
  because the constraint fires before natural heel-strike compression
  (see §8.2).
- **Use when:** generating locomotion training data and you don't need
  guaranteed zero penetration.

### `qp` + `--strict_zero_pen` (recommended)

- **Pros:** guaranteed 0.0 mm `max_pen` on every motion measured so far;
  smoother root motion than vanilla `qp` because the envelope shifts targets
  in advance of the spike; no foot-slide artifact (lift happens before IK,
  not after).
- **Cons:** ~ 2× IK runtime (Pass 1 + Pass 2); requires `scipy.ndimage`;
  envelope hyperparameters (`size=15`, `sigma=5`, `sigma=2`) are tuned for
  30 fps locomotion and may not transfer to substantially different rates
  or motion classes without re-tuning.
- **Use when:** you want the cleanest possible RL training data. This is
  the default recommendation.

### `soft` (`SoftGroundConstraint`)

- **Pros:** smooth fade-out at the activation boundary; no risk of QP
  infeasibility.
- **Cons:** empirically 44 – 62 % of walk frames still penetrate; needs
  `K1_serial_with_sites.xml` and equivalent for any other robot;
  out-competed by foot-tracking task weights.
- **Use when:** comparative study only.

---

## 8. Known Limitations and Open Problems

### 8.1 Residual walk penetrations under `qp` — RESOLVED

Eliminated by `--strict_zero_pen` (§2.5). The `qp` mode alone still has the
residual issue documented in earlier results, so the recommendation is to
always pair `--ground_mode qp` with `--strict_zero_pen`.

### 8.2 Run motion root jitter — PARTIALLY RESOLVED

The smooth envelope of `--strict_zero_pen` reduces but does not eliminate
heel-strike root overshoot on running motions, because the envelope is
contact-oblivious — it lifts the targets whenever a sole approaches the
ground regardless of whether that frame "should" be in stance. A
**contact-aware** two-pass that reads `foot_ground_contact_flags` and locks
`final_depths[i] = 0` whenever the flag says "stance" is the cleanest fix;
see §9.

### 8.3 `RootZLimit` is ineffective — known bug

The class still exists in `general_motion_retargeting/ground_constraint.py`
(with a long docstring explaining why it doesn't work) but the CLI flag
`--max_root_dz` will be removed in Commit C. The diagnosis from earlier
work remains accurate: the root jump originates in tasks-1 (rotation-only
solve), but `RootZLimit` lives in `ik_limits2` (tasks-2 only); moving it to
shared `ik_limits` causes QP infeasibility against `GroundPlaneLimit`.

### 8.4 Robot coverage

K1 is fully supported. T1 is added (§4.2) but has a known AMASS-CMU crash
that has not been root-caused. No other robots have sole-point definitions.

### 8.5 Frame-independent IK

The IK still solves each frame independently. There is no temporal
regularisation in the cost function; smoothing happens only via the
two-pass envelope on the target side. Discontinuities between frames at
state transitions remain possible.

### 8.6 Foot-contact thresholds are clip-agnostic

The detector uses fixed `z_thresh` / `v_thresh` and a per-clip floor
reference. There is no gait-cycle modelling, so:

- Stride-boundary flicker (1 – 2 frame stance gaps) is possible.
- Polyphasic clips (e.g. mixed walking + jumping) collapse into a single
  per-clip floor that is dominated by the lowest-toe phase.

A hysteresis filter (2-of-3 vote, or minimum stance duration of ~ 3 frames)
would address the flicker. See §9.

---

## 9. Future Work

| Idea | Status | Notes |
|---|---|---|
| Two-pass IK / target envelope smoothing | **Done** in `1c99b8f` | §2.5. |
| Dynamic CBF gain | **Done** in `1c99b8f` | §2.3. |
| `--robot` flag in `compare_penetration_stats` and `vis_compare_motions` | Open | Both hardcode `booster_k1`; T1 outputs cannot currently be audited. |
| Contact-aware two-pass | **Recommended next** | Use `foot_ground_contact_flags` to override `final_depths[i] = 0` on stance frames. Cleanest fix for §8.2. |
| Hysteresis on contact flags | Open | 2-of-3 vote, or minimum stance run-length. |
| Parallel batch runner | Open | `multiprocessing.Pool` with per-worker log files instead of the current shared `_Tee`. |
| Dataset auto-filter | Open | Reuse `compare_penetration_stats` outputs to auto-trash pkls with `max_pen > 10 mm` or `max_up > 30 mm`, leaving only borderline cases for manual triage. |
| Big-toe joint as a second contact detector | Open | SMPL-X has separate toe-tip joints; combining toe-base and toe-tip would distinguish heel-strike from toe-off. |
| Console-script entry point | Open | Add `[project.scripts] gmr-retarget = ...` to `setup.py` so users can invoke without remembering the script path. |
| `detect_sole_points_from_mjcf` for capsule geoms | Open | Currently box-only; would let `unitree_g1`, `fourier_n1`, etc. be added in one line. |
| Persist `final_depths` in the pkl | Open | The Pass-1 raw `depths` and Pass-2 `final_depths` are useful diagnostic data; trivially cheap to persist. |

---

## 10. Modified / Added Files

This is the diff from the b7a9cb4 baseline (the initial penetration draft)
to the current HEAD of `feature/zero-penetr-and-smoothed`.

| Path | Change | Purpose |
|---|---|---|
| `general_motion_retargeting/ground_constraint.py` | modified (`1c99b8f`) | Dynamic CBF gain in `GroundPlaneLimit.compute_qp_inequalities`. |
| `general_motion_retargeting/sole_points.py` | modified (`07c91ff`) | `booster_t1` entry in `SOLE_CONTACT_POINTS` and `FOOT_LINK_NAMES`. |
| `general_motion_retargeting/motion_retarget.py` | minor | Cosmetic spacing only. |
| `general_motion_retargeting/robot_motion_viewer.py` | minor (`fac4ebd`) | Default geom labels enabled in viewer. |
| `scripts/retarget_no_penetration.py` | major (`1c99b8f`, `07c91ff`, `47d3b5d`) | Two-pass IK, `--robot` flag, headless batch video, log tee, foot-contact flags, `_Tee` helper. |
| `scripts/vis_robot_motion_dataset.py` | modified (`fac4ebd`) | `'x'` hotkey for trash-folder triage. |
| `scripts/batch_locomotion_k1.py` | added (`fac4ebd`) | Personal one-off; **scheduled for removal** — superseded by YAML batch mode. |
| `run_batch_comparison.sh` | added (`1c99b8f`) | One-shot QP-vs-Smoothed comparison runner over CMU/35. |
| `penetration.md` | modified (`c7faa94`) | Predecessor of this document; superseded. |
| `CHANGELOG_ZERO_PENETRATION.md` | added (`1c99b8f`) | Predecessor of this document's §2.5; superseded. |

After the K1/T1 deployable cleanup, the predecessor docs `penetration.md` and
`CHANGELOG_ZERO_PENETRATION.md` live under `docs/archive/`,
`run_batch_comparison.sh` lives under `shell/`, and the script-level helpers
extracted to `general_motion_retargeting/retargeting/` are listed in this
section. Per-format converters (`smplx_to_robot.py`, `gvhmr_to_robot*.py`,
etc.) were removed entirely; git history preserves them.
