# BVH (LAFAN1) Support — EXPERIMENTAL

> **Status:** initial integration. Legs, feet, trunk, and arms now retarget
> without driving joints into limits on standard LAFAN1 walks/runs. The K1
> arm offsets were ported from `bvh_lafan1_to_t1.json` (which has correctly
> mirrored left/right quaternions) — see §7. **Head offsets remain a
> best-effort guess** and may need tuning per §4.

LAFAN1 BVH motions can be retargeted through the same pipeline as SMPL-X:

```bash
python scripts/retarget_no_penetration.py \
    --input dance.bvh --input_format bvh_lafan1 \
    --robot booster_k1 --ground_mode qp --strict_zero_pen \
    --output retargeted/dance.pkl
```

A single experimental warning is printed when `--input_format bvh_lafan1` is
selected so callers know the result needs review.

## 1. Pipeline

```
.bvh
  │  general_motion_retargeting.utils.lafan1.load_bvh_file()
  ▼  • Y-up → Z-up rotation
     • cm → m
     • injects synthetic LeftFootMod / RightFootMod bones
     • returns list[{bone: (pos, quat_wxyz)}], human_height
  ▼
GMR(src_human="bvh_lafan1", tgt_robot=...)
     • loads ik_configs/bvh_lafan1_to_{k1,t1}.json
  ▼
per-frame IK → ground-penetration pass (unchanged) → .pkl
```

The ground-constraint and two-pass IK layers are format-agnostic — nothing in
the penetration-prevention code needed changes for BVH.

## 2. IK config shape

Each task entry has the form:

```json
"<robot_link>": [
    "<bvh_bone>",
    pos_weight, orient_weight,
    [px, py, pz],                 // position offset, in BVH bone frame
    [qw, qx, qy, qz]              // BVH-bone-frame → robot-link-frame
]
```

For K1 (`bvh_lafan1_to_k1.json`):

| Source | What was reused |
|---|---|
| `smplx_to_k1.json` | robot link names, task weights (0/10 limbs, 100/50 feet), `human_scale_table`, `human_height_assumption=1.8`, position offsets (all zero on K1) |
| `bvh_lafan1_to_g1.json`, `bvh_lafan1_to_t1_29dof.json` | quaternion offsets for legs, feet, root, trunk — empirically the same across humanoids |
| **guessed** | quaternion offsets for `Left_Arm_3`, `Right_Arm_3`, `left_hand_link`, `right_hand_link`, `Head_2` |

For T1 (`bvh_lafan1_to_t1.json`): adapted from the tested
`bvh_lafan1_to_t1_29dof.json` by removing the extra arm DOFs (`AL4`, `AL5`,
`AR4`, `AR5`) and pointing `left_hand_link` / `right_hand_link` at the
forearm task instead. Leg, foot, trunk, and shoulder tasks are unchanged
between T1 and T1-29dof and should work without tuning.

### Why SMPL-X K1 quaternions can't be copied

The quaternion is a constant rotation from the **human bone's local frame**
to the robot link's local frame:

```
Q_bvh→k1_link  =  Q_bvh_bone→smplx_bone  ⊗  Q_smplx→k1_link
                  └────── per-joint ──────┘
```

`smplx_to_k1.json` uses a uniform `[-0.5, 0.5, 0.5, 0.5]` for every task.
Pasting that into the BVH config would skip the per-joint
BVH→SMPL-X bone-frame correction, which is what causes the twist.

## 3. Caveats

- **No FPS detection.** `read_bvh` parses `Frame Time:` from the header but
  doesn't expose it. The pipeline currently assumes target FPS (default 30),
  which is correct for LAFAN1 (natively 30 fps). Other BVH sources at a
  different rate need upstream resampling. To fix, plumb `frametime` through
  `Anim` in `general_motion_retargeting/utils/lafan_vendor/extract.py`.
- **Hardcoded human height.** `load_bvh_file` returns `human_height = 1.75`
  unconditionally (see `general_motion_retargeting/utils/lafan1.py:45`).
  If your skeleton is scaled differently, body-scale matching will be off.
- **No `bvh_nokov` support in this branch.** The Nokov BVH path was
  half-wired in earlier work and is not exposed in `--input_format`. Add it
  back deliberately if needed.

## 4. Validation checklist

Before trusting a K1 BVH retarget:

1. **T-pose check.** Drop a single-frame T-pose BVH through the pipeline
   with the viewer and confirm the robot stands neutrally. A 90° twist on
   a limb usually means the quaternion's longitudinal-axis component pair
   is signed wrong — swap the sign of one pair (e.g.
   `[-0.5, 0.5, 0.5, -0.5] → [0.5, -0.5, -0.5, -0.5]`) and re-check.
2. **Reference walk.** Run `lafan1/walk1_subject1.bvh` (or any walk) and
   verify the shoulders/elbows track without flipping.
3. **Side-by-side with SMPL-X.** Pick a motion that exists in both AMASS
   and LAFAN1 form (or render both your BVH and a SMPL-X equivalent on K1)
   and compare arm/head trajectories.

The five fields to focus on if tuning is needed are
`Left_Arm_3`, `Right_Arm_3`, `left_hand_link`, `right_hand_link`,
`Head_2` — each appearing twice in `ik_match_table1` and `ik_match_table2`.

## 5. Better than guessing — derive K1 offsets from G1

G1 ships both `smplx_to_g1.json` and `bvh_lafan1_to_g1.json`. The per-bone
BVH→SMPL-X frame offset is `Q_bvh_g1 ⊗ conj(Q_smplx_g1)`. Apply it to the K1
SMPL-X quaternion:

```
Q_bvh→k1  =  (Q_bvh→g1 ⊗ conj(Q_smplx→g1))  ⊗  Q_smplx→k1
```

This gives a principled starting value per joint, replacing the current
guesses for the five arm/head links.

## 6. K1 arm-offset fix (2026-04-25)

The original `bvh_lafan1_to_k1.json` reused a single quaternion
`[-0.5, 0.5, 0.5, -0.5]` for all four arm tasks (`Left_Arm_3`,
`Right_Arm_3`, `left_hand_link`, `right_hand_link`). On a `--ground_mode none`
test run across LAFAN1 walk / run / multipleActions:

| joint | pinning at limit (before) | pinning at limit (after) |
|---|---|---|
| `Left_Shoulder_Roll` | 59–74% | 0.1–1.1% |
| `Right_Shoulder_Roll` | 93–99% | 0.0–1.8% |
| `Left_Elbow_Yaw` | 90–99% | 0.0% |
| `ARight_Shoulder_Pitch` | 56% (walk) | <8% |

**Why a single quaternion was wrong.** K1's left and right arm bodies are
mirror-symmetric about the sagittal plane: `Left_Arm_3 → left_hand_link`
sits at `(0, +0.1215, 0)` (the +Y axis is "down the bone"), while
`Right_Arm_3 → right_hand_link` sits at `(0, -0.1215, 0)` (-Y is "down the
bone"). LAFAN1 BVH bones, in contrast, share a single axis convention:
`LeftArm → LeftForeArm` and `RightArm → RightForeArm` both use `(33.0, 0, 0)`
— +X down the bone for **both** sides. A constant rot_offset that maps
BVH +X → robot +Y therefore maps BVH +X to robot +Y on the right arm too,
which is the opposite of where the bone needs to go. The IK then drives the
shoulder roll/pitch into its joint limits trying to recover.

The SMPL-X K1 config doesn't have this problem because SMPL-X already
mirrors its left vs right arm bone frames (so the same rot_offset works for
both). BVH does not.

**Fix.** Port the T1 BVH arm quaternions, which already encode the mirror:

| task | offset |
|---|---|
| `Left_Arm_3`, `left_hand_link` | `[0.7071067811865476, 0, 0, -0.7071067811865476]` |
| `Right_Arm_3`, `right_hand_link` | `[0, -0.7071067811865476, 0.7071067811865476, 0]` |

`right_offset = left_offset ⊗ R_x(180°)` — the 180° about X swaps Y and Z
to convert the left-arm body frame to the right-arm body frame. K1 and T1
share the same arm-mirror structure, so the same pair of offsets works on
both robots.

The legs were unaffected by this bug because K1's `Left_Shank` and
`Right_Shank` children both sit at `(-0.014, 0, -0.117)` — they are **not**
mirrored along the bone axis, so a single rot_offset works for both legs.

## 7. Files

| Path | What |
|---|---|
| `scripts/retarget_no_penetration.py` | BVH loader branch in `load_motion_frames`, `bvh_lafan1` choice in `--input_format`, experimental warning |
| `general_motion_retargeting/utils/lafan1.py` | `load_bvh_file` (LAFAN1 / Nokov reader, FPS hardcoded) |
| `general_motion_retargeting/retargeting/batch_runner.py` | `*.bvh` recursive discovery |
| `general_motion_retargeting/params.py` | `IK_CONFIG_DICT["bvh_lafan1"]` registration |
| `general_motion_retargeting/ik_configs/bvh_lafan1_to_k1.json` | new, K1 (arms/head guessed) |
| `general_motion_retargeting/ik_configs/bvh_lafan1_to_t1.json` | new, T1 (adapted from T1-29dof) |
