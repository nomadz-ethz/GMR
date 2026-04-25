# Manual Review & Testing Guide

A walkthrough for reviewing the `feature/clean-deployable` branch by hand:
the order in which to read files, the rationale for the structure, and a
sequence of bash commands that exercise every code path before merging.

> **Conda env:** activate `gmr` first — every command below assumes it.
>
>     conda activate gmr

---

## 1. Reading order (top-down by abstraction)

The branch is layered. Read the high-level intent first, then drill down to
implementation. Reading bottom-up will buy you details before context.

### 1.1 What & why (15 min)

| File | What it answers |
|---|---|
| [`README.md`](../README.md) | What this fork is. Install + 4 quickstart commands. The 1-screen elevator pitch. |
| [`docs/overview.md`](overview.md) | Concretely, what's added vs upstream `YanjieZe/GMR` — the table in §2 is the load-bearing artefact. Lists each addition + the file it lives in + why. |
| [`CLAUDE.md`](../CLAUDE.md) | The tree map. Use this if you forget where something lives. |

After this you should be able to answer: *"What does this fork do that
upstream doesn't, and where in the tree do those additions live?"*

### 1.2 The pipeline (30–45 min)

| File | What it answers |
|---|---|
| [`docs/pipeline.md`](pipeline.md) | The canonical reference. ~800 lines, sections numbered. Skim §1 (Overview) and §2 (Ground Penetration Prevention) closely; skim §3 (Foot-Ground Contact); §5.7 (PKL schema) is the contract for downstream consumers. |
| [`scripts/retarget_no_penetration.py`](../scripts/retarget_no_penetration.py) | The CLI. **Intentionally thin glue.** Read top to bottom — top docstring first, then `load_motion_frames`, then `retarget_and_save`, then `parse_args`. |
| [`general_motion_retargeting/retargeting/__init__.py`](../general_motion_retargeting/retargeting/__init__.py) | The public surface of the modular subpackage — re-exports the five helpers the CLI uses. |
| [`general_motion_retargeting/retargeting/builder.py`](../general_motion_retargeting/retargeting/builder.py) | `build_retargeter()` — assembles `GMR` + ground constraint + sole compensation. The branching on `ground_mode` lives here. |
| [`general_motion_retargeting/retargeting/strict_zero_pen.py`](../general_motion_retargeting/retargeting/strict_zero_pen.py) | The two-pass IK helpers (`measure_penetration_depths`, `smooth_penetration_envelope`). Magic numbers (15-frame dilation, σ=5 then σ=2) are documented inline. |
| [`general_motion_retargeting/retargeting/foot_contact.py`](../general_motion_retargeting/retargeting/foot_contact.py) | Per-frame contact detection. **Operates on raw SMPL-X frames, before any retargeting** — this is the design property that keeps labels stable across `--ground_mode` and robot. |
| [`general_motion_retargeting/retargeting/fk_post.py`](../general_motion_retargeting/retargeting/fk_post.py) | Optional post-processing: height-shift, root-origin offset, `local_body_pos` computation. |
| [`general_motion_retargeting/retargeting/batch_runner.py`](../general_motion_retargeting/retargeting/batch_runner.py) | Pure plumbing — file discovery, YAML parsing, the `Tee` log helper, the `run_batch` driver. |
| [`general_motion_retargeting/ground_constraint.py`](../general_motion_retargeting/ground_constraint.py) | The ground constraint itself (`GroundPlaneLimit` for hard QP, `SoftGroundConstraint`/`GMRWithSoftGround` for soft). The dynamic CBF gain bump lives in `GroundPlaneLimit.compute_qp_inequalities` — it's the behaviour change called out in `pipeline.md` §2.3. |

After this you should be able to trace one frame of motion end-to-end:
SMPL-X load → IK → (optional Pass 2 envelope) → FK post → contact flags →
pkl write.

### 1.3 The narrowed registry (10 min)

| File | What it answers |
|---|---|
| [`general_motion_retargeting/params.py`](../general_motion_retargeting/params.py) | Four small dicts. Whole file is ~36 lines. Adding a robot ⇒ add one row to each + drop a JSON in `ik_configs/` + add an entry to `sole_points.py`. |
| [`general_motion_retargeting/sole_points.py`](../general_motion_retargeting/sole_points.py) | Per-robot sole contact points and foot link names. K1 + T1 only. |
| [`general_motion_retargeting/ik_configs/{smplx,bvh_lafan1}_to_{k1,t1}.json`](../general_motion_retargeting/ik_configs/) | Four IK configs total. Field semantics in [`docs/ik_config.md`](ik_config.md). |

### 1.4 Topic-specific docs (skim as needed)

| File | When to read |
|---|---|
| [`docs/foot_contact.md`](foot_contact.md) | When writing downstream code that consumes `foot_ground_contact_flags`. |
| [`docs/bvh.md`](bvh.md) | Before running any BVH motion or tweaking `bvh_lafan1_to_*.json`. |
| [`docs/test_motions.md`](test_motions.md) | When picking validation motions. |
| [`docs/archive/`](archive/) | Historical context only — the upstream README and the predecessor design docs (`penetration.md`, `CHANGELOG_ZERO_PENETRATION.md`). |

### 1.5 Tests

| File | What it answers |
|---|---|
| [`tests/check_baseline.py`](../tests/check_baseline.py) | Re-runs canonical motions and `np.allclose`s against pinned baselines. |
| [`tests/test_pkl_schema.py`](../tests/test_pkl_schema.py) | Pins the output pkl schema — required keys, types, shapes, dtypes. |

---

## 2. Why the layout is what it is

A few non-obvious choices worth understanding before you start grading them:

1. **The CLI is intentionally thin.** `scripts/retarget_no_penetration.py`
   is glue (~500 lines). The real logic is in
   `general_motion_retargeting/retargeting/`. The split lets you call the
   helpers from a notebook, a test, or another script without the CLI's
   argparse/path-resolution noise. Five focused modules instead of one
   monolithic 700-line script.

2. **Foot-contact runs on SMPL-X frames, not robot qpos.** Detection sits
   *before* retargeting in the pipeline (see step 8 of
   `retarget_and_save`). This means contact labels for a given clip are
   identical regardless of `--ground_mode` / `--strict_zero_pen` / robot
   choice — they describe the *human* motion, not the *robot*'s. If you
   change ground constraints, the contact flags don't move.

3. **Two IK-config "buckets" for the source format.**
   `IK_CONFIG_DICT["smplx"]` covers AMASS, AMASS CMU, *and* GVHMR — they
   all consume the SMPL-X body model, so they share the same robot mapping.
   `IK_CONFIG_DICT["bvh_lafan1"]` is separate because BVH skeletons have
   different per-joint local frames (this is what makes the BVH integration
   experimental). The `SRC_HUMAN_FOR_FORMAT` dict in
   `retarget_no_penetration.py` does the routing.

4. **Sole compensation lives in `builder.py`, not in `motion_retarget.py`.**
   The IK solver maps human ankle to robot ankle (z=0), but the robot's
   *sole* surface is below the ankle. `build_retargeter` computes
   `baseline_compensation = min_sole_z - clearance` (a negative z) and
   calls `retgt.set_ground_offset(baseline_compensation)`. The two-pass IK
   then perturbs that baseline per frame.

5. **Deletions, not history rewrites.** All 16 dropped robots and the
   `scripts/legacy/` archive were `git rm`ed in normal commits — `git log`
   on `master` still shows them, and anyone needing one can do
   `git checkout master -- assets/<robot>`. The pack size is unchanged;
   the working tree is small.

6. **The PKL schema is self-describing.** Every output records the
   `robot`, `input_format`, `source_file`, `ground_mode`,
   `strict_zero_pen`, `actual_human_height`, and `foot_contact_meta`
   alongside the numeric arrays. Downstream RL code can introspect what
   produced a pkl without re-reading CLI args. Schema is pinned by
   `tests/test_pkl_schema.py`.

7. **What was deliberately *not* unified.** The viewer script and the
   contact-overlay viz are separate (`vis_robot_motion.py` vs
   `vis_robot_motion_with_contact.py`) because the latter has an OpenCV
   dependency the former doesn't. `vis_robot_motion_debug.py` is its own
   thing because it draws sole-marker geoms and penetration colour — a
   diagnostic mode, not the default replay path.

---

## 3. Test commands

These commands exercise every active code path. Run from repo root with
`gmr` activated. Each section is self-contained; you can run them out of
order.

### 3.0 Setup — pin baselines (one-time)

If `tests/baseline/*.pkl` doesn't exist on your machine, create them now
so subsequent regression / schema tests have something to compare against:

```bash
# Generates tests/baseline/35_01_k1_amass.pkl and freekick_k1_gvhmr.pkl
python scripts/retarget_no_penetration.py \
    --input /home/shinben0327/nomadz/k1_motion_data/data/AMASS/CMU/35/35_01_stageii.npz \
    --input_format amass_cmu --robot booster_k1 \
    --ground_mode qp --strict_zero_pen --headless \
    --output tests/baseline/35_01_k1_amass.pkl

python scripts/retarget_no_penetration.py \
    --input /home/shinben0327/nomadz/archive/video_to_k1_pipeline/out_smplx/11_freekick/hmr4d_results.pt \
    --input_format gvhmr --robot booster_k1 \
    --ground_mode qp --strict_zero_pen --headless \
    --output tests/baseline/freekick_k1_gvhmr.pkl
```

The directory is gitignored — these are local-only.

### 3.1 Static checks

```bash
# Package imports cleanly, public surface intact
python -c "
import general_motion_retargeting as g
from general_motion_retargeting.retargeting import (
    build_retargeter, run_batch, detect_smplx_foot_contact,
    apply_fk_post, measure_penetration_depths, smooth_penetration_envelope,
    discover_input_files, load_yaml_paths, amass_stem,
)
print('robots:', list(g.ROBOT_XML_DICT))
print('configs:', {k: list(v) for k, v in g.IK_CONFIG_DICT.items()})
"

# Shell wrappers parse
bash -n shell/run_file.sh
bash -n shell/run_dir.sh
bash -n shell/run_yaml.sh
bash -n shell/analyse.sh

# CLI help (also confirms --input_format includes bvh_lafan1)
python scripts/retarget_no_penetration.py --help | grep -E "input_format|robot|ground_mode|strict_zero_pen"
```

Expected: only `booster_t1` and `booster_k1` listed; both `smplx` and
`bvh_lafan1` IK-config buckets present; `--input_format` choices include
`bvh_lafan1`.

### 3.2 Regression and schema (the hard checks)

```bash
# Re-run baselines and np.allclose vs pinned pkls
python tests/check_baseline.py

# Walk every pkl under tests/baseline/ and assert schema
python tests/test_pkl_schema.py
```

Expected: both print `[ OK ]` for `35_01_k1_amass.pkl` and
`freekick_k1_gvhmr.pkl`. Any `[FAIL]` is a real bug — read the diff line.

### 3.3 K1 — the main path (already validated)

```bash
# AMASS CMU walk, full pipeline, headless
python scripts/retarget_no_penetration.py \
    --input /home/shinben0327/nomadz/k1_motion_data/data/AMASS/CMU/35/35_01_stageii.npz \
    --input_format amass_cmu --robot booster_k1 \
    --ground_mode qp --strict_zero_pen \
    --headless --output /tmp/k1_smoke.pkl

# Replay it interactively (close the viewer with Ctrl-C)
python scripts/vis_robot_motion.py \
    --robot booster_k1 --robot_motion_path /tmp/k1_smoke.pkl

# Diagnostic stats: penetration / jitter
python scripts/compare_penetration_stats.py \
    --robot booster_k1 --motion_files /tmp/k1_smoke.pkl
```

Watch for: feet flush with the floor (no sinking, no float), no jittery
arms, no foot-slide during stance.

### 3.4 GVHMR — monocular video path

```bash
python scripts/retarget_no_penetration.py \
    --input /home/shinben0327/nomadz/archive/video_to_k1_pipeline/out_smplx/11_freekick/hmr4d_results.pt \
    --input_format gvhmr --robot booster_k1 \
    --ground_mode qp --strict_zero_pen \
    --headless --output /tmp/gvhmr_smoke.pkl

python scripts/vis_robot_motion.py \
    --robot booster_k1 --robot_motion_path /tmp/gvhmr_smoke.pkl
```

### 3.5 T1 — second robot, NOT yet regressed

> ⚠️ Only K1 has pinned baselines. T1 was prune-tested at the import level
> but has no end-to-end check. Validate this before merging.

```bash
# T1 single-pass (start with vanilla, no two-pass)
python scripts/retarget_no_penetration.py \
    --input /home/shinben0327/nomadz/k1_motion_data/data/AMASS/CMU/35/35_01_stageii.npz \
    --input_format amass_cmu --robot booster_t1 \
    --ground_mode qp \
    --headless --output /tmp/t1_smoke.pkl

# T1 with two-pass smoothing
python scripts/retarget_no_penetration.py \
    --input /home/shinben0327/nomadz/k1_motion_data/data/AMASS/CMU/35/35_01_stageii.npz \
    --input_format amass_cmu --robot booster_t1 \
    --ground_mode qp --strict_zero_pen \
    --headless --output /tmp/t1_smoke_smooth.pkl

# Replay with the T1 model
python scripts/vis_robot_motion.py \
    --robot booster_t1 --robot_motion_path /tmp/t1_smoke_smooth.pkl
```

Watch for: same expectations as K1 — flush feet, no slide, no jitter.

### 3.6 BVH (LAFAN1) — experimental path

> ⚠️ This path has only been smoke-tested for **import / argparse**. No
> BVH motion was retargeted end-to-end during the cleanup. The arm and
> head joint quaternions in `bvh_lafan1_to_k1.json` are best-effort
> guesses. Read [`docs/bvh.md`](bvh.md) §3–4 before trusting the result.

You'll need a LAFAN1 BVH file. If you don't have one locally:
```bash
# https://github.com/ubisoft/ubisoft-laforge-animation-dataset
# Pick something simple like walk1_subject1.bvh
```

Then:

```bash
python scripts/retarget_no_penetration.py \
    --input /path/to/walk1_subject1.bvh --input_format bvh_lafan1 \
    --robot booster_k1 --ground_mode qp \
    --headless --output /tmp/bvh_smoke.pkl

# Verify the experimental warning printed
# Verify the pkl was written and has valid contact flags
python -c "
import pickle
with open('/tmp/bvh_smoke.pkl', 'rb') as f: d = pickle.load(f)
print('input_format:', d['input_format'])
print('robot:', d['robot'])
print('frames:', len(d['root_pos']))
print('contact meta:', d['foot_contact_meta'])
"

# Visual inspection — this is where K1 arm twist (if any) becomes obvious
python scripts/vis_robot_motion.py \
    --robot booster_k1 --robot_motion_path /tmp/bvh_smoke.pkl
```

Watch for: arms / head twisted by 90° on shoulders or elbows. If yes,
that's the K1 arm/head guesses — see `docs/bvh.md` §4–5 for the fix.

### 3.7 Foot-contact overlay video

```bash
python scripts/vis_robot_motion_with_contact.py \
    --robot booster_k1 --robot_motion_path /tmp/k1_smoke.pkl
```

Outputs an mp4 next to the pkl with `L FOOT` / `R FOOT` text overlaid on
frames where the corresponding flag is `True`. Verify the overlay timing
matches the visible foot strikes.

### 3.8 Batch processing

```bash
# Tiny batch on three CMU subjects (or one)
python scripts/retarget_no_penetration.py \
    --input /home/shinben0327/nomadz/k1_motion_data/data/AMASS/CMU/35 \
    --input_format amass_cmu --robot booster_k1 \
    --ground_mode qp --strict_zero_pen \
    --output /tmp/cmu35_batch/

ls /tmp/cmu35_batch/                 # one pkl per source file
cat /tmp/cmu35_batch/*/output.txt    # tee'd log
```

### 3.9 Dataset analysis report

```bash
python scripts/analyze_locomotion_dataset.py \
    --pkl_dir /tmp/cmu35_batch --robot booster_k1

cat /tmp/cmu35_batch/REPORT.md       # auto-named
```

The Markdown report gives per-clip penetration (max, mean, %frames>tol)
and jitter — if any clip stands out, replay it with `vis_robot_motion.py`.

### 3.10 Browse the batch with the dataset viewer

```bash
python scripts/vis_robot_motion_dataset.py \
    --robot booster_k1 --robot_motion_folder /tmp/cmu35_batch/

# Keys inside the viewer:  [   previous   ]   next   x   trash current pkl
#                          space  pause/resume
```

### 3.11 Soft-ground variant (if you care)

The default everywhere is `--ground_mode qp`. The `soft` mode requires a
`<robot>_with_sites.xml` produced by `add_sole_sites.py`. K1 already has
one; T1 doesn't (open follow-up).

```bash
python scripts/retarget_no_penetration.py \
    --input /home/shinben0327/nomadz/k1_motion_data/data/AMASS/CMU/35/35_01_stageii.npz \
    --input_format amass_cmu --robot booster_k1 \
    --ground_mode soft \
    --headless --output /tmp/k1_soft.pkl
```

### 3.12 Vanilla (no ground constraint) — the upstream baseline

```bash
python scripts/retarget_no_penetration.py \
    --input /home/shinben0327/nomadz/k1_motion_data/data/AMASS/CMU/35/35_01_stageii.npz \
    --input_format amass_cmu --robot booster_k1 \
    --ground_mode none \
    --headless --output /tmp/k1_vanilla.pkl

# Compare penetration stats: should show clear penetration with 'none',
# zero with 'qp --strict_zero_pen'
python scripts/compare_penetration_stats.py \
    --robot booster_k1 \
    --motion_files /tmp/k1_vanilla.pkl /tmp/k1_smoke.pkl
```

### 3.13 Side-by-side comparison video

```bash
python scripts/vis_compare_motions.py \
    --robot booster_k1 \
    --pkl_files /tmp/k1_vanilla.pkl /tmp/k1_smoke.pkl \
    --labels vanilla zero_pen
```

---

## 4. Manual review checklist

Tick each as you go. Items marked **[code]** are read-the-file checks;
**[run]** is execute-and-eyeball; **[both]** is both.

### 4.1 Documentation

- [ ] **[code]** `README.md` reads as a fork-focused intro (no stale upstream
  marketing — no "9 robots", no GitHub video CDN links, no NEWS section).
- [ ] **[code]** `docs/overview.md` §2 table — every fork addition has a
  one-line *why it matters* and a file pointer.
- [ ] **[code]** `docs/pipeline.md` references resolve: links to
  `docs/foot_contact.md`, `../pipeline.md` from archive docs, no link to
  the deleted `CLEANUP_PLAN.md`.
- [ ] **[code]** `docs/foot_contact.md` schema dict matches what the script
  actually writes (`source`, `joints`, `columns`, `z_thresh`, `vel_thresh`,
  `floor_z`).
- [ ] **[code]** `docs/bvh.md` flags experimental status visibly and lists
  which joints are guessed (`Left_Arm_3`, `Right_Arm_3`,
  `left_hand_link`, `right_hand_link`, `Head_2`).
- [ ] **[code]** `CLAUDE.md` describes the *current* tree — no references to
  deleted scripts.

### 4.2 Code structure

- [ ] **[code]** `scripts/retarget_no_penetration.py` is glue — no algorithmic
  logic that should be in `retargeting/`. Skim each numbered step (1–9) of
  `retarget_and_save`.
- [ ] **[code]** Each module under `general_motion_retargeting/retargeting/`
  has a top docstring explaining its role.
- [ ] **[code]** `params.py` contains exactly four dicts, each with two rows
  (`booster_k1`, `booster_t1`).
- [ ] **[code]** `general_motion_retargeting/ik_configs/` has exactly four
  JSON files. No leftover dropped-robot configs.
- [ ] **[code]** `assets/` has only `booster_k1`, `booster_t1`,
  `body_models/`, plus `GMR.png` / `GMR_pipeline.png` (used by archived
  upstream README).
- [ ] **[code]** `scripts/` has 9 entries, all of them documented in
  `CLAUDE.md` §"Entry-point scripts". No `legacy/` subdirectory.

### 4.3 Static checks (3.1)

- [ ] **[run]** Package imports clean.
- [ ] **[run]** All four shell scripts pass `bash -n`.
- [ ] **[run]** `--input_format` accepts `bvh_lafan1`.

### 4.4 Regression / schema (3.2)

- [ ] **[run]** `tests/check_baseline.py` prints `[ OK ]` for both pinned
  motions.
- [ ] **[run]** `tests/test_pkl_schema.py` prints `[ OK ]` for both pkls.

### 4.5 K1 path (3.3, 3.4)

- [ ] **[run]** AMASS CMU 35_01: pkl saved, foot-contact line prints sane
  counts (`L=52/89, R=42/89` ish), foot-flush in viewer.
- [ ] **[run]** GVHMR freekick: pkl saved, viewer shows the kick with
  reasonable foot tracking.

### 4.6 T1 path (3.5) — **not pre-validated**

- [ ] **[run]** T1 single-pass `qp` produces a pkl without errors.
- [ ] **[run]** T1 two-pass `--strict_zero_pen` produces a pkl without
  errors.
- [ ] **[run]** T1 viewer plays the motion; no obvious foot penetration
  or arm twist.

### 4.7 BVH path (3.6) — **experimental, not pre-validated**

- [ ] **[run]** Experimental warning prints on `--input_format bvh_lafan1`.
- [ ] **[run]** A LAFAN1 walk produces a pkl with valid contact meta.
- [ ] **[run]** Viewer playback — note which joints look wrong (expect
  arms / head). Cross-reference `docs/bvh.md` §4 fix list.

### 4.8 Tooling and shell wrappers (3.7–3.13)

- [ ] **[run]** `vis_robot_motion_with_contact.py` produces an mp4 with
  visible per-frame `L FOOT` / `R FOOT` overlays.
- [ ] **[run]** `shell/run_dir.sh` against a small AMASS subdir works and
  produces `output.txt` log.
- [ ] **[run]** `analyze_locomotion_dataset.py` writes a `REPORT.md` with
  per-clip stats.
- [ ] **[run]** `vis_robot_motion_dataset.py` can browse a folder of pkls
  with `[`/`]`.
- [ ] **[run]** `--ground_mode soft` works on K1.
- [ ] **[run]** `--ground_mode none` shows visible penetration (sanity:
  the rest of the pipeline isn't silently doing the work).
- [ ] **[run]** `compare_penetration_stats.py` quantifies the
  `none` vs `qp --strict_zero_pen` gap.
- [ ] **[run]** `vis_compare_motions.py` produces the side-by-side mp4.

### 4.9 Cleanup hygiene

- [ ] **[code]** No `*.pyc`, `__pycache__/`, `output/`, `videos/`, or
  `tests/baseline/` is staged. (`git status` is clean.)
- [ ] **[code]** `.gitignore` lists `output/` and `tests/baseline/`.
- [ ] **[run]** `git log master..HEAD --oneline` shows 25 commits ending
  in `9808942 Pin pkl schema...` (or whatever the tip is by the time
  you read this).

---

## 5. Known gaps (intentional, document don't fix)

- **T1 has no `<robot>_with_sites.xml`** — `--ground_mode soft` works on
  K1 but errors out on T1. To fix, run `python scripts/add_sole_sites.py`
  with T1 wired in (the script is K1-only as shipped).
- **BVH FPS is hardcoded to 30.** LAFAN1 is natively 30, so this is fine
  for the only validated source. For other BVH sources, plumb `frametime`
  from `read_bvh` through `Anim` in `utils/lafan_vendor/extract.py`.
- **BVH human height is hardcoded to 1.75 m.** Scaling will be off for
  skeletons of different heights. Re-enable the commented-out calculation
  at `general_motion_retargeting/utils/lafan1.py:43–44`.
- **K1 BVH arm/head joint quaternions are guessed.** `docs/bvh.md` §5
  explains the principled G1-derived fix.
- **`bvh_nokov` is not exposed.** It was half-wired in earlier work and
  was removed deliberately. Add it back in a separate, deliberate commit
  if needed.

---

## 6. After review — what to do next

1. If everything passes, the branch is ready to merge to `master`.
2. If a checklist item fails, capture the exact command + output and
   open a follow-up. Don't fix in this branch unless it's a regression
   — keep the cleanup PR scoped.
3. Once `master` is updated, the side branches (`feature/repo-cleanup`,
   `feature/add-bvh-support`, `test/ground-contact-flag`,
   `feature/ground-penetration`) can be deleted from the remote — their
   useful content has landed.
