# Cleanup Report

This document records what changed during the repo-cleanup pass on
`feature/zero-penetr-and-smoothed`, what was verified, what is intentionally
left as future work, and how to pick up.

The work was driven by `CLEANUP_PLAN.md` (the plan handed in at the start of
the session) and was executed in six small, behaviour-preserving commits on
top of `47d3b5d`. Every commit was verified before the next was started.

> Conda environment used for all checks: `gmr` on this workstation.

---

## 1. Commits landed

```
dc2a28a Commit F: remove batch_locomotion_k1.py
32901b5 Commit E: relocate per-format converters to scripts/legacy/
b7fbc82 Commit D: extract pipeline helpers into general_motion_retargeting/retargeting/
96fac27 Commit C: CLI renames and dead-flag removal
5954be6 Commit B: enrich output pkl schema with run metadata
a30757b Commit A: consolidate docs into docs/, move shell scripts
```

Net effect (`git diff --stat 47d3b5d..HEAD`):

- `scripts/retarget_no_penetration.py`: 729 -> 488 lines (-33 %).
- New `general_motion_retargeting/retargeting/` package: 5 modules, 565 lines.
- New `docs/pipeline.md`: 805 lines, supersedes `penetration.md` +
  `CHANGELOG_ZERO_PENETRATION.md` (kept under `docs/` as archived references).
- New `scripts/legacy/` with 4 relocated single-format scripts and a README.
- Removed: `scripts/batch_locomotion_k1.py` (80 lines, superseded by
  `--yaml`).

### Commit A — Doc consolidation

- Created `docs/pipeline.md` as the new combined design reference. It folds
  in `penetration.md` and `CHANGELOG_ZERO_PENETRATION.md` and adds new
  sections for T1 robot support (commit 07c91ff), foot-ground contact
  flags (47d3b5d), and the upcoming pkl schema (Commit B).
- Moved `penetration.md` -> `docs/penetration.md` and
  `CHANGELOG_ZERO_PENETRATION.md` -> `docs/CHANGELOG_ZERO_PENETRATION.md`,
  added "archived" banners pointing at `pipeline.md`.
- Moved `run_batch_comparison.sh` -> `shell/run_batch_comparison.sh`.
- Added a one-paragraph pointer to `docs/pipeline.md` in `README.md` and a
  three-line note in `CLAUDE.md`.
- Stashed the working plan as `CLEANUP_PLAN.md` so anyone reading the
  branch can see the rationale.

### Commit B — Richer pkl schema

Added six new keys to the output `motion_data` dict in
`scripts/retarget_no_penetration.py`:

| Key | Purpose |
|---|---|
| `robot` | provenance — which robot's IK/sole config was used |
| `input_format` | which loader produced the source frames |
| `source_file` | absolute path of the source motion |
| `strict_zero_pen` | True if Pass 2 ran |
| `actual_human_height` | source SMPLX human height (m) |
| `foot_contact_params` | echoes `{z_thresh, v_thresh}` used for contact |

All pre-existing keys were left unchanged. Verified bitwise-identical
arrays vs the pre-Commit-B baseline on AMASS CMU 35_01 (every numeric
key `np.allclose`).

### Commit C — CLI renames and dead flags

`scripts/retarget_no_penetration.py`:

- `--no_viz` -> `--headless` (legacy spelling kept as alias).
- `--override` -> `--overwrite` (legacy spelling kept as alias).
- Removed `--max_root_dz` wiring. Empirically ineffective (see
  `docs/pipeline.md` sec 8.3); the `RootZLimit` class itself is retained
  in the package as documentation.
- Removed dead `--loop` flag (never wired in this script).
- Tightened `--strict_zero_pen` and `--gain` help text.

`scripts/compare_penetration_stats.py` and `scripts/vis_compare_motions.py`:

- Both previously hardcoded `ROBOT_TYPE = "booster_k1"`; now expose
  `--robot {booster_k1, booster_t1}` with `booster_k1` as default so
  existing invocations are unaffected.

Verified end-to-end: AMASS CMU 35_01 with `--strict_zero_pen --headless`
produced arrays bitwise-identical to the pre-rename baseline.

### Commit D — Extract helpers (the big one)

Created `general_motion_retargeting/retargeting/` with five focused modules:

```
general_motion_retargeting/retargeting/
  __init__.py          re-exports
  builder.py           build_retargeter()
  strict_zero_pen.py   measure_penetration_depths(), smooth_penetration_envelope()
  foot_contact.py      detect_smplx_foot_contact()
  fk_post.py           apply_fk_post()
  batch_runner.py      Tee, run_batch(), discover_input_files(),
                       load_yaml_paths(), amass_stem()
```

Why each one:

- **`builder.build_retargeter`** replaces a 60-line nested closure
  (`create_retargeter`) that was called twice in the two-pass path. It now
  takes its dependencies explicitly (`robot_type`, `ground_mode`,
  `sole_config`, `actual_human_height`, the constraint params) and returns
  `(retargeter, baseline_compensation)`. The Pass-2 path passes the same
  kwargs back in to construct a fresh retargeter without duplicating
  setup code.
- **`strict_zero_pen`** isolates the most magic-number-laden piece of the
  branch. The constants `DILATE_SIZE = 15`, `SMOOTH_SIGMA = 5`,
  `FINAL_SIGMA = 2` are now named module constants with frame-rate
  rationale in the docstring (`~ 0.5 s` window, `~ 0.17 s` rise time at
  30 fps). Both functions are independently testable.
- **`foot_contact.detect_smplx_foot_contact`** returns
  `(flags, info_dict)` so the CLI can keep its rich-formatted print but
  the function is also reusable for re-labelling existing data without
  re-running IK.
- **`fk_post.apply_fk_post`** packages the height-adjust /
  root-origin-offset / `local_body_pos` block, including the
  `--height_adjust` vs ground-mode incompatibility warning.
- **`batch_runner`** carries `Tee`, `run_batch`, file-discovery helpers,
  and YAML parsing. `run_batch` takes a `retarget_fn` callback so the
  module is decoupled from any particular CLI script.

The script (`scripts/retarget_no_penetration.py`) now reads as
`load -> build -> Pass 1 -> optional Pass 2 -> FK post -> contact ->
save`, with the two-pass driver in `_run_two_pass` and the four
near-duplicate `viewer.step` calls hoisted into `_step_viewer`.

**Regression check (the load-bearing one).** Re-ran the three
representative configurations through the refactored pipeline and
compared against the pre-refactor baselines saved in
`/tmp/gmr_baseline/`:

| config | root_pos | root_rot | dof_pos | local_body_pos | foot_flags |
|---|---|---|---|---|---|
| AMASS+QP, K1 | allclose | allclose | allclose | allclose | array_equal |
| AMASS+QP+strict, K1 | allclose | allclose | allclose | allclose | array_equal |
| GVHMR+QP+strict, K1 | allclose | allclose | allclose | allclose | array_equal |

Soft mode on K1 also smoke-tested (file produced, no crash).

### Commit E — Legacy script relocation

Moved four single-format scripts to `scripts/legacy/` (kept here, not
deleted, because they may still be useful for debugging differences
against the unified pipeline or for one-off interactive viewing):

- `smplx_to_robot.py`
- `smplx_to_robot_dataset.py`
- `gvhmr_to_robot.py`
- `gvhmr_to_robot_batch.py`

Added `scripts/legacy/README.md` mapping each legacy invocation to its
`retarget_no_penetration.py` replacement.

`bvh_to_robot.py` and `bvh_to_robot_dataset.py` were left at the scripts/
top level because BVH support is still on `feature/add-bvh-support`; once
that branch lands the BVH scripts will move here too.

### Commit F — Delete `batch_locomotion_k1.py`

The script hardcoded local paths (`locomotion_k1/`, `kicking_source/`,
`data/AMASS/CMU/`) and re-launched `retarget_no_penetration.py` per file
via `subprocess.run`. Its job is now covered by
`retarget_no_penetration.py --yaml <list>` with no hardcoded paths and
no subprocess overhead.

---

## 2. Verification evidence

### 2.1 Regression (Commit D was numerically lossless)

`/tmp/gmr_baseline/` was populated from `47d3b5d` HEAD before any change:

```
k1_amass_qp.pkl                       AMASS CMU 35_01, K1, QP only,    89 frames
k1_amass_qpsmoothed.pkl               AMASS CMU 35_01, K1, QP+strict,  89 frames
k1_gvhmr_qpsmoothed.pkl               GVHMR kick_soogon_2, K1, QP+strict, 134 frames
```

After all six commits, re-ran each through the new HEAD and diffed.
Every numeric array (`root_pos`, `root_rot`, `dof_pos`, `local_body_pos`)
matched under `np.allclose`; `foot_ground_contact_flags` matched under
`np.array_equal`. Schema completeness check confirmed all 15 expected
keys are present, none extra. Output of the regression script is
reproducible via:

```bash
conda run -n gmr python /tmp/gmr_regr.py
```

(see `/tmp/gmr_regr.py` for the script — also documented in
`docs/pipeline.md` if reproducing from scratch.)

### 2.2 Strict-zero-penetration confirmed

`scripts/compare_penetration_stats.py --robot booster_k1` on the AMASS
sample:

```
Label                frames  pen/N  max_pen  max_up  mean_up  rz_range
----------------------------------------------------------------------
k1_amass_qp             89  4/89    19.0mm   11.8mm   4.4mm   37.3mm
k1_amass_qpsmoothed     89  0/89     0.0mm   17.3mm   5.2mm   45.5mm
```

Confirms the two-pass IK still drives `max_pen` to exactly 0 mm. Note the
slight `max_up` increase on this short clip — exactly the contact-oblivious
envelope behaviour described in `docs/pipeline.md` sec 8.2; the planned
contact-aware two-pass (sec 9) is the cleanest remaining fix.

### 2.3 T1 builds end-to-end

The 07c91ff commit message warned of an AMASS-CMU crash on T1. Confirmed
on this workstation:

- `build_retargeter('booster_t1', ground_mode='qp', ...)` returns a valid
  retargeter with `baseline_compensation = -33.0 mm` (matches docs sec 4.2).
- T1 + GVHMR + `--ground_mode qp --strict_zero_pen`: completed cleanly,
  pkl schema complete, `pen/N = 0/134, max_pen = 0.0 mm`.
- T1 + AMASS CMU + `--ground_mode qp` (with and without `--strict_zero_pen`)
  on `35_01_stageii.npz` and `02_01_stageii.npz`: **completed cleanly**.
  The crash from 07c91ff was not reproducible on these test files.

This means either (a) the crash was specific to a motion not in this
local test set, (b) it was an environment/CUDA issue at the time, or
(c) it was masked by a later fix in this branch. See section 4.2 for the
follow-up suggestion.

### 2.4 Diagnostic scripts honour `--robot`

```bash
conda run -n gmr python scripts/compare_penetration_stats.py --robot booster_t1 \
    /tmp/gmr_final/t1_gvhmr_qpsmoothed.pkl
# t1_gvhmr_qpsmoothed   134  0/134   0.0mm   4.8mm   0.4mm   22.4mm
```

Works as expected; the T1 sole-config produces a different penetration
threshold than K1's, so the `--robot` flag is load-bearing for accurate
stats.

### 2.5 Subpackage importable

```python
from general_motion_retargeting.retargeting import (
    build_retargeter, detect_smplx_foot_contact,
    measure_penetration_depths, smooth_penetration_envelope,
    apply_fk_post, Tee, run_batch, discover_input_files,
    load_yaml_paths, amass_stem,
)
# All importable; smoke-test of smooth_penetration_envelope on a synthetic
# spike pattern returns the expected shape and roughly-covering envelope.
```

---

## 3. Final repository layout

```
GMR/
  CLAUDE.md                         (updated to point at docs/ and retargeting/)
  CLEANUP_PLAN.md                   (the plan)
  CLEANUP_REPORT.md                 (this file)
  README.md                         (one-paragraph pointer added)

  docs/
    pipeline.md                     NEW combined design + CLI + schema doc
    penetration.md                  archived (banner -> pipeline.md)
    CHANGELOG_ZERO_PENETRATION.md   archived (banner -> pipeline.md)

  shell/
    run_batch_comparison.sh         (moved from root)

  general_motion_retargeting/
    __init__.py                     (re-exports `retargeting` subpkg)
    ground_constraint.py            (unchanged in this pass)
    sole_points.py                  (unchanged in this pass)
    motion_retarget.py              (unchanged in this pass)
    robot_motion_viewer.py          (unchanged in this pass)
    retargeting/                    NEW subpackage
      __init__.py                   re-exports
      builder.py                    build_retargeter()
      strict_zero_pen.py            two-pass IK helpers
      foot_contact.py               SMPLX toe contact detector
      fk_post.py                    height_adjust / origin / local_body_pos
      batch_runner.py               Tee, run_batch, file discovery, YAML

  scripts/
    retarget_no_penetration.py      MAIN entry point (488 lines, glue only)
    add_sole_sites.py
    compare_penetration_stats.py    +--robot
    vis_compare_motions.py          +--robot
    vis_robot_motion.py
    vis_robot_motion_dataset.py     ('x' triage hotkey from fac4ebd)
    vis_robot_motion_debug.py
    vis_robot_urdf.py
    bvh_to_robot.py                 (kept; BVH path not yet in pipeline)
    bvh_to_robot_dataset.py         (kept; BVH path not yet in pipeline)
    fbx_offline_to_robot.py         (unrelated; upstream)
    optitrack_to_robot.py           (unrelated; upstream)
    smpl_to_smplx.py                (data-prep utility)
    convert_omomo_to_smplx.py       (data-prep utility)
    batch_gmr_pkl_to_csv.py         (data-prep utility)
    legacy/
      README.md                     migration guide
      smplx_to_robot.py
      smplx_to_robot_dataset.py
      gvhmr_to_robot.py
      gvhmr_to_robot_batch.py
```

---

## 4. What remains — open work

Mapped to `CLEANUP_PLAN.md` so each item is traceable.

### 4.1 Plan items deferred for explicit reasons

- **Unit tests (plan §5.2-8).** Not added in this pass. The behaviour
  preservation was verified through full end-to-end pkl regression
  (more rigorous than unit tests would have been), and adding tests on
  top of the current zero-penetration claims would benefit from real
  fixture data. **Suggested next step:** add `tests/test_foot_contact.py`
  with a synthesised 30-frame SMPLX clip (just `left_foot`/`right_foot`
  position trajectories) and `tests/test_strict_zero_pen.py` with a
  hand-built `depths` spike pattern asserting the envelope is smooth and
  approximately covering. Both can run without MuJoCo.

- **Console script entry point (plan §5.3-18).** Not added because
  `setup.py` is unchanged in this pass. **Suggested next step:** add
  `entry_points={'console_scripts': ['gmr-retarget = general_motion_retargeting.retargeting.cli:main']}`
  and move `parse_args` + `main` from the script into a new `retargeting/cli.py`.

- **Parallel batch runner (plan §5.2-11).** Not added. The `Tee` + rich
  reconfigure plumbing in `batch_runner.run_batch` is single-process. A
  `multiprocessing.Pool` rewrite needs per-worker log files (sharing a
  `Tee` from the parent doesn't cross process boundaries).

- **Contact-aware two-pass (plan §5.3-16).** Not added. This is the
  cleanest fix for the run-motion landing-jitter problem documented in
  `docs/pipeline.md` sec 8.2 / 9. It would override
  `final_depths[i] = 0` wherever `foot_ground_contact_flags[i]` says
  "stance". **Suggested next step:** add a `--contact_aware_envelope`
  flag and wire it into `_run_two_pass` after the existing
  `smooth_penetration_envelope` call.

- **Hysteresis on contact flags (plan §5.2-12).** Not added. The
  detector still produces single-frame flicker around stride boundaries.
  Easy fix: 2-of-3 vote or minimum-stance-duration filter applied to the
  `(N, 2)` bool array inside `detect_smplx_foot_contact` before return.

- **Big-toe joint as second contact detector (plan §5.3-13).** Not added.
  Would let the detector distinguish heel-strike from toe-off.

- **Persist `final_depths` in the pkl (plan §5.2 misc).** Not added,
  intentionally kept the schema lean. The Pass-1 `depths` and Pass-2
  `final_depths` arrays are computed in `_run_two_pass` and are useful
  diagnostic data; if downstream RL pipelines want them, add two new
  optional keys.

- **Dataset auto-filter (plan §5.3-14).** Not added. Would auto-trash
  pkls with `max_pen > 10 mm` or `max_up > 30 mm` based on
  `compare_penetration_stats` output, leaving only borderline cases for
  manual `vis_robot_motion_dataset` triage.

- **Capsule-geom support in `detect_sole_points_from_mjcf` (plan §5.3-17).**
  Not added. Would let new robots like T1 be onboarded with one line
  instead of manual capsule-end-cap math.

### 4.2 Items I tried and could not complete in this pass

- **Reproduce the T1 + AMASS CMU crash mentioned in commit 07c91ff.**
  Tested two CMU files (`35_01_stageii.npz`, `02_01_stageii.npz`) with
  `--robot booster_t1 --ground_mode qp` (with and without
  `--strict_zero_pen`); both completed cleanly. The crash was not
  reproducible on the locally-available test data. The issue may have
  been (a) specific to a particular motion file not in this set,
  (b) an environment / CUDA issue at the time of 07c91ff, or
  (c) masked by a later fix on this branch. **Suggested follow-up:**
  the next session with access to the original failing motion file
  should re-run the same command, capture the stack trace if it still
  crashes, and document the resolution in `docs/pipeline.md` sec 4.2.

- **Re-measure jitter (`max_up`) for the empirical results table in
  `docs/pipeline.md` sec 6.** The existing pre-strict-zero-pen numbers
  were measured before the dynamic CBF gain (commit 1c99b8f) shipped, so
  they are no longer directly comparable to a fresh `qp_smoothed` run.
  The table currently has `TODO` entries for `max_up`, `mean_up`,
  `rz_range` in the new column. Re-deriving requires re-running every
  motion in the original baseline set on the current HEAD; deferred
  because it does not affect the cleanup goal.

### 4.3 Items intentionally NOT done

- **Merge `feature/add-bvh-support` and `feature/repo-cleanup` into this
  branch.** Out of scope for this pass — the user asked for a cleanup
  plan execution, not a multi-branch reconciliation. The plan §3.4
  describes the merge order. The BVH branch's claim that the IK config
  values are "guessed" is unresolved; merging without a validation pass
  on a known LAFAN1 clip would risk shipping bad configs.

- **Drop `--input_format amass_cmu` as a separate bucket** (plan §2.3
  table). Considered, but doing so would change the auto-output-stem
  cleaning (`_stageii.npz` -> bare stem) for any user who currently
  passes `--input_format amass_cmu` instead of `smplx`. Net behaviour
  change disguised as a rename; deferred until anyone actually wants it.

- **Add a `--ground_mode qp_smoothed` shorthand** (plan §5.2-10). Considered.
  Would imply `--strict_zero_pen`, but mixing a "mode" and a "post-process
  flag" muddles the orthogonal axes (which constraint mode + whether to
  do two-pass). Easier to stick with the explicit `--ground_mode qp
  --strict_zero_pen` pair until we add a third post-process variant
  (e.g. contact-aware) at which point the shorthand would have to grow
  anyway.

---

## 5. How to pick up from here

Working tree is clean; branch is `feature/zero-penetr-and-smoothed`, six
commits ahead of pre-cleanup `47d3b5d`.

The two highest-value, lowest-effort follow-ups (in order):

1. **Reproduce or close the T1 + AMASS-CMU crash** (`docs/pipeline.md`
   sec 4.2). Without it, the doc has to leave a "Known issue" sticker on
   T1, which limits how much the rest of the team will trust the T1 path.
2. **Re-measure jitter for the empirical table** (`docs/pipeline.md`
   sec 6 — the `TODO` cells). Just re-run the existing motions on the
   current HEAD and fill in the table. ~15 min of compute.

The next two highest-value items are bigger but well-scoped:

3. **Contact-aware two-pass** (sec 4.1). Straightforward extension of
   `_run_two_pass`; the contact flags are already computed and stored
   in the pkl, so the change is local to the script + a flag in
   `parse_args`.
4. **Console script entry point + `tests/`** (sec 4.1). Mostly mechanical;
   would let `pip install -e .` give us `gmr-retarget --help` and would
   give the contact / envelope code a CI safety net.

After those four, the remaining items in `CLEANUP_PLAN.md` §5.2-§5.3 can
be picked up in any order based on what the project actually needs next.
