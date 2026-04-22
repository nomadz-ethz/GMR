# Repo Cleanup & Next-Steps Plan

This file captures the read-through and plan before any edits. It is meant to be handed to the next Claude Code session (with `conda env gmr` active) to execute against.

> **Scope window:** everything on `feature/zero-penetr-and-smoothed` since `b7a9cb4` (`Add draft of algorithms preventing foot ground penetration.`). Commits covered:
>
> - `c7faa94` docs-only refresh of `penetration.md`
> - `1c99b8f` strict-zero-pen two-pass IK + dynamic CBF gain
> - `fac4ebd` dataset-viewer triage (`x` to trash) + `batch_locomotion_k1.py`
> - `07c91ff` T1 support + headless batch video recording
> - `47d3b5d` SMPLX toe-based foot-ground contact flags (current HEAD)
>
> Sibling branches worth looking at before merging:
> - `feature/repo-cleanup` (`d48f774`) — already moves `penetration.md`/`CHANGELOG_ZERO_PENETRATION.md` to `docs/`, shell scripts to `shell/`, adds `scripts/analyze_locomotion_dataset.py`. Good reference for directory layout.
> - `feature/add-bvh-support` (`f3df32a`) — adds `bvh_lafan1`/`bvh_nokov` paths in `retarget_no_penetration.py` + `bvh_lafan1_to_{k1,t1}.json` configs + `BVH_SUPPORT.md`. Tests are still marked as "guessed values"; needs validation.

---

## 1. Understanding Recap — what actually happened

1. **Ground penetration prevention (baseline from b7a9cb4)** added three layers in `general_motion_retargeting/ground_constraint.py`:
   - `GroundPlaneLimit` (hard QP inequality row per sole corner)
   - `RootZLimit` (kept for history, empirically ineffective — see `penetration.md §4.3`)
   - `SoftGroundConstraint` + `GMRWithSoftGround` (weaker, needs `K1_serial_with_sites.xml` produced by `scripts/add_sole_sites.py`)
2. **Strict zero penetration (1c99b8f)** layered a **two-pass IK** on top of QP: Pass 1 measures per-frame penetration with `mj_forward`, dilates/blurs it into a smooth envelope with `scipy.ndimage`, then Pass 2 re-runs IK with `retarget.set_ground_offset(baseline_compensation - final_depths[i])` per frame. Also bumped `GroundPlaneLimit.gain` to a `dynamic_gain = min(1.0, self.gain + 2*|margin|)` when penetrating, so deep violations get corrected faster. Result: 0.0 mm max penetration on CMU walks with no visible foot-slide.
3. **T1 + headless (07c91ff)** added `booster_t1` to `SOLE_CONTACT_POINTS`/`FOOT_LINK_NAMES`, `--robot` flag (replacing hardcoded `ROBOT_TYPE`), and re-routed batch video recording through `output_dir/videos/` so offscreen MP4s stay close to their pkl. Also added log tee-ing (`_Tee`) into `<output_dir>/<top_level>/output.txt` for batch runs.
4. **Dataset triage (fac4ebd)** added the `'x'` key to `vis_robot_motion_dataset.py` (moves the current pkl into `trash/`), and `batch_locomotion_k1.py` — a thin wrapper that re-sources `.npz` originals for a set of "good" pkls. This latter script is ad-hoc and tied to my local directory layout.
5. **Foot-ground contact flags (47d3b5d)** reads SMPLX `left_foot`/`right_foot` joint positions (indices 10/11) from `smplx_data_frames`, computes per-frame speed with `np.gradient`, establishes a per-motion `floor_z = min observed toe z`, then thresholds `(height < z_thresh) AND (speed < v_thresh)` into a `(N, 2)` bool mask saved as `foot_ground_contact_flags` in the output pkl. It intentionally runs on raw SMPLX frames so the labels are identical across `ground_mode` / `strict_zero_pen` variants.

The story of the branch is: **"Make K1 (and now T1) retargeted pkls that are physically plausible AND come with contact labels, for AMASS + GVHMR + [soon] BVH."**

Everything in this plan should keep that story coherent.

---

## 2. `scripts/retarget_no_penetration.py` — structural cleanup

At 732 lines and counting, the script mixes CLI parsing, file discovery, IK building, two-pass correction, viewer handling, FK post-processing, contact-flag extraction, and batch logging. Break it up, but do not over-abstract — the file is fine as the single entry point; just extract coherent pieces into small helpers or sibling modules.

### 2.1 Recommended module layout

Add a small `general_motion_retargeting/retargeting/` subpackage (or keep as sibling files in the top-level package) and move pure-logic pieces there. The script should only be glue.

| New home | Contents | Why |
| --- | --- | --- |
| `general_motion_retargeting/retargeting/builder.py` | `build_retargeter(args, robot_type, sole_config, actual_human_height) -> (retgt, baseline_compensation)` — current `create_retargeter()` nested closure | Removes the 70-line closure from `retarget_and_save`, lets you unit-test the "assemble GMR + ground_limit + offset" path without running IK. |
| `general_motion_retargeting/retargeting/strict_zero_pen.py` | `measure_penetration_depths(qpos_list, model, sole_config, clearance) -> np.ndarray` and `smooth_penetration_envelope(depths, dilate_size=15, gauss_sigma=5, final_sigma=2) -> np.ndarray` | The two `scipy.ndimage` calls are the single most magic-number-laden piece of the branch. Isolate them so the numbers are discoverable, documented, and can be tuned. |
| `general_motion_retargeting/retargeting/foot_contact.py` | `detect_smplx_foot_contact(smplx_data_frames, fps, z_thresh, v_thresh) -> (N,2) bool` | Stand-alone + unit-testable. Lets the function grow (hysteresis, left/right from `left_foot`/`right_foot` *and* `left_toe_base`/`right_toe_base`, fallback to ankle if toes missing, etc.) without touching the script. |
| `general_motion_retargeting/retargeting/fk_post.py` | `apply_height_adjust(root_pos, body_pos, ground_mode) -> root_pos`, `apply_root_origin_offset(root_pos)`, `compute_local_body_pos(km, rp, rr, dp)` | Current FK post-processing block is 40 lines inside `retarget_and_save`. Extracting it also exposes the `height_adjust` vs `ground_mode != "none"` compatibility warning clearly. |
| `general_motion_retargeting/retargeting/batch_runner.py` | `_Tee`, `_run_batch`, YAML loader, file discovery (`discover_input_files`, `load_yaml_paths`, `_amass_stem`) | These are pure plumbing; the script is cleaner if `main()` only resolves args → calls one of `run_single_file` / `run_batch` / `run_yaml`. |

Then `scripts/retarget_no_penetration.py` becomes roughly: `parse_args → resolve paths → call retargeting.run_single(args) or retargeting.run_batch(args)`, well under 300 lines.

### 2.2 In-script refactors (even if we keep it as one file)

- **Remove the closure.** `create_retargeter()` captures `args`, `ROBOT_TYPE`, `sole_config`, `actual_human_height` by closure so that `strict_zero_pen` can call it twice. Make it a top-level function that takes these explicitly. The current code is 40 lines of indent-drift inside `retarget_and_save`.
- **Move the Pass-1 / Pass-2 block into its own function.** `run_strict_zero_pen(qpos_list, retargeter_factory, sole_config, ...)`. This also clarifies that the viewer stepping is intentionally inside the loop (pass-2 only) because pass-1 qpos gets thrown away.
- **Collapse duplicated viewer-stepping.** Currently the `viewer.step(...)` call is written four times (normal, pass-2, pass-2 no-penetration branch, normal no-strict branch) with slightly different arguments. Hoist a `step_viewer(viewer, qpos_list, scaled_human, rate_limit)` helper.
- **Drop `ROBOT_TYPE = args.robot` alias inside `retarget_and_save`.** Just pass `args.robot` through or name the parameter `robot_type`.
- **Delete `RootZLimit` wiring path from the script** (keep the class + docstring for reference). `--max_root_dz` is documented as ineffective in `penetration.md §4.3`; continuing to branch on it is dead code. Keep the class in the package but remove the CLI flag and `ik_limits2.append(...)` block.
- **`import` hygiene.** Every `retarget_and_save` call re-imports `torch`, `mujoco`, `scipy.ndimage`, `KinematicsModel`, etc. That's fine for the batch-import-cost reason (delay torch load unless we need FK), but make it explicit — add a comment `# deferred to avoid importing torch in pure-IK-only runs` above the import.

### 2.3 CLI flags — what to rename / drop / re-default

| Current flag | Proposed | Rationale |
| --- | --- | --- |
| `--ground_mode {none,qp,soft}` | **Keep**, but **change default to `qp`** | Everything we ship in this branch assumes QP. `none` is debugging-only. |
| `--strict_zero_pen` | Rename to `--smooth_two_pass` (or `--zero_penetration`); **document it in `--help`** so it doesn't look optional-nice-to-have | The name describes the outcome ("strict zero"), not the mechanism ("two-pass smoothing"). Users reading `--help` need to know this costs 2× IK. Whichever name you pick — **stay consistent** between the pkl's `"ground_mode"` field (right now it only stores `args.ground_mode`, but a smoothed run should be distinguishable; see §2.4). |
| `--foot_contact_z_thresh`, `--foot_contact_vel_thresh`, `--no_foot_contact` | Keep names. Consider grouping under `--foot_contact_method {off, smplx_toe, disabled}` later if more methods appear. Defaults (0.08 m, 0.5 m/s) are still "tuning" per commit message. | The current names match what's in the pkl. Don't change them until we have more detectors. |
| `--max_root_dz` | **Delete** (along with wiring). | Documented ineffective; adds a confusing knob. |
| `--gain` default 0.5 | Keep | Matches `penetration.md §4.2`. |
| `--clearance` default 0.003 | Keep | 3 mm matches everything. |
| `--activation_distance` default 0.02 | Keep | Matches. |
| `--ground_height` default 0.0 | Keep | Matches. |
| `--robot {booster_k1, booster_t1}` | **Keep**; make default `booster_k1` (already). Consider accepting a generic MuJoCo robot name as long as `sole_points.py` has an entry; raise a helpful error otherwise. | The valid-list is tight because of `SOLE_CONTACT_POINTS`. |
| `--input_format {smplx,gvhmr,amass_cmu}` | Drop `amass_cmu` as a separate bucket; fold it into `smplx` + a `--dataset amass_cmu` hint for filename stripping ONLY. | `amass_cmu` vs `smplx` use the same loader (`load_smplx_file`); the only difference is `_stageii.npz` stem cleaning. Keeping it as an input format is misleading. Better: `--input_format {smplx, gvhmr}` and let the stem helper auto-strip `_stageii` regardless. (The `repo-cleanup` branch already calls it just `amass`; either rename is fine, but don't have two format names that take the same code path.) |
| `--no_viz` | Rename to `--headless` | "no_viz" is a double negative; `--headless` is the industry-standard toggle and pairs naturally with `--record_video`. |
| `--override` | Rename to `--overwrite` (standard spelling) | Minor, but `--override` reads like "override a default" not "overwrite output". |
| `--rate_limit` | Keep | Fine. |
| `--loop` | Currently parsed but unused (single-file viewer doesn't loop) | Delete or wire it up. |
| `--record_video` | Keep. Document that it implies a `viewer` is constructed even with `--headless`. | Already works. |

### 2.4 Output pkl schema — make it self-describing

Right now the pkl carries `ground_mode`, `ground_clearance`, `foot_ground_contact_flags` but **not** `strict_zero_pen`, `robot`, `input_format`, or source-file stem. Future-you (or downstream RL code) will regret that.

Proposed dict:

```python
{
    "fps": float,
    "robot": "booster_k1",                       # NEW
    "input_format": "smplx" | "gvhmr" | ...,     # NEW
    "source_file": str(input_path),              # NEW — absolute path of the npz/pt source
    "ground_mode": "qp" | "soft" | "none",
    "strict_zero_pen": bool,                     # NEW
    "ground_clearance": float,
    "actual_human_height": float,                # NEW — already computed, nice to keep
    "root_pos": (N,3) float32,
    "root_rot": (N,4) float32,                   # xyzw
    "dof_pos": (N,J) float32,
    "local_body_pos": (N,B,3) float32 | None,
    "link_body_list": list[str] | None,
    "foot_ground_contact_flags": (N,2) bool | None,
    "foot_contact_params": {"z_thresh": 0.08, "v_thresh": 0.5} | None,  # NEW
}
```

Add a `README_PKL_SCHEMA.md` section (or embed in `docs/penetration.md`) so downstream code can pin against a known layout.

### 2.5 Comments & docstrings

- `retarget_and_save` is missing a top-level docstring describing **what's new** (two-pass, contact flags). Rewrite: "Loads one motion, runs single- or two-pass IK against the ground constraint, optionally drives the viewer/recorder, and saves a pkl with the schema described in docs/penetration.md §5.5."
- Every `scipy.ndimage` call (`maximum_filter1d(size=15)`, `gaussian_filter1d(sigma=5)`, then again `sigma=2`) deserves a WHY comment: "15 frames ≈ 0.5 s at 30 fps, captures heel-strike width; σ=5 produces a ~0.3 s rise time so the root lifts before the spike and settles after it; the second σ=2 smooths the kink where `np.maximum(smoothed, depths)` reintroduces sharp points." Without that, these are unexplainable magic numbers. Also make them kwargs with defaults (not bare literals) so users can override from CLI if needed later.
- The `dynamic_gain = min(1.0, self.gain + 2.0 * abs(margin))` line in `ground_constraint.py` is a *behaviour change*, not a cleanup. Docstring needs a new bullet: "When penetrating (margin<0), effective gain is boosted proportional to depth, so deep violations converge in fewer IK iterations. See CHANGELOG_ZERO_PENETRATION.md §1."

### 2.6 Things NOT to change

- The ordering "run foot-contact detection on `smplx_data_frames` before FK post-processing" is intentional (per the commit message) — do not move it into FK post or into the `strict_zero_pen` block. Leave the comment that says so.
- The fact that `create_retargeter` is called twice in strict-zero-pen mode (once for Pass 1, once for Pass 2, fresh state) is intentional. Don't try to reuse the first retargeter.
- Pass-2 viewer-stepping is inside the IK loop rather than after — that's so the viewer shows the final smoothed result, not Pass 1 (which has visible foot slides). Keep it.

---

## 3. Repo-wide structure — what to stash, consolidate, delete

### 3.1 Scripts directory — 20 files, many unrelated to our pipeline

Inventory (by whether our branch uses them):

| Script | Used by our pipeline? | Recommendation |
| --- | --- | --- |
| `retarget_no_penetration.py` | **Yes** (main entry) | Refactor per §2. |
| `add_sole_sites.py` | Only for `--ground_mode soft` | Keep. |
| `compare_penetration_stats.py` | Yes (diagnostics) | Keep. Currently hardcodes `ROBOT_TYPE = "booster_k1"` — add `--robot` flag. |
| `vis_compare_motions.py` | Yes (diagnostics) | Keep. Same `--robot` fix. |
| `vis_robot_motion_dataset.py` | Yes (triage tool from fac4ebd) | Keep. |
| `vis_robot_motion.py` | Yes (single-file viewer) | Keep. |
| `batch_locomotion_k1.py` | **Our branch only**, but hardcodes `locomotion_k1/`, `locomotion_k1_retargeted_smooth/`, `CMU/`, `kicking_source/`, `data/AMASS/CMU` | This is a personal one-off. Either delete, or move to `scripts/personal/` / `scripts/examples/` and add a big `# one-off example; not part of the pipeline` header. Prefer DELETE — its job is better done by `retarget_no_penetration.py --yaml <list>` once a YAML is written. |
| `gvhmr_to_robot.py`, `gvhmr_to_robot_batch.py` | Superseded by `retarget_no_penetration.py --input_format gvhmr` | Move to `scripts/legacy/` with a README pointing at the replacement; or delete outright once we're sure parity exists (they predate the ground-penetration work). |
| `smplx_to_robot.py`, `smplx_to_robot_dataset.py` | Superseded similarly | Same as above. |
| `bvh_to_robot.py`, `bvh_to_robot_dataset.py` | Not yet replaced (BVH path only lives on `feature/add-bvh-support`) | Keep on master. After merging BVH support, move to `scripts/legacy/`. |
| `fbx_offline_to_robot.py`, `optitrack_to_robot.py` | Unrelated to our project | Leave untouched (upstream). |
| `vis_robot_urdf.py`, `vis_robot_motion_debug.py` | Diagnostic; upstream | Leave untouched. |
| `smpl_to_smplx.py`, `convert_omomo_to_smplx.py`, `batch_gmr_pkl_to_csv.py` | Data-prep utilities | Leave untouched. |

Creating a `scripts/legacy/` (or `scripts/single_format/`) folder and moving the `*_to_robot*.py` per-format scripts there will make it obvious that `retarget_no_penetration.py` is now the pipeline. **Don't delete** until the replacement is verified for each format end-to-end (AMASS / GVHMR / BVH), which is a TODO on its own — see §4.

### 3.2 Directory-level tidy-up

Borrow the layout from `feature/repo-cleanup` (branch `d48f774`):

```
GMR/
  docs/
    penetration.md              ← moved from root
    CHANGELOG_ZERO_PENETRATION.md ← moved from root (consolidate: see §4)
    pkl_schema.md               ← NEW, §2.4
  shell/
    run_batch_comparison.sh     ← moved from root
    run_dir.sh, run_file.sh, run_yaml.sh (optional — from repo-cleanup branch)
  scripts/
    retarget_no_penetration.py  (refactored)
    compare_penetration_stats.py
    vis_compare_motions.py
    vis_robot_motion_dataset.py
    vis_robot_motion.py
    add_sole_sites.py
    legacy/                     ← pre-pipeline single-format scripts
      smplx_to_robot.py
      gvhmr_to_robot.py
      gvhmr_to_robot_batch.py
      bvh_to_robot.py
      ...
  general_motion_retargeting/
    retargeting/                ← new per §2.1
      __init__.py
      builder.py
      strict_zero_pen.py
      foot_contact.py
      fk_post.py
      batch_runner.py
    ground_constraint.py
    sole_points.py
    motion_retarget.py
    ...
  output/                       ← gitignored (already gitignored? check)
  videos/                       ← gitignored
```

Check `.gitignore` before moving; `output/` and `videos/` contain local runs and should stay out of git (current `.gitignore` needs verification).

### 3.3 Root-level files

- `CHANGELOG_ZERO_PENETRATION.md` + `penetration.md` — **merge into one doc** under `docs/` (see §4).
- `DOC.md` — tiny upstream stub, leave untouched.
- `TEST_MOTIONS.md` — upstream, leave untouched.
- `README.md` — add a short "K1/T1 ground-contact pipeline" section near the top that links to `docs/penetration.md`. Don't rewrite the upstream README.

### 3.4 Branches to clean up after merging

Once this plan lands on `feature/zero-penetr-and-smoothed`:

1. Rebase `feature/repo-cleanup` on top (or cherry-pick the directory moves it already did).
2. Rebase `feature/add-bvh-support` on top (validate the "guessed values" caveat — see §4.6).
3. Delete the merged branches.

---

## 4. Documentation — one combined design doc

Right now we have:

- `penetration.md` — the detailed design doc (646 lines with TOC).
- `CHANGELOG_ZERO_PENETRATION.md` — a short add-on describing the two-pass IK.
- `BVH_SUPPORT.md` (on `feature/add-bvh-support`) — design note for the BVH input format.
- Nothing about the `foot_ground_contact_flags` feature from `47d3b5d`.
- Nothing about the T1 addition from `07c91ff`.
- Nothing about the pkl schema (§2.4).

**Proposal:** collapse to `docs/pipeline.md` with the structure below, and leave `penetration.md` / `CHANGELOG_ZERO_PENETRATION.md` as redirect stubs (`see docs/pipeline.md`) or delete them.

### 4.1 Proposed structure for `docs/pipeline.md`

```
1. Overview
   1.1 What this pipeline produces (K1/T1 pkls, contact labels, ground-safe qpos)
   1.2 Supported inputs (AMASS SMPLX, GVHMR, BVH* planned)
   1.3 Quick-start single-file and batch examples

2. Ground Penetration Prevention
   2.1 The Problem (from penetration.md §1)
   2.2 Architecture (from penetration.md §2)
   2.3 GroundPlaneLimit (QP) [include dynamic gain from CHANGELOG]
   2.4 Soft mode (abridged — can link out to archived penetration.md §4.4)
   2.5 Strict Zero Penetration (two-pass IK) [from CHANGELOG, expanded]
       - Surveyor pass mechanism
       - Envelope extraction (what the scipy.ndimage numbers mean)
       - Solver pass with per-frame set_ground_offset
       - Why this eliminates heel-strike jitter AND foot slide
   2.6 Sole compensation / ground_offset (from penetration.md §4.5)

3. Foot-Ground Contact Flags                 ← NEW
   3.1 What it is (per-foot bool labels from SMPLX toe positions)
   3.2 Algorithm (height-below-floor AND velocity thresholds)
   3.3 Why on raw SMPLX frames, not on robot qpos
   3.4 Default thresholds and known tuning needs (per commit 47d3b5d)
   3.5 Caveat: currently only populated for smplx and gvhmr input formats

4. Supported Robots
   4.1 Booster K1 — sole box geometry (24 mm below ankle)
   4.2 Booster T1 — dual-capsule sole (30 mm below ankle), added in 07c91ff
   4.3 Adding another robot: SOLE_CONTACT_POINTS + FOOT_LINK_NAMES + matching ik_config

5. Running the Pipeline
   5.1 CLI reference for retarget_no_penetration.py
   5.2 Single file
   5.3 Directory batch
   5.4 YAML batch
   5.5 Comparison (vis_compare_motions, compare_penetration_stats)
   5.6 Dataset triage (vis_robot_motion_dataset with 'x' key)
   5.7 Output pkl schema                    ← reference §2.4

6. Empirical Results                         ← from penetration.md §6, extend with strict_zero_pen column
   - Add a qp_smoothed column to the CMU walk / run / kick tables
   - Add a T1 row (once validated)

7. Known Limitations & Open Problems         ← from penetration.md §8, updated
   - Mark §8.1 (residual walk penetration) as RESOLVED by two-pass
   - Mark §8.2 (run jitter) as PARTIALLY RESOLVED
   - Add: foot-contact thresholds are per-motion-agnostic (no gait-cycle awareness)
   - Add: T1 sole points were derived from the MJCF capsules manually — needs verification against a reference dataset

8. Future Work                               ← from penetration.md §9
```

### 4.2 What to **remove** from the merged doc

- Section 9.1 "Temporal smoothing (post-processing)" from `penetration.md` — the two-pass IK now handles this. Keep only a short "legacy savgol removed" note.
- Section 9.6 "Better CBF gain scheduling" — done (dynamic gain shipped in `1c99b8f`). Move to "Completed Work" section.
- `CHANGELOG_ZERO_PENETRATION.md` §2.1–§2.2 — the explanatory prose is worth preserving but lives more naturally in §2.5 of the new doc.

### 4.3 What to **add** from commits not yet documented

- **T1 support** (07c91ff): note sole geometry, verify sole compensation magnitude (30 mm vs 24 mm), known "AMASS CMU crash" (see commit message: "Crashes for AMASS CMU for some reason"). **Action item:** reproduce and diagnose before claiming T1 production-ready.
- **Dataset triage tool** (fac4ebd): one paragraph describing the `'x'` hotkey and `trash/` convention.
- **Two-pass IK** (1c99b8f): absorbed from `CHANGELOG_ZERO_PENETRATION.md`.
- **Foot contact flags** (47d3b5d): brand new doc section §3 above.
- **Headless batch video + log tee** (07c91ff): short subsection under §5.3.

### 4.4 `README.md`

Add a short section near the top:

```md
## Ground-safe retargeting (Booster K1, T1)

This fork adds a unified pipeline that retargets SMPLX/GVHMR motions to
Booster K1 or T1 with zero foot-ground penetration and per-foot contact
labels. See [docs/pipeline.md](docs/pipeline.md) for design details and
[scripts/retarget_no_penetration.py](scripts/retarget_no_penetration.py)
for the entry point.
```

Do **not** rewrite the upstream README content; keep changes minimal.

### 4.5 `CLAUDE.md`

Add 3–4 lines so future sessions see the new pipeline immediately:

```md
### Ground-safe retargeting pipeline (this fork)
- Entry point: `scripts/retarget_no_penetration.py` (supports --robot {booster_k1, booster_t1}, --ground_mode, --strict_zero_pen, foot-contact flags).
- Constraint code: `general_motion_retargeting/ground_constraint.py`, `sole_points.py`.
- Design doc: `docs/pipeline.md`.
- Legacy per-format scripts (`smplx_to_robot.py`, `gvhmr_to_robot*.py`, `bvh_to_robot*.py`) live in `scripts/legacy/`.
```

### 4.6 BVH branch integration

`feature/add-bvh-support` is still flagged as "guessed values" by its own commit message. Before merging:

1. Run a known LAFAN1 clip through `retarget_no_penetration.py --input_format bvh_lafan1 --robot booster_k1` and compare to the original `scripts/bvh_to_robot.py --robot booster_k1` output (needs a k1 entry in `IK_CONFIG_DICT["bvh_lafan1"]`, which the branch adds).
2. Sanity-check in `vis_robot_motion.py`.
3. Then fold `BVH_SUPPORT.md` into `docs/pipeline.md` §1.2 + §5 (supported inputs + examples).

---

## 5. Additional things to change / features to add

Ordered roughly by value-to-effort.

### 5.1 High value, low effort

1. **Store `strict_zero_pen` + `robot` + `source_file` in the pkl.** Already argued in §2.4. Five-minute change that removes a lot of future "what was this file generated with?" archaeology.
2. **Add `--robot` to `compare_penetration_stats.py` and `vis_compare_motions.py`.** Both hardcode `booster_k1`. Since T1 now exists, this will hit you soon.
3. **Auto-skip `--height_adjust` silently when `--ground_mode != none` is already done with a yellow print; promote it to a `warnings.warn`.** Small, but keeps log output clean in batch.
4. **Fix `--loop` flag (dead code) OR remove it.** Documented as "Loop motion in viewer (single file mode)" but never wired.
5. **Make the T1 AMASS CMU crash reproducible.** Commit `07c91ff` says it crashes; we need a repro command + stack trace before merging further. Run one CMU file through T1 with `--ground_mode qp --strict_zero_pen` and capture the error.
6. **Gitignore** `output/`, `videos/`, `retargeted*/`, `*.pkl` at repo root (if not already). Verify.
7. **Remove `_Tee` log reconfiguration hack for rich** or wrap it behind a helper — `rich.reconfigure()` twice inside `_run_batch` is brittle.

### 5.2 Medium value, medium effort

8. **Unit tests for `detect_smplx_foot_contact` and `measure_penetration_depths`.** These are the two pieces with the most magic numbers and are the most likely to drift. pytest fixtures with a single small SMPLX clip (10 frames synthesised) would be enough. Run under `conda run -n gmr pytest tests/` in CI.
9. **YAML schema for runs.** Currently `amass_cmu_locomotion_list.yaml` just lists paths + descriptions. Extend to optionally specify `--strict_zero_pen`, `--clearance`, etc. per entry so one YAML can mix conditions (or at least per-section). Important for dataset-level comparisons.
10. **A `--ground_mode qp_smoothed` shorthand** that implies `--ground_mode qp --strict_zero_pen`. This is what we actually always run; reduce the mental overhead. Reflect that in the pkl's `ground_mode` field so downstream code can filter on it.
11. **Parallelise the batch runner.** `_run_batch` processes files sequentially. A `multiprocessing.Pool` with `cpu_count() // 2` workers would roughly halve dataset-level runtimes on a typical workstation. Only tricky bit: the rich stdout tee; either serialise prints or give each worker its own log file.
12. **Foot-contact hysteresis.** The two-threshold AND filter produces flicker near stride boundaries. A 2-of-3 vote or a minimum-stance-duration filter (e.g. ≥ 3 frames at 30 fps) would clean this up. See `47d3b5d` note "Still need tuning."

### 5.3 Larger projects

13. **Add `left_big_toe`/`right_big_toe` SMPLX joints as a second contact detector.** SMPLX has toe tip joints (indices 10/11 are actually `left_foot`/`right_foot` which are the toe base; the big-toe tips are at higher indices — verify). A weighted combination of foot-base + toe-tip contact will detect heel-strike vs toe-off cleanly.
14. **Dataset-level quality filtering.** Your `vis_robot_motion_dataset.py` x-key triage is manual. We already have `compare_penetration_stats.py` producing numbers; plug it into a `filter_dataset.py` that auto-moves pkls with `max_pen > 10 mm` or `max_up > 30 mm` into `trash/`, then let manual review focus on borderline cases. Cuts triage time by ~5×.
15. **Per-frame ground clearance trajectory in the pkl.** `depths[i]` from Pass 1 is useful data (it's the raw penetration depth). Storing it (or the `final_depths` smoothed curve) lets downstream RL pipelines use it as a reward signal or loss target. Trivially cheap to persist.
16. **Add a `--ground_mode qp_smoothed_contact_aware` mode** that reads `foot_ground_contact_flags` (either precomputed or on-the-fly from SMPLX) and locks `final_depths[i]` to zero when the contact flag says "should be on ground". The current blur/dilate approach is contact-oblivious. This would be the cleanest fix for `penetration.md §8.2` (run-motion landing jitter) and maps naturally to gait-cycle-aware retargeting.
17. **Cross-robot portability of sole points.** `detect_sole_points_from_mjcf()` exists but only handles box geoms. Extend it to capsule geoms (T1's geometry), which would make adding `unitree_g1`, `fourier_n1`, etc. a one-liner.
18. **Package as a proper CLI entry point.** `setup.py` already registers the package; adding `[project.scripts]` `gmr-retarget = general_motion_retargeting.cli:main` means users can just run `gmr-retarget --input ... --robot booster_k1 --ground_mode qp_smoothed` without remembering the script path.

### 5.4 Things to investigate, not necessarily change

- **GVHMR's post-rotation** note in the `47d3b5d` commit comment — worth a one-paragraph explanation in the doc so someone doesn't re-debug "why the floor isn't at z=0 for GVHMR inputs."
- **The "Cleanup" commit message on `fac4ebd`** ("Better visualization in robot retargeted dataset to filter good motions") could be expanded into an official "Dataset curation guide" section in the pipeline doc.
- **Do we still need the soft mode at all?** Empirically it's worse than QP everywhere (`penetration.md §6`). Consider marking it "archived / not recommended" in the doc, keep the code around for completeness, and stop surfacing it in default examples.

---

## 6. Verification checklist for the next session

Before declaring the cleanup done, the next Claude session should confirm the following using `conda run -n gmr`:

1. `conda run -n gmr python scripts/retarget_no_penetration.py --input <AMASS_CMU_npz> --input_format smplx --ground_mode qp --strict_zero_pen --headless --output /tmp/out.pkl` — succeeds and the pkl contains all §2.4 keys.
2. `conda run -n gmr python scripts/retarget_no_penetration.py --input <AMASS_CMU_npz> --input_format smplx --ground_mode qp --strict_zero_pen --robot booster_t1 --headless --output /tmp/out_t1.pkl` — succeeds OR reproduces the 07c91ff crash with a clean stack trace for debugging.
3. `conda run -n gmr python scripts/retarget_no_penetration.py --input <GVHMR_pt> --input_format gvhmr --ground_mode qp --strict_zero_pen --headless --output /tmp/gvhmr.pkl` — succeeds; `foot_ground_contact_flags` is a `(N,2)` bool array with some `True` entries.
4. `conda run -n gmr python scripts/compare_penetration_stats.py /tmp/out.pkl` — reports `0/N` penetrations (smoothed mode should be strict zero).
5. `conda run -n gmr python scripts/vis_compare_motions.py --motion_a <old_qp.pkl> --motion_b /tmp/out.pkl --video_path /tmp/cmp.mp4` — renders.
6. `conda run -n gmr python scripts/vis_robot_motion_dataset.py --robot_motion_folder /tmp/retargeted_batch/` — `'x'` moves files to `trash/`.
7. `conda run -n gmr python -c "import pickle; d = pickle.load(open('/tmp/out.pkl','rb')); print(sorted(d.keys()))"` — matches §2.4 schema.
8. `conda run -n gmr pytest tests/ -q` — if unit tests from §5.2-8 are added.

Retain the pre-cleanup reference outputs by running steps 1, 3, 4 on `feature/zero-penetr-and-smoothed@47d3b5d` first, archiving the pkls, then re-running after the refactor and diffing with `numpy.allclose` on every array field. Refactor should be behaviour-preserving modulo floating-point noise.

---

## 7. Suggested execution order for the next session

To keep diffs small and reviewable, do the refactor in this order (one commit per step):

1. **Commit A — doc consolidation.** Create `docs/pipeline.md`, move `penetration.md`/`CHANGELOG_ZERO_PENETRATION.md` into `docs/`, leave stub files at root pointing to the new location. No code changes.
2. **Commit B — pkl schema.** Add new keys (§2.4) to the output dict + update any reader. Keep all old keys so downstream code doesn't break.
3. **Commit C — CLI renames / defaults.** `--no_viz → --headless`, `--override → --overwrite`, remove `--max_root_dz`, remove dead `--loop`, add `--robot` to `compare_penetration_stats.py` / `vis_compare_motions.py`. Add backwards-compatible aliases for the flags so existing shell scripts don't break immediately.
4. **Commit D — extract helpers.** `retargeting/builder.py`, `strict_zero_pen.py`, `foot_contact.py`, `fk_post.py`, `batch_runner.py`. `retarget_no_penetration.py` shrinks to glue.
5. **Commit E — scripts/legacy/ move.** Relocate single-format scripts; add a README in that folder.
6. **Commit F — delete `batch_locomotion_k1.py`** (or move to `scripts/examples/` with a disclaimer).
7. **Commit G — tests** (§5.2-8).
8. **Commit H — merge prep.** Reconcile with `feature/repo-cleanup` and `feature/add-bvh-support`.

Each commit should pass the §6 verification for the pieces it touches.

---

## 8. TL;DR

- Keep the two-pass IK + contact-flag + T1 work — these are the contributions of this branch.
- Split `retarget_no_penetration.py` into a thin CLI + 4–5 small helper modules; remove the `create_retargeter` closure; remove `RootZLimit` wiring.
- Merge `penetration.md` + `CHANGELOG_ZERO_PENETRATION.md` into `docs/pipeline.md`, add sections for T1, foot contact, and the pkl schema.
- Move per-format legacy scripts to `scripts/legacy/`; delete `batch_locomotion_k1.py`.
- Richen the pkl schema (`strict_zero_pen`, `robot`, `source_file`, optional depth trajectory).
- Next features worth shipping: contact-aware two-pass (§5.3-16), dataset auto-filter (§5.3-14), parallel batch (§5.2-11), hysteresis on contact flags (§5.2-12).
- Validate everything under `conda run -n gmr ...` per §6 before merging.
