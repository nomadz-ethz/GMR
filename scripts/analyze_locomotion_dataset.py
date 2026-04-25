"""
Analyze retargeted locomotion PKL files and produce a Markdown report.

Reads all PKL files from a mode-specific output directory, computes per-file
penetration and jitter statistics using MuJoCo FK, optionally maps files to
motion categories via a YAML index, and writes a Markdown report.

Output directory convention (matches retarget_no_penetration.py):
  output/AMASS/CMU/<mode>/...
  output/AMASS/HDM05/<mode>/...

Usage:
    # With YAML category mapping
    python scripts/analyze_locomotion_dataset.py \\
        --pkl_dir output --mode qp_smoothed \\
        --yaml ../k1_motion_data/index/amass_cmu_locomotion.yaml

    # Multiple YAMLs (stacked into one report)
    python scripts/analyze_locomotion_dataset.py \\
        --pkl_dir output --mode qp_smoothed \\
        --yaml ../k1_motion_data/index/amass_cmu_locomotion.yaml \\
               ../k1_motion_data/index/amass_hdm05_locomotion.yaml

    # No YAML (all files reported under 'unknown' category)
    python scripts/analyze_locomotion_dataset.py \\
        --pkl_dir output --mode qp_smoothed

    # Custom output file
    python scripts/analyze_locomotion_dataset.py \\
        --pkl_dir output --mode qp_smoothed \\
        --yaml ../k1_motion_data/index/amass_cmu_locomotion.yaml \\
        --output reports/cmu_qp_smoothed.md
"""

import argparse
import os
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Default target; overridable via --robot.
DEFAULT_ROBOT = "booster_k1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_mode_from_stem(stem: str, mode: str) -> str:
    """Remove the mode component inserted at position 2 by retarget_no_penetration.py.

    AMASS/CMU/qp_smoothed/02/02_01  →  AMASS/CMU/02/02_01
    AMASS/HDM05/qp/bk/HDM_bk_01     →  AMASS/HDM05/bk/HDM_bk_01
    """
    parts = pathlib.Path(stem).parts
    if len(parts) >= 3 and parts[2] == mode:
        remaining = parts[3:]
        if remaining:
            return str(pathlib.Path(*parts[:2]) / pathlib.Path(*remaining))
        return str(pathlib.Path(*parts[:2]))
    return stem


def _clean_stem(name: str) -> str:
    stem = pathlib.Path(name).stem
    for suffix in ("_stageii", "_stagei"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def load_yaml_category_map(yaml_files: list) -> dict:
    """Build {output_stem: category} from one or more YAML index files.

    Each YAML file's paths are resolved relative to the YAML's own directory,
    then made relative to the adjacent ``data/`` directory — matching the
    stems produced by retarget_no_penetration.py.

    Example stem:  AMASS/CMU/02/02_01  (relative to output/<mode>/)
    """
    import yaml

    result = {}
    for yaml_file in yaml_files:
        yaml_path = pathlib.Path(yaml_file).resolve()
        yaml_dir = yaml_path.parent

        # data_root is the sibling data/ directory of the yaml's parent (index/)
        data_root = (yaml_dir.parent / "data").resolve()
        if not data_root.is_dir():
            data_root = None

        with open(yaml_file, "r") as f:
            cfg = yaml.safe_load(f)

        for category, items in cfg.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or "path" not in item:
                    continue
                abs_path = (yaml_dir / item["path"]).resolve()
                stem = _clean_stem(abs_path.name)
                if data_root is not None:
                    try:
                        rel = abs_path.relative_to(data_root)
                        stem = str(rel.parent / stem)
                    except ValueError:
                        pass
                result[stem] = category

    return result


def compute_stats(pkl_path: str, model, sole_config, ground_height=0.0, clearance=0.003):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    import mujoco as mj

    root_pos = d["root_pos"]
    root_rot = d["root_rot"]
    dof_pos  = d["dof_pos"]
    fps      = float(d.get("fps", 30.0))
    N        = len(root_pos)

    data = mj.MjData(model)

    penetrations = []
    root_z_vals  = []

    for i in range(N):
        data.qpos[:3]  = root_pos[i]
        data.qpos[3:7] = root_rot[i][[3, 0, 1, 2]]   # xyzw → wxyz
        data.qpos[7:]  = dof_pos[i]
        mj.mj_forward(model, data)
        root_z_vals.append(data.qpos[2])

        min_z = np.inf
        for body_name, local_pts in sole_config.items():
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
            if bid < 0:
                continue
            bp = data.xpos[bid]
            br = data.xmat[bid].reshape(3, 3)
            for lp in local_pts:
                world_z = (br @ np.array(lp) + bp)[2]
                if world_z < min_z:
                    min_z = world_z
        penetrations.append(min_z)

    pen_arr = np.array(penetrations)
    rz_arr  = np.array(root_z_vals)

    pen_depth  = np.maximum(0.0, clearance - pen_arr)
    n_pen      = int(np.sum(pen_depth > 1e-4))
    max_pen_mm = float(pen_depth.max()) * 1000

    drz       = np.diff(rz_arr)
    up_jumps  = drz[drz > 0]
    max_up_mm  = float(up_jumps.max())  * 1000 if len(up_jumps) else 0.0
    mean_up_mm = float(up_jumps.mean()) * 1000 if len(up_jumps) else 0.0
    rz_range_mm = float(rz_arr.max() - rz_arr.min()) * 1000

    return {
        "n_frames":    N,
        "fps":         fps,
        "duration_s":  N / fps,
        "n_pen":       n_pen,
        "pen_pct":     100.0 * n_pen / N if N > 0 else 0.0,
        "max_pen_mm":  max_pen_mm,
        "max_up_mm":   max_up_mm,
        "mean_up_mm":  mean_up_mm,
        "rz_range_mm": rz_range_mm,
    }


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def fmt_duration(total_seconds: float) -> str:
    m = int(total_seconds) // 60
    s = total_seconds - m * 60
    return f"{m}m {s:.1f}s" if m > 0 else f"{s:.1f}s"


def _find_report_dir(pkl_dir: pathlib.Path, pkl_files: list, mode: str | None) -> pathlib.Path:
    """Pick where to write the report.

    PKL layout: pkl_dir/DATASET/SUBSET/<mode>/subject/file.pkl
    If all found PKLs share a single mode folder (e.g. output/AMASS/CMU/qp_smoothed),
    the report goes there.  If PKLs span multiple datasets (CMU + HDM05), the report
    goes in pkl_dir itself.
    """
    if not mode:
        return pkl_dir

    mode_dirs = set()
    for p in pkl_files:
        try:
            parts = p.relative_to(pkl_dir).parts
        except ValueError:
            continue
        # Expected: DATASET/SUBSET/<mode>/subject/file.pkl  → parts[2] == mode
        if len(parts) >= 3 and parts[2] == mode:
            mode_dirs.add(pkl_dir / parts[0] / parts[1] / parts[2])

    if len(mode_dirs) == 1:
        return mode_dirs.pop()
    return pkl_dir


def _stem_short(stem: str) -> str:
    """Return the last 2 path components for compact display in tables."""
    parts = pathlib.Path(stem).parts
    return "/".join(parts[-2:]) if len(parts) > 1 else stem


def build_markdown(
    results_by_category: dict,
    ground_mode: str,
    clearance_mm: float,
    pkl_dir: str,
) -> str:
    lines = []

    # ── Header ──
    lines += [
        "# Locomotion Retargeting Dataset Analysis",
        "",
        f"> **Robot**: Booster K1  ",
        f"> **Ground mode**: `{ground_mode}`  ",
        f"> **Clearance**: {clearance_mm:.1f} mm  ",
        f"> **Output directory**: `{pkl_dir}`",
        "",
    ]

    all_cats = [c for c in results_by_category if c != "unknown"] + (
        ["unknown"] if "unknown" in results_by_category else []
    )

    # ── Summary table ──
    lines += [
        "## Summary by Motion Category",
        "",
        "| Category | Files | Total frames | Duration | Pen. frames | Pen. rate | Max pen. | Max root jump |",
        "|----------|------:|-------------:|---------:|------------:|----------:|---------:|-------------:|",
    ]

    grand = {"files": 0, "frames": 0, "duration_s": 0.0,
             "n_pen": 0, "max_pen_mm": 0.0, "max_up_mm": 0.0}

    for cat in all_cats:
        rows = results_by_category[cat]
        nf   = len(rows)
        nfr  = sum(r["n_frames"]   for r in rows)
        dur  = sum(r["duration_s"] for r in rows)
        np_  = sum(r["n_pen"]      for r in rows)
        pr   = 100.0 * np_ / nfr if nfr else 0.0
        mp   = max(r["max_pen_mm"] for r in rows)
        mu   = max(r["max_up_mm"]  for r in rows)
        lines.append(
            f"| {cat} | {nf} | {nfr:,} | {fmt_duration(dur)} "
            f"| {np_:,} | {pr:.1f}% | {mp:.1f} mm | {mu:.1f} mm |"
        )
        grand["files"]      += nf
        grand["frames"]     += nfr
        grand["duration_s"] += dur
        grand["n_pen"]      += np_
        grand["max_pen_mm"]  = max(grand["max_pen_mm"], mp)
        grand["max_up_mm"]   = max(grand["max_up_mm"],  mu)

    gpr = 100.0 * grand["n_pen"] / grand["frames"] if grand["frames"] else 0.0
    lines.append(
        f"| **Total** | **{grand['files']}** | **{grand['frames']:,}** "
        f"| **{fmt_duration(grand['duration_s'])}** "
        f"| **{grand['n_pen']:,}** | **{gpr:.1f}%** "
        f"| **{grand['max_pen_mm']:.1f} mm** | **{grand['max_up_mm']:.1f} mm** |"
    )
    lines.append("")

    # ── Duration breakdown ──
    lines += [
        "## Motion Duration Breakdown",
        "",
        "| Category | Files | Total duration | Avg per clip |",
        "|----------|------:|---------------:|-------------:|",
    ]
    for cat in all_cats:
        rows = results_by_category[cat]
        dur  = sum(r["duration_s"] for r in rows)
        avg  = dur / len(rows) if rows else 0.0
        lines.append(f"| {cat} | {len(rows)} | {fmt_duration(dur)} | {fmt_duration(avg)} |")
    gf   = grand["files"]
    gdur = grand["duration_s"]
    lines.append(
        f"| **Total** | **{gf}** | **{fmt_duration(gdur)}** "
        f"| **{fmt_duration(gdur / gf) if gf else '—'}** |"
    )
    lines += [
        "",
        f"**Total motion data: {gdur/60:.2f} minutes ({gdur:.1f} s) across {gf} clips.**",
        "",
    ]

    # ── Per-category detailed tables ──
    lines += [
        "## Per-File Statistics",
        "",
        "Columns: **frames** | **fps** | **dur** | **pen/N** | "
        "**pen%** | **max_pen (mm)** | **max_up (mm)** | **mean_up** | **rz_range**",
        "",
    ]

    for cat in all_cats:
        rows = results_by_category[cat]
        lines += [
            f"### {cat}",
            "",
            "| File | frames | fps | dur | pen/N | pen% | max_pen | max_up | mean_up | rz_range |",
            "|------|-------:|----:|----:|------:|-----:|--------:|-------:|--------:|---------:|",
        ]
        for r in sorted(rows, key=lambda x: x["stem"]):
            lines.append(
                f"| {_stem_short(r['stem'])} "
                f"| {r['n_frames']} "
                f"| {r['fps']:.1f} "
                f"| {fmt_duration(r['duration_s'])} "
                f"| {r['n_pen']}/{r['n_frames']} "
                f"| {r['pen_pct']:.1f}% "
                f"| {r['max_pen_mm']:.1f} "
                f"| {r['max_up_mm']:.1f} "
                f"| {r['mean_up_mm']:.1f} "
                f"| {r['rz_range_mm']:.1f} |"
            )
        lines.append("")

    # ── Key observations ──
    lines += ["## Key Observations", ""]

    all_rows = [r for rows in results_by_category.values() for r in rows]
    if all_rows:
        zero_pen    = [r for r in all_rows if r["n_pen"] == 0]
        nonzero_pen = [r for r in all_rows if r["n_pen"] > 0]
        high_pen    = sorted(nonzero_pen, key=lambda r: r["max_pen_mm"], reverse=True)[:5]
        high_jitter = sorted(all_rows,    key=lambda r: r["max_up_mm"],  reverse=True)[:5]

        lines.append(f"- **{len(zero_pen)}/{len(all_rows)} clips** have zero penetrating frames.")
        if nonzero_pen:
            avg_pr = np.mean([r["pen_pct"] for r in nonzero_pen])
            lines.append(
                f"- Among clips with any penetration, average penetration rate: **{avg_pr:.1f}%**."
            )
        if high_pen:
            lines.append("- **Worst penetration clips** (max depth, mm):")
            for r in high_pen:
                lines.append(
                    f"  - `{_stem_short(r['stem'])}`: {r['max_pen_mm']:.1f} mm "
                    f"({r['n_pen']}/{r['n_frames']} frames)"
                )
        if high_jitter:
            lines.append("- **Highest root jitter clips** (max upward jump, mm):")
            for r in high_jitter[:3]:
                lines.append(
                    f"  - `{_stem_short(r['stem'])}`: {r['max_up_mm']:.1f} mm max, "
                    f"{r['mean_up_mm']:.1f} mm mean"
                )

        lines.append("")
        for cat in all_cats:
            rows = results_by_category[cat]
            nfr  = sum(r["n_frames"]   for r in rows)
            np_  = sum(r["n_pen"]      for r in rows)
            pr   = 100.0 * np_ / nfr if nfr else 0.0
            mp   = max(r["max_pen_mm"] for r in rows)
            mu   = max(r["max_up_mm"]  for r in rows)
            dur  = sum(r["duration_s"] for r in rows) / 60.0
            lines.append(
                f"- **{cat}** ({len(rows)} clips, {dur:.1f} min): "
                f"{pr:.1f}% frames penetrate, max depth {mp:.1f} mm, "
                f"max root jump {mu:.1f} mm."
            )

    lines += [
        "",
        "---",
        "",
        "*Generated by `scripts/analyze_locomotion_dataset.py`. "
        "Penetration measured at 8 sole corners (4 per foot) using MuJoCo FK. "
        f"Clearance threshold: {clearance_mm:.1f} mm.*",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze retargeted PKL files and write a Markdown report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pkl_dir", type=str, required=True,
        help=(
            "Output directory containing PKL files. "
            "Typically 'output/' (the base output dir). "
            "PKL stems are computed relative to this directory."
        ),
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        help=(
            "Processing mode (e.g. qp_smoothed, qp, vanilla, soft). "
            "When provided, the mode component is stripped from PKL stems "
            "before YAML category lookup, matching the layout: "
            "output/AMASS/CMU/<mode>/subject/file.pkl."
        ),
    )
    parser.add_argument(
        "--yaml", type=str, nargs="*", default=None,
        help=(
            "One or more YAML index files for category mapping. "
            "Examples: ../k1_motion_data/index/amass_cmu_locomotion.yaml "
            "../k1_motion_data/index/amass_hdm05_locomotion.yaml. "
            "If omitted, all clips are grouped under 'unknown'."
        ),
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=(
            "Output Markdown file. "
            "Default: REPORT_<mode>.md inside the mode folder if all PKLs share one "
            "(e.g. output/AMASS/CMU/qp_smoothed/REPORT_qp_smoothed.md), "
            "otherwise <pkl_dir>/REPORT_<mode>.md."
        ),
    )
    parser.add_argument("--ground_height", type=float, default=0.0)
    parser.add_argument(
        "--clearance", type=float, default=0.003,
        help="Clearance threshold in metres (default: 3 mm)",
    )
    parser.add_argument(
        "--robot", choices=["booster_k1", "booster_t1"], default=DEFAULT_ROBOT,
        help="Robot whose MJCF + sole config to use for FK/penetration stats.",
    )
    args = parser.parse_args()

    pkl_dir = pathlib.Path(os.path.abspath(args.pkl_dir))

    # Load MuJoCo model and sole config
    import mujoco as mj
    from general_motion_retargeting import ROBOT_XML_DICT
    from general_motion_retargeting.sole_points import get_sole_points

    model       = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    sole_config = get_sole_points(args.robot)

    # Build category map (stem → category) from YAML(s), if provided
    category_map = load_yaml_category_map(args.yaml) if args.yaml else {}

    # Collect PKL files — filter by mode so other modes under the same output/ root
    # are not included (e.g. qp_smoothed files are excluded when --mode qp).
    all_pkl_files = sorted(pkl_dir.rglob("*.pkl"))
    if args.mode:
        pkl_files = [
            p for p in all_pkl_files
            if len(p.relative_to(pkl_dir).parts) >= 3
            and p.relative_to(pkl_dir).parts[2] == args.mode
        ]
    else:
        pkl_files = all_pkl_files

    if not pkl_files:
        print(f"No PKL files found in {pkl_dir}" + (f" for mode '{args.mode}'" if args.mode else ""))
        sys.exit(1)

    # When --yaml is provided, restrict to files whose (mode-stripped) stem is in the
    # YAML.  This prevents CMU files leaking in when analysing HDM05 (or vice-versa)
    # and ensures _find_report_dir sees only a single dataset directory.
    if category_map:
        def _stem_for(p):
            s = str(p.relative_to(pkl_dir).with_suffix(""))
            return _strip_mode_from_stem(s, args.mode) if args.mode else s
        pkl_files = [p for p in pkl_files if _stem_for(p) in category_map]
        if not pkl_files:
            print(f"No PKL files matched the provided YAML entries (mode='{args.mode}').")
            sys.exit(1)

    # Determine report output path now that we know the PKL structure
    if args.output:
        output_path = args.output
    else:
        report_dir = _find_report_dir(pkl_dir, pkl_files, args.mode)
        report_name = f"REPORT_{args.mode}.md" if args.mode else "REPORT.md"
        output_path = str(report_dir / report_name)

    print(f"Found {len(pkl_files)} PKL files in {pkl_dir}. Computing stats...")
    print(f"Report will be written to: {output_path}")

    results_by_category: dict = {}
    errors = []

    try:
        from tqdm import tqdm
        iterator = tqdm(pkl_files, desc="Analyzing")
    except ImportError:
        iterator = pkl_files

    for pkl_path in iterator:
        # Stem relative to pkl_dir: e.g. AMASS/CMU/qp_smoothed/02/02_01
        # Strip mode to get AMASS/CMU/02/02_01 for YAML category lookup
        stem = str(pkl_path.relative_to(pkl_dir).with_suffix(""))
        if args.mode:
            stem = _strip_mode_from_stem(stem, args.mode)

        try:
            stats = compute_stats(
                str(pkl_path), model, sole_config, args.ground_height, args.clearance
            )
        except Exception as e:
            errors.append((stem, str(e)))
            continue

        stats["stem"] = stem
        category = category_map.get(stem, "unknown")
        results_by_category.setdefault(category, []).append(stats)

    if errors:
        print(f"\n{len(errors)} errors:")
        for s, e in errors:
            print(f"  {s}: {e}")

    # Infer ground_mode from first readable PKL
    ground_mode = "unknown"
    for pkl_path in pkl_files:
        try:
            with open(pkl_path, "rb") as f:
                d = pickle.load(f)
            ground_mode = d.get("ground_mode", "unknown")
            break
        except Exception:
            continue

    report_parent = os.path.dirname(os.path.abspath(output_path))
    if report_parent:
        os.makedirs(report_parent, exist_ok=True)
    md = build_markdown(
        results_by_category,
        ground_mode=ground_mode,
        clearance_mm=args.clearance * 1000,
        pkl_dir=str(pkl_dir),
    )

    with open(output_path, "w") as f:
        f.write(md)

    total_clips = sum(len(v) for v in results_by_category.values())
    total_dur   = sum(r["duration_s"] for rows in results_by_category.values() for r in rows)
    print(f"\nReport written to: {output_path}")
    print(f"  {total_clips} clips, {total_dur/60:.2f} minutes of motion data")


if __name__ == "__main__":
    main()
