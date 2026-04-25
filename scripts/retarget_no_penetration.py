"""Unified retargeting CLI with ground-penetration prevention.

This is intentionally a thin script. Heavy lifting lives in
`general_motion_retargeting.retargeting.*`:
    - builder.build_retargeter        - build GMR + ground constraint + sole offset
    - strict_zero_pen.measure_*       - Pass 1 surveyor (forward-kinematics)
    - strict_zero_pen.smooth_*        - dilation + Gaussian envelope
    - foot_contact.detect_smplx_*     - SMPL-X toe contact labels
    - fk_post.apply_fk_post           - height/origin shifts + local_body_pos
    - batch_runner.run_batch / Tee    - directory & YAML batch driver

Inputs:   AMASS SMPL-X .npz/.pkl, AMASS CMU _stageii.npz, GVHMR .pt,
          BVH LAFAN1 (experimental — see docs/bvh.md)
Outputs:  pkl with the schema documented in docs/pipeline.md sec 5.7
Robots:   booster_k1 (default), booster_t1
Modes:    --ground_mode {none, qp, soft}, --strict_zero_pen for two-pass IK

Usage examples:
    # Recommended: K1 + AMASS CMU + two-pass smoothing, headless
    python scripts/retarget_no_penetration.py \\
        --input /data/AMASS/CMU/35/35_14_stageii.npz \\
        --input_format amass_cmu \\
        --ground_mode qp --strict_zero_pen \\
        --headless --output retargeted/35_14.pkl

    # GVHMR with viewer
    python scripts/retarget_no_penetration.py \\
        --input GVHMR/outputs/demo/freekick/hmr4d_results.pt \\
        --input_format gvhmr --ground_mode qp --strict_zero_pen --rate_limit

    # AMASS CMU batch via YAML
    python scripts/retarget_no_penetration.py \\
        --yaml /path/to/amass_cmu_locomotion_list.yaml \\
        --input_format amass_cmu \\
        --ground_mode qp --strict_zero_pen \\
        --height_adjust --root_origin_offset --output retargeted_cmu/

    # Directory batch
    python scripts/retarget_no_penetration.py \\
        --input motion_data/ACCAD/ \\
        --input_format smplx --ground_mode qp --strict_zero_pen --output retargeted/

    # BVH LAFAN1 (experimental; arm/head joint offsets are best-effort,
    # see docs/bvh.md for caveats)
    python scripts/retarget_no_penetration.py \\
        --input dance.bvh --input_format bvh_lafan1 \\
        --robot booster_k1 --ground_mode qp --strict_zero_pen \\
        --output retargeted/dance.pkl
"""

import argparse
import os
import pathlib
import pickle
import sys

import numpy as np
from rich import print

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

SMPLX_FOLDER = pathlib.Path(__file__).parent.parent / "assets" / "body_models"


# ---------------------------------------------------------------------------
# Format-aware loader
# ---------------------------------------------------------------------------

def load_motion_frames(input_path: str, input_format: str, tgt_fps: int = 30):
    """Load human motion from SMPL-X, GVHMR, or BVH.

    Returns:
        smplx_data_frames: list of per-frame dicts {body_name: (pos, quat_wxyz)}
        aligned_fps: float
        actual_human_height: float
    """
    if input_format in ("smplx", "amass_cmu"):
        from general_motion_retargeting.utils.smpl import (
            load_smplx_file, get_smplx_data_offline_fast,
        )
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            input_path, str(SMPLX_FOLDER)
        )
        smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
        )
    elif input_format == "gvhmr":
        from general_motion_retargeting.utils.smpl import (
            load_gvhmr_pred_file, get_gvhmr_data_offline_fast,
        )
        smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
            input_path, str(SMPLX_FOLDER)
        )
        smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
        )
    elif input_format == "bvh_lafan1":
        # BVH header frame-time is parsed inside read_bvh but not plumbed
        # through Anim; LAFAN1 is natively 30 fps so this is exact for that
        # source. See docs/bvh.md for the assumption.
        from general_motion_retargeting.utils.lafan1 import load_bvh_file
        smplx_data_frames, actual_human_height = load_bvh_file(
            input_path, format="lafan1",
        )
        aligned_fps = float(tgt_fps)
    else:
        raise ValueError(f"Unknown input format: {input_format!r}")

    return smplx_data_frames, aligned_fps, actual_human_height


# Map --input_format to the IK-config bucket. SMPL-X-based formats (AMASS,
# AMASS CMU, GVHMR) all share the "smplx" bucket; BVH gets its own.
SRC_HUMAN_FOR_FORMAT = {
    "smplx": "smplx",
    "amass_cmu": "smplx",
    "gvhmr": "smplx",
    "bvh_lafan1": "bvh_lafan1",
}


# ---------------------------------------------------------------------------
# Viewer step helper (used by single-pass and Pass 2)
# ---------------------------------------------------------------------------

def _step_viewer(viewer, qpos_list, retarget, rate_limit: bool):
    if viewer is None:
        return
    scaled_human = getattr(retarget, "scaled_human_data", None)
    for qpos in qpos_list:
        viewer.step(
            root_pos=qpos[:3], root_rot=qpos[3:7], dof_pos=qpos[7:],
            human_motion_data=scaled_human,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False, rate_limit=rate_limit,
        )


def _make_viewer(robot_type, output_path, aligned_fps, args):
    if args.headless and not args.record_video:
        return None
    from general_motion_retargeting import RobotMotionViewer
    out_path_obj = pathlib.Path(output_path)
    video_dir = out_path_obj.parent / "videos"
    video_path = str(video_dir / f"{robot_type}_{out_path_obj.stem}.mp4")
    return RobotMotionViewer(
        robot_type=robot_type,
        motion_fps=aligned_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=video_path if args.record_video else None,
    )


# ---------------------------------------------------------------------------
# Two-pass IK driver (uses helpers from retargeting.strict_zero_pen)
# ---------------------------------------------------------------------------

def _run_two_pass(qpos_list, smplx_data_frames, retarget, sole_config,
                  sole_compensation, args, viewer, robot_type, build_kwargs):
    """Strict-zero-penetration: survey -> envelope -> Pass 2.

    `build_kwargs` is the dict the caller used to construct the Pass-1
    retargeter; we use it again to construct a fresh Pass-2 retargeter.

    Returns the (possibly new) qpos_list and the final retargeter (so its
    `scaled_human_data` is available for the FK post-step).
    """
    from general_motion_retargeting.retargeting import (
        measure_penetration_depths, smooth_penetration_envelope, build_retargeter,
    )

    depths = measure_penetration_depths(
        qpos_list, retarget.model, sole_config, args.clearance,
    )

    if depths.max() <= 0.0:
        print("[bold cyan]Pass 1 max penetration: 0.0mm. Pass 2 skipped![/bold cyan]")
        _step_viewer(viewer, qpos_list, retarget, args.rate_limit)
        return qpos_list, retarget

    print(
        f"[bold cyan]Pass 1 max penetration: {depths.max()*1000:.1f}mm. "
        "Extracting smooth envelope..."
        "[/bold cyan]"
    )
    final_depths = smooth_penetration_envelope(depths)

    # Fresh retargeter for Pass 2 (Pass 1 state is discarded; warm-start chain
    # would otherwise still reflect spike-driven corrections).
    retarget, _ = build_retargeter(robot_type, **build_kwargs)
    qpos_list2 = []
    for i, frame_data in enumerate(smplx_data_frames):
        # Subtract final_depths to push the ENTIRE target skeleton up by the
        # smooth envelope. QP IK then solves cleanly with no spike-driven
        # correction, no foot-slide, and zero penetration.
        retarget.set_ground_offset(sole_compensation - final_depths[i])
        qpos = retarget.retarget(frame_data)
        qpos_list2.append(qpos.copy())

        # Step the viewer here (not after) so the user sees the smoothed
        # Pass 2 result, not Pass 1's foot-slide artifact.
        if viewer is not None:
            scaled_human = getattr(retarget, "scaled_human_data", None)
            viewer.step(
                root_pos=qpos[:3], root_rot=qpos[3:7], dof_pos=qpos[7:],
                human_motion_data=scaled_human,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False, rate_limit=args.rate_limit,
            )

    return qpos_list2, retarget


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------

def retarget_and_save(input_path: str, output_path: str, input_format: str, args) -> bool:
    """Load -> IK (single or two-pass) -> FK post -> contact flags -> save pkl.

    Returns True on success, False on failure. See docs/pipeline.md for the
    output schema.
    """
    from general_motion_retargeting.sole_points import get_sole_points
    from general_motion_retargeting.retargeting import (
        build_retargeter, detect_smplx_foot_contact, apply_fk_post,
    )

    robot_type = args.robot

    # 1. Load source frames.
    try:
        smplx_data_frames, aligned_fps, actual_human_height = load_motion_frames(
            input_path, input_format,
        )
    except Exception as e:
        print(f"[red]Error loading {input_path}: {e}[/red]")
        return False

    # 2. Build the retargeter.
    sole_config = get_sole_points(robot_type)
    build_kwargs = dict(
        ground_mode=args.ground_mode,
        sole_config=sole_config,
        actual_human_height=actual_human_height,
        src_human=SRC_HUMAN_FOR_FORMAT[input_format],
        gain=args.gain,
        ground_height=args.ground_height,
        clearance=args.clearance,
        activation_distance=args.activation_distance,
        max_weight=args.max_weight,
    )
    retarget, sole_compensation = build_retargeter(robot_type, **build_kwargs)
    if retarget is None:
        return False

    if args.ground_mode != "none":
        min_sole_z = min(pt[2] for pts in sole_config.values() for pt in pts)
        print(
            f"[bold]Sole compensation: {-sole_compensation*1000:.1f}mm "
            f"(ankle-to-sole={-min_sole_z*1000:.1f}mm + "
            f"clearance={args.clearance*1000:.1f}mm)[/bold]"
        )

    # 3. Optional viewer (also created when --record_video is on).
    viewer = _make_viewer(robot_type, output_path, aligned_fps, args)

    # 4. Pass 1 IK.
    qpos_list = [retarget.retarget(frame_data).copy() for frame_data in smplx_data_frames]

    # 5. Optional Pass 2 (strict zero penetration).
    if getattr(args, "strict_zero_pen", False):
        qpos_list, retarget = _run_two_pass(
            qpos_list, smplx_data_frames, retarget, sole_config,
            sole_compensation, args, viewer, robot_type, build_kwargs,
        )
    else:
        _step_viewer(viewer, qpos_list, retarget, args.rate_limit)

    if viewer is not None:
        viewer.close()

    # 6. Assemble arrays (root_rot stored as xyzw).
    root_pos = np.array([q[:3] for q in qpos_list])
    root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz -> xyzw
    dof_pos = np.array([q[7:] for q in qpos_list])

    # 7. FK post-processing (height_adjust, root_origin_offset, local_body_pos).
    height_adjust = getattr(args, "height_adjust", False)
    root_origin_offset = getattr(args, "root_origin_offset", False)
    needs_fk = (
        height_adjust or root_origin_offset or input_format == "amass_cmu"
        or getattr(args, "strict_zero_pen", False)
    )
    if needs_fk:
        root_pos, local_body_pos, link_body_list = apply_fk_post(
            root_pos, root_rot, dof_pos,
            robot_type=robot_type,
            height_adjust=height_adjust,
            root_origin_offset=root_origin_offset,
            ground_mode_active=(args.ground_mode != "none"),
        )
    else:
        local_body_pos = None
        link_body_list = None

    # 8. Foot-ground contact flags from raw SMPL-X kinematics.
    foot_ground_contact_flags = None
    foot_contact_meta = None
    if not getattr(args, "no_foot_contact", False):
        flags, info = detect_smplx_foot_contact(
            smplx_data_frames, aligned_fps,
            z_thresh=args.foot_contact_z_thresh,
            v_thresh=args.foot_contact_vel_thresh,
        )
        if flags is None:
            reason = info.get("reason", "unknown")
            print(f"[yellow][Foot contact] skipped: {reason}.[/yellow]")
        else:
            foot_ground_contact_flags = flags
            foot_contact_meta = {
                "source": "smplx_toe_kinematics",
                "joints": ["left_foot", "right_foot"],
                "columns": ["left", "right"],
                "z_thresh": float(args.foot_contact_z_thresh),
                "vel_thresh": float(args.foot_contact_vel_thresh),
                "floor_z": float(info["floor_z"]),
            }
            print(
                f"[Foot contact] L={info['l_count']}/{info['n_frames']}, "
                f"R={info['r_count']}/{info['n_frames']} "
                f"(floor_z={info['floor_z']:.3f} m, z<{info['z_thresh']:.3f} m, "
                f"v<{info['v_thresh']:.3f} m/s)"
            )

    # 9. Save the pkl. Schema: see docs/pipeline.md sec 5.7.
    motion_data = {
        "fps": float(aligned_fps),
        "robot": robot_type,
        "input_format": input_format,
        "source_file": os.path.abspath(input_path),
        "ground_mode": args.ground_mode,
        "strict_zero_pen": bool(getattr(args, "strict_zero_pen", False)),
        "ground_clearance": args.clearance,
        "actual_human_height": float(actual_human_height),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos,
        "link_body_list": link_body_list,
        "foot_ground_contact_flags": foot_ground_contact_flags,
        "foot_contact_meta": foot_contact_meta,
    }

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(motion_data, f)

    print(f"[green]Saved: {output_path} ({len(qpos_list)} frames @ {aligned_fps:.1f} fps)[/green]")
    return True


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Retarget human motion to a humanoid robot with ground penetration prevention."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", type=str,
        help="Path to a single motion file or a directory for batch processing.",
    )
    input_group.add_argument(
        "--yaml", type=str,
        help="Path to a YAML locomotion list (AMASS CMU only).",
    )

    parser.add_argument(
        "--input_format", choices=["smplx", "gvhmr", "amass_cmu", "bvh_lafan1"],
        required=True,
        help="'smplx' = AMASS/OMOMO .npz/.pkl, 'gvhmr' = GVHMR .pt, "
             "'amass_cmu' = AMASS CMU _stageii.npz with stem cleaning, "
             "'bvh_lafan1' = LAFAN1 BVH (EXPERIMENTAL, see docs/bvh.md).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path. Single file: a .pkl path. Batch/YAML: an output directory.",
    )

    parser.add_argument(
        "--robot", choices=["booster_k1", "booster_t1"], default="booster_k1",
        help="Target robot model.",
    )
    parser.add_argument(
        "--ground_mode", choices=["none", "qp", "soft"], default="none",
        help="'none'=vanilla GMR, 'qp'=hard QP inequality, 'soft'=soft repulsive tasks.",
    )
    parser.add_argument("--ground_height", type=float, default=0.0)
    parser.add_argument("--clearance", type=float, default=0.003,
                        help="Minimum clearance above ground in metres (default 3mm).")
    parser.add_argument("--gain", type=float, default=0.5,
                        help="QP CBF base gain. Boosted dynamically when penetrating.")
    parser.add_argument("--max_weight", type=float, default=500.0,
                        help="Soft mode: maximum task weight.")
    parser.add_argument("--activation_distance", type=float, default=0.02,
                        help="Distance above ground at which constraint activates.")

    # FK post-processing
    parser.add_argument("--height_adjust", action="store_true",
                        help="Shift root z so the lowest body sits at z=0. "
                             "Skipped if --ground_mode != none.")
    parser.add_argument("--root_origin_offset", action="store_true",
                        help="Translate root XY so the first frame is at the origin.")
    parser.add_argument("--strict_zero_pen", action="store_true",
                        help="Two-pass IK: survey penetration, build a smooth target-shift "
                             "envelope, re-run IK so soles stay flush with no foot-slide. "
                             "~2x runtime. See docs/pipeline.md sec 2.5.")

    # Foot-ground contact detection from SMPLX toe kinematics
    parser.add_argument("--foot_contact_z_thresh", type=float, default=0.08,
                        help="Max SMPLX toe height above the per-motion floor (m).")
    parser.add_argument("--foot_contact_vel_thresh", type=float, default=0.5,
                        help="Max SMPLX toe 3D speed (m/s).")
    parser.add_argument("--no_foot_contact", action="store_true",
                        help="Skip writing foot_ground_contact_flags to the output pkl.")

    # Processing. --no_viz / --override are legacy aliases.
    parser.add_argument("--headless", "--no_viz", dest="headless", action="store_true",
                        help="Disable MuJoCo viewer (auto-set in batch mode).")
    parser.add_argument("--record_video", action="store_true",
                        help="Record an MP4 even in headless mode (offscreen). "
                             "Output: <output_dir>/videos/<robot>_<stem>.mp4.")
    parser.add_argument("--rate_limit", action="store_true",
                        help="Rate-limit visualisation to motion FPS.")
    parser.add_argument("--overwrite", "--override", dest="overwrite", action="store_true",
                        help="Overwrite existing output files in batch mode.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _stem_for_yaml_entry(rel_path: str, robot: str) -> str:
    """Build the output stem for a YAML entry. Mirrors the AMASS sub-tree
    under <robot>_AMASS/... so K1 and T1 outputs don't collide."""
    from general_motion_retargeting.retargeting import amass_stem
    parts = pathlib.Path(rel_path).parts
    if "AMASS" in parts:
        idx = parts.index("AMASS")
        tail = pathlib.Path(*parts[idx + 1:])
    else:
        tail = pathlib.Path(pathlib.Path(rel_path).name)
    stem = str(tail.parent / amass_stem(str(tail)))
    stem_parts = pathlib.Path(stem).parts
    if len(stem_parts) > 1:
        stem = str(pathlib.Path(f"{robot}_{stem_parts[0]}", *stem_parts[1:]))
    else:
        stem = f"{robot}_{stem}"
    return stem


def main():
    args = parse_args()

    from general_motion_retargeting.retargeting import (
        run_batch, discover_input_files, load_yaml_paths, amass_stem,
    )

    if args.input_format == "bvh_lafan1":
        print(
            "[yellow]bvh_lafan1 is EXPERIMENTAL. Arm/head joint offsets in "
            "bvh_lafan1_to_{k1,t1}.json are best-effort guesses derived from "
            "the SMPL-X configs and have not been visually validated. "
            "See docs/bvh.md.[/yellow]"
        )

    # YAML batch (AMASS CMU locomotion list)
    if args.yaml:
        if args.input_format != "amass_cmu":
            print("[red]--yaml requires --input_format amass_cmu[/red]")
            sys.exit(1)
        yaml_path = os.path.abspath(args.yaml)
        yaml_dir = pathlib.Path(yaml_path).parent
        entries = load_yaml_paths(yaml_path)

        files = []
        for rel_path, _ in entries:
            full = str((yaml_dir / rel_path).resolve())
            stem = _stem_for_yaml_entry(rel_path, args.robot)
            files.append((full, stem))

        output_dir = args.output if args.output else str(yaml_dir / "retargeted")
        run_batch(files, output_dir, args.input_format, args, retarget_and_save)
        return

    # File / directory mode
    input_path = os.path.abspath(args.input)

    if os.path.isfile(input_path):
        if args.output:
            output_path = args.output
        elif args.input_format == "amass_cmu":
            output_path = amass_stem(input_path) + ".pkl"
        else:
            stem = input_path.rsplit(".", 1)[0]
            output_path = stem + "_retargeted.pkl"
        retarget_and_save(input_path, output_path, args.input_format, args)
    elif os.path.isdir(input_path):
        output_dir = args.output or (input_path.rstrip("/").rstrip("\\") + "_retargeted")
        files = discover_input_files(input_path, args.input_format)
        if not files:
            print(f"[red]No {args.input_format} files found in {input_path}[/red]")
            return
        run_batch(files, output_dir, args.input_format, args, retarget_and_save)
    else:
        print(f"[red]'{input_path}' is neither a file nor a directory[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
