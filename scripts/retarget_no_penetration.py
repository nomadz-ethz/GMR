"""
Unified retargeting script with ground penetration prevention.

Supports:
  - SMPL-X (AMASS .npz/.pkl) and GVHMR (.pt) input formats
  - AMASS CMU dataset (_stageii.npz files) with YAML-based batch discovery
  - Single file, batch directory, or YAML batch mode
  - Three ground modes: none (vanilla), qp (hard constraint), soft (repulsive tasks)

Target robot is selectable via --robot (default: booster_k1).

Usage examples:
    # Single GVHMR file, QP constraint
    python scripts/retarget_no_penetration.py \\
        --input GVHMR/outputs/demo/freekick/hmr4d_results.pt \\
        --input_format gvhmr --ground_mode qp --output retargeted/freekick.pkl

    # Batch SMPL-X directory
    python scripts/retarget_no_penetration.py \\
        --input motion_data/ACCAD/ \\
        --input_format smplx --ground_mode qp --output retargeted_qp/

    # AMASS CMU: single file with height/origin adjustment
    python scripts/retarget_no_penetration.py \\
        --input /data/AMASS/CMU/02/02_01_stageii.npz \\
        --input_format amass_cmu --ground_mode qp \\
        --height_adjust --root_origin_offset --output retargeted/02_01.pkl

    # AMASS CMU: batch via YAML locomotion list
    python scripts/retarget_no_penetration.py \\
        --yaml /path/to/amass_cmu_locomotion_list.yaml \\
        --input_format amass_cmu --ground_mode qp \\
        --height_adjust --root_origin_offset --output retargeted_cmu/

    # Single file with viewer
    python scripts/retarget_no_penetration.py \\
        --input motion.npz --input_format smplx --ground_mode qp --rate_limit
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
# AMASS CMU helpers
# ---------------------------------------------------------------------------

def _amass_stem(npz_path: str) -> str:
    """Strip _stageii/_stagei suffix: '02_01_stageii.npz' -> '02_01'."""
    stem = pathlib.Path(npz_path).stem
    return stem.replace("_stageii", "").replace("_stagei", "")


def load_yaml_paths(yaml_file: str) -> list:
    """Parse a locomotion YAML and return [(rel_path, description), ...]."""
    import yaml
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)
    entries = []
    for _, items in cfg.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and "path" in item:
                entries.append((item["path"], item.get("description", "")))
    return entries


# ---------------------------------------------------------------------------
# Format-aware loader
# ---------------------------------------------------------------------------

def load_motion_frames(input_path: str, input_format: str, tgt_fps: int = 30):
    """Load human motion data from either SMPL-X or GVHMR format.

    Returns:
        smplx_data_frames: list of per-frame dicts {body_name: (pos, quat_wxyz)}
        aligned_fps: float
        actual_human_height: float
    """
    from general_motion_retargeting.utils.smpl import (
        load_smplx_file, get_smplx_data_offline_fast,
        load_gvhmr_pred_file, get_gvhmr_data_offline_fast,
    )

    if input_format in ("smplx", "amass_cmu"):
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            input_path, str(SMPLX_FOLDER)
        )
        smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
        )
    elif input_format == "gvhmr":
        smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
            input_path, str(SMPLX_FOLDER)
        )
        smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
        )
    else:
        raise ValueError(f"Unknown input format: {input_format!r}")

    return smplx_data_frames, aligned_fps, actual_human_height


# ---------------------------------------------------------------------------
# File discovery for batch mode
# ---------------------------------------------------------------------------

def discover_input_files(input_dir: str, input_format: str):
    """Walk directory and find motion files.

    Returns list of (input_path, relative_stem).
    The relative_stem is used to construct the output path.
    """
    results = []
    input_dir = os.path.abspath(input_dir)
    base = pathlib.Path(input_dir)

    if input_format == "gvhmr":
        # Recursively find hmr4d_results.pt files
        for pt_path in sorted(base.rglob("hmr4d_results.pt")):
            stem = pt_path.parent.name
            if stem == base.name:
                stem = pt_path.stem
            results.append((str(pt_path), stem))

        # Fallback: any .pt file in the directory
        if not results:
            for pt_path in sorted(base.glob("*.pt")):
                results.append((str(pt_path), pt_path.stem))

    elif input_format in ("smplx", "amass_cmu"):
        for dirpath, _, filenames in os.walk(input_dir):
            for filename in sorted(filenames):
                if filename.endswith("_stagei.npz"):
                    continue
                if filename.endswith((".npz", ".pkl")):
                    full_path = os.path.join(dirpath, filename)
                    rel = os.path.relpath(full_path, input_dir)
                    if input_format == "amass_cmu":
                        # subject/file structure: preserve subject dir, clean filename
                        rel_path = pathlib.Path(rel)
                        stem = str(rel_path.parent / _amass_stem(rel_path.name))
                    else:
                        stem = os.path.splitext(rel)[0]
                    results.append((full_path, stem))

    return results


# ---------------------------------------------------------------------------
# Core retargeting function
# ---------------------------------------------------------------------------

def retarget_and_save(input_path: str, output_path: str, input_format: str, args):
    """Retarget a single motion file with optional ground constraint, save to pkl.

    Returns True on success, False on failure.
    """
    ROBOT_TYPE = args.robot
    # 1. Load frames
    try:
        smplx_data_frames, aligned_fps, actual_human_height = load_motion_frames(
            input_path, input_format
        )
    except Exception as e:
        print(f"[red]Error loading {input_path}: {e}[/red]")
        return False

    # 2. Helper to create retargeter with ground constraint
    from general_motion_retargeting.sole_points import get_sole_points, get_sole_site_names

    sole_config = get_sole_points(ROBOT_TYPE)

    def create_retargeter():
        if args.ground_mode == "soft":
            from general_motion_retargeting.ground_constraint import GMRWithSoftGround
            import general_motion_retargeting.params as gmr_params

            original_xml = gmr_params.ROBOT_XML_DICT[ROBOT_TYPE]
            sites_xml = str(original_xml).replace(".xml", "_with_sites.xml")
            if not os.path.exists(sites_xml):
                print(
                    f"[red]Soft mode requires {sites_xml}. "
                    f"Run: python scripts/add_sole_sites.py[/red]"
                )
                return None, None

            gmr_params.ROBOT_XML_DICT[ROBOT_TYPE] = pathlib.Path(sites_xml)
            try:
                retgt = GMRWithSoftGround(
                    site_names=get_sole_site_names(ROBOT_TYPE),
                    ground_height=args.ground_height,
                    clearance=args.clearance,
                    max_weight=args.max_weight,
                    activation_distance=args.activation_distance,
                    actual_human_height=actual_human_height,
                    src_human="smplx",
                    tgt_robot=ROBOT_TYPE,
                )
            finally:
                gmr_params.ROBOT_XML_DICT[ROBOT_TYPE] = original_xml

        else:
            from general_motion_retargeting import GeneralMotionRetargeting as GMR
            retgt = GMR(
                actual_human_height=actual_human_height,
                src_human="smplx",
                tgt_robot=ROBOT_TYPE,
            )

            if args.ground_mode == "qp":
                from general_motion_retargeting.ground_constraint import GroundPlaneLimit

                ground_limit = GroundPlaneLimit(
                    model=retgt.model,
                    sole_contact_points=sole_config,
                    ground_height=args.ground_height,
                    clearance=args.clearance,
                    gain=args.gain,
                    activation_distance=args.activation_distance,
                )
                retgt.ik_limits.append(ground_limit)

                max_root_dz = getattr(args, "max_root_dz", None)
                if max_root_dz is not None:
                    from general_motion_retargeting.ground_constraint import RootZLimit
                    retgt.ik_limits2.append(RootZLimit(
                        model=retgt.model,
                        sole_contact_points=sole_config,
                        ground_height=args.ground_height,
                        clearance=args.clearance,
                        activation_distance=args.activation_distance,
                        max_dz=max_root_dz,
                    ))
                    
        # Calculate baseline sole compensation
        baseline_compensation = 0.0
        if args.ground_mode != "none":
            min_sole_z = min(pt[2] for pts in sole_config.values() for pt in pts)
            baseline_compensation = min_sole_z - args.clearance  # negative → raises targets
            retgt.set_ground_offset(baseline_compensation)
            
        return retgt, baseline_compensation
        
    retarget, sole_compensation = create_retargeter()
    if retarget is None:
        return False
        
    if args.ground_mode != "none":
        min_sole_z = min(pt[2] for pts in sole_config.values() for pt in pts)
        print(f"[bold]Sole compensation: {-sole_compensation*1000:.1f}mm "
              f"(ankle-to-sole={-min_sole_z*1000:.1f}mm + clearance={args.clearance*1000:.1f}mm)[/bold]")

    # 3. Optional viewer (created if visualizing OR recording video)
    viewer = None
    if (not args.no_viz) or args.record_video:
        from general_motion_retargeting import RobotMotionViewer

        # Place video alongside the output pkl, mirroring its stem.
        out_path_obj = pathlib.Path(output_path)
        video_dir = out_path_obj.parent / "videos"
        video_path = str(video_dir / f"{ROBOT_TYPE}_{out_path_obj.stem}.mp4")

        viewer = RobotMotionViewer(
            robot_type=ROBOT_TYPE,
            motion_fps=aligned_fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=video_path if args.record_video else None,
        )

    # 4. First Pass: Initial IK solve
    qpos_list = []
    for frame_data in smplx_data_frames:
        qpos = retarget.retarget(frame_data)
        qpos_list.append(qpos.copy())
        
    # -- Strict Zero Penetration: TWO-PASS IK --
    if getattr(args, "strict_zero_pen", False):
        import mujoco as mj
        import scipy.ndimage
        
        # Surveyor Pass: Evaluate exact penetrations
        depths = np.zeros(len(qpos_list))
        data_mj = mj.MjData(retarget.model)
        for i, qpos in enumerate(qpos_list):
            data_mj.qpos[:3] = qpos[:3]
            data_mj.qpos[3:7] = qpos[3:7][[1, 2, 3, 0]]  # wxyz -> xyzw
            data_mj.qpos[7:] = qpos[7:]
            mj.mj_forward(retarget.model, data_mj)

            min_z = np.inf
            for body_name, local_pts in sole_config.items():
                bid = mj.mj_name2id(retarget.model, mj.mjtObj.mjOBJ_BODY, body_name)
                if bid < 0: continue
                bp = data_mj.xpos[bid]
                br = data_mj.xmat[bid].reshape(3, 3)
                for lp in local_pts:
                    world_z = (br @ np.array(lp) + bp)[2]
                    if world_z < min_z:
                        min_z = world_z
            
            if min_z < args.clearance:
                depths[i] = args.clearance - min_z
        
        if np.max(depths) > 0.0:
            print(f"[bold cyan]Pass 1 max penetration: {np.max(depths)*1000:.1f}mm. Extracting smooth envelope directly...[/bold cyan]")
            # Dilate peaks to capture wider foot-strike curve, then heavily smooth.
            dilated = scipy.ndimage.maximum_filter1d(depths, size=15)
            smoothed = scipy.ndimage.gaussian_filter1d(dilated, sigma=5)
            # Ensure it strictly covers the spikes, then smooth the kinks.
            final_depths = np.maximum(smoothed, depths)
            final_depths = scipy.ndimage.gaussian_filter1d(final_depths, sigma=2)
            
            # Second Pass: Reset IK solver and feed the final_depths as continuous target offsets!
            retarget, _ = create_retargeter()
            qpos_list = []
            for i, frame_data in enumerate(smplx_data_frames):
                # We subtract final_depths to push the ENTIRE target skeleton up precisely, 
                # mathematically guaranteeing flawless motion smoothness for RL!
                retarget.set_ground_offset(sole_compensation - final_depths[i])
                qpos = retarget.retarget(frame_data)
                qpos_list.append(qpos.copy())
                
                # Step the viewer here instead on the final pass!
                if viewer is not None:
                    scaled_human = retarget.scaled_human_data if hasattr(retarget, "scaled_human_data") else None
                    viewer.step(
                        root_pos=qpos[:3], root_rot=qpos[3:7], dof_pos=qpos[7:],
                        human_motion_data=scaled_human, human_pos_offset=np.array([0.0, 0.0, 0.0]),
                        show_human_body_name=False, rate_limit=args.rate_limit,
                    )
        else:
            print("[bold cyan]Pass 1 max penetration: 0.0mm. Pass 2 skipped![/bold cyan]")
            # Already done, step viewer if it exists.
            if viewer is not None:
                for qpos in qpos_list:
                    viewer.step(
                        root_pos=qpos[:3], root_rot=qpos[3:7], dof_pos=qpos[7:],
                        human_pos_offset=np.array([0.0, 0.0, 0.0]),
                        show_human_body_name=False, rate_limit=args.rate_limit,
                    )
    else:
        # Normal visualization logic for single-pass
        if viewer is not None:
            for qpos in qpos_list:
                scaled_human = retarget.scaled_human_data if hasattr(retarget, "scaled_human_data") else None
                viewer.step(
                    root_pos=qpos[:3], root_rot=qpos[3:7], dof_pos=qpos[7:],
                    human_motion_data=scaled_human, human_pos_offset=np.array([0.0, 0.0, 0.0]),
                    show_human_body_name=False, rate_limit=args.rate_limit,
                )

    if viewer is not None:
        viewer.close()

    # 5. Assemble arrays (root_rot stored as xyzw)
    root_pos = np.array([q[:3] for q in qpos_list])
    root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz -> xyzw
    dof_pos = np.array([q[7:] for q in qpos_list])

    # 6. Optional FK post-processing (height_adjust, root_origin_offset, local_body_pos)
    local_body_pos = None
    link_body_list = None

    height_adjust = getattr(args, "height_adjust", False)
    root_origin_offset = getattr(args, "root_origin_offset", False)

    if height_adjust or root_origin_offset or input_format == "amass_cmu" or getattr(args, "strict_zero_pen", False):
        import torch
        from general_motion_retargeting.kinematics_model import KinematicsModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        xml_file = str(__import__("general_motion_retargeting").ROBOT_XML_DICT[ROBOT_TYPE])
        km = KinematicsModel(xml_file, device=device)

        rp = torch.from_numpy(root_pos).to(device=device, dtype=torch.float)
        rr = torch.from_numpy(root_rot).to(device=device, dtype=torch.float)
        dp = torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)

        if height_adjust or root_origin_offset:
            body_pos, _ = km.forward_kinematics(rp, rr, dp)
            if height_adjust:
                if args.ground_mode != "none":
                    print(
                        "[yellow]--height_adjust skipped: incompatible with "
                        f"--ground_mode {args.ground_mode}. The ground constraint "
                        "already positions the robot; height_adjust would lower it "
                        "back underground.[/yellow]"
                    )
                else:
                    lowest = torch.min(body_pos[..., 2]).item()
                    root_pos[:, 2] -= lowest
                    rp = torch.from_numpy(root_pos).to(device=device, dtype=torch.float)
            if root_origin_offset:
                root_pos[:, :2] -= root_pos[0, :2]
                rp = torch.from_numpy(root_pos).to(device=device, dtype=torch.float)

        # Legacy smoothing and aggressive bound clipping removed: Two-Pass IK handles this natively.

        # Local body positions (root fixed at origin — shape/pose only)
        fk_root_pos = torch.zeros((root_pos.shape[0], 3), device=device)
        fk_root_rot = torch.zeros((root_pos.shape[0], 4), device=device)
        fk_root_rot[:, -1] = 1.0  # xyzw identity
        local_body_pos, _ = km.forward_kinematics(fk_root_pos, fk_root_rot, dp)
        local_body_pos = local_body_pos.detach().cpu().numpy()
        link_body_list = km.body_names
        torch.cuda.empty_cache()

    # Per-foot ground-contact detection from SMPLX toe kinematics.
    # Runs on the unified smplx_data_frames (unaffected by ground_mode / strict_zero_pen),
    # so labels stay stable across retargeting variants.
    foot_ground_contact_flags = None
    if not getattr(args, "no_foot_contact", False) and len(smplx_data_frames) >= 2:
        first_frame = smplx_data_frames[0]
        if "left_foot" in first_frame and "right_foot" in first_frame:
            l_toe = np.array([f["left_foot"][0]  for f in smplx_data_frames], dtype=np.float32)
            r_toe = np.array([f["right_foot"][0] for f in smplx_data_frames], dtype=np.float32)
            dt = 1.0 / float(aligned_fps)
            l_vel = np.linalg.norm(np.gradient(l_toe, dt, axis=0), axis=1)
            r_vel = np.linalg.norm(np.gradient(r_toe, dt, axis=0), axis=1)
            # Per-motion floor reference: raw SMPLX z is not guaranteed to hit 0 at the
            # ground (AMASS sequences and GVHMR's post-rotation both shift it), so compare
            # toe height against the lowest observed toe z in the clip.
            floor_z = float(min(l_toe[:, 2].min(), r_toe[:, 2].min()))
            l_stance = ((l_toe[:, 2] - floor_z) < args.foot_contact_z_thresh) & (l_vel < args.foot_contact_vel_thresh)
            r_stance = ((r_toe[:, 2] - floor_z) < args.foot_contact_z_thresh) & (r_vel < args.foot_contact_vel_thresh)
            foot_ground_contact_flags = np.stack([l_stance, r_stance], axis=1).astype(bool)
            n = len(smplx_data_frames)
            print(
                f"[Foot contact] L={int(l_stance.sum())}/{n}, R={int(r_stance.sum())}/{n} "
                f"(floor_z={floor_z:.3f} m, z<{args.foot_contact_z_thresh:.3f} m, "
                f"v<{args.foot_contact_vel_thresh:.3f} m/s)"
            )
        else:
            print("[yellow][Foot contact] left_foot/right_foot missing from SMPLX frames; skipping.[/yellow]")

    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos,
        "link_body_list": link_body_list,
        "ground_mode": args.ground_mode,
        "ground_clearance": args.clearance,
        "foot_ground_contact_flags": foot_ground_contact_flags,
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

    # Input: one of --input (file/dir) or --yaml (AMASS CMU locomotion list)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", type=str,
        help=(
            "Path to a single motion file (.npz/.pkl for SMPL-X/AMASS, .pt for GVHMR) "
            "OR a directory for batch processing."
        ),
    )
    input_group.add_argument(
        "--yaml", type=str,
        help=(
            "Path to a YAML locomotion list (e.g. amass_cmu_locomotion_list.yaml). "
            "Paths in the YAML are resolved relative to the YAML file's directory. "
            "Only valid with --input_format amass_cmu."
        ),
    )

    parser.add_argument(
        "--input_format", choices=["smplx", "gvhmr", "amass_cmu"], required=True,
        help=(
            "'smplx' for AMASS/OMOMO .npz/.pkl, "
            "'gvhmr' for GVHMR .pt, "
            "'amass_cmu' for AMASS CMU _stageii.npz files"
        ),
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=(
            "Output path. Single file: path to .pkl. "
            "Batch/YAML: output directory. Defaults to input_path + '_retargeted'."
        ),
    )

    # Ground constraint
    parser.add_argument(
        "--robot", choices=["booster_k1", "booster_t1"], default="booster_k1",
        help="Target robot model.",
    )
    parser.add_argument(
        "--ground_mode", choices=["none", "qp", "soft"], default="none",
        help="'none'=vanilla GMR, 'qp'=hard QP inequality, 'soft'=soft repulsive tasks",
    )
    parser.add_argument("--ground_height", type=float, default=0.0)
    parser.add_argument("--clearance", type=float, default=0.003,
                        help="Minimum clearance above ground in meters (default: 3mm)")
    parser.add_argument("--gain", type=float, default=0.5,
                        help="QP mode: CBF gain (0<gain<=1). 0.5 = half-correction per IK step.")
    parser.add_argument("--max_root_dz", type=float, default=None,
                        help="QP mode: cap upward root-z displacement per IK step (metres). "
                             "Reduces heel-strike jitter in locomotion. Typical: 0.003. "
                             "Default: None (disabled).")
    parser.add_argument("--max_weight", type=float, default=500.0,
                        help="Soft mode: maximum task weight")
    parser.add_argument("--activation_distance", type=float, default=0.02,
                        help="Distance above ground at which constraint activates")

    # Post-processing (FK-based; relevant for amass_cmu)
    parser.add_argument("--height_adjust", action="store_true",
                        help="Shift root z so the lowest body point sits at z=0.")
    parser.add_argument("--root_origin_offset", action="store_true",
                        help="Translate root XY so the first frame is at the origin.")
    parser.add_argument("--strict_zero_pen", action="store_true",
                        help="Post-processing: smooth root Z and rigidly eliminate penetration.")

    # Per-foot ground contact detection (from SMPLX toe kinematics)
    parser.add_argument("--foot_contact_z_thresh", type=float, default=0.08,
                        help="Max SMPLX toe height above the per-motion floor to count as "
                             "ground contact, in meters. Default 0.08.")
    parser.add_argument("--foot_contact_vel_thresh", type=float, default=0.5,
                        help="Max SMPLX toe 3D speed to count as ground contact, in m/s. "
                             "Default 0.5.")
    parser.add_argument("--no_foot_contact", action="store_true",
                        help="Skip writing foot_ground_contact_flags to the output pkl.")

    # Processing
    parser.add_argument("--no_viz", action="store_true",
                        help="Disable MuJoCo viewer (default for batch)")
    parser.add_argument("--record_video", action="store_true",
                        help="Record MP4 video. Works in single-file, directory, and YAML batch modes. "
                             "Videos are saved to <output_dir>/videos/<robot>_<stem>.mp4.")
    parser.add_argument("--rate_limit", action="store_true",
                        help="Rate-limit visualization to motion FPS")
    parser.add_argument("--loop", action="store_true",
                        help="Loop motion in viewer (single file mode)")
    parser.add_argument("--override", action="store_true",
                        help="Overwrite existing output files in batch mode")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

class _Tee:
    """Duplicate writes to multiple streams (e.g. real stdout + a log file)."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass
    def isatty(self):
        return False


def _run_batch(files, output_dir, input_format, args):
    """Process a list of (src_path, stem) pairs and save to output_dir."""
    # Suppress interactive viewer in batch, but allow video recording (offscreen render).
    args.no_viz = True
    args.rate_limit = False

    # Tee stdout to <output_dir>/<top-level-stem-dir>/output.txt so the run's
    # terminal log lives next to the pkl/video outputs.
    log_file = None
    saved_stdout = None
    if files:
        first_parts = pathlib.Path(files[0][1]).parts
        log_subdir = first_parts[0] if len(first_parts) > 1 else ""
        log_dir = pathlib.Path(output_dir) / log_subdir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "output.txt"
        log_file = open(log_path, "w", buffering=1)
        saved_stdout = sys.stdout
        sys.stdout = _Tee(saved_stdout, log_file)
        # Reset rich's cached console so it picks up the new sys.stdout.
        try:
            import rich
            rich.reconfigure()
        except Exception:
            pass

    try:
        print(
            f"[bold]Found {len(files)} files. "
            f"Ground mode: {args.ground_mode}. "
            f"Output: {output_dir}[/bold]"
        )

        try:
            from tqdm import tqdm
            file_iter = tqdm(files, desc="Retargeting")
        except ImportError:
            file_iter = files

        success = 0
        for src_path, rel_stem in file_iter:
            out_path = os.path.join(output_dir, rel_stem + ".pkl")

            if os.path.exists(out_path) and not args.override:
                continue

            try:
                ok = retarget_and_save(src_path, out_path, input_format, args)
                if ok:
                    success += 1
            except Exception as e:
                print(f"[red]Error processing {src_path}: {e}[/red]")

        print(f"[bold green]Done! {success}/{len(files)} files -> {output_dir}[/bold green]")
    finally:
        if log_file is not None:
            sys.stdout = saved_stdout
            log_file.close()
            try:
                import rich
                rich.reconfigure()
            except Exception:
                pass


def main():
    args = parse_args()

    # ======= YAML BATCH MODE (AMASS CMU locomotion list) =======
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
            # Stem: prefer the path under an "AMASS/" component for clean output layout.
            parts = pathlib.Path(rel_path).parts
            if "AMASS" in parts:
                idx = parts.index("AMASS")
                tail = pathlib.Path(*parts[idx + 1:])
            else:
                tail = pathlib.Path(pathlib.Path(rel_path).name)
            stem = str(tail.parent / _amass_stem(str(tail)))
            # Prefix the top-level dataset directory with the robot type so
            # different robots' outputs don't collide (e.g. booster_t1_CMU/...).
            stem_parts = pathlib.Path(stem).parts
            if len(stem_parts) > 1:
                stem = str(pathlib.Path(f"{args.robot}_{stem_parts[0]}", *stem_parts[1:]))
            else:
                stem = f"{args.robot}_{stem}"
            files.append((full, stem))

        output_dir = args.output if args.output else str(yaml_dir / "retargeted")
        _run_batch(files, output_dir, args.input_format, args)
        return

    # ======= FILE / DIRECTORY MODE =======
    input_path = os.path.abspath(args.input)

    if os.path.isfile(input_path):
        # ======= SINGLE FILE MODE =======
        if args.output:
            output_path = args.output
        elif args.input_format == "amass_cmu":
            output_path = _amass_stem(input_path) + ".pkl"
        else:
            stem = input_path.rsplit(".", 1)[0]
            output_path = stem + "_retargeted.pkl"
        retarget_and_save(input_path, output_path, args.input_format, args)

    elif os.path.isdir(input_path):
        # ======= DIRECTORY BATCH MODE =======
        if args.output:
            output_dir = args.output
        else:
            output_dir = input_path.rstrip("/").rstrip("\\") + "_retargeted"

        files = discover_input_files(input_path, args.input_format)

        if not files:
            print(f"[red]No {args.input_format} files found in {input_path}[/red]")
            return

        _run_batch(files, output_dir, args.input_format, args)

    else:
        print(f"[red]'{input_path}' is neither a file nor a directory[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
