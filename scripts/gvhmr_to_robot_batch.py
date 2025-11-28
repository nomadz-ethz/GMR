"""
Modified copy of gvhmr_to_robot.py to run a batch of videos in a folder.
"""

import argparse
import pathlib
import os
import time
import pickle
from tqdm import tqdm 

import numpy as np
from rich import print

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

HERE = pathlib.Path(__file__).parent
SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

def process_motion(gvhmr_file_path, save_path, args):
    """
    Handles the loading, retargeting, and saving for a single file.
    """
    # Use parent folder name (e.g. '11_freekick') for the video name, 
    # because the file itself is always named 'hmr4d_results.pt'
    file_stem = pathlib.Path(gvhmr_file_path).parent.name
    
    # --- 1. Load SMPLX trajectory ---
    try:
        smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
            str(gvhmr_file_path), SMPLX_FOLDER
        )
    except Exception as e:
        print(f"[red]Error loading {gvhmr_file_path}: {e}[/red]")
        return

    # --- 2. Align FPS ---
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )
    
    # --- 3. Initialize the retargeting system ---
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    
    # Determine video output path if recording
    # It will save as videos/booster_k1_11_freekick.mp4
    video_out = f"videos/{args.robot}_{file_stem}.mp4"
    
    # Initialize Viewer
    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=aligned_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=video_out,
    )
    
    qpos_list = []
    
    # --- 4. Main Loop ---
    should_loop = args.loop and (save_path is None) 
    i = 0
    
    while True:
        if should_loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        smplx_data_frame = smplx_data_frames[i]

        # Retarget
        qpos = retarget.retarget(smplx_data_frame)

        # Visualize / Step
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
        )
        
        if save_path is not None:
            qpos_list.append(qpos)
            
    # --- 5. Save Results ---
    if save_path is not None and len(qpos_list) > 0:
        root_pos = np.array([q[:3] for q in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([q[3:7][[1,2,3,0]] for q in qpos_list])
        dof_pos = np.array([q[7:] for q in qpos_list])
        
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"[green]Saved {file_stem} to {save_path}[/green]")
            
    robot_motion_viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # -- Input Mode Arguments --
    parser.add_argument(
        "--gvhmr_pred_file",
        type=str,
        default=None,
        help="Single SMPLX motion file to load.",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default=None,
        help="Directory containing subfolders with hmr4d_results.pt files.",
    )
    
    # -- Output Arguments --
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion (Single file mode).",
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        help="Directory to save robot motions (Batch mode).",
    )

    # -- Configuration Arguments --
    parser.add_argument(
        "--robot",
        default="booster_k1",
        help="Robot name (e.g. booster_k1, unitree_g1)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the motion (only works in single file visualization mode).",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Record the video.",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        help="Limit the rate of the retargeted robot motion.",
    )

    args = parser.parse_args()

    # --- Execution Logic ---
    
    if args.input_folder:
        # === BATCH PROCESSING (Fixed for recursive structure) ===
        input_path = pathlib.Path(args.input_folder)
        
        # Matches: out_smplx/11_freekick/hmr4d_results.pt
        motion_files = sorted(list(input_path.rglob("hmr4d_results.pt")))
        
        if not motion_files:
            # Fallback: try looking for flat .pt files just in case
            motion_files = sorted(list(input_path.glob("*.pt")))
        
        if not motion_files:
            print(f"[red]No 'hmr4d_results.pt' files found in {args.input_folder}[/red]")
            exit(1)
            
        print(f"[bold green]Found {len(motion_files)} motion files in {args.input_folder}[/bold green]")
        
        # Ensure output folder exists
        out_root = args.output_folder if args.output_folder else "output_retargeted"
        os.makedirs(out_root, exist_ok=True)
        
        for motion_file in tqdm(motion_files, desc="Retargeting"):
            video_name = motion_file.parent.name
            
            # If the file was found in the root (flat structure fallback), use stem
            if video_name == input_path.name: 
                video_name = motion_file.stem
            
            # --- PRINT STATEMENTS ADDED HERE ---
            print(f"\n[bold cyan]Found Source File:[/bold cyan] {motion_file}")
            print(f"[bold cyan]Processing Video Name:[/bold cyan] {video_name}")

            out_filename = f"{video_name}.pkl"
            out_path = os.path.join(out_root, out_filename)
            
            process_motion(motion_file, out_path, args)
            
    else:
        # === SINGLE FILE PROCESSING ===
        if not args.gvhmr_pred_file:
            print("[red]Please provide either --input_folder or --gvhmr_pred_file[/red]")
            exit(1)
            
        process_motion(args.gvhmr_pred_file, args.save_path, args)