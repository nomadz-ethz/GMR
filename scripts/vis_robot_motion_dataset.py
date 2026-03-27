from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
from rich import print

paused = False
motion_num = 0
motion_id = 0
current_motion_id = -1
delete_requested = False

def keyboard_callback(keycode):
    global paused, motion_id, motion_num, delete_requested
    if chr(keycode) == ' ':
        paused = not paused
    if chr(keycode) == '[':
        motion_id = (motion_id - 1) % motion_num
    if chr(keycode) == ']':
        motion_id = (motion_id + 1) % motion_num
    if chr(keycode) == 'x':
        delete_requested = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_folder", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_folder = args.robot_motion_folder
    
    if not os.path.exists(robot_motion_folder):
        raise FileNotFoundError(f"Motion data dir {robot_motion_folder} does not exist.")
    
    motion_files = [f for f in os.listdir(robot_motion_folder) if f.endswith('.pkl')]
    motion_files = sorted(motion_files)
    motion_num = len(motion_files)
    print(f"Found {motion_num} motion files in {robot_motion_folder}, loading...")
    motion_dataset = []
    for motion_file in tqdm(motion_files):
        motion_path = os.path.join(robot_motion_folder, motion_file)
        motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(motion_path)
        motion_dataset.append({
            "motion_file": motion_file,
            "motion_data": motion_data,
            "motion_fps": motion_fps,
            "motion_root_pos": motion_root_pos,
            "motion_root_rot": motion_root_rot,
            "motion_dof_pos": motion_dof_pos,
            "motion_local_body_pos": motion_local_body_pos,
            "motion_link_body_list": motion_link_body_list,
        })
    print("Loading done.")
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path, 
                            keyboard_callback=keyboard_callback)
    
    # Create a trash directory for bad motions
    trash_dir = Path(robot_motion_folder) / "trash"
    
    frame_idx = 0
    while True:
        # get current motion
        if current_motion_id != motion_id:
            current_motion_id = motion_id
            frame_idx = 0
            motion_data = motion_dataset[motion_id]
            motion_file = motion_data["motion_file"]
            motion_fps = motion_data["motion_fps"]
            motion_root_pos = motion_data["motion_root_pos"]
            motion_root_rot = motion_data["motion_root_rot"]
            motion_dof_pos = motion_data["motion_dof_pos"]
            print(f"\n[bold green]Switching to motion {motion_id}: {motion_file}[/bold green]")
            print(f"  > Frames: {len(motion_root_pos)}, FPS: {motion_fps}")
            print(f"  > Press [bold red]'x'[/bold red] to move this file to trash.")
        
        # Check for deletion request
        if delete_requested:
            delete_requested = False
            pkl_to_trash = Path(robot_motion_folder) / motion_files[motion_id]
            trash_dir.mkdir(exist_ok=True)
            new_path = trash_dir / motion_files[motion_id]
            print(f"[bold red]Moving {motion_files[motion_id]} to trash...[/bold red]")
            os.rename(pkl_to_trash, new_path)
            
            # Remove from our local lists and update counts
            motion_dataset.pop(motion_id)
            motion_files.pop(motion_id)
            motion_num -= 1
            if motion_num == 0:
                print("All motions deleted/reviewed!")
                break
            motion_id %= motion_num  # Stay on same index or wrap
            current_motion_id = -1 # Trigger reload
            continue

        if not paused:
            # We pass a dummy human_motion_data to RobotMotionViewer.step 
            # to leverage its internal draw_frame logic for our filename label
            # We place it at root_pos + [0, 0, 0.5] to float above the robot's head
            label_pos = motion_root_pos[frame_idx].copy()
            label_pos[2] += 0.5 # Floating 0.5m above root
            label_data = {f"FILE: {motion_file}": (label_pos, np.array([1, 0, 0, 0]))}
            
            env.step(motion_root_pos[frame_idx], 
                    motion_root_rot[frame_idx], 
                    motion_dof_pos[frame_idx], 
                    human_motion_data=label_data,
                    show_human_body_name=True,
                    human_point_scale=0.01, # Invisible anchor
                    rate_limit=True)
            frame_idx += 1
            if frame_idx >= len(motion_root_pos):
                frame_idx = 0
    env.close()