"""
Side-by-side comparison video from two robot motion pkl files.

Usage:
    python scripts/vis_compare_motions.py \\
        --motion_a retargeted/vanilla.pkl \\
        --motion_b retargeted/qp.pkl \\
        --label_a "Vanilla GMR" \\
        --label_b "QP Constraint" \\
        --video_path videos/comparison.mp4 \\
        --show_sole_points

Pass --robot to choose between booster_k1 and booster_t1.
"""

import argparse
import pathlib
import sys

import numpy as np
import mujoco as mj
import imageio

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from general_motion_retargeting import (
    ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT, load_robot_motion
)
from general_motion_retargeting.sole_points import get_sole_points


# ---------------------------------------------------------------------------
# Drawing helpers (reuse from vis_robot_motion_debug)
# ---------------------------------------------------------------------------

def draw_sphere(pos, scene, radius=0.008, rgba=(0.0, 1.0, 0.0, 0.6)):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[radius, radius, radius],
        pos=np.asarray(pos, dtype=float),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=float),
    )
    scene.ngeom += 1


def draw_debug_geoms_simple(scene, model, data, sole_config, ground_height=0.0):
    """Draw colored sole point spheres: red if penetrating, green if ok."""
    for body_name, local_points in sole_config.items():
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        body_pos = data.xpos[body_id]
        body_rot = data.xmat[body_id].reshape(3, 3)

        for local_pt in local_points:
            world_pt = body_rot @ np.array(local_pt, dtype=float) + body_pos
            is_penetrating = world_pt[2] < ground_height
            color = [1.0, 0.0, 0.0, 0.9] if is_penetrating else [0.0, 1.0, 0.0, 0.6]
            draw_sphere(world_pt, scene, radius=0.024, rgba=color)


def _add_text_label(img: np.ndarray, text: str) -> np.ndarray:
    """Add a text label to the top-left of an image. Uses cv2 if available."""
    try:
        import cv2
        img = img.copy()
        cv2.putText(
            img, text, (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
    except ImportError:
        pass  # labels won't appear but video still works
    return img


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison video from two robot motion pkl files."
    )
    parser.add_argument("--motion_a", type=str, required=True,
                        help="Path to pkl file A (e.g., vanilla).")
    parser.add_argument("--motion_b", type=str, required=True,
                        help="Path to pkl file B (e.g., ground-constrained).")
    parser.add_argument("--label_a", type=str, default="Vanilla GMR",
                        help="Label for panel A.")
    parser.add_argument("--label_b", type=str, default="No Penetration",
                        help="Label for panel B.")
    parser.add_argument("--video_path", type=str, default="videos/comparison.mp4",
                        help="Output video path.")
    parser.add_argument("--video_width", type=int, default=640,
                        help="Per-panel video width (default: 640).")
    parser.add_argument("--video_height", type=int, default=480,
                        help="Per-panel video height (default: 480).")
    parser.add_argument("--show_sole_points", action="store_true",
                        help="Draw colored sole point debug spheres.")
    parser.add_argument("--ground_height", type=float, default=0.0,
                        help="z-coordinate of the ground plane (default: 0.0).")
    parser.add_argument("--robot", choices=["booster_k1", "booster_t1"],
                        default="booster_k1",
                        help="Target robot model. Should match the pkl's 'robot' "
                             "key when present.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load both motions
    _, fps_a, pos_a, rot_a, dof_a, _, _ = load_robot_motion(args.motion_a)
    _, fps_b, pos_b, rot_b, dof_b, _, _ = load_robot_motion(args.motion_b)

    num_frames = min(len(pos_a), len(pos_b))
    fps = fps_a
    print(
        f"[bold]Motion A: {len(pos_a)} frames, Motion B: {len(pos_b)} frames. "
        f"Rendering {num_frames} frames.[/bold]"
    )

    robot_type = args.robot
    xml_path = str(ROBOT_XML_DICT[robot_type])

    # Two separate model/data pairs (same MJCF, independent physics states)
    model_a = mj.MjModel.from_xml_path(xml_path)
    data_a = mj.MjData(model_a)
    model_b = mj.MjModel.from_xml_path(xml_path)
    data_b = mj.MjData(model_b)

    # Two offscreen renderers
    renderer_a = mj.Renderer(model_a, height=args.video_height, width=args.video_width)
    renderer_b = mj.Renderer(model_b, height=args.video_height, width=args.video_width)

    # Sole config for debug markers
    sole_config = get_sole_points(robot_type) if args.show_sole_points else {}

    # Output video (combined width = 2 * per-panel width)
    vid_dir = pathlib.Path(args.video_path).parent
    vid_dir.mkdir(parents=True, exist_ok=True)
    mp4_writer = imageio.get_writer(args.video_path, fps=int(fps))

    base_name = ROBOT_BASE_DICT[robot_type]
    cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]

    try:
        from tqdm import tqdm
        frame_iter = tqdm(range(num_frames), desc="Rendering comparison")
    except ImportError:
        frame_iter = range(num_frames)

    for i in frame_iter:
        # --- Panel A ---
        data_a.qpos[:3] = pos_a[i]
        data_a.qpos[3:7] = rot_a[i]  # wxyz
        data_a.qpos[7:] = dof_a[i]
        mj.mj_forward(model_a, data_a)

        cam_a = mj.MjvCamera()
        base_id_a = mj.mj_name2id(model_a, mj.mjtObj.mjOBJ_BODY, base_name)
        if base_id_a >= 0:
            cam_a.lookat[:] = data_a.xpos[base_id_a]
        cam_a.distance = cam_distance
        cam_a.elevation = -15
        cam_a.azimuth = 90

        renderer_a.update_scene(data_a, camera=cam_a)
        if sole_config:
            draw_debug_geoms_simple(
                renderer_a.scene, model_a, data_a, sole_config, args.ground_height
            )
        img_a = renderer_a.render().copy()

        # --- Panel B ---
        data_b.qpos[:3] = pos_b[i]
        data_b.qpos[3:7] = rot_b[i]  # wxyz
        data_b.qpos[7:] = dof_b[i]
        mj.mj_forward(model_b, data_b)

        cam_b = mj.MjvCamera()
        base_id_b = mj.mj_name2id(model_b, mj.mjtObj.mjOBJ_BODY, base_name)
        if base_id_b >= 0:
            cam_b.lookat[:] = data_b.xpos[base_id_b]
        cam_b.distance = cam_distance
        cam_b.elevation = -15
        cam_b.azimuth = 90

        renderer_b.update_scene(data_b, camera=cam_b)
        if sole_config:
            draw_debug_geoms_simple(
                renderer_b.scene, model_b, data_b, sole_config, args.ground_height
            )
        img_b = renderer_b.render().copy()

        # Add labels
        img_a = _add_text_label(img_a, args.label_a)
        img_b = _add_text_label(img_b, args.label_b)

        # Combine side by side with a thin separator
        separator = np.full((args.video_height, 2, 3), 128, dtype=np.uint8)
        combined = np.concatenate([img_a, separator, img_b], axis=1)
        mp4_writer.append_data(combined)

    mp4_writer.close()
    print(f"[green]Comparison video saved: {args.video_path}[/green]")


if __name__ == "__main__":
    main()
