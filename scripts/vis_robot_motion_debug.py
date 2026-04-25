"""
Debug viewer: replay a saved robot motion pkl with sole contact point markers
and penetration coloring.

Usage:
    python scripts/vis_robot_motion_debug.py \\
        --robot_motion_path retargeted/kick.pkl \\
        --show_sole_points --show_penetration --show_ground_plane

    # With video recording:
    python scripts/vis_robot_motion_debug.py \\
        --robot_motion_path retargeted/kick.pkl \\
        --show_sole_points --show_penetration \\
        --record_video --video_path videos/debug.mp4
"""

import argparse
import pathlib
import sys

import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
import imageio
from loop_rate_limiters import RateLimiter

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from general_motion_retargeting import (
    ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT, load_robot_motion
)
from general_motion_retargeting.sole_points import get_sole_points

ROBOT_TYPE = "booster_k1"


# ---------------------------------------------------------------------------
# Drawing helpers (work on any mjvScene)
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


def draw_line(from_pos, to_pos, scene, width=0.005, rgba=(1.0, 1.0, 0.0, 0.8)):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_LINE,
        size=[width, width, width],
        pos=np.asarray(from_pos, dtype=float),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=float),
    )
    mj.mjv_connector(
        scene.geoms[scene.ngeom],
        type=mj.mjtGeom.mjGEOM_LINE,
        width=width,
        from_=np.asarray(from_pos, dtype=float),
        to=np.asarray(to_pos, dtype=float),
    )
    geom.rgba[:] = np.asarray(rgba, dtype=float)
    scene.ngeom += 1


def draw_debug_geoms(scene, model, data, sole_config, args):
    """Draw sole point markers and optional ground plane on the given scene.

    Works for both the interactive viewer (viewer.user_scn) and the offscreen
    renderer (renderer.scene).
    """
    for body_name, local_points in sole_config.items():
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        body_pos = data.xpos[body_id]
        body_rot = data.xmat[body_id].reshape(3, 3)

        for local_pt in local_points:
            world_pt = body_rot @ np.array(local_pt, dtype=float) + body_pos
            is_penetrating = world_pt[2] < args.ground_height

            if args.show_sole_points or args.show_penetration:
                if args.show_penetration and is_penetrating:
                    color = [1.0, 0.0, 0.0, 0.9]  # red = penetrating
                else:
                    color = [0.0, 1.0, 0.0, 0.6]  # green = ok
                draw_sphere(world_pt, scene, radius=args.sole_point_radius, rgba=color)

            # Vertical line from point down to ground when penetrating
            if args.show_penetration and is_penetrating:
                ground_pt = world_pt.copy()
                ground_pt[2] = args.ground_height
                draw_line(world_pt, ground_pt, scene, width=0.003,
                          rgba=[1.0, 0.3, 0.0, 0.8])

    # Translucent ground plane indicator
    if args.show_ground_plane:
        robot_base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY,
                                       ROBOT_BASE_DICT[ROBOT_TYPE])
        if robot_base_id >= 0 and scene.ngeom < scene.maxgeom:
            center = data.xpos[robot_base_id].copy()
            center[2] = args.ground_height
            geom = scene.geoms[scene.ngeom]
            mj.mjv_initGeom(
                geom,
                type=mj.mjtGeom.mjGEOM_PLANE,
                size=[2.0, 2.0, 0.001],
                pos=center,
                mat=np.eye(3).flatten(),
                rgba=np.array([0.5, 0.5, 0.5, 0.12], dtype=float),
            )
            scene.ngeom += 1


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug viewer for robot motion pkl with sole point markers."
    )
    parser.add_argument("--robot_motion_path", type=str, required=True,
                        help="Path to the .pkl robot motion file.")
    parser.add_argument("--record_video", action="store_true",
                        help="Record MP4 video.")
    parser.add_argument("--video_path", type=str, default="videos/debug.mp4",
                        help="Output video path (default: videos/debug.mp4).")
    parser.add_argument("--show_sole_points", action="store_true",
                        help="Draw spheres at sole contact point positions.")
    parser.add_argument("--show_penetration", action="store_true",
                        help="Color sole points red if penetrating, green if ok. "
                             "Draw vertical lines for penetrating points.")
    parser.add_argument("--show_ground_plane", action="store_true",
                        help="Draw a translucent ground plane.")
    parser.add_argument("--sole_point_radius", type=float, default=0.024,
                        help="Radius of sole point debug spheres (default: 0.024).")
    parser.add_argument("--ground_height", type=float, default=0.0,
                        help="z-coordinate of the ground plane (default: 0.0).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-frame penetration statistics.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load motion data
    motion_data, fps, root_pos, root_rot, dof_pos, _, _ = load_robot_motion(
        args.robot_motion_path
    )
    # root_rot is now wxyz (load_robot_motion converts from pkl's xyzw)
    num_frames = len(root_pos)
    print(f"[bold]Loaded {num_frames} frames @ {fps:.1f} fps from {args.robot_motion_path}[/bold]")

    # Load MuJoCo model
    xml_path = str(ROBOT_XML_DICT[ROBOT_TYPE])
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    # Sole config for debug rendering
    sole_config = {}
    if args.show_sole_points or args.show_penetration:
        sole_config = get_sole_points(ROBOT_TYPE)

    # Create passive viewer
    viewer = mjv.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    )
    rate_limiter = RateLimiter(frequency=fps, warn=False)

    # Video recording setup
    renderer = None
    mp4_writer = None
    if args.record_video:
        vid_dir = pathlib.Path(args.video_path).parent
        vid_dir.mkdir(parents=True, exist_ok=True)
        renderer = mj.Renderer(model, height=480, width=640)
        mp4_writer = imageio.get_writer(args.video_path, fps=int(fps))
        print(f"Recording video to {args.video_path}")

    base_name = ROBOT_BASE_DICT[ROBOT_TYPE]
    cam_distance = VIEWER_CAM_DISTANCE_DICT[ROBOT_TYPE]

    frame_idx = 0
    while viewer.is_running():
        # Set robot pose from saved motion
        data.qpos[:3] = root_pos[frame_idx]
        data.qpos[3:7] = root_rot[frame_idx]   # wxyz
        data.qpos[7:] = dof_pos[frame_idx]
        mj.mj_forward(model, data)

        # Camera follow
        base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, base_name)
        if base_id >= 0:
            viewer.cam.lookat[:] = data.xpos[base_id]
        viewer.cam.distance = cam_distance
        viewer.cam.elevation = -15

        # Draw debug geoms on interactive viewer scene
        viewer.user_scn.ngeom = 0
        draw_debug_geoms(viewer.user_scn, model, data, sole_config, args)
        viewer.sync()

        # Draw debug geoms on offscreen renderer scene (for video)
        if renderer is not None:
            renderer.update_scene(data, camera=viewer.cam)
            draw_debug_geoms(renderer.scene, model, data, sole_config, args)
            img = renderer.render()
            mp4_writer.append_data(img)

        # Per-frame penetration stats
        if args.verbose and sole_config:
            min_z = float("inf")
            for body_name, local_points in sole_config.items():
                bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
                if bid < 0:
                    continue
                bpos = data.xpos[bid]
                brot = data.xmat[bid].reshape(3, 3)
                for lp in local_points:
                    wz = (brot @ np.array(lp) + bpos)[2]
                    min_z = min(min_z, wz)
            penetration_cm = max(0.0, args.ground_height - min_z) * 100
            if penetration_cm > 0:
                print(f"Frame {frame_idx}: penetration = {penetration_cm:.2f} cm")

        rate_limiter.sleep()
        frame_idx = (frame_idx + 1) % num_frames

    viewer.close()
    if mp4_writer is not None:
        mp4_writer.close()
        print(f"[green]Video saved: {args.video_path}[/green]")


if __name__ == "__main__":
    main()
