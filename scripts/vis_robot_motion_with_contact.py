"""Render an mp4 from a retargeted pkl and overlay per-frame foot-ground contact.

Reads ``foot_ground_contact_flags`` from the pkl (shape (T, 2); col 0 = left,
col 1 = right) and, after the mp4 is written, re-encodes it with per-frame
``L FOOT`` / ``R FOOT`` text whenever the corresponding flag is True.

Two modes:

    1. Render + overlay (default):
         python scripts/vis_robot_motion_with_contact.py \
             --robot_motion_path path/to/retarget.pkl --robot booster_k1

    2. Overlay only (skip rendering — reuse an existing mp4):
         python scripts/vis_robot_motion_with_contact.py \
             --robot_motion_path path/to/retarget.pkl --robot booster_k1 \
             --video_path path/to/existing.mp4 --no_render

Use ``--input_dir`` to process a directory of pkls recursively; set
``--video_dir`` to redirect outputs. If the pkl has no
``foot_ground_contact_flags`` key the overlay is skipped silently.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def render_from_pkl(pkl_path: Path, robot: str, video_path: Path, rate_limit: bool = False) -> None:
    """Render a robot-motion pkl to ``video_path`` using RobotMotionViewer."""
    from general_motion_retargeting import RobotMotionViewer, load_robot_motion

    _, fps, root_pos, root_rot, dof_pos, _, _ = load_robot_motion(str(pkl_path))
    video_path.parent.mkdir(parents=True, exist_ok=True)
    viewer = RobotMotionViewer(
        robot_type=robot,
        motion_fps=fps,
        camera_follow=False,
        record_video=True,
        video_path=str(video_path),
    )
    try:
        for i in range(len(root_pos)):
            viewer.step(root_pos[i], root_rot[i], dof_pos[i], rate_limit=rate_limit)
    finally:
        viewer.close()


def overlay_foot_contact(video_path: Path, foot_flags: np.ndarray) -> None:
    """Re-encode ``video_path`` in-place with per-frame L/R FOOT text.

    ``foot_flags`` shape (T, 2), column 0 = left foot, column 1 = right.
    Silent no-op if OpenCV is unavailable or the flags look wrong.
    """
    try:
        import cv2
    except ImportError:
        print("[WARN] OpenCV not available; skipping foot-contact overlay.")
        return

    foot_flags = np.asarray(foot_flags, dtype=bool)
    if foot_flags.ndim != 2 or foot_flags.shape[1] < 2:
        print(f"[WARN] foot_ground_contact_flags has unexpected shape {foot_flags.shape}; skipping overlay.")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open {video_path} for overlay; skipping.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp_out = video_path.with_name(video_path.stem + "_footoverlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_out), fourcc, fps, (W, H))
    font = cv2.FONT_HERSHEY_SIMPLEX
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if t < foot_flags.shape[0]:
            if foot_flags[t, 0]:
                cv2.putText(frame, "L FOOT", (20, 50), font, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            if foot_flags[t, 1]:
                cv2.putText(frame, "R FOOT", (20, 100), font, 1.2, (0, 165, 255), 3, cv2.LINE_AA)
        writer.write(frame)
        t += 1
    cap.release()
    writer.release()
    os.replace(tmp_out, video_path)
    print(f"[INFO] Foot-contact overlay applied: {video_path}")


def _default_video_path(pkl_path: Path, robot: str, input_root: Path = None, video_dir: Path = None) -> Path:
    """Place the mp4 mirroring the pkl's stem.

    - If ``video_dir`` is given and ``input_root`` is given, preserve subdir
      structure under ``video_dir``.
    - Otherwise default to ``videos/{stem}_{robot}.mp4`` in the repo root.
    """
    name = f"{pkl_path.stem}_{robot}.mp4"
    if video_dir is not None and input_root is not None:
        rel = pkl_path.relative_to(input_root)
        return video_dir / rel.parent / name
    if video_dir is not None:
        return video_dir / name
    return Path(__file__).resolve().parents[1] / "videos" / name


def process_one(pkl_path: Path, robot: str, video_path: Path, *, do_render: bool, do_overlay: bool, rate_limit: bool) -> None:
    if do_render:
        print(f"[INFO] Rendering {pkl_path} -> {video_path}")
        render_from_pkl(pkl_path, robot, video_path, rate_limit=rate_limit)

    if not do_overlay:
        return

    with open(pkl_path, "rb") as f:
        motion_data = pickle.load(f)
    flags = motion_data.get("foot_ground_contact_flags")
    if flags is None:
        print(f"[INFO] {pkl_path.name}: no 'foot_ground_contact_flags' key; overlay skipped.")
        return
    if not video_path.exists():
        print(f"[WARN] {video_path} does not exist; cannot overlay. Did you pass --no_render?")
        return
    overlay_foot_contact(video_path, flags)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--robot_motion_path", type=str, help="Single retargeted pkl.")
    src.add_argument("--input_dir", type=str, help="Directory of pkls (recursive).")

    p.add_argument("--robot", type=str, default="booster_k1",
                   help="Robot type used by RobotMotionViewer. Must match the pkl.")
    p.add_argument("--video_path", type=str, default=None,
                   help="Output mp4 (single pkl mode). Defaults to videos/{stem}_{robot}.mp4.")
    p.add_argument("--video_dir", type=str, default=None,
                   help="Output directory mirroring the input_dir subtree.")
    p.add_argument("--no_render", action="store_true",
                   help="Skip rendering; only overlay on the existing --video_path.")
    p.add_argument("--no_overlay", action="store_true",
                   help="Render without applying the L/R FOOT text overlay.")
    p.add_argument("--rate_limit", action="store_true",
                   help="Step RobotMotionViewer at the motion FPS (slower; only useful if watching live).")

    args = p.parse_args()
    do_render = not args.no_render
    do_overlay = not args.no_overlay

    if args.robot_motion_path:
        pkl = Path(args.robot_motion_path).resolve()
        if not pkl.exists():
            p.error(f"--robot_motion_path does not exist: {pkl}")
        video_path = Path(args.video_path) if args.video_path else _default_video_path(pkl, args.robot)
        process_one(pkl, args.robot, video_path, do_render=do_render, do_overlay=do_overlay, rate_limit=args.rate_limit)
        return

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        p.error(f"--input_dir is not a directory: {input_dir}")
    pkls = sorted(input_dir.rglob("*.pkl"))
    if not pkls:
        print(f"[WARN] No pkls under {input_dir}")
        return
    video_dir = Path(args.video_dir).resolve() if args.video_dir else None
    for pkl in pkls:
        vpath = _default_video_path(pkl, args.robot, input_root=input_dir, video_dir=video_dir)
        try:
            process_one(pkl, args.robot, vpath, do_render=do_render, do_overlay=do_overlay, rate_limit=args.rate_limit)
        except Exception as e:
            print(f"[ERROR] {pkl}: {e}")


if __name__ == "__main__":
    main()
