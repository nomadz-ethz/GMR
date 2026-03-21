"""
Compare ground penetration and root jitter stats across multiple pkl files.

Usage:
    python scripts/compare_penetration_stats.py file1.pkl file2.pkl ...
    python scripts/compare_penetration_stats.py --glob 'data/locomotion_amass_cmu/*.pkl'
"""

import argparse
import glob
import pickle
import sys
import pathlib

import numpy as np
import mujoco as mj

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

ROBOT_TYPE = "booster_k1"


def load_sole_config():
    from general_motion_retargeting.sole_points import get_sole_points
    return get_sole_points(ROBOT_TYPE)


def compute_stats(pkl_path, model, sole_config, ground_height=0.0, clearance=0.003):
    """Return dict of stats for one pkl file."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    root_pos = d["root_pos"]       # (N, 3)
    root_rot = d["root_rot"]       # (N, 4) xyzw
    dof_pos  = d["dof_pos"]        # (N, J)
    N = len(root_pos)

    data = mj.MjData(model)

    penetrations = []
    root_z_vals = []

    for i in range(N):
        data.qpos[:3] = root_pos[i]
        # convert xyzw -> wxyz for qpos[3:7]
        data.qpos[3:7] = root_rot[i][[3, 0, 1, 2]]
        data.qpos[7:] = dof_pos[i]
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

    pen_depth = np.maximum(0.0, clearance - pen_arr)  # depth below clearance
    n_pen = int(np.sum(pen_depth > 1e-4))
    max_pen = float(pen_depth.max()) * 1000  # mm

    # Root jitter: per-frame upward jump in root_z
    drz = np.diff(rz_arr)
    up_jumps = drz[drz > 0]
    max_up = float(up_jumps.max()) * 1000 if len(up_jumps) else 0.0  # mm
    mean_up = float(up_jumps.mean()) * 1000 if len(up_jumps) else 0.0  # mm
    rz_range = float(rz_arr.max() - rz_arr.min()) * 1000  # mm

    return {
        "n_frames": N,
        "n_pen": n_pen,
        "max_pen_mm": max_pen,
        "max_up_mm": max_up,
        "mean_up_mm": mean_up,
        "rz_range_mm": rz_range,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="PKL files to compare")
    parser.add_argument("--glob", type=str, default=None, help="Glob pattern for pkl files")
    parser.add_argument("--ground_height", type=float, default=0.0)
    parser.add_argument("--clearance", type=float, default=0.003)
    args = parser.parse_args()

    files = list(args.files)
    if args.glob:
        files += sorted(glob.glob(args.glob))
    if not files:
        print("No files specified.")
        sys.exit(1)

    from general_motion_retargeting import ROBOT_XML_DICT
    xml_path = str(ROBOT_XML_DICT[ROBOT_TYPE])
    model = mj.MjModel.from_xml_path(xml_path)
    sole_config = load_sole_config()

    header = ("Label              "
              "  frames  pen/N  max_pen  max_up  mean_up  rz_range")
    print(header)
    print("-" * len(header))

    for f in files:
        label = pathlib.Path(f).stem
        try:
            s = compute_stats(f, model, sole_config, args.ground_height, args.clearance)
        except Exception as e:
            print(f"{label:18s}  ERROR: {e}")
            continue

        pen_frac = f"{s['n_pen']}/{s['n_frames']}"
        print(
            f"{label:18s}  "
            f"{s['n_frames']:6d}  "
            f"{pen_frac:7s}  "
            f"{s['max_pen_mm']:6.1f}mm  "
            f"{s['max_up_mm']:5.1f}mm  "
            f"{s['mean_up_mm']:6.1f}mm  "
            f"{s['rz_range_mm']:7.1f}mm"
        )


if __name__ == "__main__":
    main()
