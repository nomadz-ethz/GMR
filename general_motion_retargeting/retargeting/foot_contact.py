"""Per-foot ground contact detection from raw SMPL-X toe kinematics.

Operates on `smplx_data_frames` (the per-frame dicts produced by
`general_motion_retargeting.utils.smpl.get_smplx_data_offline_fast`) so the
labels are stable across `--ground_mode` / `--strict_zero_pen` retargeting
variants. See `docs/pipeline.md` section 3 for the full rationale.
"""

from typing import Sequence, Tuple, Optional

import numpy as np


def detect_smplx_foot_contact(
    smplx_data_frames: Sequence[dict],
    fps: float,
    *,
    z_thresh: float = 0.08,
    v_thresh: float = 0.5,
    left_joint: str = "left_foot",
    right_joint: str = "right_foot",
) -> Tuple[Optional[np.ndarray], dict]:
    """Compute per-frame, per-foot contact bool flags.

    A foot is "in contact" at frame i iff
        (toe_z[i] - floor_z) < z_thresh    AND    speed[i] < v_thresh

    where `floor_z` is the per-clip minimum observed toe-z (left or right).
    Both thresholds together — height-only fires through clips whose source
    keeps the feet near the floor stylistically; speed-only fires at every
    momentary mid-air direction change.

    Args:
        smplx_data_frames: per-frame dict {joint_name: (pos, quat_wxyz)}.
        fps: aligned target fps used to compute `dt = 1 / fps`.
        z_thresh: max toe height above the per-clip floor (m).
        v_thresh: max toe 3D speed (m/s).
        left_joint, right_joint: SMPL-X joint names. Defaults match the
            standard names produced by `get_smplx_data_offline_fast`.

    Returns:
        (flags, info)
            flags: (N, 2) bool array (col 0 = left, col 1 = right) — or
                None if the source frames do not carry the requested joints
                or there are fewer than 2 frames.
            info: {"floor_z": float, "l_count": int, "r_count": int,
                   "n_frames": int, "z_thresh": float, "v_thresh": float}
    """
    n = len(smplx_data_frames)
    if n < 2:
        return None, {"reason": "fewer than 2 frames"}

    first = smplx_data_frames[0]
    if left_joint not in first or right_joint not in first:
        return None, {"reason": f"{left_joint}/{right_joint} missing"}

    l_toe = np.array([f[left_joint][0] for f in smplx_data_frames], dtype=np.float32)
    r_toe = np.array([f[right_joint][0] for f in smplx_data_frames], dtype=np.float32)
    dt = 1.0 / float(fps)
    l_vel = np.linalg.norm(np.gradient(l_toe, dt, axis=0), axis=1)
    r_vel = np.linalg.norm(np.gradient(r_toe, dt, axis=0), axis=1)

    # Per-motion floor reference: raw SMPL-X z is not anchored to z=0 (AMASS
    # sequences and GVHMR's post-rotation both shift it), so compare against
    # the lowest observed toe.
    floor_z = float(min(l_toe[:, 2].min(), r_toe[:, 2].min()))
    l_stance = ((l_toe[:, 2] - floor_z) < z_thresh) & (l_vel < v_thresh)
    r_stance = ((r_toe[:, 2] - floor_z) < z_thresh) & (r_vel < v_thresh)
    flags = np.stack([l_stance, r_stance], axis=1).astype(bool)

    info = {
        "floor_z": floor_z,
        "l_count": int(l_stance.sum()),
        "r_count": int(r_stance.sum()),
        "n_frames": n,
        "z_thresh": float(z_thresh),
        "v_thresh": float(v_thresh),
    }
    return flags, info
