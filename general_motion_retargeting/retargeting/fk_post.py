"""FK-based post-processing: optional height_adjust / root_origin_offset, plus
the always-computed `local_body_pos` (root pinned to origin, used by some
downstream RL pipelines).
"""

from typing import Optional, Tuple, List

import numpy as np

from rich import print


def apply_fk_post(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
    *,
    robot_type: str,
    height_adjust: bool,
    root_origin_offset: bool,
    ground_mode_active: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[str]]]:
    """Apply optional root shifts and compute local body positions.

    Args:
        root_pos: (N, 3) float32 -- mutated in place if shifts apply.
        root_rot: (N, 4) float32 xyzw.
        dof_pos:  (N, J) float32.
        robot_type: e.g. "booster_k1".
        height_adjust: shift root z so the lowest body sits at z=0.
            Silently skipped (with a warning) when ground_mode_active=True.
        root_origin_offset: subtract frame-0 XY from every frame's root XY.
        ground_mode_active: True if `--ground_mode != none`.

    Returns:
        (root_pos, local_body_pos, link_body_list).
        local_body_pos: (N, B, 3) float32 with root pinned at origin.
        link_body_list: list of body names matching the body axis.
    """
    import torch
    from ..kinematics_model import KinematicsModel
    from .. import ROBOT_XML_DICT

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    xml_file = str(ROBOT_XML_DICT[robot_type])
    km = KinematicsModel(xml_file, device=device)

    rp = torch.from_numpy(root_pos).to(device=device, dtype=torch.float)
    rr = torch.from_numpy(root_rot).to(device=device, dtype=torch.float)
    dp = torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)

    if height_adjust or root_origin_offset:
        body_pos, _ = km.forward_kinematics(rp, rr, dp)
        if height_adjust:
            if ground_mode_active:
                print(
                    "[yellow]--height_adjust skipped: incompatible with active "
                    "ground constraint. The ground constraint already positions "
                    "the robot; height_adjust would lower it back underground."
                    "[/yellow]"
                )
            else:
                lowest = torch.min(body_pos[..., 2]).item()
                root_pos[:, 2] -= lowest
                rp = torch.from_numpy(root_pos).to(device=device, dtype=torch.float)
        if root_origin_offset:
            root_pos[:, :2] -= root_pos[0, :2]
            rp = torch.from_numpy(root_pos).to(device=device, dtype=torch.float)

    # Local body positions (root pinned at origin -- shape/pose only).
    fk_root_pos = torch.zeros((root_pos.shape[0], 3), device=device)
    fk_root_rot = torch.zeros((root_pos.shape[0], 4), device=device)
    fk_root_rot[:, -1] = 1.0  # xyzw identity
    local_body_pos, _ = km.forward_kinematics(fk_root_pos, fk_root_rot, dp)
    local_body_pos = local_body_pos.detach().cpu().numpy()
    link_body_list = km.body_names

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return root_pos, local_body_pos, link_body_list
