"""
Sole contact point definitions for ground-plane constraints.
Currently only Booster K1 and T1 are defined. Add other robots by extending the dicts.

Points are in the foot link's LOCAL frame.
z-values are negative (below the ankle/link origin = the sole surface).

For K1: foot box geom size=[0.08, 0.035, 0.016] (half-extents), pos=[0.014, 0.0, -0.008]
Sole surface z = -0.008 - 0.016 = -0.024 in foot local frame.
x range: 0.014 ± 0.08 → [0.094, -0.066]; y range: 0.0 ± 0.035 → [0.035, -0.035]
"""

import xml.etree.ElementTree as ET
import numpy as np

SOLE_CONTACT_POINTS = {
    "booster_k1": {
        "left_foot_link": [
            [0.094, 0.035, -0.024],   # front-left  (fl)
            [0.094, -0.035, -0.024],  # front-right (fr)
            [-0.066, 0.035, -0.024],  # rear-left   (rl)
            [-0.066, -0.035, -0.024], # rear-right  (rr)
        ],
        "right_foot_link": [
            [0.094, 0.035, -0.024],
            [0.094, -0.035, -0.024],
            [-0.066, 0.035, -0.024],
            [-0.066, -0.035, -0.024],
        ],
    },
    # For T1: each foot has two collision capsules (size="0.02 0.0915",
    # axis along local x via quat), centers at (0.01, ±0.035, -0.01).
    # Sole z = -0.01 - 0.02 = -0.03.
    # x range: 0.01 ± 0.0915 → [-0.0815, 0.1015]
    # y outer edges: ±(0.035 + 0.02) = ±0.055
    "booster_t1": {
        "left_foot_link": [
            [0.1015, 0.055, -0.03],   # fl
            [0.1015, -0.055, -0.03],  # fr
            [-0.0815, 0.055, -0.03],  # rl
            [-0.0815, -0.055, -0.03], # rr
        ],
        "right_foot_link": [
            [0.1015, 0.055, -0.03],
            [0.1015, -0.055, -0.03],
            [-0.0815, 0.055, -0.03],
            [-0.0815, -0.055, -0.03],
        ],
    },
}

FOOT_LINK_NAMES = {
    "booster_k1": ["left_foot_link", "right_foot_link"],
    "booster_t1": ["left_foot_link", "right_foot_link"],
}

_CORNER_NAMES = ["fl", "fr", "rl", "rr"]


def get_sole_points(robot_type: str) -> dict:
    """Return sole contact points dict for the given robot type.

    Returns:
        {body_name: [[x, y, z], ...], ...}

    Raises:
        ValueError: if robot_type is not in SOLE_CONTACT_POINTS.
    """
    if robot_type not in SOLE_CONTACT_POINTS:
        available = list(SOLE_CONTACT_POINTS.keys())
        raise ValueError(
            f"No sole contact points defined for robot '{robot_type}'. "
            f"Available robots: {available}"
        )
    return SOLE_CONTACT_POINTS[robot_type]


def get_foot_link_names(robot_type: str):
    """Return list of foot link body names for the given robot type, or None."""
    return FOOT_LINK_NAMES.get(robot_type, None)


def get_sole_site_names(robot_type: str) -> list:
    """Generate site names for sole contact points (used in soft mode).

    Convention: "{side}_sole_{corner}" where side is 'left'/'right'
    and corner is 'fl'/'fr'/'rl'/'rr'.

    Returns:
        list of site name strings, e.g. ["left_sole_fl", ..., "right_sole_rr"]
    """
    sole_config = get_sole_points(robot_type)
    site_names = []
    for body_name, points in sole_config.items():
        if "left" in body_name:
            side = "left"
        elif "right" in body_name:
            side = "right"
        else:
            side = body_name
        for i, _ in enumerate(points):
            corner = _CORNER_NAMES[i] if i < len(_CORNER_NAMES) else str(i)
            site_names.append(f"{side}_sole_{corner}")
    return site_names


def detect_sole_points_from_mjcf(xml_path: str, foot_body_names: list) -> dict:
    """Best-effort auto-detection of sole contact points from MJCF collision geoms.

    Parses XML, finds <geom> children of foot bodies, computes corner points
    from geom type/size/pos. Useful for calibrating placeholder values.

    Returns:
        {body_name: [[x, y, z], ...], ...}
    """
    import mujoco as mj

    model = mj.MjModel.from_xml_path(xml_path)
    result = {}

    for body_name in foot_body_names:
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            print(f"[warn] Body '{body_name}' not found in {xml_path}")
            continue

        corners = []
        for g in range(model.body_geomadr[body_id],
                       model.body_geomadr[body_id] + model.body_geomnum[body_id]):
            gtype = model.geom_type[g]
            gsize = model.geom_size[g]
            gpos = model.geom_pos[g]

            # Box geom (type 6 = mjGEOM_BOX): size = [hx, hy, hz]
            if gtype == 6:
                hx, hy, hz = gsize[0], gsize[1], gsize[2]
                cx, cy, cz = gpos[0], gpos[1], gpos[2]
                # Four bottom corners (min z)
                z_sole = cz - hz
                corners = [
                    [cx + hx, cy + hy, z_sole],
                    [cx + hx, cy - hy, z_sole],
                    [cx - hx, cy + hy, z_sole],
                    [cx - hx, cy - hy, z_sole],
                ]
                break  # use first box geom found

        if not corners:
            print(f"[warn] No box geom found for '{body_name}', skipping")

        result[body_name] = corners

    return result
