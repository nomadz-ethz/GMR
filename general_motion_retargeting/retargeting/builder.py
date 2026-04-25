"""Build a `GeneralMotionRetargeting` (or `GMRWithSoftGround`) instance with
the correct ground constraint wired in and the sole-compensation `ground_offset`
already applied.

Returns `(retargeter, baseline_compensation)`. `baseline_compensation` is the
negative-z target shift the caller can later perturb (e.g. by per-frame
`final_depths[i]`) when running the strict-zero-penetration two-pass.

`retargeter` is `None` if soft mode was requested but the
`<robot>_with_sites.xml` is missing — this is reported to stdout and the
caller is expected to bail out.
"""

import os
import pathlib

from rich import print


def build_retargeter(
    robot_type: str,
    *,
    ground_mode: str,
    sole_config: dict,
    actual_human_height: float,
    src_human: str = "smplx",
    gain: float = 0.5,
    ground_height: float = 0.0,
    clearance: float = 0.003,
    activation_distance: float = 0.02,
    max_weight: float = 500.0,
):
    """Construct an IK-ready retargeter with the requested ground constraint.

    Args:
        robot_type: e.g. "booster_k1", "booster_t1".
        ground_mode: "none" | "qp" | "soft".
        sole_config: output of `general_motion_retargeting.sole_points.get_sole_points`.
        actual_human_height: source human height in metres.
        src_human: IK config bucket name. SMPL-X covers AMASS / OMOMO / GVHMR.
        gain, ground_height, clearance, activation_distance, max_weight:
            constraint-mode parameters.

    Returns:
        (retargeter, baseline_compensation).
    """
    if ground_mode == "soft":
        from ..ground_constraint import GMRWithSoftGround
        from ..sole_points import get_sole_site_names
        from .. import params as gmr_params

        original_xml = gmr_params.ROBOT_XML_DICT[robot_type]
        sites_xml = str(original_xml).replace(".xml", "_with_sites.xml")
        if not os.path.exists(sites_xml):
            print(
                f"[red]Soft mode requires {sites_xml}. "
                f"Run: python scripts/add_sole_sites.py[/red]"
            )
            return None, None

        gmr_params.ROBOT_XML_DICT[robot_type] = pathlib.Path(sites_xml)
        try:
            retgt = GMRWithSoftGround(
                site_names=get_sole_site_names(robot_type),
                ground_height=ground_height,
                clearance=clearance,
                max_weight=max_weight,
                activation_distance=activation_distance,
                actual_human_height=actual_human_height,
                src_human=src_human,
                tgt_robot=robot_type,
            )
        finally:
            gmr_params.ROBOT_XML_DICT[robot_type] = original_xml

    else:
        from ..motion_retarget import GeneralMotionRetargeting as GMR
        retgt = GMR(
            actual_human_height=actual_human_height,
            src_human=src_human,
            tgt_robot=robot_type,
        )

        if ground_mode == "qp":
            from ..ground_constraint import GroundPlaneLimit
            ground_limit = GroundPlaneLimit(
                model=retgt.model,
                sole_contact_points=sole_config,
                ground_height=ground_height,
                clearance=clearance,
                gain=gain,
                activation_distance=activation_distance,
            )
            retgt.ik_limits.append(ground_limit)

    # Sole compensation: the IK config maps human ankle to robot ankle (z=0),
    # but the sole surface is `min_sole_z` below the ankle. Without this shift
    # the robot's sole sits underground. set_ground_offset() raises every body
    # z target by `-baseline_compensation`.
    baseline_compensation = 0.0
    if ground_mode != "none":
        min_sole_z = min(pt[2] for pts in sole_config.values() for pt in pts)
        baseline_compensation = min_sole_z - clearance  # negative -> raises targets
        retgt.set_ground_offset(baseline_compensation)

    return retgt, baseline_compensation
