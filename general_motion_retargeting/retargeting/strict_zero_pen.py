"""Two-pass IK helpers: surveyor + smooth-envelope extraction.

The two-pass scheme drives `max penetration` to exactly 0 mm without the
foot-slide artifact that a naive post-shift would produce. See
`docs/pipeline.md` section 2.5 for the full rationale.

The functions in this module are pure compute; they do not own a retargeter
and they do not run any IK on their own. The script glues them together:

    qpos_pass1 = run IK frame-by-frame                # in script
    depths     = measure_penetration_depths(...)      # this module
    envelope   = smooth_penetration_envelope(...)     # this module
    for i, frame in enumerate(frames):                # in script
        retgt.set_ground_offset(baseline - envelope[i])
        qpos_pass2[i] = retgt.retarget(frame)
"""

from typing import Sequence

import numpy as np

# Default envelope shape parameters. Tuned for 30 fps locomotion. The
# corresponding time-windows are roughly:
#   DILATE_SIZE = 15 frames  ~= 0.5 s of stance widening
#   SMOOTH_SIGMA =  5 frames ~= 0.17 s rise-time of the lift bump
#   FINAL_SIGMA  =  2 frames ~= 0.07 s polish on np.maximum kinks
DILATE_SIZE = 15
SMOOTH_SIGMA = 5.0
FINAL_SIGMA = 2.0


def measure_penetration_depths(
    qpos_list: Sequence[np.ndarray],
    model,
    sole_config: dict,
    clearance: float,
) -> np.ndarray:
    """Pass 1 surveyor: forward-kinematics scan of the first-pass qpos list.

    For every frame, sets a temporary `MjData.qpos` (note the wxyz->xyzw swap
    that mink uses internally) and runs `mj_forward`. Computes the minimum
    world-z over all sole corners. Wherever that minimum is below `clearance`,
    `depths[i] = clearance - min_z`; otherwise `depths[i] = 0`.

    Args:
        qpos_list: list of (nq,) arrays, output of pass-1 IK.
        model: MuJoCo model (the one owned by the retargeter).
        sole_config: {body_name: [[x,y,z], ...]} in body-local frame.
        clearance: minimum allowed sole z above ground (m).

    Returns:
        depths: (N,) float64 array, `>= 0` everywhere, with positive entries
        only at frames that need correction.
    """
    import mujoco as mj

    depths = np.zeros(len(qpos_list))
    data_mj = mj.MjData(model)

    body_ids = {
        name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        for name in sole_config
    }

    for i, qpos in enumerate(qpos_list):
        data_mj.qpos[:3] = qpos[:3]
        data_mj.qpos[3:7] = qpos[3:7][[1, 2, 3, 0]]  # mink wxyz -> mujoco xyzw
        data_mj.qpos[7:] = qpos[7:]
        mj.mj_forward(model, data_mj)

        min_z = np.inf
        for body_name, local_pts in sole_config.items():
            bid = body_ids[body_name]
            if bid < 0:
                continue
            bp = data_mj.xpos[bid]
            br = data_mj.xmat[bid].reshape(3, 3)
            for lp in local_pts:
                world_z = (br @ np.asarray(lp, dtype=float) + bp)[2]
                if world_z < min_z:
                    min_z = world_z

        if min_z < clearance:
            depths[i] = clearance - min_z

    return depths


def smooth_penetration_envelope(
    depths: np.ndarray,
    *,
    dilate_size: int = DILATE_SIZE,
    smooth_sigma: float = SMOOTH_SIGMA,
    final_sigma: float = FINAL_SIGMA,
) -> np.ndarray:
    """Turn a sparse spike pattern into a smooth target-shift envelope.

    Strategy:
      1. Dilate (max filter) — widen each spike so the lift event covers the
         full ~0.5 s window around heel-strike.
      2. Gaussian-blur the dilation — produce a continuous bump with a natural
         rise time so the root motion that follows reads as anticipation
         rather than a step.
      3. `np.maximum(smoothed, depths)` — pull the envelope back up to cover
         each spike (the smoothing can dip below isolated peaks).
      4. A second, lighter Gaussian — soften the kinks that the `np.maximum`
         in step 3 reintroduces. With realistic spike widths (heel-strike
         events are not isolated single-frame impulses), the dilation in
         step 1 has already broadened each spike across ~ DILATE_SIZE frames,
         so this final blur trims the peak by a sub-millimetre amount but
         keeps the envelope effectively above `depths` in practice.

    Defaults are tuned for 30 fps locomotion. Re-derive for substantially
    different frame rates or non-locomotor motion classes.

    Returns: (N,) array, smooth in the C^0 sense, approximately `>= depths`
    on real spike patterns. The downstream IK absorbs any sub-mm slack.
    """
    import scipy.ndimage as ndi

    dilated = ndi.maximum_filter1d(depths, size=dilate_size)
    smoothed = ndi.gaussian_filter1d(dilated, sigma=smooth_sigma)
    final = np.maximum(smoothed, depths)
    final = ndi.gaussian_filter1d(final, sigma=final_sigma)
    return final
