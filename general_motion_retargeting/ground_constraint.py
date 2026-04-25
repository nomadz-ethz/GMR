"""
Ground penetration prevention constraints for GMR.

Two modes:
  - GroundPlaneLimit: hard QP inequality constraint (add to retarget.ik_limits)
  - SoftGroundConstraint: soft repulsive mink tasks (requires <site> elements in MJCF)
  - GMRWithSoftGround: subclass of GeneralMotionRetargeting that injects soft constraint
    into the IK iteration loop
"""

import numpy as np
import mujoco as mj
import mink
from mink.limits.limit import Constraint, Limit


class GroundPlaneLimit(Limit):
    """Hard QP inequality: keeps sole contact points above the ground plane.

    For each active sole point at world position p(q) with z-coordinate z_p:
        Linearize: z_p + J_z · dq · dt >= ground_height + clearance
        QP form (G·dq <= h):  -J_z · dq <= gain * (z_p - ground_height - clearance) / dt

    Add an instance of this to retarget.ik_limits to activate the constraint.
    """

    def __init__(
        self,
        model: mj.MjModel,
        sole_contact_points: dict,
        ground_height: float = 0.0,
        clearance: float = 0.003,
        gain: float = 0.5,
        activation_distance: float = 0.05,
    ):
        """
        Args:
            model: MuJoCo model.
            sole_contact_points: {body_name: [[x, y, z], ...]} in body local frame.
            ground_height: z-coordinate of the ground plane (default 0.0).
            clearance: minimum clearance above ground in meters (default 3mm).
            gain: CBF-style correction per IK step (0 < gain <= 1).
                  gain=1.0 requests full correction in one step (may drift root upward);
                  gain=0.5 requests half correction per step, converges in ~3 iterations.
                  For motions that use offset_to_ground=True, this primarily handles
                  dynamic excursions and should be ≥0.5.
            activation_distance: only enforce constraint when point is within this
                distance above ground_height + clearance. Points higher are ignored.
        """
        self.model = model
        self.sole_contact_points = sole_contact_points
        self.ground_height = ground_height
        self.clearance = clearance
        self.gain = gain
        self.activation_distance = activation_distance
        self.nv = model.nv

        # Precompute body IDs
        self.body_ids = {}
        for body_name in sole_contact_points:
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
            if bid == -1:
                available = [
                    mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
                    for i in range(model.nbody)
                ]
                raise ValueError(
                    f"Body '{body_name}' not found in MuJoCo model. "
                    f"Available bodies: {available}"
                )
            self.body_ids[body_name] = bid

        # Preallocate Jacobian buffers to avoid per-iteration allocation
        self._jacp = np.zeros((3, self.nv))
        self._jacr = np.zeros((3, self.nv))

    def compute_qp_inequalities(self, configuration, dt: float) -> Constraint:
        """Compute ground-plane QP inequality rows.

        Returns:
            Constraint(G, h) where G·dq <= h enforces z_p >= ground_height + clearance,
            or Constraint() (inactive) if no sole points are near the ground.
        """
        data = configuration.data
        threshold = self.ground_height + self.clearance + self.activation_distance

        G_rows = []
        h_vals = []

        for body_name, local_points in self.sole_contact_points.items():
            body_id = self.body_ids[body_name]
            body_pos = data.xpos[body_id]          # (3,) world position
            body_rot = data.xmat[body_id].reshape(3, 3)  # world rotation matrix

            for local_pt in local_points:
                world_pt = body_rot @ np.array(local_pt, dtype=float) + body_pos
                z = world_pt[2]

                if z > threshold:
                    continue  # well above ground, no constraint needed

                # Compute translation Jacobian at this world point w.r.t. all DoFs
                self._jacp[:] = 0.0
                self._jacr[:] = 0.0
                mj.mj_jac(self.model, data, self._jacp, self._jacr, world_pt, body_id)
                J_z = self._jacp[2, :].copy()  # z-row; copy because buffer is reused

                # mink's QP variable is Δq (joint displacement), NOT velocity.
                # Constraint: -J_z · Δq <= gain * margin
                # where margin = z - ground_height - clearance.
                # When margin < 0 (penetrating): right side is negative, forcing
                # the foot to displace upward by at least gain*|margin| per IK step.
                # gain=1.0 → full correction per step; gain<1 → partial (CBF-style).
                # Do NOT divide by dt here — the QP variable is already displacement.
                margin = z - self.ground_height - self.clearance
                # gain=1.0 → full correction per step; gain<1 → partial (CBF-style).
                dynamic_gain = min(1.0, self.gain + 2.0 * abs(margin)) if margin < 0 else self.gain
                G_rows.append(-J_z)
                h_vals.append(dynamic_gain * margin)

        if not G_rows:
            return Constraint()

        return Constraint(
            G=np.stack(G_rows, axis=0),
            h=np.array(h_vals),
        )


class RootZLimit(Limit):
    """Caps upward root-z displacement per IK step when any sole is near the ground.

    Adds one QP row:  e_z · Δq ≤ max_dz  (free-joint z is DOF index 2).

    NOTE ON EFFECTIVENESS: In practice this constraint has limited effect on
    locomotion jitter because:
      1. Most root-z upward movement during heel-strike happens in tasks-1
         (pure-rotation solve), not tasks-2.  Placing this limit in ik_limits2
         means it never fires during the phase that causes the jump.
      2. Placing it in shared ik_limits (tasks-1 + tasks-2) causes QP infeasibility:
         GroundPlaneLimit requires root to go UP to keep soles above ground, while
         RootZLimit prevents upward movement beyond max_dz — these conflict.
      3. Per-step caps (3-5mm) are rarely binding because actual per-step movements
         are below the threshold; the 10-25mm frame-to-frame jumps accumulate across
         multiple IK iterations.

    For walk motions, QP alone (without this class) already reduces jitter vs vanilla.
    For run motions, the jitter increase comes from the foot-landing dynamics and
    is not addressable with per-step capping.

    Usage (if you still want to try it):
        retarget.ik_limits2.append(
            RootZLimit(retarget.model, sole_config, max_dz=0.003)
        )
    """

    def __init__(
        self,
        model: mj.MjModel,
        sole_contact_points: dict,
        ground_height: float = 0.0,
        clearance: float = 0.003,
        activation_distance: float = 0.05,
        max_dz: float = 0.003,
    ):
        """
        Args:
            model: MuJoCo model.
            sole_contact_points: {body_name: [[x,y,z], ...]} in body local frame
                (same dict as GroundPlaneLimit).
            ground_height: z-coordinate of the ground plane.
            clearance: clearance threshold (should match GroundPlaneLimit).
            activation_distance: cap is only active when at least one sole point
                is within this distance of ground_height + clearance.
            max_dz: maximum allowed upward root-z displacement per IK step (m).
                Typical value for walking: 0.003 (3 mm).
        """
        self.ground_height = ground_height
        self.clearance = clearance
        self.activation_distance = activation_distance
        self.max_dz = max_dz
        self.nv = model.nv

        self._body_ids = {}
        for body_name in sole_contact_points:
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
            if bid == -1:
                raise ValueError(f"Body '{body_name}' not found in MuJoCo model.")
            self._body_ids[body_name] = bid
        self._sole_contact_points = sole_contact_points

    def compute_qp_inequalities(self, configuration, dt: float) -> Constraint:
        data = configuration.data
        threshold = self.ground_height + self.clearance + self.activation_distance

        any_active = False
        for body_name, local_points in self._sole_contact_points.items():
            body_id = self._body_ids[body_name]
            body_pos = data.xpos[body_id]
            body_rot = data.xmat[body_id].reshape(3, 3)
            for local_pt in local_points:
                world_z = (body_rot @ np.array(local_pt, dtype=float) + body_pos)[2]
                if world_z <= threshold:
                    any_active = True
                    break
            if any_active:
                break

        if not any_active:
            return Constraint()

        e_z = np.zeros(self.nv)
        e_z[2] = 1.0
        return Constraint(G=e_z[np.newaxis], h=np.array([self.max_dz]))


class SoftGroundConstraint:
    """Soft repulsive tasks: pushes sole sites above the ground plane.

    Requires <site> elements at sole positions in the MJCF.
    Run scripts/add_sole_sites.py to generate K1_serial_with_sites.xml first.

    The tasks are standard mink.FrameTask instances with dynamic weights.
    Call update(configuration) before each solve_ik call to refresh weights
    and targets based on the current configuration state.
    """

    def __init__(
        self,
        model: mj.MjModel,
        site_names: list,
        ground_height: float = 0.0,
        clearance: float = 0.003,
        max_weight: float = 500.0,
        activation_distance: float = 0.02,
    ):
        """
        Args:
            model: MuJoCo model (must contain the named sites).
            site_names: list of site names (e.g. ["left_sole_fl", ...]).
            ground_height: z-coordinate of the ground plane.
            clearance: minimum clearance above ground.
            max_weight: maximum task weight at ground level.
            activation_distance: weight linearly ramps from 0 to max_weight
                as site z drops from (ground_height + clearance + activation_distance)
                down to (ground_height + clearance).
        """
        self.ground_height = ground_height
        self.clearance = clearance
        self.max_weight = max_weight
        self.activation_distance = activation_distance

        self.site_ids = []
        self.tasks = []

        for name in site_names:
            sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
            if sid == -1:
                raise ValueError(
                    f"Site '{name}' not found in MuJoCo model. "
                    f"Run 'python scripts/add_sole_sites.py' to inject site elements "
                    f"into the MJCF, then use K1_serial_with_sites.xml."
                )
            self.site_ids.append(sid)

            task = mink.FrameTask(
                frame_name=name,
                frame_type="site",
                position_cost=0.0,      # starts inactive; updated dynamically
                orientation_cost=0.0,
                lm_damping=1.0,
            )
            self.tasks.append(task)

    def update(self, configuration):
        """Refresh task weights and targets based on current configuration.

        Call this before each solve_ik inside the IK iteration loop.
        """
        data = configuration.data
        target_z = self.ground_height + self.clearance
        threshold = target_z + self.activation_distance

        for site_id, task in zip(self.site_ids, self.tasks):
            z = data.site_xpos[site_id][2]

            if z >= threshold:
                task.position_cost = 0.0  # deactivate: site is well above ground
                continue

            # Linear ramp: full weight at z == target_z, zero weight at z == threshold
            alpha = 1.0 - max(0.0, z - target_z) / self.activation_distance
            alpha = float(np.clip(alpha, 0.0, 1.0))
            task.position_cost = alpha * self.max_weight

            # Push site up to target_z
            target_pos = data.site_xpos[site_id].copy()
            target_pos[2] = target_z
            task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(np.array([1.0, 0.0, 0.0, 0.0])),  # identity wxyz
                    target_pos,
                )
            )


class GMRWithSoftGround:
    """GeneralMotionRetargeting with soft ground constraint injected into the IK loop.

    This is a composition wrapper (not a subclass) that replicates the retarget()
    loop from GeneralMotionRetargeting and injects soft_ground.update() before
    each solve_ik call.

    Usage:
        from general_motion_retargeting.ground_constraint import GMRWithSoftGround
        retarget = GMRWithSoftGround(
            site_names=get_sole_site_names("booster_k1"),
            ground_height=0.0,
            clearance=0.003,
            max_weight=500.0,
            activation_distance=0.02,
            src_human="smplx",
            tgt_robot="booster_k1",
            actual_human_height=1.7,
        )
    """

    def __init__(
        self,
        site_names: list,
        ground_height: float = 0.0,
        clearance: float = 0.003,
        max_weight: float = 500.0,
        activation_distance: float = 0.02,
        **gmr_kwargs,
    ):
        """
        Args:
            site_names: site names for SoftGroundConstraint.
            ground_height: z of ground plane.
            clearance: minimum clearance.
            max_weight: maximum repulsive task weight.
            activation_distance: activation zone above clearance.
            **gmr_kwargs: all keyword arguments passed to GeneralMotionRetargeting
                (src_human, tgt_robot, actual_human_height, solver, damping, verbose).
        """
        from .motion_retarget import GeneralMotionRetargeting

        self._gmr = GeneralMotionRetargeting(**gmr_kwargs)

        self.soft_ground = SoftGroundConstraint(
            model=self._gmr.model,
            site_names=site_names,
            ground_height=ground_height,
            clearance=clearance,
            max_weight=max_weight,
            activation_distance=activation_distance,
        )

        # Add soft ground tasks to tasks2 (where foot tracking lives)
        self._gmr.tasks2.extend(self.soft_ground.tasks)

        # Expose attributes used externally
        self.model = self._gmr.model
        self.scaled_human_data = None

    def retarget(self, human_data, offset_to_ground=False):
        """Retarget with soft ground tasks injected into the IK loop.

        Preserves the exact convergence logic from GeneralMotionRetargeting.retarget()
        and inserts soft_ground.update() before every solve_ik call.
        """
        gmr = self._gmr
        gmr.update_targets(human_data, offset_to_ground)
        self.scaled_human_data = gmr.scaled_human_data

        if gmr.use_ik_match_table1:
            curr_error = gmr.error1()
            dt = gmr.configuration.model.opt.timestep
            self.soft_ground.update(gmr.configuration)
            vel1 = mink.solve_ik(
                gmr.configuration, gmr.tasks1, dt, gmr.solver, gmr.damping, limits=gmr.ik_limits
            )
            gmr.configuration.integrate_inplace(vel1, dt)
            next_error = gmr.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < gmr.max_iter:
                curr_error = next_error
                dt = gmr.configuration.model.opt.timestep
                self.soft_ground.update(gmr.configuration)
                vel1 = mink.solve_ik(
                    gmr.configuration, gmr.tasks1, dt, gmr.solver, gmr.damping, gmr.ik_limits
                )
                gmr.configuration.integrate_inplace(vel1, dt)
                next_error = gmr.error1()
                num_iter += 1

        if gmr.use_ik_match_table2:
            curr_error = gmr.error2()
            dt = gmr.configuration.model.opt.timestep
            self.soft_ground.update(gmr.configuration)
            vel2 = mink.solve_ik(
                gmr.configuration, gmr.tasks2, dt, gmr.solver, gmr.damping, limits=gmr.ik_limits
            )
            gmr.configuration.integrate_inplace(vel2, dt)
            next_error = gmr.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < gmr.max_iter:
                curr_error = next_error
                dt = gmr.configuration.model.opt.timestep
                self.soft_ground.update(gmr.configuration)
                vel2 = mink.solve_ik(
                    gmr.configuration, gmr.tasks2, dt, gmr.solver, gmr.damping, gmr.ik_limits
                )
                gmr.configuration.integrate_inplace(vel2, dt)
                next_error = gmr.error2()
                num_iter += 1

        return gmr.configuration.data.qpos.copy()

    def set_ground_offset(self, offset):
        self._gmr.set_ground_offset(offset)
