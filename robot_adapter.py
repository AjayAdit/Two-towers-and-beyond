"""
RobotAdapter: a thin adapter around a Genesis robot entity.

Features:
  • Transparent pass-through to underlying robot
  • OMPL motion planning bridge via PlannerInterface
  • move_to_pose(..., ignore_collisions=False)
  • attach_object / detach_object + auto hand-follow
  • pick(pos, quat, obj=None) / place(pos, quat, obj=None)
"""

from typing import Any, Optional
import numpy as np

from genesis.utils.misc import tensor_to_array  # robust tensor → np converter


def _to_np(x) -> np.ndarray:
    """Safely convert Genesis / torch tensors or arrays to np.ndarray."""
    try:
        return tensor_to_array(x)
    except Exception:
        return np.asarray(x, dtype=float)


class RobotAdapter:
    def __init__(self, robot: Any, scene: Any = None):
        """Wrap a genesis robot entity.

        Args:
            robot: the raw genesis robot entity (e.g., returned from scene.add_entity)
            scene: optional scene reference (some callers use scene alongside robot)
        """
        self.robot = robot
        self.scene = scene

        # --- grasp/attachment state ---
        self.attached_object: Optional[Any] = None
        self._attach_offset: Optional[np.ndarray] = None  # obj_pos - hand_pos
        self._sync_cb = None

        # For fallback mode: ensure we only wrap scene.step() ONCE
        self._step_wrapped = False

        # ------------------------------------------------------------------
        # Hand → gripper-center offset along world Z.
        # Tune this so that when the hand link is at:
        #   hand_pos = block_center + [0, 0, grip_center_offset_z]
        # the block sits exactly in the middle of the jaws.
        # ------------------------------------------------------------------
        self.grip_center_offset_z = 0.12  # meters (INITIAL GUESS; TUNE IN SIM)

    # ------------------------------------------------------------------
    # Transparent forwarding
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute access to the underlying robot."""
        return getattr(self.robot, name)

    # Convenience wrappers (stable names)
    def get_pos(self):
        return self.robot.get_pos()

    def set_pos(self, pos):
        return self.robot.set_pos(pos)

    def get_qpos(self):
        return self.robot.get_qpos()

    def set_qpos(self, qpos):
        return self.robot.set_qpos(qpos)

    def control_dofs_position(self, *a, **kw):
        return self.robot.control_dofs_position(*a, **kw)

    def control_dofs_force(self, *a, **kw):
        return self.robot.control_dofs_force(*a, **kw)

    def get_link(self, *a, **kw):
        return self.robot.get_link(*a, **kw)

    def inverse_kinematics(self, *a, **kw):
        return self.robot.inverse_kinematics(*a, **kw)

    def detect_collision(self, *a, **kw):
        return self.robot.detect_collision(*a, **kw)

    # ------------------------------------------------------------------
    # Planning interface bridge
    # ------------------------------------------------------------------
    def plan_path(self, *args, **kwargs):
        """
        Use PlannerInterface (planning.py) for motion planning.
        """
        from planning import PlannerInterface

        print("[DEBUG][RobotAdapter] plan_path() called")
        planner_interface = PlannerInterface(self.robot, self.scene)
        path = planner_interface.plan_path(*args, **kwargs)
        print(f"[DEBUG][RobotAdapter] OMPL returned {len(path)} waypoints")
        return path

    # ------------------------------------------------------------------
    # Debug utils
    # ------------------------------------------------------------------
    def print_ee_pose(self, string_label: str = ""):
        """Print current end-effector (EE) position and orientation."""
        ee_link = self.get_link("hand")
        ee_pos = _to_np(ee_link.get_pos())
        ee_quat = _to_np(ee_link.get_quat())
        print(f"\n[POSE] {string_label}")
        print(f"  Position : {np.round(ee_pos, 4)}")
        print(f"  Quaternion: {np.round(ee_quat, 4)}")

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------
    def open_gripper(self, open_pos: float = 0.04, steps: int = 10):
        """Open parallel gripper (assumes last 2 DOFs)."""
        self.print_ee_pose("Before open_gripper()")
        qpos = _to_np(self.get_qpos())
        qpos[-2:] = open_pos
        self.control_dofs_position(qpos)
        if self.scene is not None:
            for _ in range(steps):
                self.scene.step()
        self.print_ee_pose("After open_gripper()")

    def close_gripper(self, close_pos: float = 0.0, steps: int = 10):
        """Close parallel gripper (assumes last 2 DOFs)."""
        self.print_ee_pose("Before close_gripper()")
        qpos = _to_np(self.get_qpos())
        qpos[-2:] = close_pos
        self.control_dofs_position(qpos)
        if self.scene is not None:
            for _ in range(steps):
                self.scene.step()
        self.print_ee_pose("After close_gripper()")

    # ------------------------------------------------------------------
    # ATTACH / DETACH (virtual, version-agnostic)
    # ------------------------------------------------------------------
    def attach_object(self, obj: Any):
        """Save full hand→obj offset and start kinematic sync."""
        self.attached_object = obj

        hand = self.get_link("hand")
        hand_pos = _to_np(hand.get_pos())
        obj_pos = _to_np(obj.get_pos())

        # Full 3D offset from hand to object center
        offset = obj_pos - hand_pos
        self._attach_offset = offset.copy()

        self._enable_sync()
        print("[ROBOT][ATTACH] attached; offset =", np.round(self._attach_offset, 4))

    def detach_object(self):
        """Stop kinematic sync and clear state."""
        self._disable_sync()
        self.attached_object = None
        self._attach_offset = None
        print("[ROBOT][DETACH] detached")

    def _sync_attached_object(self):
        """Keep the attached object rigidly at the hand each sim step."""
        if self.attached_object is None or self._attach_offset is None:
            return

        hand = self.get_link("hand")
        hand_pos = _to_np(hand.get_pos())
        hand_quat = _to_np(hand.get_quat())

        target_pos = hand_pos + self._attach_offset
        self.attached_object.set_pos(target_pos)
        self.attached_object.set_quat(hand_quat)

    def _enable_sync(self):
        """Register a post-step callback (idempotent)."""
        if self._sync_cb is not None:
            return

        if self.scene is None:
            print("[WARN] RobotAdapter._enable_sync(): scene is None, cannot sync.")
            self._sync_cb = None
            return

        # Preferred: real callbacks if Genesis supports them
        if hasattr(self.scene, "add_post_step_callback"):
            self._sync_cb = self.scene.add_post_step_callback(
                self._sync_attached_object
            )
        elif hasattr(self.scene, "add_post_step_cb"):
            self._sync_cb = self.scene.add_post_step_cb(self._sync_attached_object)
        else:
            # Fallback: wrap scene.step(), but only ONCE
            if not self._step_wrapped:
                orig_step = self.scene.step

                def step_wrapper(*args, **kwargs):
                    r = orig_step(*args, **kwargs)
                    self._sync_attached_object()
                    return r

                self.scene.step = step_wrapper
                self._step_wrapped = True

            # Using sentinel to indicate wrapper mode
            self._sync_cb = "wrapped"

    def _disable_sync(self):
        """Unregister post-step callback (idempotent)."""
        if self._sync_cb is None:
            return

        if self._sync_cb != "wrapped":
            if hasattr(self.scene, "remove_post_step_callback"):
                self.scene.remove_post_step_callback(self._sync_cb)
            elif hasattr(self.scene, "remove_post_step_cb"):
                self.scene.remove_post_step_cb(self._sync_cb)

        # For the wrapper case, we DO NOT unwrap scene.step.
        # We just clear state; _sync_attached_object() is a no-op when nothing attached.
        self._sync_cb = None

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def move_to_pose(
        self,
        qpos_goal,
        steps: int = 25,
        ignore_collisions: bool = False,
    ):
        """Plan and execute path using OMPL planner."""
        self.print_ee_pose("Before move_to_pose()")

        qpos_start = _to_np(self.get_qpos())
        print(f"[DEBUG] attached_object: {self.attached_object}")
        print(
            f"[DEBUG] move_to_pose(): ignore_collisions={ignore_collisions}"
        )

        if ignore_collisions:
            print("[DEBUG] Ignoring collisions during motion planning (explicit flag).")
            from planning import PlannerInterface

            planner_interface = PlannerInterface(self.robot, self.scene)
            path = planner_interface.plan_path(
                qpos_goal=qpos_goal,
                qpos_start=qpos_start,
                num_waypoints=steps,
                attached_object=self.attached_object,
            )
        else:
            path = self.plan_path(
                qpos_goal=qpos_goal,
                num_waypoints=steps,
                attached_object=self.attached_object,
            )

        if len(path) == 0:
            print(
                "[WARN] OMPL returned empty path; executing direct control to goal."
            )
            self.control_dofs_position(qpos_goal)
            if self.scene is not None:
                for _ in range(100):
                    self.scene.step()
            return

        for waypoint in path:
            self.control_dofs_position(waypoint)
            if self.scene is not None:
                self.scene.step()

        self.print_ee_pose("After move_to_pose()")

    def _servo_to_q(self, q_goal, steps: int = 80):
        """Simple joint-space interpolation from current q to q_goal."""
        q_start = _to_np(self.get_qpos())
        q_goal = _to_np(q_goal)

        for alpha in np.linspace(0.0, 1.0, steps):
            q = (1.0 - alpha) * q_start + alpha * q_goal
            self.control_dofs_position(q)
            if self.scene is not None:
                self.scene.step()

    def pre_grasp_pose(
        self,
        pos,
        quat,
        approach_dir: np.ndarray = np.array([0.0, 0.0, 1.0]),
        offset: float = 0.10,
        steps: int = 60,
    ):
        """
        Move the end-effector to a pre-grasp position offset from the given pos
        along the specified approach direction.

        NOTE: Here `pos` is interpreted as a target hand position (not block center).
        Uses local joint-space servo (no OMPL) for this short motion.
        """
        pos = _to_np(pos)
        approach_dir = _to_np(approach_dir)
        self.print_ee_pose("Before pre_grasp_pose()")

        pregrasp_pos = pos + approach_dir * offset
        print(f"[DEBUG] Pre-grasp hand position: {pregrasp_pos}")

        qpos_pregrasp = self.inverse_kinematics(
            link=self.get_link("hand"),
            pos=pregrasp_pos,
            quat=quat,
        )
        self._servo_to_q(qpos_pregrasp, steps=steps)

        self.print_ee_pose("After pre_grasp_pose()")

    def post_grasp_pose(
        self,
        pos,
        quat,
        approach_dir: np.ndarray = np.array([0.0, 0.0, -1.0]),
        offset: float = 0.10,
        steps: int = 60,
    ):
        """
        Move the end-effector away from a grasped pose (opposite of approach_dir).

        NOTE: Here `pos` is interpreted as a target hand position (not block center).
        Uses local joint-space servo (no OMPL).
        """
        pos = _to_np(pos)
        approach_dir = _to_np(approach_dir)
        self.print_ee_pose("Before post_grasp_pose()")

        retreat_pos = pos - approach_dir * offset
        print(f"[DEBUG] Post-grasp hand position: {retreat_pos}")

        qpos_postgrasp = self.inverse_kinematics(
            link=self.get_link("hand"),
            pos=retreat_pos,
            quat=quat,
        )
        self._servo_to_q(qpos_postgrasp, steps=steps)

        self.print_ee_pose("After post_grasp_pose()")

    # ------------------------------------------------------------------
    # High-level actions (obj is optional for backward compatibility)
    # ------------------------------------------------------------------
    def pick(
        self,
        pos,
        quat: np.ndarray = np.array([0.0, 1.0, 0.0, 0.0]),
        obj: Optional[Any] = None,
    ):
        """
        PICK SEQUENCE (NO OMPL BETWEEN PRE-GRASP / GRASP / RETREAT)

        Steps:
          1) Pre-grasp above block center
          2) Open gripper
          3) Descend so gripper center aligns with block center
          4) Close gripper
          5) Attach object (if given)
          6) Retreat upward

        `pos` is the desired *block center* world position (nominal COM).
        This function computes the hand pose so that the gripper center
        coincides with the block center, so the block sits in the middle
        of the jaws.
        """
        # pos = block center (do NOT mutate it)
        block_center = _to_np(pos).copy()

        # Where the HAND should be so that gripper center ≈ block center
        hand_at_center = block_center.copy()
        hand_at_center[2] += self.grip_center_offset_z

        self.print_ee_pose("Before pick()")

        # 1) Pre-grasp ABOVE the centered hand pose (local, no OMPL)
        pregrasp_target = hand_at_center.copy()
        pregrasp_target[2] += 0.10  # 10 cm above grasp
        print(f"[DEBUG] Pre-grasp hand target: {pregrasp_target}")

        qpos_pregrasp = self.inverse_kinematics(
            link=self.get_link("hand"),
            pos=pregrasp_target,
            quat=quat,
        )
        self._servo_to_q(qpos_pregrasp, steps=60)

        # 2) Open gripper
        self.open_gripper()

        # 3) Move hand so that gripper center lines up with block center
        qpos_grasp = self.inverse_kinematics(
            link=self.get_link("hand"),
            pos=hand_at_center,
            quat=quat,
        )
        print(f"[DEBUG] Grasp hand target (centered on block): {hand_at_center}")
        self._servo_to_q(qpos_grasp, steps=60)

        # 4) Close gripper (should now be around block center)
        self.close_gripper()

        # 5) Attach if provided (we'll measure the actual offset here)
        if obj is not None:
            self.attach_object(obj)

        # 6) Retreat straight up from the centered grasp (local, no OMPL)
        retreat_target = hand_at_center + np.array([0.0, 0.0, 0.15])
        qpos_retreat = self.inverse_kinematics(
            link=self.get_link("hand"),
            pos=retreat_target,
            quat=quat,
        )
        print("[DEBUG] Retreat target (from centered grasp):", retreat_target)
        self._servo_to_q(qpos_retreat, steps=60)

        self.print_ee_pose("After pick()")

    def place(
        self,
        pos,
        quat: np.ndarray = np.array([0.0, 1.0, 0.0, 0.0]),
        obj: Optional[Any] = None,
        hover_height: float = 0.08,
        release_clearance: float = 0.004,   # release a few mm above target
    ):
        """
        PLACE SEQUENCE

        `pos` is the desired object center at the final stack position.

        Steps:
          1) Move hand to hover above target (via OMPL)
          2) Iterative XY correction using PD control on object center feedback
          3) Descend to a release height slightly above final Z (still attached)
          4) Detach + open gripper + settling time (block drops a few mm)
          5) Retreat upward (via OMPL)
        """
        self.print_ee_pose("Before place()")
        pos = _to_np(pos).copy()

        # === 1) target object center hover above final stack position ===
        pos_above_obj = pos.copy()
        pos_above_obj[2] += hover_height

        # If we have an attached object, we know the offset between hand and object center
        off = np.zeros(3, dtype=float)
        if self.attached_object is not None and self._attach_offset is not None:
            off = self._attach_offset.astype(float)

        print("[DEBUG] attach offset:", np.round(off, 4))
        print("[DEBUG] desired object pos:", np.round(pos, 4))
        print("[DEBUG] desired object pos above:", np.round(pos_above_obj, 4))

        # Where the HAND should be if object center = pos_above_obj
        target_hand_pos_above = pos_above_obj - off

        # === 2) Move to hover pose via OMPL (coarse motion) ===
        q_above = self.inverse_kinematics(
            link=self.get_link("hand"),
            pos=target_hand_pos_above,
            quat=quat,
        )
        print("[DEBUG] Initial hover qpos:", q_above)
        self.move_to_pose(q_above, steps=30)
        self.print_ee_pose("[DEBUG] After initial hover move")

        # === 3) Iterative XY correction (single PD stage, local servo) ===
        tol_xy = 2e-3      # 2 mm target
        max_iters = 10
        stall_eps = 1e-6
        k_p = 1.5
        k_d = 0.18
        step_max = 7e-1    # up to 7 mm per correction

        prev_err_xy = None
        prev_err_norm = None

        for it in range(max_iters):
            hand = self.get_link("hand")
            ee_pos = _to_np(hand.get_pos())

            if self.attached_object is not None:
                obj_pos = _to_np(self.attached_object.get_pos())
            else:
                obj_pos = ee_pos + off

            target_xy = pos[:2]
            err_xy = target_xy - obj_pos[:2]
            err_norm = float(np.linalg.norm(err_xy))

            print(
                f"[CORRECTION {it}] XY error = "
                f"({err_xy[0]:+.6f}, {err_xy[1]:+.6f}), "
                f"‖err‖ = {err_norm:.6f} m"
            )

            if err_norm < tol_xy:
                print(f"[CORRECTION] Reached tolerance {tol_xy} m, stopping.")
                break

            if prev_err_norm is not None and abs(prev_err_norm - err_norm) < stall_eps:
                print(
                    f"[CORRECTION] Stalled at ~{err_norm:.6f} m "
                    f"(Δ‖err‖ < {stall_eps}), breaking."
                )
                break

            if prev_err_xy is None:
                derr_xy = np.zeros_like(err_xy)
            else:
                derr_xy = err_xy - prev_err_xy

            prev_err_xy = err_xy.copy()
            prev_err_norm = err_norm

            u_xy = k_p * err_xy + k_d * derr_xy
            dx = float(u_xy[0])
            dy = float(u_xy[1])

            dx = float(np.clip(dx, -step_max, step_max))
            dy = float(np.clip(dy, -step_max, step_max))

            print(
                f"[CORRECTION {it}] Applying Δhand = "
                f"({dx:+.6f}, {dy:+.6f})"
            )

            new_hand_pos = ee_pos.copy()
            new_hand_pos[0] += dx
            new_hand_pos[1] += dy
            new_hand_pos[2] = target_hand_pos_above[2]

            q_fix = self.inverse_kinematics(
                link=hand,
                pos=new_hand_pos,
                quat=quat,
            )
            # local servo, no OMPL
            self._servo_to_q(q_fix, steps=30)

        # === 4) Descend to *release* height while still attached ===
        # We want the block center slightly above the nominal target by release_clearance
        final_obj_pos = pos.copy()
        final_obj_pos[2] += release_clearance   # release higher than nominal
        final_hand_pos = final_obj_pos - off

        q_final = self.inverse_kinematics(
            link=self.get_link("hand"),
            pos=final_hand_pos,
            quat=quat,
        )
        print("[DEBUG] Descend to release pose qpos:", q_final)
        # Slower descent for gentler approach on tall stacks
        self._servo_to_q(q_final, steps=50)
        self.print_ee_pose("[DEBUG] At release pose (attached)")

        # Detach first, then open gripper
        if obj is not None and obj is self.attached_object:
            self.detach_object()
        elif obj is None and self.attached_object is not None:
            self.detach_object()

        self.open_gripper()

        # Let block drop release_clearance and contacts settle
        if self.scene is not None:
            for _ in range(30):
                self.scene.step()

        # Log final object error but DO NOT snap/teleport
        try:
            if obj is not None:
                final_obj_pos_meas = _to_np(obj.get_pos())
                xy_err_vec = pos[:2] - final_obj_pos_meas[:2]
                xy_err_norm = float(np.linalg.norm(xy_err_vec))

                print(
                    "[DEBUG] Final placed object center:",
                    np.round(final_obj_pos_meas, 4),
                    "‖XY err‖ =",
                    f"{xy_err_norm:.6f} m",
                )
        except Exception as e:
            print("[DEBUG] Could not read final object pose:", e)

        # === 5) Retreat upward so we don't collide with the stack ===
        pos_retreat = pos.copy()
        pos_retreat[2] += 0.30
        q_retreat = self.inverse_kinematics(
            link=self.get_link("hand"),
            pos=pos_retreat,
            quat=quat,
        )
        print("[DEBUG] move_to_pose retreat called, pos_retreat:", pos_retreat)
        self.move_to_pose(q_retreat, steps=30)
        self.print_ee_pose("After place()")

    # ------------------------------------------------------------------
    # Raw object access
    # ------------------------------------------------------------------
    @property
    def raw(self):
        """Direct access to underlying Genesis robot entity."""
        return self.robot
