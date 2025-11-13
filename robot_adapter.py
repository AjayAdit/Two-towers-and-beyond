"""
RobotAdapter: a thin adapter around a Genesis robot entity.

Now with:
  • move_to_pose(..., ignore_collisions=False)
  • attach_object / detach_object + auto hand-follow
  • pick(pos, quat, obj=None) / place(pos, quat, obj=None)
"""

from typing import Any, Optional
import time
import numpy as np
from genesis.utils.misc import tensor_to_array  # TODO: use this utility for conversions


class RobotAdapter:
    def __init__(self, robot: Any, scene: Any = None):
        """Wrap a genesis robot entity.

        Args:
            robot: raw genesis robot entity (e.g., scene.add_entity return)
            scene: optional scene reference
        """
        self.robot = robot
        self.scene = scene

        # --- grasp/attachment state ---
        self.attached_object: Optional[Any] = None
        self._attached_offset_world: Optional[np.ndarray] = None  # obj_pos - hand_pos

    # ------------------------------------------------------------------
    # Transparent forwarding
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(self.robot, name)

    # Convenience wrappers (stable names)
    def get_pos(self): return self.robot.get_pos()
    def set_pos(self, pos): return self.robot.set_pos(pos)
    def get_qpos(self): return self.robot.get_qpos()
    def set_qpos(self, qpos): return self.robot.set_qpos(qpos)
    def control_dofs_position(self, *a, **kw): return self.robot.control_dofs_position(*a, **kw)
    def control_dofs_force(self, *a, **kw): return self.robot.control_dofs_force(*a, **kw)
    def get_link(self, *a, **kw): return self.robot.get_link(*a, **kw)
    def inverse_kinematics(self, *a, **kw): return self.robot.inverse_kinematics(*a, **kw)
    def detect_collision(self, *a, **kw): return self.robot.detect_collision(*a, **kw)

    # ------------------------------------------------------------------
    # Planning interface bridge
    # ------------------------------------------------------------------
    def plan_path(self, *args, **kwargs):
        """
        Default path planning via planning.PlannerInterface
        """
        from planning import PlannerInterface
        print("[DEBUG][RobotAdapter] plan_path() called")
        planner_interface = PlannerInterface(self.robot, self.scene)
        print("[DEBUG][RobotAdapter] Forwarding to PlannerInterface.plan_path()")
        waypoints = planner_interface.plan_path(*args, **kwargs)
        print(f"[DEBUG][RobotAdapter] OMPL returned {len(waypoints)} waypoints")
        return waypoints

    # ------------------------------------------------------------------
    # Debug utils
    # ------------------------------------------------------------------
    def print_ee_pose(self, string_label=""):
        ee_link = self.get_link("hand")
        ee_pos = ee_link.get_pos().cpu().numpy()
        ee_quat = ee_link.get_quat().cpu().numpy()
        print(f"\n[POSE] {string_label}")
        print(f"  Position : {np.round(ee_pos, 4)}")
        print(f"  Quaternion: {np.round(ee_quat, 4)}")

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------
    def open_gripper(self, open_pos=0.04, steps=20):
        self.print_ee_pose("Before open_gripper()")
        qpos = self.get_qpos()
        qpos[-2:] = open_pos
        self.control_dofs_position(qpos)
        for _ in range(steps):
            self._sync_attached_object()
            self.scene.step()
        self.print_ee_pose("After open_gripper()")

    def close_gripper(self, close_pos=0.0, steps=20):
        self.print_ee_pose("Before close_gripper()")
        qpos = self.get_qpos()
        qpos[-2:] = close_pos
        self.control_dofs_position(qpos)
        for _ in range(steps):
            self._sync_attached_object()
            self.scene.step()
        self.print_ee_pose("After close_gripper()")

    # ------------------------------------------------------------------
    # ATTACH / DETACH (virtual, version-agnostic)
    # ------------------------------------------------------------------
    def attach_object(self, obj: Any):
        """
        Virtually attach an object to the gripper (hand). We store a world-space
        offset so we can keep the object aligned during motions.
        """
        hand = self.get_link("hand")
        hand_pos = hand.get_pos().cpu().numpy()
        obj_pos = obj.get_pos().cpu().numpy()

        self._attached_offset_world = obj_pos - hand_pos
        self.attached_object = obj

        # Soften collisions if API exists (varies by Genesis version)
        if hasattr(obj, "set_collision_enabled"):
            try: obj.set_collision_enabled(False)
            except Exception: pass
        elif hasattr(obj, "disable_collisions"):
            try: obj.disable_collisions()
            except Exception: pass
        elif hasattr(obj, "set_collision_mask"):
            try: obj.set_collision_mask(0)
            except Exception: pass

        print(f"[INFO] Attached {getattr(obj, 'name', 'object')} to gripper.")

    def detach_object(self):
        """
        Release the currently attached object, if any, and re-enable collisions.
        """
        obj = self.attached_object
        if obj is None:
            return

        # Re-enable collisions when possible
        if hasattr(obj, "set_collision_enabled"):
            try: obj.set_collision_enabled(True)
            except Exception: pass
        elif hasattr(obj, "enable_collisions"):
            try: obj.enable_collisions()
            except Exception: pass
        elif hasattr(obj, "set_collision_mask"):
            try: obj.set_collision_mask(1)
            except Exception: pass

        print(f"[INFO] Detached {getattr(obj, 'name', 'object')} from gripper.")
        self.attached_object = None
        self._attached_offset_world = None

    def _sync_attached_object(self):
        """
        If an object is attached, keep it aligned to the hand using the stored offset.
        This is called every simulation step while we are moving.
        """
        if self.attached_object is None or self._attached_offset_world is None:
            return
        hand = self.get_link("hand")
        hand_pos = hand.get_pos().cpu().numpy()
        target = hand_pos + self._attached_offset_world
        try:
            self.attached_object.set_pos(target)
        except Exception:
            # Some Genesis versions require set_pose or similar — fall back if needed
            if hasattr(self.attached_object, "set_pose"):
                self.attached_object.set_pose(pos=target)

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def move_to_pose(self, qpos_goal, steps=300, ignore_collisions=False):
        """Plan and execute path using OMPL planner."""
        self.print_ee_pose("Before move_to_pose()")
        qpos_start = self.get_qpos()

        # --- auto ignore heuristic ---
        fingers = qpos_start[-2:]
        if hasattr(fingers, "cpu"):  # convert torch.Tensor → numpy
            fingers = fingers.detach().cpu().numpy()
        gripper_closed = np.all(fingers < 0.01)

        auto_ignore = (getattr(self, "attached_object", None) is not None) or gripper_closed
        use_ignore = ignore_collisions or auto_ignore

        if use_ignore:
            print("[DEBUG] Ignoring collisions during motion planning (grasp context).")
            from planning import PlannerInterface
            planner_interface = PlannerInterface(self.robot, self.scene)
            planner_interface._is_ompl_state_valid = lambda s: True
            path = planner_interface.plan_path(
                qpos_goal=qpos_goal,
                qpos_start=qpos_start,
                num_waypoints=steps,
            )
        else:
            path = self.plan_path(qpos_goal=qpos_goal, num_waypoints=steps)

        if len(path) == 0:
            print("[WARN] OMPL returned empty path; executing direct control to goal.")
            self.control_dofs_position(qpos_goal)
            for _ in range(100):
                self._sync_attached_object()
                self.scene.step()
            return

        for waypoint in path:
            self.control_dofs_position(waypoint)
            self._sync_attached_object()
            self.scene.step()

        self.print_ee_pose("After move_to_pose()")
        time.sleep(0.1)


    def pre_grasp_pose(self, pos, quat, approach_dir=np.array([0, 0, 1]), offset=0.10, steps=300):
        """
        Move the end-effector to a pre-grasp position offset from the object
        along the specified approach direction.
        """
        self.print_ee_pose("Before pre_grasp_pose()")
        pregrasp_pos = pos + approach_dir * offset
        print(f"[DEBUG] Pre-grasp position: {pregrasp_pos}")
        qpos_pregrasp = self.inverse_kinematics(
            link=self.get_link("hand"),
            pos=pregrasp_pos,
            quat=quat,
        )
        self.move_to_pose(qpos_pregrasp, steps=steps)
        self.print_ee_pose("After pre_grasp_pose()")

    def post_grasp_pose(self, pos, quat, approach_dir=np.array([0, 0, -1]), offset=0.10, steps=300):
        """
        Move the end-effector away from the grasped object (opposite of approach direction).
        """
        self.print_ee_pose("Before post_grasp_pose()")
        retreat_pos = pos - approach_dir * offset
        qpos_postgrasp = self.inverse_kinematics(
            link=self.get_link("hand"),
            pos=retreat_pos,
            quat=quat,
        )
        # If attached_object exists, collisions are ignored automatically inside move_to_pose
        self.move_to_pose(qpos_postgrasp, steps=steps)
        self.print_ee_pose("After post_grasp_pose()")

    # ------------------------------------------------------------------
    # High-level actions (obj is optional for backward compatibility)
    # ------------------------------------------------------------------
    def pick(self, pos, quat=np.array([0, 1, 0, 0]), obj: Optional[Any] = None):
        """
        Approach → open → descend → close → retreat.
        If obj is provided, it will be attached after closing the gripper.
        """
        # compensate for fingertip-to-EE offset (approx 12 cm)
        pos = np.array(pos, dtype=float)
        pos[2] += 0.12

        self.print_ee_pose("Before pick()")
        self.pre_grasp_pose(pos, quat)
        self.open_gripper()

        qpos_grasp = self.inverse_kinematics(link=self.get_link("hand"), pos=pos, quat=quat)
        self.move_to_pose(qpos_grasp, steps=200)
        self.close_gripper()

        # Attach if provided
        if obj is not None:
            self.attach_object(obj)

        # Retreat up (auto ignores collisions if attached)
        qpos_retreat = self.inverse_kinematics(
            link=self.get_link("hand"),
            pos=pos + np.array([0, 0, 0.15]),
            quat=quat,
        )
        self.move_to_pose(qpos_retreat, steps=200)
        self.print_ee_pose("After pick()")

    def place(self, pos, quat=np.array([0, 1, 0, 0]), obj: Optional[Any] = None):
        """
        Move over target → descend → open → retreat.
        If obj is provided, it will be detached after opening the gripper.
        If no obj is provided but an object is currently attached, that one is detached.
        """
        self.print_ee_pose("Before place()")

        # Move above target
        q_place = self.inverse_kinematics(link=self.get_link("hand"), pos=pos, quat=quat)
        self.move_to_pose(q_place, steps=200)

        self.open_gripper()

        # Detach whichever applies
        if obj is not None and obj is self.attached_object:
            self.detach_object()
        elif obj is None and self.attached_object is not None:
            self.detach_object()

        # Retreat (now without attached object; normal collision checking resumes)
        self.post_grasp_pose(pos, quat)
        self.print_ee_pose("After place()")

    # ------------------------------------------------------------------
    # Raw object access
    # ------------------------------------------------------------------
    @property
    def raw(self):
        return self.robot
