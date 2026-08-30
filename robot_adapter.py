"""
RobotAdapter: a thin adapter around a Genesis robot entity.

Grasping is real: the block is held by contact friction between the fingers, never
welded or teleported. attached_object is bookkeeping only, set after pick() confirms by
test-lift that the block actually came up, and cleared if the grasp is later lost.

  • Transparent pass-through to the underlying robot
  • pick(pos, quat, obj=None) / place(pos, quat, obj=None)
  • Wrist-tilt ladder so blocks past the ~0.79 m top-down grasp radius stay reachable
  • Carries along straight Cartesian lines at a height that clears every other block,
    because OMPL cannot see the cube in the gripper
  • Chooses the place orientation whose open-finger footprint misses same-level
    neighbours, which matters at goal 4's 2 mm face gaps
  • Closed-loop placement: XY corrected at hover AND re-checked at final height before
    release; a metre-scale error means the grasp was lost, not that we are misaligned
  • Fingers held closed in position control for the whole carry
"""

from typing import Any, Optional, List
import numpy as np

import recording
from genesis.utils.misc import tensor_to_array  # robust tensor → np converter


def _to_np(x) -> np.ndarray:
    """Safely convert Genesis / torch tensors or arrays to np.ndarray."""
    try:
        return tensor_to_array(x)
    except Exception:
        return np.asarray(x, dtype=float)


# ======================================================================
# Quaternion / rotation helpers (W, X, Y, Z convention)
# ======================================================================

def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions in [w, x, y, z] format."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=float)


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate for [w, x, y, z]."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def quat_to_rot_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert [w, x, y, z] quaternion to 3×3 rotation matrix."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def rotate_quat_world_z_deg(quat_in: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate `quat_in` by `angle_deg` about WORLD Z axis.
    All quats are [w, x, y, z].

    q_total = qz ⊗ quat_in
    """
    quat_in = np.asarray(quat_in, dtype=float)
    angle = np.deg2rad(angle_deg)
    half = 0.5 * angle
    sin_half = np.sin(half)
    cos_half = np.cos(half)

    # rotation about WORLD Z: axis (0,0,1) → [w, x, y, z]
    qz = np.array([cos_half, 0.0, 0.0, sin_half], dtype=float)

    return quat_mul_wxyz(qz, quat_in)


def quat_about_axis(axis, angle_deg: float) -> np.ndarray:
    """Quaternion [w,x,y,z] for a rotation of `angle_deg` about `axis`."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12 or angle_deg == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    half = np.deg2rad(angle_deg) / 2.0
    return np.array([np.cos(half), *(np.sin(half) * axis)])


# Straight-down grasp reaches only ~0.79 m but blocks spawn as far as 0.832 m. Leaning
# the fingers outward puts the hand closer to the base while the grasp centre still
# lands on the block. Yaw is left alone: rotating 90 deg fails at every tilt.
GRASP_TILTS_DEG = (0.0, 15.0, 25.0, 35.0)


class RobotAdapter:
    def __init__(self, robot: Any, scene: Any = None):
        """Wrap a genesis robot entity.

        Args:
            robot: the raw genesis robot entity (e.g., returned from scene.add_entity)
            scene: optional scene reference (some callers use scene alongside robot)
        """
        self.robot = robot
        self.scene = scene

        # --- DOF index groups for arm vs fingers (Franka) ---
        jnt_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ]
        dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in jnt_names]

        self._arm_dofs = np.array(dofs_idx[:7], dtype=int)
        self._finger_dofs = np.array(dofs_idx[7:], dtype=int)

        print("[DEBUG] arm dofs   :", self._arm_dofs)
        print("[DEBUG] finger dofs:", self._finger_dofs)

        self.block_size = 0.04  # meters
        self.block_half = 0.5 * self.block_size

        # Grasp height: commanding the hand to block_z + 0.10 seats the fingers on a
        # table cube (measured width 0.0396). True fingertip offset is 0.1124.
        self.hand_to_finger_tip_z = 0.12
        self.hand_to_block_center_z = self.hand_to_finger_tip_z - self.block_half

        # Keep fingers closed after grasp
        self._fingers_locked: bool = False
        self._finger_lock_value: Optional[float] = None  # scalar for last 2 DOFs

        # Registered for neighbour search during placement
        self.blocks: List[Any] = []

        # Which block we are holding, and its measured offset in the hand frame
        self.attached_object: Optional[Any] = None
        self._attach_offset_pos: Optional[np.ndarray] = None  # block in HAND frame
        self._attach_offset_quat: Optional[np.ndarray] = None  # q_rel = q_h*conj(q_o)

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
        # Used for squeezing fingers in close_gripper
        return self.robot.control_dofs_force(*a, **kw)

    def get_link(self, *a, **kw):
        return self.robot.get_link(*a, **kw)

    def inverse_kinematics(self, *a, **kw):
        return self.robot.inverse_kinematics(*a, **kw)

    def detect_collision(self, *a, **kw):
        return self.robot.detect_collision(*a, **kw)

    # ------------------------------------------------------------------
    # Blocks registration + neighbor-orientation logic
    # ------------------------------------------------------------------
    def register_blocks(self, blocks: List[Any]):
        """
        Register all block entities so we can search for neighbors
        using only geometry (no TAMP).
        """
        self.blocks = list(blocks)
        print(f"[ROBOT][BLOCKS] Registered {len(self.blocks)} blocks.")

    def _descent_blocked(self, target_pos, closing_axis_xy, ignore_obj) -> bool:
        """Would descending the OPEN gripper at `target_pos`, with the fingers closing
        along `closing_axis_xy`, strike a same-level neighbouring block?

        Geometry: with the gripper open each fingertip sits ~0.04 m from the grasp
        centre along the closing axis, and the finger body presents a cross-section of
        roughly 26 mm (Panda finger; half-envelope 0.013 m). A neighbour cube (half
        0.02 m) is struck when it overlaps the swept rectangle of either finger:
            |along| in (0.04 - 0.033, 0.04 + 0.033)  and  |perp| < 0.033
        where 0.033 = block_half + finger half-envelope.

        Goal 4 places blocks 42 mm apart, so at those positions one orientation always
        descends a finger onto the neighbour and knocks it off its spot.
        """
        target_pos = _to_np(target_pos)
        ax = np.asarray(closing_axis_xy, dtype=float)
        ax = ax / (np.linalg.norm(ax) + 1e-12)
        perp_ax = np.array([-ax[1], ax[0]])

        OPEN_HALF = 0.04           # fingertip offset from grasp centre when open
        ENVELOPE = self.block_half + 0.013   # cube half-width + finger half-envelope
        z = float(target_pos[2])

        for blk in self.blocks:
            if ignore_obj is not None and blk is ignore_obj:
                continue
            try:
                p = _to_np(blk.get_pos())
            except Exception:
                continue
            if abs(float(p[2]) - z) > 0.5 * self.block_size:
                continue  # a level up/down (incl. the support) cannot meet the fingers
            rel = p[:2] - target_pos[:2]
            along = abs(float(np.dot(rel, ax)))
            perp = abs(float(np.dot(rel, perp_ax)))
            if perp < ENVELOPE and (OPEN_HALF - ENVELOPE) < along < (OPEN_HALF + ENVELOPE):
                return True
        return False

    def _choose_place_quat_from_neighbor(
        self,
        base_quat: np.ndarray,
        target_pos: np.ndarray,
        ignore_obj: Optional[Any] = None,
        extra_deg: float = 90.0,
    ) -> np.ndarray:
        """
        Choose the wrist orientation whose open-finger descent envelope does not strike
        a same-level neighbour (see _descent_blocked for the geometry).

        With the base quat [0,1,0,0] the fingers close along world Y; rotated +90 deg
        about world Z they close along world X.
        """
        base_quat = np.asarray(base_quat, dtype=float)
        target_pos = _to_np(target_pos)

        if not self.blocks:
            return base_quat

        blocked_base = self._descent_blocked(target_pos, (0.0, 1.0), ignore_obj)
        blocked_rot = self._descent_blocked(target_pos, (1.0, 0.0), ignore_obj)

        if not blocked_base:
            if blocked_rot:
                print("[PLACE-ORIENT] base orientation clear, rotated blocked ⇒ keep base.")
            return base_quat
        if not blocked_rot:
            print(f"[PLACE-ORIENT] base orientation would strike a neighbour ⇒ "
                  f"rotate +{extra_deg}° about world Z.")
            return rotate_quat_world_z_deg(base_quat, extra_deg)

        print("[PLACE-ORIENT] WARNING: both orientations strike a neighbour; keeping "
              "base orientation. Expect contact.")
        return base_quat

    # ------------------------------------------------------------------
    # Logical + kinematic attach / detach
    # ------------------------------------------------------------------
    def attach_object(self, obj: Any):
        """Record that the robot is holding `obj`, and measure how it is being held.

        Bookkeeping, not actuation. Set only after pick() confirms by test-lift that the
        block came up. place() aims with the measured offset.
        """
        self.attached_object = obj

        hand = self.get_link("hand")
        p_h = _to_np(hand.get_pos())
        q_h = _to_np(hand.get_quat())
        p_o = _to_np(obj.get_pos())
        q_o = _to_np(obj.get_quat())

        R_h = quat_to_rot_wxyz(q_h)

        # block center in hand frame
        self._attach_offset_pos = R_h.T @ (p_o - p_h)
        # relative orientation: q_rel = conj(q_h) ⊗ q_o
        self._attach_offset_quat = quat_mul_wxyz(
            quat_conj_wxyz(q_h), q_o
        )

        print(
            "[ROBOT][ATTACH-LOGIC] attached_object set to",
            obj,
            "offset_pos (hand frame)=",
            np.round(self._attach_offset_pos, 4),
        )

    def detach_object(self):
        """
        Logically mark that the robot is not holding anything
        and disable kinematic weld.
        """
        print(
            "[ROBOT][DETACH-LOGIC] clearing attached_object (was",
            self.attached_object,
            ")",
        )
        self.attached_object = None
        self._attach_offset_pos = None
        self._attach_offset_quat = None

    def _sync_attached_object(self):
        """Refresh the measured hand->object offset. Observes only; never writes a pose.

        The grasp is carried by contact friction, verified in isolation: the Franka
        lifts a 4 cm, 12.8 g cube 150 mm with the fingers position-closed onto it.
        """
        if self.attached_object is None:
            return

        hand = self.get_link("hand")
        p_h = _to_np(hand.get_pos())
        q_h = _to_np(hand.get_quat())
        R_h = quat_to_rot_wxyz(q_h)

        p_o = _to_np(self.attached_object.get_pos())
        q_o = _to_np(self.attached_object.get_quat())

        self._attach_offset_pos = R_h.T @ (p_o - p_h)
        self._attach_offset_quat = quat_mul_wxyz(quat_conj_wxyz(q_h), q_o)

    # ------------------------------------------------------------------
    # Helpers: stepping + "lock fingers"
    # ------------------------------------------------------------------
    def _step_scene(self, steps: int = 1):
        """Step the simulator and enforce kinematic weld if attached."""
        if self.scene is None:
            return
        for _ in range(steps):
            self.scene.step()
            self._sync_attached_object()
            recording.capture(self.scene)

    def _maybe_lock_fingers(self, q) -> np.ndarray:
        """
        Ensure that if fingers are locked, the last 2 DOFs are held at the
        closed value, regardless of what IK/path planner produced.
        """
        q_np = _to_np(q).copy()
        if (
            self._fingers_locked
            and q_np.shape[0] >= 2
            and self._finger_lock_value is not None
        ):
            q_np[-2:] = self._finger_lock_value
        return q_np

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
        """
        Open parallel gripper (assumes last 2 DOFs).
        Also UNLOCK fingers so they can move normally.
        """
        self.print_ee_pose("Before open_gripper()")
        qpos = _to_np(self.get_qpos())
        qpos[-2:] = open_pos
        self.control_dofs_position(qpos)

        # Opening: stop locking
        self._fingers_locked = False
        self._finger_lock_value = None

        self._step_scene(steps)
        self.print_ee_pose("After open_gripper()")

    def close_gripper(self, close_pos: float = 0.0, settle_steps: int = 60):
        """Close the gripper onto the object using position control only.

        A force squeeze fights the position controller on the same DOFs and wins: at
        20 N the fingers close past the cube (width -0.0006 m) and drop it. Position
        control stalls them against the object at width 0.0396 m on a 4 cm cube, which
        is ample: the cube weighs 0.126 N and needs 0.063 N per finger at mu = 1.
        """
        self.print_ee_pose("Before close_gripper()")

        q_current = _to_np(self.get_qpos())

        # Hold the arm exactly where it is so closing cannot drag the arm off the grasp.
        self.robot.control_dofs_position(q_current[:-2], self._arm_dofs)
        # Drive the fingers shut; they will stall on the object.
        self.robot.control_dofs_position(
            np.array([close_pos, close_pos], dtype=float),
            self._finger_dofs,
        )

        # Let the contact establish and the fingers come to rest against the object.
        self._step_scene(settle_steps)

        width = float(_to_np(self.get_qpos())[-2] + _to_np(self.get_qpos())[-1])
        print(f"[GRIP] Closed onto object: finger width = {width:.4f} m "
              f"(a seated 4 cm cube reads ~0.040)")

        # Keep commanding closed for the rest of the carry: this is the continuous grip.
        self._fingers_locked = True
        self._finger_lock_value = float(close_pos)

        self.print_ee_pose("After close_gripper()")

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def move_to_pose(
        self,
        qpos_goal,
        steps: int = 30,
        ignore_collisions: bool = False,
    ):
        """
        Plan and execute path using OMPL planner.

        If we are holding an object (fingers locked), use
        more waypoints (slower, smoother) to reduce slip.
        """
        self.print_ee_pose("Before move_to_pose()")

        # If grasping, increase resolution of motion
        if self._fingers_locked:
            steps = max(steps, 60)

        qpos_start = _to_np(self.get_qpos())
        print(
            f"[DEBUG] move_to_pose(): ignore_collisions={ignore_collisions}, "
            f"num_waypoints={steps}"
        )

        if ignore_collisions:
            print(
                "[DEBUG] Ignoring collisions during motion planning (explicit flag)."
            )
            from planning import PlannerInterface

            planner_interface = PlannerInterface(self.robot, self.scene)
            path = planner_interface.plan_path(
                qpos_goal=qpos_goal,
                qpos_start=qpos_start,
                num_waypoints=steps,
            )
        else:
            # Assumes underlying robot or wrapper has plan_path(...)
            path = self.plan_path(
                qpos_goal=qpos_goal,
                num_waypoints=steps,
            )

        if len(path) == 0:
            print(
                "[WARN] OMPL returned empty path; executing direct control to goal."
            )
            q_goal = self._maybe_lock_fingers(qpos_goal)
            self.control_dofs_position(q_goal)
            self._step_scene(100)
            return

        # OPTIONAL: subdivide between waypoints when grasping
        if self._fingers_locked and len(path) >= 2:
            dense_path = []
            for i in range(len(path) - 1):
                q0 = _to_np(path[i])
                q1 = _to_np(path[i + 1])
                # 3 sub-steps between waypoints → smoother
                for alpha in np.linspace(0.0, 1.0, 4, endpoint=False):
                    dense_path.append((1 - alpha) * q0 + alpha * q1)
            dense_path.append(_to_np(path[-1]))
            path = dense_path

        for waypoint in path:
            q = self._maybe_lock_fingers(waypoint)
            self.control_dofs_position(q)
            self._step_scene(1)

        self.print_ee_pose("After move_to_pose()")

    def _select_grasp_pose(self, block_center, base_quat):
        """Pick the least-tilted wrist pose whose IK actually reaches `block_center`.

        Returns (hand_pos, quat, tilt_deg). Tilt 0 is a plain straight-down grasp; the
        larger tilts only apply at the edge of the workspace.
        """
        block_center = _to_np(block_center)
        bx, by, bz = float(block_center[0]), float(block_center[1]), float(block_center[2])
        r = float(np.hypot(bx, by))
        off = self.hand_to_block_center_z

        if r < 1e-6:
            return block_center + np.array([0.0, 0.0, off]), np.asarray(base_quat, float), 0.0

        radial = np.array([bx / r, by / r, 0.0])
        perp = np.array([-by / r, bx / r, 0.0])

        last = None
        for tilt in GRASP_TILTS_DEG:
            th = np.deg2rad(tilt)
            hand_pos = np.array([bx, by, bz]) + off * np.array(
                [-np.sin(th) * radial[0], -np.sin(th) * radial[1], np.cos(th)]
            )
            quat = (np.asarray(base_quat, float) if tilt == 0.0
                    else quat_mul_wxyz(quat_about_axis(perp, -tilt), np.asarray(base_quat, float)))

            _, err = self._ik(hand_pos, quat, label=f"grasp-tilt{tilt:.0f}")
            last = (hand_pos, quat, tilt, err)
            if not np.isnan(err) and err < 0.002:
                if tilt > 0.0:
                    print(f"[GRASP] block at r={r:.4f} m is beyond the straight-down reach; "
                          f"tilting the wrist {tilt:.0f} deg (hand r={np.hypot(*hand_pos[:2]):.4f} m)")
                return hand_pos, quat, tilt

        hand_pos, quat, tilt, err = last
        print(f"[GRASP] WARNING: block at r={r:.4f} m is unreachable at every tilt "
              f"(best IK residual {err*1000:.1f} mm). The grasp will fail.")
        return hand_pos, quat, tilt

    def _safe_transit_hand_z(self, margin: float = 0.03) -> float:
        """Hand height at which a carried block clears every other block in the scene.

        OMPL collision-checks the robot but not the cube in the gripper, so transits must
        stay above the tallest stack. The carried cube's underside sits 0.12 m below the
        hand frame; add half a block to clear the obstacle's top face, plus a margin.
        """
        top_z = self.block_half  # bare table
        for blk in self.blocks:
            if self.attached_object is not None and blk is self.attached_object:
                continue
            try:
                top_z = max(top_z, float(_to_np(blk.get_pos())[2]))
            except Exception:
                continue
        return top_z + self.block_half + self.hand_to_block_center_z + self.block_half + margin

    def _cartesian_transit(self, target_xy, quat, z: Optional[float] = None,
                           step_m: float = 0.02, settle: int = 10):
        """Carry the hand to target_xy along a straight Cartesian line at transit height.

        OMPL plans in joint space and cannot see the held block, so its paths can swing
        the arm wide or dip the cube through a stack. A straight line above everything is
        collision-free by construction, and 2 cm steps keep accelerations low enough that
        the friction grasp holds.
        """
        z = float(self._safe_transit_hand_z() if z is None else z)
        hand = self.get_link("hand")
        cur = _to_np(hand.get_pos())

        # 1) vertical lift to transit height, if below it
        if cur[2] < z - 1e-4:
            q_up, _ = self._ik(np.array([cur[0], cur[1], z]), quat, label="transit-lift")
            self._servo_to_q(q_up, steps=60)
            cur = _to_np(hand.get_pos())

        # 2) straight XY line at constant z, in small steps
        start_xy = cur[:2].copy()
        delta = np.asarray(target_xy, dtype=float) - start_xy
        dist = float(np.linalg.norm(delta))
        n = max(1, int(np.ceil(dist / step_m)))
        for i in range(1, n + 1):
            xy = start_xy + delta * (i / n)
            q_i, _ = self._ik(np.array([xy[0], xy[1], z]), quat,
                              label=f"transit-{i}/{n}")
            self._servo_to_q(q_i, steps=8)
        self._step_scene(settle)

    def _ik(self, pos, quat, label: str = ""):
        """Solve IK and return (qpos, position_residual_metres).

        Genesis returns a best-effort qpos even when it fails to converge, and only
        reports that via return_error=True. Unreachable targets otherwise look fine.
        """
        try:
            q, err = self.robot.inverse_kinematics(
                link=self.get_link("hand"), pos=pos, quat=quat, return_error=True
            )
            e = _to_np(err).ravel()
            pos_err = float(np.linalg.norm(e[:3]))
        except TypeError:
            # Older signature without return_error: fall back to unchecked.
            q = self.robot.inverse_kinematics(link=self.get_link("hand"), pos=pos, quat=quat)
            return q, float("nan")

        if pos_err > 0.002:
            print(f"[IK] {label} UNCONVERGED: residual {pos_err*1000:.2f} mm "
                  f"for target {np.round(_to_np(pos), 4)}")
        return q, pos_err

    def _servo_to_q(self, q_goal, steps: int = 80):
        """
        Simple joint-space interpolation from current q to q_goal.

        When fingers are locked (grasping), we automatically increase
        the number of interpolation steps to reduce accelerations and slip.
        """
        # Slow down if grasping
        if self._fingers_locked:
            steps = int(steps * 2.0)  # 2× slower when holding a block

        q_start = _to_np(self.get_qpos())
        q_goal = _to_np(q_goal)

        for alpha in np.linspace(0.0, 1.0, steps):
            q = (1.0 - alpha) * q_start + alpha * q_goal
            q = self._maybe_lock_fingers(q)
            self.control_dofs_position(q)
            self._step_scene(1)

        # Report joint-space tracking. If the arm does not reach the commanded qpos there
        # is no point correcting against the pose it was supposed to have reached.
        q_end = _to_np(self.get_qpos())
        arm_track_err = np.abs(q_end[:7] - q_goal[:7])
        if arm_track_err.max() > 0.01:
            worst = int(np.argmax(arm_track_err))
            print(f"[SERVO] arm did not track: joint{worst+1} off by "
                  f"{arm_track_err[worst]:.5f} rad "
                  f"(commanded {q_goal[worst]:+.4f}, reached {q_end[worst]:+.4f}); "
                  f"per-joint err={np.round(arm_track_err, 5)}")

    # ------------------------------------------------------------------
    # High-level actions (obj is optional for logging / logic)
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
          3) Slow vertical descent to grasp height
          4) Let contacts settle
          5) Close gripper (lock fingers closed)
          6) Small test-lift; ONLY IF block lifts, attach_object(obj)
          7) Retreat upward

        `pos` is the desired *block center* world position (4 cm cube).
        """
        block_center = _to_np(pos).copy()

        # Choose the wrist orientation. Straight down whenever it reaches; leaning
        # outward when the block sits beyond the straight-down workspace radius.
        hand_at_center, quat, grasp_tilt = self._select_grasp_pose(block_center, quat)
        target_grasp_z = float(hand_at_center[2])

        hand_link = self.get_link("hand")

        # For grasp success check, cache pre-grasp block pose
        obj_pos_before = None
        if obj is not None:
            try:
                obj_pos_before = _to_np(obj.get_pos())
            except Exception as e:
                print("[PICK] Could not read obj_pos_before:", e)

        self.print_ee_pose("Before pick()")

        # Approach in three stages: lift clear, translate above the block, then descend.
        # A direct joint-space move sweeps the gripper through whatever lies between and
        # pushes blocks off the positions we just perceived them at.
        pregrasp_target = hand_at_center.copy()
        pregrasp_target[2] = target_grasp_z + 1.5 * self.block_size  # ~6 cm above

        safe_z = self._safe_transit_hand_z()
        cur_hand = _to_np(hand_link.get_pos())

        if cur_hand[2] < safe_z - 1e-4:
            lift_here = cur_hand.copy()
            lift_here[2] = safe_z
            q_lift, _ = self._ik(lift_here, quat, label="pick-lift-clear")
            self._servo_to_q(q_lift, steps=60)

        print(f"[DEBUG] Cartesian transit above block at z={max(safe_z, pregrasp_target[2]):.4f}, "
              f"then pre-grasp at {pregrasp_target}")
        self._cartesian_transit(pregrasp_target[:2], quat,
                                z=max(safe_z, pregrasp_target[2]))

        qpos_pregrasp, _ = self._ik(pregrasp_target, quat, label="pick-pregrasp")
        self._servo_to_q(qpos_pregrasp, steps=60)

        # 2) Open gripper (fully open before going down) and unlock fingers
        self.open_gripper()

        # 3) SLOW VERTICAL DESCENT from pre-grasp to grasp height
        current_hand_pos = _to_np(hand_link.get_pos())
        z_start = current_hand_pos[2]
        z_goal = hand_at_center[2] - 0.001

        # ~90 mm at 150 steps is 0.06 m/s. At 30 steps the kp=4500 controller overshot
        # 6.6 mm, driving the fingertips into the table so they could not close.
        descent_steps = 150
        print(
            f"[DEBUG] Slow vertical descent from z={z_start:.4f} "
            f"→ z={z_goal:.4f} in {descent_steps} steps"
        )

        for i in range(descent_steps):
            alpha = (i + 1) / descent_steps
            z = z_start + alpha * (z_goal - z_start)

            target_pos = hand_at_center.copy()
            target_pos[2] = z

            q_step = self.inverse_kinematics(
                link=hand_link,
                pos=target_pos,
                quat=quat,
            )
            q_step = self._maybe_lock_fingers(q_step)
            self.control_dofs_position(q_step)
            self._step_scene(1)

        # 4) Let contacts settle at final grasp pose
        self._step_scene(60)

        # Verify we actually arrived before closing. Closing from the wrong height is the
        # difference between a grasp and a collision, so this is reported, not patched.
        z_reached = float(_to_np(hand_link.get_pos())[2])
        z_err = z_reached - z_goal
        tip_drop = 0.1124 * float(np.cos(np.deg2rad(grasp_tilt)))
        print(f"[PICK] Descent target z={z_goal:.4f}, reached z={z_reached:.4f} "
              f"(error {z_err*1000:+.1f} mm, tilt {grasp_tilt:.0f} deg, "
              f"fingertips at z={z_reached - tip_drop:.4f})")
        if abs(z_err) > 0.005:
            print(f"[PICK] WARNING: hand is {abs(z_err)*1000:.1f} mm off the grasp height; "
                  f"the grasp is likely to fail.")

        # 5) Close gripper (now we are at grasp height) and LOCK fingers
        self.close_gripper()

        # 6) Small test-lift: attach ONLY IF block actually comes up
        grasp_succeeded = False
        if obj is not None and obj_pos_before is not None:
            # target: small lift of the hand (e.g., +1.5 cm)
            test_lift_target = hand_at_center.copy()
            test_lift_target[2] += 0.015  # 1.5 cm up from grasp height
            print(f"[PICK] Test-lift target hand pos: {test_lift_target}")

            q_test_lift = self.inverse_kinematics(
                link=hand_link,
                pos=test_lift_target,
                quat=quat,
            )
            self._servo_to_q(q_test_lift, steps=40)
            self._step_scene(20)

            try:
                obj_pos_after = _to_np(obj.get_pos())
                dz = float(obj_pos_after[2] - obj_pos_before[2])
                print(f"[PICK] Test-lift dz for obj: {dz:.4f} m")

                # threshold: block must move up at least 8 mm to consider grasp success
                if dz > 0.008:
                    print("[PICK] Grasp succeeded ⇒ attaching object.")
                    self.attach_object(obj)
                    grasp_succeeded = True
                else:
                    print("[PICK] Grasp failed (block did not lift) ⇒ NOT attaching.")
            except Exception as e:
                print("[PICK] Could not evaluate grasp success:", e)
        else:
            print("[PICK] Skipping grasp success check (no obj or no pre pose).")

        # 7) Retreat straight up (slow, to reduce slip), high enough that the block we are
        #    now carrying will clear every other block during the transit that follows.
        retreat_target = hand_at_center.copy()
        retreat_target[2] = max(
            hand_at_center[2] + 2.0 * self.block_size,
            self._safe_transit_hand_z(),
        )
        qpos_retreat = self.inverse_kinematics(
            link=hand_link,
            pos=retreat_target,
            quat=quat,
        )
        print("[DEBUG] Retreat target (from centered grasp):", retreat_target)
        # extra steps; _servo_to_q will 2× it again because fingers_locked
        self._servo_to_q(qpos_retreat, steps=80)

        self.print_ee_pose("After pick()")

    def place(
        self,
        pos,
        quat: np.ndarray = np.array([0.0, 1.0, 0.0, 0.0]),  # [w,x,y,z]
        obj: Optional[Any] = None,
        hover_height: float = 0.08,
        release_clearance: float = 0.0,  # no free drop; drive to final Z
    ):
        """
        PLACE SEQUENCE (kinematically attached block + neighbor-aware orientation)

        `pos` is the desired object center at the final stack position.

        Steps:
          1) Decide orientation using neighbors and |y - b| / |x - a| rule
          2) Move hand to hover above target (via OMPL, smoothed if grasping)
          3) Iterative XY correction using object pose (if given)
          4) Descend to final placement height (no free drop)
          5) Open gripper (unlock fingers) + logically detach + settling time
          6) Retreat upward
        """
        self.print_ee_pose("Before place()")
        pos = _to_np(pos).copy()

        base_quat = np.asarray(quat, dtype=float)
        print("[DEBUG] Incoming base quat (w,x,y,z):", base_quat)

        # Decide whether to keep base_quat or rotate by +90° about WORLD Z
        quat_place = self._choose_place_quat_from_neighbor(
            base_quat=base_quat,
            target_pos=pos,
            ignore_obj=obj,
            extra_deg=90.0,
        )

        print("[DEBUG] Final quat_place passed to IK:", quat_place)

        # === 1) target object center hover above final stack position ===
        pos_above_obj = pos.copy()
        pos_above_obj[2] += hover_height

        # Default approximate hand↔block offset (if no attach info)
        off = np.array([0.0, 0.0, -self.hand_to_block_center_z], dtype=float)

        # If we have a real measured attach offset, use that instead (world frame)
        if (
            obj is not None
            and obj is self.attached_object
            and self._attach_offset_pos is not None
        ):
            R_place = quat_to_rot_wxyz(quat_place)
            off = R_place @ self._attach_offset_pos

        print("[DEBUG] used hand↔block offset (world):", np.round(off, 4))
        print("[DEBUG] desired object pos:", np.round(pos, 4))
        print("[DEBUG] desired object pos above:", np.round(pos_above_obj, 4))

        target_hand_pos_above = pos_above_obj - off
        # Hover high enough that the carried block clears every other stack; OMPL
        # collision-checks the robot but not the cube in the gripper.
        safe_z = self._safe_transit_hand_z()
        if target_hand_pos_above[2] < safe_z:
            print(f"[PLACE] raising approach from z={target_hand_pos_above[2]:.4f} to "
                  f"{safe_z:.4f} so the carried block clears the tallest stack")
            target_hand_pos_above[2] = safe_z

        # 2) Carry to the hover point along a straight line at the safe height.
        self._cartesian_transit(target_hand_pos_above[:2], quat_place,
                                z=target_hand_pos_above[2])
        self.print_ee_pose("[DEBUG] After transit to hover")

        # === 3) Iterative XY correction (adaptive step_max) ===
        tol_xy = 1e-3      # 1 mm target
        max_iters = 20
        stall_eps = 1e-6
        k_p = 1.3
        k_d = 0.18

        prev_err_xy = None
        prev_err_norm = None

        for it in range(max_iters):
            hand = self.get_link("hand")
            ee_pos = _to_np(hand.get_pos())

            if obj is not None:
                try:
                    obj_pos = _to_np(obj.get_pos())
                except Exception:
                    obj_pos = ee_pos - off  # fallback guess
            else:
                obj_pos = ee_pos - off  # approximate block center

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

            if err_norm > 0.05 and obj is not None and obj is self.attached_object:
                # An error this large at hover means the block is not in the fingers any
                # more, so correcting toward it just walks the empty hand across the table.
                print(f"[CORRECTION] error {err_norm:.3f} m implies the grasp was lost, "
                      f"abandoning this placement for replanning.")
                self.open_gripper()
                self.detach_object()  # align the logical state with the physical one
                return

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

            # adaptive step_max: big steps when far, tiny when close
            adaptive = 0.3 * err_norm
            step_max = min(0.02, max(0.001, adaptive))  # [1mm, 2cm]

            u_xy = k_p * err_xy + k_d * derr_xy
            dx = float(np.clip(u_xy[0], -step_max, step_max))
            dy = float(np.clip(u_xy[1], -step_max, step_max))

            print(
                f"[CORRECTION {it}] step_max={step_max:.4f} "
                f"Δhand=({dx:+.6f}, {dy:+.6f})"
            )

            new_hand_pos = ee_pos.copy()
            new_hand_pos[0] += dx
            new_hand_pos[1] += dy
            new_hand_pos[2] = target_hand_pos_above[2]

            q_fix, ik_err = self._ik(new_hand_pos, quat_place, label=f"place-correct[{it}]")
            hand_before_xy = ee_pos[:2].copy()
            self._servo_to_q(q_fix, steps=40)

            # Did the hand actually go where we asked? If the arm does not track the
            # command there is no point issuing more corrections against it.
            hand_after_xy = _to_np(self.get_link("hand").get_pos())[:2]
            moved = hand_after_xy - hand_before_xy
            print(f"[CORRECTION {it}] hand moved ({moved[0]:+.6f}, {moved[1]:+.6f}) "
                  f"vs commanded ({dx:+.6f}, {dy:+.6f}), ik_residual={ik_err*1000:.2f}mm")

        # 4) Descend to final height, re-checking XY at the bottom. The correction
        # above runs at hover; the block can still shift ~3 mm on the way down, which is
        # over the 2 mm at-position threshold. Measure before releasing, not after.
        final_obj_pos = pos.copy()
        final_obj_pos[2] += release_clearance  # 0.0 by default
        final_hand_pos = final_obj_pos - off

        for attempt in range(3):
            q_final, _ = self._ik(final_hand_pos, quat_place, label="place-final")
            self._servo_to_q(q_final, steps=80)

            if obj is None:
                break
            try:
                obj_now = _to_np(obj.get_pos())
            except Exception:
                break
            bottom_err = pos[:2] - obj_now[:2]
            bottom_norm = float(np.linalg.norm(bottom_err))
            print(f"[PLACE-BOTTOM {attempt}] XY error at final height = "
                  f"{bottom_norm*1000:.2f} mm")
            if bottom_norm <= 0.0015:
                break

            if bottom_norm > 0.05:
                # This size of error means the block is gone, not misaligned. Chasing
                # it commanded hand positions metres outside the workspace.
                print(f"[PLACE-BOTTOM] error {bottom_norm:.3f} m implies the grasp was "
                      f"lost mid-place, abandoning this placement for replanning.")
                self.open_gripper()
                self.detach_object()
                return

            # Misaligned but still held: lift, shift by the measured error, retry.
            lift = final_hand_pos.copy()
            lift[2] += 0.03
            q_lift, _ = self._ik(lift, quat_place, label="place-relift")
            self._servo_to_q(q_lift, steps=40)
            final_hand_pos[0] += float(bottom_err[0])
            final_hand_pos[1] += float(bottom_err[1])

        self.print_ee_pose("[DEBUG] At final pose")

        # 5) Open gripper to release (unlock fingers) and logically detach
        self.open_gripper()
        if obj is not None and obj is self.attached_object:
            self.detach_object()
        elif obj is None and self.attached_object is not None:
            self.detach_object()

        self._step_scene(30)

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

        # === 6) Retreat upward so we don't collide with the stack ===
        pos_retreat = pos.copy()
        pos_retreat[2] += 0.30

        q_retreat = self.inverse_kinematics(
            link=self.get_link("hand"),
            pos=pos_retreat,
            quat=quat_place,
        )
        print("[DEBUG] move_to_pose retreat called, pos_retreat:", pos_retreat)
        self._servo_to_q(q_retreat, steps=40)
        self.print_ee_pose("After place()")

    # ------------------------------------------------------------------
    # Raw object access
    # ------------------------------------------------------------------
    @property
    def raw(self):
        """Direct access to underlying Genesis robot entity."""
        return self.robot
