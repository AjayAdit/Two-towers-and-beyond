#!/usr/bin/env python3
"""
motion_primitives.py
High-level motion primitives using RobotAdapter + PlannerInterface.

Primitives:
  - pick_block(adapter, block)
  - putdown_block(adapter, block, target_height=0.02)
  - stack_block(adapter, block_top, block_bottom)
  - unstack_block(adapter, block_top, block_bottom)

Notes:
  * Uses a universal "virtual attach" so the block follows the hand even
    if your Genesis build lacks create_constraint / set_collision_enabled APIs.
  * Plans via adapter.plan_path (PlannerInterface), but executes with a local
    loop that also syncs any attached object each simulation step.
"""

import numpy as np
import torch
import time

# Fingertip offset from Franka "hand" link (tune Z if needed)
GRIPPER_OFFSET = np.array([0.0, 0.0, -0.045])
# Default end-effector orientation (matches your adapter’s usage)
DEFAULT_QUAT = np.array([0, 1, 0, 0])


# ----------------------------- utils -----------------------------
def _to_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x, dtype=float)


def _move_joints(adapter, q_goal, steps=200, sleep=0.0):
    """
    Plan + execute a joint motion using adapter.plan_path().
    Keeps any virtually attached object aligned to the hand each step.
    """
    q_goal = np.array(q_goal).flatten()
    path = adapter.plan_path(qpos_goal=q_goal, num_waypoints=steps)
    hand = adapter.get_link("hand")

    for wp in path:
        adapter.control_dofs_position(wp)

        # keep attached block synced to hand
        obj = getattr(adapter, "attached_object", None)
        if obj is not None and getattr(obj, "_attached_to_hand", False):
            offset = getattr(obj, "_relative_offset", np.zeros(3))
            obj.set_pos(hand.get_pos() + offset)

        adapter.scene.step()
        if sleep > 0:
            time.sleep(sleep)


def _move_cartesian(adapter, pos, quat=DEFAULT_QUAT, steps=200):
    """IK → joint motion with attached-object sync."""
    q_goal = adapter.inverse_kinematics(link=adapter.get_link("hand"), pos=pos, quat=quat)
    _move_joints(adapter, q_goal, steps=steps)


def _set_gripper(adapter, width, steps=25):
    """
    Smooth-ish gripper actuation by sending a final setpoint and letting
    the controller settle over a few steps.
    """
    qpos = adapter.get_qpos()
    qpos[-2:] = width
    adapter.control_dofs_position(qpos)
    for _ in range(steps):
        adapter.scene.step()


# --------------------- universal attach / detach ------------------
def _attach(adapter, obj):
    """Virtual attach: record offset, mark as attached, try to disable collisions."""
    hand = adapter.get_link("hand")
    obj._relative_offset = _to_np(obj.get_pos()) - _to_np(hand.get_pos())
    obj._attached_to_hand = True

    # Try several APIs to disable collisions, depending on Genesis version.
    if hasattr(obj, "set_collision_enabled"):
        obj.set_collision_enabled(False)
    elif hasattr(obj, "disable_collisions"):
        obj.disable_collisions()
    elif hasattr(obj, "set_collision_mask"):
        try:
            obj.set_collision_mask(0)
        except Exception:
            pass

    adapter.attached_object = obj
    print(f"[INFO] Attached {getattr(obj, 'name', 'object')} to gripper.")


def _detach(adapter, obj):
    """Virtual detach: unmark, re-enable collisions, clear adapter.attached_object."""
    obj._attached_to_hand = False
    obj._relative_offset = None

    if hasattr(obj, "set_collision_enabled"):
        obj.set_collision_enabled(True)
    elif hasattr(obj, "enable_collisions"):
        obj.enable_collisions()
    elif hasattr(obj, "set_collision_mask"):
        try:
            obj.set_collision_mask(1)
        except Exception:
            pass

    if getattr(adapter, "attached_object", None) is obj:
        adapter.attached_object = None

    print(f"[INFO] Detached {getattr(obj, 'name', 'object')} from gripper.")


# --------------------------- primitives ---------------------------
def pick_block(adapter, block):
    """
    Approach → grasp → gentle lift.
    Uses a hover and grasp pose with GRIPPER_OFFSET and stabilizes after attaching.
    """
    hand = adapter.get_link("hand")
    bpos = _to_np(block.get_pos())

    # Hover above block with TCP offset
    hover = bpos + np.array([0, 0, 0.08]) + GRIPPER_OFFSET
    _move_cartesian(adapter, hover, DEFAULT_QUAT, steps=200)

    # Descend to grasp pose
    grasp = bpos + np.array([0, 0, 0.015]) + GRIPPER_OFFSET
    _move_cartesian(adapter, grasp, DEFAULT_QUAT, steps=150)

    # Close and attach (virtual)
    _set_gripper(adapter, width=0.0, steps=25)
    _attach(adapter, block)

    # Let “constraint” stabilize
    for _ in range(20):
        adapter.scene.step()

    # Gentle lift a few cm, then return to hover
    safe = grasp.copy(); safe[2] += 0.03
    _move_cartesian(adapter, safe, DEFAULT_QUAT, steps=180)
    _move_cartesian(adapter, hover, DEFAULT_QUAT, steps=180)


def putdown_block(adapter, block, target_height=0.02):
    """
    Move over the target XY, descend to place height, detach, open, and retreat to hover.
    """
    hand = adapter.get_link("hand")
    bpos = _to_np(block.get_pos())

    place = bpos.copy()
    place[2] = target_height
    hover = place + np.array([0, 0, 0.08]) + GRIPPER_OFFSET

    _move_cartesian(adapter, hover, DEFAULT_QUAT, steps=200)
    _move_cartesian(adapter, place + GRIPPER_OFFSET, DEFAULT_QUAT, steps=150)

    _detach(adapter, block)
    _set_gripper(adapter, width=0.04, steps=25)

    _move_cartesian(adapter, hover, DEFAULT_QUAT, steps=180)


def stack_block(adapter, block_top, block_bottom):
    """
    Place top block onto bottom block by aiming to the top center plus block height.
    """
    hand = adapter.get_link("hand")
    bbot = _to_np(block_bottom.get_pos())

    stack_pos = bbot + np.array([0, 0, 0.05]) + GRIPPER_OFFSET
    hover = stack_pos + np.array([0, 0, 0.08])

    _move_cartesian(adapter, hover, DEFAULT_QUAT, steps=200)
    _move_cartesian(adapter, stack_pos, DEFAULT_QUAT, steps=150)

    _detach(adapter, block_top)
    _set_gripper(adapter, width=0.04, steps=25)

    _move_cartesian(adapter, hover, DEFAULT_QUAT, steps=180)


def unstack_block(adapter, block_top, block_bottom):
    """
    Take the top block from a stack: hover, descend, close+attach, gentle lift, hover.
    """
    hand = adapter.get_link("hand")
    btop = _to_np(block_top.get_pos())

    hover = btop + np.array([0, 0, 0.08]) + GRIPPER_OFFSET
    grasp = btop + np.array([0, 0, 0.015]) + GRIPPER_OFFSET

    _move_cartesian(adapter, hover, DEFAULT_QUAT, steps=200)
    _move_cartesian(adapter, grasp, DEFAULT_QUAT, steps=150)

    _set_gripper(adapter, width=0.0, steps=25)
    _attach(adapter, block_top)

    for _ in range(20):
        adapter.scene.step()

    safe = grasp.copy(); safe[2] += 0.03
    _move_cartesian(adapter, safe, DEFAULT_QUAT, steps=180)
    _move_cartesian(adapter, hover, DEFAULT_QUAT, steps=180)


# -------------------------- convenience --------------------------
def dispatch_primitive(action: str, args, adapter, BlocksState):
    """
    Call the right primitive using symbolic action + args from a .soln plan.
      - (pick X)
      - (putdown X)
      - (stack X Y)
      - (unstack X Y)
    """
    a = action.lower()
    if a == "pick":
        pick_block(adapter, BlocksState[args[0]])
    elif a == "putdown":
        putdown_block(adapter, BlocksState[args[0]])
    elif a == "stack":
        stack_block(adapter, BlocksState[args[0]], BlocksState[args[1]])
    elif a == "unstack":
        unstack_block(adapter, BlocksState[args[0]], BlocksState[args[1]])
    else:
        print(f"[WARN] Unknown action: {action} {args}")
