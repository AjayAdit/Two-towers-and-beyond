#!/usr/bin/env python3
"""
demo.py — Execute symbolic plan (.soln) directly using RobotAdapter (no motion_primitives).
"""

import os
import sys
import time
import numpy as np
import genesis as gs
from scenes import create_scene_6blocks
from robot_adapter import RobotAdapter


# ------------------------------------------------------
# 1️⃣ Initialize Genesis
# ------------------------------------------------------
if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    gs.init(backend=gs.gpu, logging_level='Warning', logger_verbose_time=False)
else:
    gs.init(backend=gs.cpu, logging_level='Warning', logger_verbose_time=False)

print("[INFO] ✅ Genesis initialized successfully.")


# ------------------------------------------------------
# 2️⃣ Build Scene and Wrap Robot
# ------------------------------------------------------
scene, franka_raw, BlocksState = create_scene_6blocks()
franka = RobotAdapter(franka_raw, scene)

# Controller tuning
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])
)

print("[INFO] ✅ Scene and Franka ready.")

# ------------------------------------------------------
# 3️⃣ Move to initial hover pose
# ------------------------------------------------------
print("[INFO] Moving Franka to initial hover pose...")
hover_qpos = franka.inverse_kinematics(
    link=franka.get_link("hand"),
    pos=np.array([0.65, 0.0, 0.25]),
    quat=np.array([0, 1, 0, 0]),
)
hover_qpos[-2:] = 0.04  # open gripper
hover_path = franka.plan_path(qpos_goal=hover_qpos, num_waypoints=150)
for waypoint in hover_path:
    franka.control_dofs_position(waypoint)
    scene.step()
print("[INFO] ✅ Reached hover pose.")


# ------------------------------------------------------
# 4️⃣ Load plan from .soln file
# ------------------------------------------------------
soln_file = "problem.pddl.soln"
if not os.path.exists(soln_file):
    print(f"[WARN] '{soln_file}' not found — creating default plan example.")
    with open(soln_file, "w") as f:
        f.write("(pick r)\n(stack r g)\n(pick m)\n(stack m c)\n")

print(f"[INFO] Reading plan from {soln_file}...")

plan_steps = []
with open(soln_file, "r") as f:
    for line in f:
        line = line.strip().lower()
        if not line or not line.startswith("("):
            continue
        tokens = line.replace("(", "").replace(")", "").split()
        action = tokens[0]
        args = tokens[1:]
        plan_steps.append((action, args))

print(f"[INFO] ✅ Loaded {len(plan_steps)} actions:")
for i, (a, args) in enumerate(plan_steps, 1):
    print(f"  {i:02d}. ({a} {' '.join(args)})")


# ------------------------------------------------------
# 5️⃣ Execute Plan
# ------------------------------------------------------
print("\n[EXECUTION] ▶ Starting plan execution...\n")

for (action, args) in plan_steps:
    try:
        print(f"\n[EXEC] ▶ Action: ({action} {' '.join(args)})")  # <-- ADD THIS LINE
        if action == "pick":
            obj = BlocksState[args[0]]
            pos = obj.get_pos().cpu().numpy()
            franka.pick(pos, obj=obj)

        elif action == "stack":
            top = BlocksState[args[0]]
            bottom = BlocksState[args[1]]
            place_pos = bottom.get_pos().cpu().numpy().copy()
            place_pos[2] += 0.12 # stack height offset
            print(f"[DEBUG][STACK] Goal place position for {args[0]} on {args[1]} → {place_pos}")
            print(f"[DEBUG][STACK-Z] bottom_z={bottom.get_pos().cpu().numpy()[2]:.4f}, target_z={place_pos[2]:.4f}")
            franka.place(place_pos, obj=top)

        elif action == "unstack":
            top = BlocksState[args[0]]
            bottom = BlocksState[args[1]]
            pos = top.get_pos().cpu().numpy()
            franka.pick(pos, obj=top)

        elif action == "putdown":
            obj = BlocksState[args[0]]
            drop_pos = np.array([0.55, 0.0, 0.05])
            print(f"[DEBUG][PUTDOWN] Goal drop position for {args[0]} → {drop_pos}")
            franka.place(drop_pos, obj=obj)

    except Exception as e:
        print(f"[ERROR] ❌ Failed to execute ({action} {args}): {e}")


print("\n✅ [DONE] Plan execution complete — Two towers assembled!\n")


# ------------------------------------------------------
# 6️⃣ Keep Simulation Alive for Observation
# ------------------------------------------------------
print("[INFO] Keeping simulation running for observation...")
for _ in range(600):
    scene.step()
    time.sleep(0.01)
