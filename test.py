#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import genesis as gs
from scenes import create_scene_6blocks
from robot_adapter import RobotAdapter

if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    gs.init(backend=gs.gpu, logging_level='Warning', logger_verbose_time=False)
else:
    gs.init(backend=gs.cpu, logging_level='Warning', logger_verbose_time=False)
print("[INFO] ✅ Genesis initialized successfully.")

scene, franka_raw, BlocksState = create_scene_6blocks()
franka = RobotAdapter(franka_raw, scene)
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])
)
print("[INFO]  Scene and Franka ready.")

hover_qpos = franka.inverse_kinematics(
    link=franka.get_link("hand"),
    pos=np.array([0.65, 0.0, 0.25]),
    quat=np.array([0, 1, 0, 0]),
)
hover_qpos[-2:] = 0.04
hover_path = franka.plan_path(qpos_goal=hover_qpos, num_waypoints=150)
for waypoint in hover_path:
    franka.control_dofs_position(waypoint)
    scene.step()
print("[INFO] ✅ Reached hover pose.\n")

soln_file = "test.pddl.soln"
if not os.path.exists(soln_file):
    print(f"[WARN] '{soln_file}' not found — creating fallback plan example.")
    with open(soln_file, "w") as f:
        f.write("(pick r)\n(putdown r)\n(pick g)\n(putdown g)\n")

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

print(f"[INFO]  Loaded {len(plan_steps)} actions:")
for i, (a, args) in enumerate(plan_steps, 1):
    print(f"  {i:02d}. ({a} {' '.join(args)})")

print("\n[EXECUTION] ▶ Starting plan execution...\n")

for (action, args) in plan_steps:
    try:
        if action == "pick":
            obj = BlocksState[args[0]]
            pos = obj.get_pos().cpu().numpy()
            print(f"[EXEC] Picking {args[0]} at {pos}")
            franka.pick(pos, obj=obj)

        elif action == "putdown":
            print("\n[EXECUTION] ▶ Starting place execution...\n")
            obj = BlocksState[args[0]]

            if args[0] == "r":
                drop_pos = np.array([0.55, -0.05, 0.15])  # slightly left
            elif args[0] == "g":
                drop_pos = np.array([0.55, 0.05, 0.15])   # slightly right
            else:
                drop_pos = np.array([0.55, 0.0, 0.15])    # default

            print(f"[EXEC] Putting down {args[0]} at {drop_pos}")
            franka.place(drop_pos, obj=obj)

        else:
            print(f"[WARN] Unknown action '{action}' — skipping.")

    except Exception as e:
        print(f"[ERROR] ❌ Failed to execute ({action} {args}): {e}")

print("\n✅ [DONE] Plan execution complete — all moved blocks handled!\n")

print("[INFO] Keeping simulation running for observation...")
for _ in range(600):
    scene.step()
    time.sleep(0.01)
