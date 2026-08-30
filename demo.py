#!/usr/bin/env python3
"""
Unified TAMP Demo for All Goals

Supports:
- Goals 1-3: Simple block stacking (6 or 8 blocks)
- Goal 4: Two-structure building (18 blocks: yellow cross + green square)

Usage:
    python3 demo.py --goal 1          # Single tower
    python3 demo.py --goal 4          # Two structures (Goal 4 demo)
"""

import argparse
import os
import numpy as np
import genesis as gs

import recording
from scenes import (
    create_scene_6blocks,
    create_scene_stacked,
    create_scene_8blocks,
    create_scene_goal4_initial,
)
from robot_adapter import RobotAdapter
from symbolic_abstraction import abstract_state, visualize_predicates
from task_planner import plan_symbolic

BLOCK_SIZE = 0.04


def print_block_positions(BlocksState, goal_id=1, label="[BLOCK POS]"):
    """Print block positions for debugging. Adapts to goal type."""
    print(f"\n{label}")

    if goal_id in [4, 41, 42]:
        # Goal 4: 18 blocks (y1-y12, g1-g6)
        yellow_blocks = [f"y{i}" for i in range(1, 13)]
        green_blocks = [f"g{i}" for i in range(1, 7)]

        print("Yellow blocks:")
        for name in yellow_blocks:
            if name in BlocksState:
                obj = BlocksState[name]
                pos = obj.get_pos()
                try:
                    pos = pos.cpu().numpy()
                except Exception:
                    pos = np.array(pos, dtype=float)
                print(f"  {name.upper()} @ {pos}")

        print("Green blocks:")
        for name in green_blocks:
            if name in BlocksState:
                obj = BlocksState[name]
                pos = obj.get_pos()
                try:
                    pos = pos.cpu().numpy()
                except Exception:
                    pos = np.array(pos, dtype=float)
                print(f"  {name.upper()} @ {pos}")
    else:
        # Goals 1-2: 6 blocks (b, c, g, m, r, y)
        # Goal 3: 8 blocks (b, c, g, m, r, y, o, p)
        block_list = ["b", "c", "g", "m", "r", "y"]
        if goal_id == 3:
            block_list = ["b", "c", "g", "m", "r", "y", "o", "p"]

        for name in block_list:
            if name in BlocksState:
                obj = BlocksState[name]
                pos = obj.get_pos()
                try:
                    pos = pos.cpu().numpy()
                except Exception:
                    pos = np.array(pos, dtype=float)
                print(f"  {name.upper()} @ {pos}")
    print()


# 8 cm between centres leaves a 4 cm gap, clearing the open gripper (span 0.08 m).
# Reach band from a measured workspace sweep: top-down grasp reaches ~0.79 m.
PUTDOWN_CLEARANCE = 0.08
PUTDOWN_R_MIN = 0.32
PUTDOWN_R_MAX = 0.72
TABLE_BLOCK_Z = BLOCK_SIZE / 2  # a 4 cm block resting on the table has its centre at 0.02


def find_free_table_spot(BlocksState, moving_block):
    """Pick a clear, reachable spot on the table to put `moving_block` down.

    Scans a grid and returns the reachable candidate furthest from every other block.
    Deterministic, so runs stay reproducible.
    """
    others = []
    for name, obj in BlocksState.items():
        if name == moving_block:
            continue
        try:
            p = obj.get_pos().cpu().numpy()
        except Exception:
            p = np.array(obj.get_pos(), dtype=float)
        others.append(p[:2])

    best, best_score = None, -1.0
    for x in np.arange(0.32, 0.721, 0.02):
        for y in np.arange(-0.40, 0.501, 0.02):
            r = float(np.hypot(x, y))
            if r < PUTDOWN_R_MIN or r > PUTDOWN_R_MAX:
                continue
            if others:
                d = min(float(np.hypot(x - o[0], y - o[1])) for o in others)
            else:
                d = 1.0
            if d < PUTDOWN_CLEARANCE:
                continue
            # Prefer roomy spots, and among equals prefer closer to the robot (easier IK).
            score = d - 0.15 * r
            if score > best_score:
                best_score, best = score, (float(x), float(y), d)

    if best is None:
        # No clear spot: report it rather than dropping the block onto a neighbour.
        print(f"[PUTDOWN] WARNING: no free table spot with {PUTDOWN_CLEARANCE*100:.0f} cm "
              f"clearance for '{moving_block}'; falling back to the least-bad candidate.")
        return np.array([0.55, 0.0, TABLE_BLOCK_Z])

    x, y, clearance = best
    print(f"[PUTDOWN] free spot for '{moving_block}': ({x:.3f}, {y:.3f}) "
          f"r={np.hypot(x, y):.3f} m, nearest block {clearance*100:.1f} cm away")
    return np.array([x, y, TABLE_BLOCK_Z])


def execute_action(franka, scene, BlocksState, action):
    """
    Execute a single action.

    Supports:
      - Goals 1–3: PICKUP, PUTDOWN, STACK, UNSTACK
      - Goal 4: PICKUP, PUTDOWN, STACK, UNSTACK,
                PICKUP-AT, PUTDOWN-AT, STACK-AT, UNSTACK-AT
    """
    name = action[0]
    args = action[1:]

    print(f"\n[EXEC] ▶ Action: {action}")

    # Allow both with/without hyphens (e.g., "PICKUP-AT" vs "PICKUPAT")
    name_normalized = name.replace("-", "")

    # ---------- Basic actions ----------
    if name_normalized == "PICKUP":
        blk = args[0]
        obj = BlocksState[blk]
        pos = obj.get_pos().cpu().numpy()
        print(f"[EXEC][PICK] {blk} @ {pos}")
        franka.pick(pos, obj=obj)
        return True

    elif name_normalized == "PUTDOWN":
        blk = args[0]
        obj = BlocksState[blk]

        # A clear patch, not a fixed point: successive putdowns must not stack.
        target = find_free_table_spot(BlocksState, blk)
        print(f"[EXEC][PUTDOWN] {blk} at {target}")
        franka.place(target, obj=obj)
        return True

    elif name_normalized == "STACK":
        block_top = args[0]
        block_bottom = args[1]

        top_obj = BlocksState[block_top]
        bottom_obj = BlocksState[block_bottom]

        bottom_link = bottom_obj.links[0]
        bottom_geom = bottom_link.geoms[0]
        bottom_pos = bottom_geom.get_pos().cpu().numpy()

        place_pos = bottom_pos.copy()
        place_pos[2] += BLOCK_SIZE

        print(f"[EXEC][STACK] {block_top} on {block_bottom} @ {place_pos}")
        franka.place(place_pos, obj=top_obj)
        return True

    elif name_normalized == "UNSTACK":
        blk = args[0]
        obj = BlocksState[blk]
        pos = obj.get_pos().cpu().numpy()
        print(f"[EXEC][UNSTACK] {blk} @ {pos}")
        franka.pick(pos, obj=obj)
        return True

    # ---------- Position-aware actions for Goal 4 ----------
    elif name_normalized == "PICKUPAT":
        # (pickup-at ?x ?p)
        blk = args[0]
        pos_name = args[1]
        obj = BlocksState[blk]
        pos = obj.get_pos().cpu().numpy()
        print(f"[EXEC][PICKUP-AT] {blk} from position {pos_name} @ {pos}")
        franka.pick(pos, obj=obj)
        return True

    elif name_normalized == "UNSTACKAT":
        # (unstack-at ?x ?y ?p)
        blk = args[0]
        bottom_blk = args[1]
        pos_name = args[2]
        obj = BlocksState[blk]
        pos = obj.get_pos().cpu().numpy()
        print(f"[EXEC][UNSTACK-AT] {blk} from {bottom_blk} at position {pos_name} @ {pos}")
        franka.pick(pos, obj=obj)
        return True

    elif name_normalized == "PUTDOWNAT":
        # (putdown-at ?x ?p)
        blk = args[0]
        pos_name = args[1]
        obj = BlocksState[blk]
        try:
            from goal4_config import get_position_coords
            target = np.array(get_position_coords(pos_name))
            print(f"[EXEC][PUTDOWN-AT] {blk} at position {pos_name} → {target}")
            franka.place(target, obj=obj)
            return True
        except ImportError:
            print("[EXEC] ERROR: goal4_config not available for PUTDOWN-AT")
            return False
        except KeyError:
            print(f"[EXEC] ERROR: Unknown position {pos_name}")
            return False

    elif name_normalized == "STACKAT":
        # (stack-at ?x ?y ?p-bottom ?p-top)
        block_top = args[0]
        block_bottom = args[1]
        pos_bottom = args[2]  # logical name; we use actual pose
        pos_top = args[3]

        top_obj = BlocksState[block_top]
        bottom_obj = BlocksState[block_bottom]

        bottom_link = bottom_obj.links[0]
        bottom_geom = bottom_link.geoms[0]
        bottom_pos = bottom_geom.get_pos().cpu().numpy()

        place_pos = bottom_pos.copy()
        place_pos[2] += BLOCK_SIZE

        print(f"[EXEC][STACK-AT] {block_top} on {block_bottom} at bottom={pos_bottom}, top={pos_top} → {place_pos}")
        franka.place(place_pos, obj=top_obj)
        return True

    print(f"[EXEC] Unknown action {action}")
    return False


def run_simple_goals(scene, franka, BlocksState, goal_id, max_iterations=20):
    """
    Execute Goals 1-3 with single-action execution per iteration.
    """
    print("\n" + "=" * 80)
    print(f"GOAL {goal_id} DEMO - Simple Block Stacking")
    print("=" * 80)
    print(f"Max iterations: {max_iterations}")
    print("=" * 80)

    recent_actions = []
    loop_detection_window = 6

    for step_idx in range(max_iterations):
        print("\n" + "-" * 80)
        print(f"Iteration {step_idx}")
        print("-" * 80)

        # 1. Symbolic abstraction
        preds = abstract_state(scene, franka, BlocksState, goal_id=goal_id)
        visualize_predicates(preds)
        print_block_positions(BlocksState, goal_id=goal_id, label=f"[BLOCK POS AFTER STEP {step_idx}]")

        # 2. Task planning
        try:
            plan = plan_symbolic(preds, goal_id=goal_id)
        except RuntimeError as e:
            print(f"[task_planner] ERROR during planning: {e}")
            break

        # 3. Check if goal reached
        if not plan:
            print("\n[INFO] ✓ GOAL REACHED!")
            print("=" * 80)
            break

        # 4. Display plan
        print("\n[TASK PLAN]")
        for i, step in enumerate(plan):
            print(f"  {i}: {step}")

        # 5. Loop detection on first action
        action = plan[0]
        recent_actions.append(action)
        if len(recent_actions) > loop_detection_window:
            recent_actions.pop(0)

        if len(recent_actions) >= 3:
            if recent_actions[-1] == recent_actions[-2] == recent_actions[-3]:
                print("\n[WARNING] Detected infinite loop: same action 3 times consecutively")
                print(f"[WARNING] Recent actions: {recent_actions[-3:]}")
                print("[INFO] Stopping execution to avoid infinite loop.")
                print("=" * 80)
                break

        # 6. Execute only the first action
        print(f"\n[EXEC-STEP {step_idx}.0] ▶ {action}")
        action_name = action[0]
        action_args = action[1:]
        action_name_normalized = action_name.replace("-", "")

        positions_before = {
            name: obj.get_pos().cpu().numpy().copy()
            for name, obj in BlocksState.items()
        }

        ok = execute_action(franka, scene, BlocksState, action)
        if not ok:
            print(f"[EXEC] Action failed: {action}. Stopping execution loop.")
            break

        # 7. Physics settling
        if goal_id == 2:
            settling_steps = 200
        elif goal_id in [1, 3]:
            settling_steps = 100
        else:
            settling_steps = 60

        try:
            for _ in range(settling_steps):
                scene.step()
                recording.capture(scene)
        except Exception as e:
            if "Viewer closed" in str(e):
                print("\n[INFO] Viewer was closed. Exiting demo.")
                return
            raise

        # 8. Check for crazy movement when stacking
        if action_name_normalized in ["STACK", "STACKAT"]:
            block_name = action_args[0]
            pos_after = BlocksState[block_name].get_pos().cpu().numpy()
            pos_before = positions_before[block_name]
            displacement = np.linalg.norm(pos_after - pos_before)

            if displacement > 0.5:
                print(f"\n[WARNING] Block {block_name} moved {displacement:.2f}m during physics simulation!")
                print("[WARNING] This suggests placement failure or physics instability.")
                print(f"[WARNING] Before: {pos_before}, After: {pos_after}")

    print("\n[INFO] Demo complete.")


def run_goal4(scene, franka, BlocksState):
    """
    Goal 4 demo: exactly the two-phase behavior from the standalone script.

    Phase 1: Build yellow cross tower (goal_id=41)
    Phase 2: Build green hollow square (goal_id=42)
    """
    print("=" * 80)
    print("GOAL 4 DEMO - Task and Motion Planning")
    print("=" * 80)
    print("\nTarget structures:")
    print("  - Yellow cross/plus tower (12 blocks)")
    print("  - Green hollow square (6 blocks)")
    print("\n" + "=" * 80)

    MAX_STEPS_PER_STRUCTURE = 60
    # One action per perceive-plan cycle, like goals 1-3. Batching 16 acted on
    # perception that was 16 actions stale and drove cubes into each other.
    ACTIONS_PER_BATCH = 1

    # ---------------- PHASE 1: Yellow tower ----------------
    print("\n" + "=" * 80)
    print("PHASE 1: Building Yellow Cross Tower")
    print("=" * 80)

    yellow_done = False
    recent_actions = []

    for step_idx in range(MAX_STEPS_PER_STRUCTURE):
        print("\n" + "-" * 80)
        print(f"YELLOW TOWER - Iteration {step_idx}")
        print("-" * 80)

        # 1. Abstraction for yellow tower
        preds = abstract_state(scene, franka, BlocksState, goal_id=41)
        visualize_predicates(preds, title=f"Yellow tower state - step {step_idx}")

        # 2. Planning
        try:
            plan = plan_symbolic(preds, goal_id=41, problem_name=f"goal41_step{step_idx}")
        except RuntimeError as e:
            print(f"\n[task_planner] ERROR during planning: {e}")
            print("\n[INFO] Planning failed. Stopping execution.")
            break

        if not plan:
            print("\n[INFO] ✓ Yellow tower COMPLETE!")
            yellow_done = True
            break

        print(f"\n[TASK PLAN] ({len(plan)} actions)")
        for i, step in enumerate(plan[:5]):
            print(f"  {i}: {step}")
        if len(plan) > 5:
            print(f"  ... and {len(plan) - 5} more")

        recent_actions.append(plan[0])
        # 3 recurrences in 8 cycles, not 3 in a row: livelocks alternate
        # PICKUP x / PUTDOWN-AT x and never repeat consecutively.
        if len([a for a in recent_actions[-8:] if a == plan[0]]) >= 3:
            print(f"\n[WARNING] Action replanned 3x within 8 cycles with no lasting "
                  f"effect: {plan[0]}")
            print("[INFO] Aborting yellow phase to avoid an infinite loop.")
            break

        actions_to_execute = plan[:ACTIONS_PER_BATCH]
        print(f"\n[BATCH EXECUTION] Executing {len(actions_to_execute)} actions...")

        for action_idx, action in enumerate(actions_to_execute):
            print(f"\n[EXEC-STEP {step_idx}.{action_idx}] Executing: {action}")
            ok = execute_action(franka, scene, BlocksState, action)
            if not ok:
                print(f"\n[EXEC] Action failed: {action}")
                print("[INFO] Stopping execution loop.")
                break

            for _ in range(20):
                scene.step()
                recording.capture(scene)
        else:
            # finished batch with no break, go to next outer iteration
            continue

        # if we broke from the inner loop, break outer loop
        break

    # ---------------- PHASE 2: Green square ----------------
    print("\n" + "=" * 80)
    print("PHASE 2: Building Green Hollow Square")
    print("=" * 80)

    recent_actions = []
    for step_idx in range(MAX_STEPS_PER_STRUCTURE):
        print("\n" + "-" * 80)
        print(f"GREEN SQUARE - Iteration {step_idx}")
        print("-" * 80)

        preds = abstract_state(scene, franka, BlocksState, goal_id=42)
        visualize_predicates(preds, title=f"Green square state - step {step_idx}")

        try:
            plan = plan_symbolic(preds, goal_id=42, problem_name=f"goal42_step{step_idx}")
        except RuntimeError as e:
            print(f"\n[task_planner] ERROR during planning: {e}")
            print("\n[INFO] Planning failed. Stopping execution.")
            break

        if not plan:
            print("\n[INFO] ✓ Green square COMPLETE!")
            # Both structures must be done, not just this phase.
            if yellow_done:
                print("\n" + "=" * 80)
                print("GOAL 4 FULLY COMPLETED!")
                print("=" * 80)
            else:
                print("\n[WARNING] Green square done, but the yellow phase did NOT "
                      "complete, so goal 4 is NOT fully achieved.")
            break

        print(f"\n[TASK PLAN] ({len(plan)} actions)")
        for i, step in enumerate(plan[:5]):
            print(f"  {i}: {step}")
        if len(plan) > 5:
            print(f"  ... and {len(plan) - 5} more")

        recent_actions.append(plan[0])
        if len([a for a in recent_actions[-8:] if a == plan[0]]) >= 3:
            print(f"\n[WARNING] Action replanned 3x within 8 cycles with no lasting "
                  f"effect: {plan[0]}")
            print("[INFO] Aborting green phase to avoid an infinite loop.")
            break

        actions_to_execute = plan[:ACTIONS_PER_BATCH]
        print(f"\n[BATCH EXECUTION] Executing {len(actions_to_execute)} actions...")

        for action_idx, action in enumerate(actions_to_execute):
            print(f"\n[EXEC-STEP {step_idx}.{action_idx}] Executing: {action}")
            ok = execute_action(franka, scene, BlocksState, action)
            if not ok:
                print(f"\n[EXEC] Action failed: {action}")
                print("[INFO] Stopping execution loop.")
                break

            for _ in range(20):
                scene.step()
                recording.capture(scene)
        else:
            continue

        break

    print("\n[INFO] Goal 4 demo complete. Close viewer to exit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified TAMP Demo for All Goals")
    parser.add_argument(
        "--goal",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Goal ID (1-3: simple stacking, 4: two structures)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Genesis backend (cpu or gpu)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum iterations for simple goals (1-3)",
    )
    parser.add_argument(
        "--scene",
        type=int,
        default=1,
        choices=[1, 2],
        help="Initial scene for Goals 1-3 (1: scattered, 2: stacked tower)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the interactive viewer and exit when finished (for automated verification)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to write an mp4 of the run, e.g. runs/goal1/run.mp4",
    )
    parser.add_argument(
        "--video-res",
        type=str,
        default="960x540",
        help="Recording resolution as WxH (default 960x540)",
    )
    parser.add_argument(
        "--video-every",
        type=int,
        default=4,
        help=(
            "Capture one frame every N simulation steps (default 4 = 25 fps of sim time). "
            "Genesis buffers every frame in RAM until the video is encoded, so long runs "
            "must use a larger value: goal 4 at the default was OOM-killed after ~25 min, "
            "having buffered roughly 18000 frames (~12 GB at 640x360)."
        ),
    )
    args = parser.parse_args()

    # Configure recording before the scene is built; scenes.py reads this to decide
    # whether to open a viewer and whether to attach an offscreen camera.
    _w, _h = (int(v) for v in args.video_res.lower().split("x"))
    if args.video:
        os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
    recording.configure(
        headless=args.headless,
        video_path=args.video,
        res=(_w, _h),
        every=args.video_every,
    )

    # Initialize Genesis
    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, logging_level="Warning", logger_verbose_time=False)
    print(f"\n[INFO] Genesis initialized (backend={args.backend})")

    # Scene selection
    if args.goal == 4:
        scene, franka_raw, BlocksState = create_scene_goal4_initial()
    elif args.goal == 3:
        print("[INFO] Using 8-block scene for Goal 3: All blocks scattered on table")
        scene, franka_raw, BlocksState = create_scene_8blocks()
    else:
        if args.scene == 1:
            print("[INFO] Using Scene 1: All blocks scattered on table")
            scene, franka_raw, BlocksState = create_scene_6blocks()
        else:
            print("[INFO] Using Scene 2: Six blocks pre-stacked in tower")
            scene, franka_raw, BlocksState = create_scene_stacked()

    # Begin recording (no-op unless --video was given). Must follow scene.build(),
    # which the scene factories have already done.
    recording.start(scene)

    # Wrap robot in adapter
    franka = RobotAdapter(franka_raw, scene)
    franka.register_blocks(list(BlocksState.values()))

    # Control parameters
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    print("[INFO] Scene + robot ready")
    print(f"[INFO] Total blocks: {len(BlocksState)}")
    print()

    # Run appropriate goal
    if args.goal == 4:
        run_goal4(scene, franka, BlocksState)
    else:
        run_simple_goals(
            scene,
            franka,
            BlocksState,
            goal_id=args.goal,
            max_iterations=args.max_iterations,
        )

    # Settle so the video ends on the final state, not mid-motion.
    for _ in range(300):
        scene.step()
        recording.capture(scene)

    video_path = recording.finish(scene)

    if args.headless:
        # Automated runs must terminate so the exit code is meaningful.
        print("\n[INFO] Headless run complete.")
        if video_path:
            print(f"[INFO] Video: {video_path}")
    else:
        print("\n[INFO] Keeping viewer open. Press Ctrl+C to exit.")
        try:
            while True:
                scene.step()
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down...")
