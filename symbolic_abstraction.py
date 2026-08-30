"""Symbolic abstraction: continuous scene geometry -> discrete PDDL predicates.

This is the "lifting" step of the TAMP loop. It reports what it OBSERVES, never what
was intended: holding is ground-truthed against the gripper's attachment state, and a
block that is neither on the table nor cleanly on another block is reported as neither.

Emits (ontable ?x), (on ?x ?y), (clear ?x), (holding ?x), (handempty), plus
(at-position ?x ?p) and (position-free ?p) for goal 4.
"""

from typing import Dict, Set, Tuple, Any
import numpy as np


# Constants - Block properties
BLOCK_SIZE = 0.04  # 4cm cubes
BLOCK_HALF_HEIGHT = BLOCK_SIZE / 2  # 0.02m

# Tuning thresholds (experimentally determined)
TABLE_HEIGHT = 0.0  # Table Z-coordinate
TABLE_THRESHOLD = 0.03  # Tolerance for "on table" check
XY_ALIGNMENT_THRESHOLD = 0.015  # XY centering tolerance for stacking
Z_STACK_THRESHOLD = 0.005  # Z-distance tolerance for "on" relation
HOLDING_THRESHOLD = 0.05  # Distance threshold for gripper holding block


def get_block_positions(blocks_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Query all block positions from the Genesis scene.

    Args:
        blocks_state: Dictionary mapping block names to Genesis entities
                     {"r": cubeR, "g": cubeG, "b": cubeB, ...}

    Returns:
        Dictionary mapping block names to [x, y, z] positions
        Example: {"r": [0.65, 0.0, 0.02], "g": [0.65, 0.2, 0.02], ...}
    """
    positions = {}
    for block_name, block_entity in blocks_state.items():
        pos = block_entity.get_pos()
        # Convert to numpy array for easier manipulation
        try:
            positions[block_name] = pos.cpu().numpy()
        except:
            positions[block_name] = np.array(pos, dtype=float)

    return positions


def get_gripper_state(robot: Any) -> Tuple[np.ndarray, bool]:
    """Get robot gripper position and state.

    Args:
        robot: RobotAdapter instance

    Returns:
        Tuple of (gripper_position, is_closed)
        - gripper_position: [x, y, z] numpy array
        - is_closed: True if gripper is closed (grasping), False if open
    """
    # Get end effector (hand) link position
    hand_link = robot.get_link("hand")
    pos = hand_link.get_pos()
    try:
        gripper_pos = pos.cpu().numpy()
    except Exception:
        gripper_pos = np.array(pos, dtype=float)

    # Get gripper joint positions (last 2 DOFs are gripper fingers)
    qpos = robot.get_qpos()
    if hasattr(qpos, "cpu"):
        try:
            qpos_cpu = qpos.detach().cpu().numpy()
        except Exception:
            qpos_cpu = np.array(qpos, dtype=float)
    else:
        qpos_cpu = np.array(qpos, dtype=float)

    # Defensive: ensure we have at least two values for fingers
    if qpos_cpu.size >= 2:
        gripper_width = float(qpos_cpu[-2] + qpos_cpu[-1])
    else:
        gripper_width = float(np.sum(qpos_cpu)) if qpos_cpu.size > 0 else 0.0

    # Gripper is considered closed if width is small
    is_closed = gripper_width < 0.04

    return gripper_pos, is_closed


def is_on_table(block_pos: np.ndarray, table_height: float = TABLE_HEIGHT,
                threshold: float = TABLE_THRESHOLD) -> bool:
    """Check if a block is on the table.

    Args:
        block_pos: [x, y, z] position of block center
        table_height: Z-coordinate of table surface (default: 0.0)
        threshold: Tolerance for height check

    Returns:
        True if block is on table, False otherwise

    Note:
        Block positions are at their center. A 4cm block sitting on the table
        has its center at Z = 0.02 (half the block height above table).
    """
    expected_z = table_height + BLOCK_HALF_HEIGHT
    return abs(block_pos[2] - expected_z) < threshold


def is_on(block_a_pos: np.ndarray, block_b_pos: np.ndarray,
          xy_threshold: float = XY_ALIGNMENT_THRESHOLD,
          z_threshold: float = Z_STACK_THRESHOLD) -> bool:
    """Check if block_a is directly on top of block_b.

    Args:
        block_a_pos: [x, y, z] position of top block (block A)
        block_b_pos: [x, y, z] position of bottom block (block B)
        xy_threshold: Tolerance for XY alignment (centering)
        z_threshold: Tolerance for Z distance

    Returns:
        True if block_a is on block_b, False otherwise

    Logic:
        - XY distance must be small (blocks are centered)
        - Z distance must equal one block height (0.04m)
    """
    # Check XY alignment (blocks should be centered on each other)
    xy_distance = np.linalg.norm(block_a_pos[:2] - block_b_pos[:2])
    if xy_distance > xy_threshold:
        return False

    # Check Z distance (should be one block height)
    z_distance = block_a_pos[2] - block_b_pos[2]
    expected_z_diff = BLOCK_SIZE

    return abs(z_distance - expected_z_diff) < z_threshold


def is_clear(block_name: str, all_block_positions: Dict[str, np.ndarray]) -> bool:
    """Check if a block has nothing on top of it.

    Args:
        block_name: Name of block to check (e.g., "r", "g", "b")
        all_block_positions: Dictionary of all block positions

    Returns:
        True if no other block is on top of this block, False otherwise
    """
    target_pos = all_block_positions[block_name]

    # Check if any other block is on top of this one
    for other_name, other_pos in all_block_positions.items():
        if other_name == block_name:
            continue

        if is_on(other_pos, target_pos):
            return False

    return True

def is_holding(block_name: str,
               robot: Any,
               blocks_state: Dict[str, Any]) -> bool:
    """
    Ground-truth holding check using RobotAdapter.attached_object.
    """
    attached = robot.attached_object
    block_obj = blocks_state[block_name]

    is_holding_block = (attached is block_obj)

    print(f"[HOLDING] block={block_name}, "
          f"attached_obj={attached}, "
          f"is_holding={is_holding_block}")

    return is_holding_block



def abstract_state(scene: Any, robot: Any, blocks_state: Dict[str, Any],
                   goal_id: int = 1) -> Set[Tuple]:
    """Convert continuous scene state into symbolic PDDL predicates.

    This is the main function for symbolic abstraction (lifting).

    Args:
        scene: Genesis scene object
        robot: RobotAdapter instance
        blocks_state: Dictionary mapping block names to Genesis entities
        goal_id: Goal identifier (1-3: standard blocks, 4: position-aware)

    Returns:
        Set of tuples representing true predicates

    Example output:
        {
            ("ontable", "r"),     # Red block on table
            ("ontable", "b"),     # Blue block on table
            ("on", "g", "r"),     # Green block on red block
            ("clear", "b"),       # Blue block is clear
            ("clear", "g"),       # Green block is clear
            ("handempty",)        # Robot not holding anything
        }

    Predicates (Goals 1-3):
        - (ontable, block): block is on the table
        - (on, block_a, block_b): block_a is on top of block_b
        - (clear, block): nothing is on top of block
        - (holding, block): robot is holding block
        - (handempty,): robot gripper is empty

    Additional Predicates (Goal 4):
        - (at-position, block, position): block is at named position
        - (position-free, position): position is not occupied
    """
    predicates = set()

    # Get all current positions
    block_positions = get_block_positions(blocks_state)
    gripper_pos, gripper_closed = get_gripper_state(robot)

    # Evaluate predicates for each block
    for block_name in blocks_state.keys():
        block_pos = block_positions[block_name]

        # Check if block is on table
        if is_on_table(block_pos):
            predicates.add(("ontable", block_name))

        # Check if block is clear (nothing on top)
        if is_clear(block_name, block_positions):
            predicates.add(("clear", block_name))

        # Check if robot is holding this block
        if is_holding(block_name, robot, blocks_state):
            predicates.add(("holding", block_name))

        # Check "on" relationships with all other blocks
        for other_name in blocks_state.keys():
            if other_name == block_name:
                continue

            other_pos = block_positions[other_name]
            if is_on(block_pos, other_pos):
                predicates.add(("on", block_name, other_name))

    # Check if hand is empty
    if robot.attached_object is None:
        predicates.add(("handempty",))

    # Goal 4 (and sub-goals 41, 42): Add position predicates
    if goal_id in [4, 41, 42]:
        try:
            from goal4_config import (
                GOAL4_POSITION_COORDS,
                POSITION_XY_THRESHOLD,
                POSITION_Z_THRESHOLD,
                get_all_position_names
            )

            # Track which positions are occupied
            occupied_positions = set()

            # DEBUG: Track yellow blocks detection
            yellow_blocks_detected = []
            yellow_blocks_missing = []

            # OPTIMIZATION: Pre-convert all position coords to numpy arrays (cache)
            position_coords_np = {
                pos_name: np.array(coords)
                for pos_name, coords in GOAL4_POSITION_COORDS.items()
            }

            # Check each block's position at goal locations
            for block_name in blocks_state.keys():
                block_pos = block_positions[block_name]

                # OPTIMIZED: Manual position search with cached numpy arrays
                pos_name = None
                for candidate_pos_name, target_pos in position_coords_np.items():
                    # Fast vectorized distance check
                    xy_distance = np.linalg.norm(block_pos[:2] - target_pos[:2])
                    if xy_distance <= POSITION_XY_THRESHOLD:
                        z_distance = abs(block_pos[2] - target_pos[2])
                        if z_distance <= POSITION_Z_THRESHOLD:
                            pos_name = candidate_pos_name
                            break

                if pos_name is not None:
                    predicates.add(("at-position", block_name, pos_name))
                    occupied_positions.add(pos_name)
                    if block_name.startswith('y'):
                        yellow_blocks_detected.append((block_name, pos_name))
                else:
                    if block_name.startswith('y'):
                        yellow_blocks_missing.append((block_name, block_pos))

            # DEBUG: Print summary for yellow blocks
            if goal_id == 41 and (yellow_blocks_missing or len(yellow_blocks_detected) < 12):
                print(f"\n[YELLOW BLOCKS DEBUG] Goal 41 perception:")
                print(f"  Detected: {len(yellow_blocks_detected)}/12 yellow blocks")
                for name, pos in sorted(yellow_blocks_detected):
                    print(f"    ✓ {name} at {pos}")
                if yellow_blocks_missing:
                    print(f"  Missing: {len(yellow_blocks_missing)} yellow blocks")
                    from goal4_config import GOAL4_POSITION_COORDS
                    for name, actual_pos in sorted(yellow_blocks_missing):
                        print(f"    ✗ {name} at [{actual_pos[0]:.4f}, {actual_pos[1]:.4f}, {actual_pos[2]:.4f}]")
                        # Find closest expected position
                        min_dist = float('inf')
                        closest_pos = None
                        for expected_name, expected_coords in GOAL4_POSITION_COORDS.items():
                            if not expected_name.startswith('pos_r'):
                                continue
                            expected = np.array(expected_coords)
                            xy_dist = np.linalg.norm(actual_pos[:2] - expected[:2])
                            z_dist = abs(actual_pos[2] - expected[2])
                            total_dist = np.sqrt(xy_dist**2 + z_dist**2)
                            if total_dist < min_dist:
                                min_dist = total_dist
                                closest_pos = (expected_name, xy_dist, z_dist)
                        if closest_pos:
                            print(f"      Closest: {closest_pos[0]} (XY={closest_pos[1]*1000:.1f}mm, Z={closest_pos[2]*1000:.1f}mm)")
                print()

            # Mark free positions
            all_positions = get_all_position_names()
            for pos_name in all_positions:
                if pos_name not in occupied_positions:
                    predicates.add(("position-free", pos_name))

        except ImportError:
            # Goal 4 config not available, skip position predicates
            pass

    return predicates


def visualize_predicates(predicates: Set[Tuple], title: str = "Current State") -> None:
    """Print predicates in human-readable format.

    Args:
        predicates: Set of predicate tuples
        title: Optional title for the output
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

    # Sort predicates by type for better readability
    ontable = [p for p in predicates if p[0] == "ontable"]
    on_relations = [p for p in predicates if p[0] == "on"]
    clear = [p for p in predicates if p[0] == "clear"]
    holding = [p for p in predicates if p[0] == "holding"]
    handempty = [p for p in predicates if p[0] == "handempty"]
    at_position = [p for p in predicates if p[0] == "at-position"]
    position_free = [p for p in predicates if p[0] == "position-free"]

    if ontable:
        print("\nBlocks on table:")
        for pred in sorted(ontable):
            print(f"  - {pred[1].upper()} is on table")

    if on_relations:
        print("\nStacking relationships:")
        for pred in sorted(on_relations):
            print(f"  - {pred[1].upper()} is on {pred[2].upper()}")

    if clear:
        print("\nClear blocks (nothing on top):")
        for pred in sorted(clear):
            print(f"  - {pred[1].upper()} is clear")

    if holding:
        print("\nRobot holding:")
        for pred in sorted(holding):
            print(f"  - Holding {pred[1].upper()}")

    if handempty:
        print("\nRobot hand: EMPTY")

    if at_position:
        print(f"\nBlocks at positions ({len(at_position)} occupied):")
        for pred in sorted(at_position):
            print(f"  - {pred[1].upper()} at {pred[2]}")

    if position_free:
        print(f"\nFree positions ({len(position_free)} available):")
        # Only show first 10 to avoid clutter
        for pred in sorted(position_free)[:10]:
            print(f"  - {pred[1]}")
        if len(position_free) > 10:
            print(f"  ... and {len(position_free) - 10} more")

    print(f"{'='*60}\n")


