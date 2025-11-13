#!/usr/bin/env python3
import numpy as np
import torch

def extract_predicates(scene, franka, blocks_state, grasp_threshold=0.03, stack_tolerance=0.01):
    """
    Convert continuous Genesis scene state into PDDL-style predicates.
    """
    predicates = set()
    block_keys = list(blocks_state.keys())

    # 🔧 Ensure all positions are numpy arrays
    block_positions = {}
    for k in block_keys:
        pos = blocks_state[k].get_pos()
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().numpy()
        block_positions[k] = np.array(pos, dtype=float)

    # 1️⃣ Detect ON(x,y) and ONTABLE(x)
    block_height = 0.04
    table_z = 0.0

    for a in block_keys:
        pos_a = block_positions[a]
        found_support = False

        for b in block_keys:
            if a == b:
                continue
            pos_b = block_positions[b]
            same_xy = np.linalg.norm(pos_a[:2] - pos_b[:2]) < 0.03
            height_diff = abs((pos_a[2] - pos_b[2]) - block_height)
            if same_xy and height_diff < stack_tolerance:
                predicates.add(f"ON({a.upper()},{b.upper()})")
                found_support = True
                break

        if not found_support and pos_a[2] - table_z < block_height * 1.5:
            predicates.add(f"ONTABLE({a.upper()})")

    # 2️⃣ Detect CLEAR(x)
    for a in block_keys:
        pos_a = block_positions[a]
        clear = True
        for b in block_keys:
            if a == b:
                continue
            pos_b = block_positions[b]
            same_xy = np.linalg.norm(pos_a[:2] - pos_b[:2]) < 0.03
            height_diff = (pos_b[2] - pos_a[2])
            if same_xy and 0 < height_diff < block_height * 1.5:
                clear = False
                break
        if clear:
            predicates.add(f"CLEAR({a.upper()})")

    # 3️⃣ Detect HANDEMPTY() or HOLDING(x)
    def _to_np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x, dtype=float)

    gripper_left = _to_np(franka.get_link("left_finger").get_pos())
    gripper_right = _to_np(franka.get_link("right_finger").get_pos())
    grip_center = (gripper_left + gripper_right) / 2.0

    holding_block = None
    for k, pos in block_positions.items():
        # 🔧 Ensure same type before subtraction
        if np.linalg.norm(pos - grip_center) < grasp_threshold:
            holding_block = k
            break

    if holding_block:
        predicates.add(f"HOLDING({holding_block.upper()})")
    else:
        predicates.add("HANDEMPTY()")

    return predicates
