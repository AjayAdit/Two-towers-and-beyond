"""Scene factory helpers.

Provide functions to create common demo scenes. Each factory returns a
tuple (scene, franka, blocks_state, end_effector) to be used by demos.
"""
from typing import Any, Dict, Tuple
import random
import time
random.seed(time.time())

import numpy as np
import genesis as gs
import recording
from robot_adapter import RobotAdapter


def _build_base_scene(camera_pos=(3, -1, 1.5), camera_lookat=(0.0, 0.0, 0.5)) -> gs.Scene:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=8),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=not recording.is_headless(),
    )
    # Offscreen recording camera, framed like the viewer. No-op unless recording
    # was configured by the entry point; must be added before scene.build().
    recording.attach_camera(scene, pos=camera_pos, lookat=camera_lookat, fov=30)
    return scene

def _elevate_robot_base(franka: Any) -> None:
    """Slightly raise robot base to avoid initial collisions."""
    # Get position as tensor (might be on GPU)
    base_pos_tensor = franka.get_pos()
    
    # Move to CPU if necessary and convert to numpy
    if base_pos_tensor.is_cuda:
        base_pos = base_pos_tensor.cpu().numpy()
    else:
        base_pos = base_pos_tensor.numpy()
    
    new_pos = base_pos.copy()
    new_pos[2] += 0.01
    franka.set_pos(new_pos) 


def _rand_xy(base, noise=0.05):
        dx = random.uniform(-noise, noise)
        dy = random.uniform(-noise, noise)
        return (base[0] + dx, base[1] + dy, base[2])

def create_scene_6blocks() -> Tuple[Any, Any, Dict[str, Any], Any]:
    """Create the default demo scene (layout 1).

    Returns:
        scene, franka_adapter, blocks_state, end_effector
    """
    scene = _build_base_scene()

    # basic geometry
    plane = scene.add_entity(gs.morphs.Plane())
    # add some random noise up to 5 cm in x/y


    posR = _rand_xy((0.65, 0.0, 0.02))
    posG = _rand_xy((0.65, 0.2, 0.02))
    posB = _rand_xy((0.65, 0.4, 0.02))
    posY = _rand_xy((0.45, 0.0, 0.02))
    posM = _rand_xy((0.45, 0.2, 0.02))
    posC = _rand_xy((0.45, 0.4, 0.02))

    cubeR = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posR),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cubeG = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posG),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cubeB = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posB),
        surface=gs.options.surfaces.Plastic(color=(0.0, 0.0, 1.0)),
    )
    cubeY = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posY),
        surface=gs.options.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
    )
    cubeM = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posM),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0, 1.0)),
    )

    cubeC = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posC),
        surface=gs.options.surfaces.Plastic(color=(0, 1.0, 1.0)),
    )

    franka_raw = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        # The real Panda compensates gravity in its controller. Without this joint 2
        # sags 0.0158 rad below every command, swamping millimetre corrections.
        material=gs.materials.Rigid(gravity_compensation=1.0),
    )
    franka = RobotAdapter(franka_raw, scene)

    # build scene (construct physics/visuals)
    scene.build()

    # initial robot pose (7 arm joints + 2 gripper fingers)
    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))

    # slightly raise robot base to avoid initial collisions
    _elevate_robot_base(franka)

    blocks_state: Dict[str, Any] = {"r": cubeR, "g": cubeG, "b": cubeB, "y": cubeY, "m": cubeM, "c": cubeC}

    return scene, franka, blocks_state


def create_scene_stacked() -> Tuple[Any, Any, Dict[str, Any], Any]:
    """Create an alternative demo scene (layout 2) with cube positions. one on top of the other."""
    scene = _build_base_scene(camera_pos=(2.5, -1.2, 1.2), camera_lookat=(0.6, 0.0, 0.2))

    plane = scene.add_entity(gs.morphs.Plane())

    # slightly different positions
    startx, starty, _ = _rand_xy((0.45, 0.0, 0.02), noise=0.2)
    cubeR = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.02)),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cubeG = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.06)),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cubeB = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.10)),
        surface=gs.options.surfaces.Plastic(color=(0.0, 0.0, 1.0)),
    )
    cubeY = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.14)),
        surface=gs.options.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
    )

    cubeM = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.18)),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0, 1.0)),
    )

    cubeC = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.22)),
        surface=gs.options.surfaces.Plastic(color=(0, 1.0, 1.0)),
    )

    franka_raw = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        # The real Panda compensates gravity in its controller. Without this joint 2
        # sags 0.0158 rad below every command, swamping millimetre corrections.
        material=gs.materials.Rigid(gravity_compensation=1.0),
    )
    franka = RobotAdapter(franka_raw, scene)
    scene.build()

    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))

    _elevate_robot_base(franka)

    blocks_state: Dict[str, Any] = {"r": cubeR, "g": cubeG, "b": cubeB, "y": cubeY, "m": cubeM, "c": cubeC}

    return scene, franka, blocks_state

def create_scene_8blocks() -> Tuple[Any, Any, Dict[str, Any]]:
    """Create scene with 8 blocks scattered on table (for Goal 3).

    Adds 2 new blocks to the original 6:
    - o (orange): Orange block
    - p (purple): Purple block

    Returns:
        scene, franka_adapter, blocks_state
    """
    scene = _build_base_scene()

    # basic geometry
    plane = scene.add_entity(gs.morphs.Plane())

    # Original 6 blocks positions
    posR = _rand_xy((0.65, 0.0, 0.02))
    posG = _rand_xy((0.65, 0.2, 0.02))
    posB = _rand_xy((0.65, 0.4, 0.02))
    posY = _rand_xy((0.45, 0.0, 0.02))
    posM = _rand_xy((0.45, 0.2, 0.02))
    posC = _rand_xy((0.45, 0.4, 0.02))

    # 2 new blocks positions
    posO = _rand_xy((0.55, -0.2, 0.02))  # Orange
    posP = _rand_xy((0.55, 0.5, 0.02))   # Purple

    # Original 6 blocks
    cubeR = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posR),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cubeG = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posG),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cubeB = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posB),
        surface=gs.options.surfaces.Plastic(color=(0.0, 0.0, 1.0)),
    )
    cubeY = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posY),
        surface=gs.options.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
    )
    cubeM = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posM),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 1.0)),
    )
    cubeC = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posC),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 1.0)),
    )

    # 2 new blocks
    cubeO = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posO),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.5, 0.0)),  # Orange
    )
    cubeP = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posP),
        surface=gs.options.surfaces.Plastic(color=(0.5, 0.0, 0.5)),  # Purple
    )

    franka_raw = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        # The real Panda compensates gravity in its controller. Without this joint 2
        # sags 0.0158 rad below every command, swamping millimetre corrections.
        material=gs.materials.Rigid(gravity_compensation=1.0),
    )
    franka = RobotAdapter(franka_raw, scene)

    # build scene (construct physics/visuals)
    scene.build()

    # initial robot pose (7 arm joints + 2 gripper fingers)
    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))

    # slightly raise robot base to avoid initial collisions
    _elevate_robot_base(franka)

    blocks_state: Dict[str, Any] = {
        "r": cubeR, "g": cubeG, "b": cubeB,
        "y": cubeY, "m": cubeM, "c": cubeC,
        "o": cubeO, "p": cubeP
    }

    return scene, franka, blocks_state

def create_scene_goal4_initial() -> Tuple[Any, Any, Dict[str, Any]]:
    """Create Goal 4 scene with blocks SCATTERED on table (initial state for TAMP).

    This creates 18 blocks randomly positioned on the table:
    - Yellow blocks: y1-y12 (12 blocks)
    - Green blocks: g1-g6 (6 blocks)

    Total: 18 blocks, all scattered on the table surface

    Returns:
        scene, franka_adapter, blocks_state
    """
    scene = _build_base_scene(camera_pos=(2.5, -1.5, 1.2), camera_lookat=(0.6, 0.0, 0.15))

    plane = scene.add_entity(gs.morphs.Plane())

    BLOCK_SIZE = 0.04  # 4cm cubes

    # Yellow color (pure yellow)
    yellow_color = (1.0, 1.0, 0.0)

    # Green color (pure green)
    green_color = (0.0, 1.0, 0.0)

    blocks_state: Dict[str, Any] = {}

    # ===== CREATE 12 YELLOW BLOCKS (scattered) =====
    # Place scattered blocks AWAY from build zone (build zone is x=0.5-0.8)
    # Increased spacing between blocks (x gap 0.10, y gap 0.20)
    yellow_base_positions = [
        (0.35, -0.20), (0.35, 0.00), (0.35, 0.20), (0.35, 0.40),
        (0.45, -0.20), (0.45, 0.00), (0.45, 0.20), (0.45, 0.40),
        (0.55, -0.20), (0.55, 0.00), (0.55, 0.20), (0.55, 0.40),
    ]

    for i, (base_x, base_y) in enumerate(yellow_base_positions, start=1):
        block_name = f"y{i}"
        pos = _rand_xy((base_x, base_y, 0.02), noise=0.03)  # Reduced noise for cleaner separation

        cube = scene.add_entity(
            gs.morphs.Box(size=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), pos=pos),
            surface=gs.options.surfaces.Plastic(color=yellow_color),
        )
        blocks_state[block_name] = cube

    # ===== CREATE 6 GREEN BLOCKS (scattered) =====
    # Place green blocks on the sides, aligned with the wider grid
    green_base_positions = [
        (0.25, -0.40), (0.35, -0.40), (0.45, -0.40),
        (0.25, 0.60), (0.35, 0.60), (0.45, 0.60),
    ]

    for i, (base_x, base_y) in enumerate(green_base_positions, start=1):
        block_name = f"g{i}"
        pos = _rand_xy((base_x, base_y, 0.02), noise=0.03)

        cube = scene.add_entity(
            gs.morphs.Box(size=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), pos=pos),
            surface=gs.options.surfaces.Plastic(color=green_color),
        )
        blocks_state[block_name] = cube

    # Add robot
    franka_raw = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        # The real Panda compensates gravity in its controller. Without this joint 2
        # sags 0.0158 rad below every command, swamping millimetre corrections.
        material=gs.materials.Rigid(gravity_compensation=1.0),
    )
    franka = RobotAdapter(franka_raw, scene)

    # Build scene
    scene.build()

    # Initial robot pose (retracted, out of the way)
    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))

    _elevate_robot_base(franka)

    return scene, franka, blocks_state
