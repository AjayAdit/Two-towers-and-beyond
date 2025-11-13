import genesis as gs
import numpy as np
import torch
from typing import Any

from genesis.utils.misc import tensor_to_array
from robot_adapter import RobotAdapter


def _ensure_adapter(robot: Any, scene: Any) -> RobotAdapter:
    """Wrap raw genesis robot in RobotAdapter if needed.

    This keeps PlannerInterface backward-compatible with callers that may
    still pass the raw genesis entity.
    """
    if isinstance(robot, RobotAdapter):
        return robot
    return RobotAdapter(robot, scene)

class PlannerInterface:
    def __init__(self, robot: Any, scene: Any):
        # ensure we have a RobotAdapter so the rest of the code can rely on a
        # stable interface (but attribute access is forwarded to the raw robot)
        self.robot = _ensure_adapter(robot, scene)
        self.scene = scene
        self.attached_object = None

    def diagnose_bounds_violation(self, si, state):
        # print the bounds the current state is violating
        violated_bounds = []
        for i_q in range(self.robot.n_qs):
            val = state[i_q]
            low = si.getStateSpace().getBounds().low[i_q]
            high = si.getStateSpace().getBounds().high[i_q]
            if val < low or val > high:
                violated_bounds.append((i_q, val, low, high))
        gs.logger.warning(f"State violates bounds on joints: {violated_bounds}")

    def diagnose_valid_violation(self, state):
        # set robot to the candidate start and check collisions / joint violations
        self.robot.set_qpos(self._ompl_state_to_tensor(state))
        print(self.robot.get_qpos())
        collision_pairs = self.robot.detect_collision()
        if collision_pairs.any() and len(collision_pairs) > 0:
            bad_links = set()
            for a, b in collision_pairs:
                # print(self.scene.rigid_solver.geoms[a])
                print(self.scene.rigid_solver.geoms[a].link.name)
                # print(self.scene.rigid_solver.geoms[b])
                print(self.scene.rigid_solver.geoms[b].link.name)
                bad_links.add(self.scene.rigid_solver.geoms[a].link.name)
                bad_links.add(self.scene.rigid_solver.geoms[b].link.name)
            gs.logger.warning(f"State causes collisions between links: {sorted(bad_links)}")

    def plan_path(
            self,
            qpos_goal,
            qpos_start=None,
            timeout=5.0,
            smooth_path=True,
            num_waypoints=100,
            attached_object=None,
            planner="RRTConnect",
    ):
        """
        Plan a path from `qpos_start` to `qpos_goal`.

        Parameters
        ----------
        qpos_goal : array_like
            The goal state.
        qpos_start : None | array_like, optional
            The start state. If None, the current state of the rigid entity will be used. Defaults to None.
        timeout : float, optional
            The maximum time (in seconds) allowed for the motion planning algorithm to find a solution. Defaults to 5.0.
        smooth_path : bool, optional
            Whether to smooth the path after finding a solution. Defaults to True.
        num_waypoints : int, optional
            The number of waypoints to interpolate the path. If None, no interpolation will be performed. Defaults to 100.
        ignore_collision : bool, optional
            Whether to ignore collision checking during motion planning. Defaults to False.
        ignore_joint_limit : bool, optional
            Whether to ignore joint limits during motion planning. Defaults to False.
        planner : str, optional
            The name of the motion planning algorithm to use. Supported planners: 'PRM', 'RRT', 'RRTConnect', 'RRTstar', 'EST', 'FMT', 'BITstar', 'ABITstar'. Defaults to 'RRTConnect'.

        Returns
        -------
        waypoints : list
            A list of waypoints representing the planned path. Each waypoint is an array storing the entity's qpos of a single time step.
        """

        ########## validate ##########
        try:
            from ompl import base as ob
            from ompl import geometric as og
            from ompl import util as ou

            ou.setLogLevel(ou.LOG_ERROR)
        except:
            gs.raise_exception(
                    "Failed to import OMPL. Did you install? (For installation instructions, see https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html#optional-motion-planning)"
            )

        supported_planners = [
            "PRM",
            "RRT",
            "RRTConnect",
            "RRTstar",
            "EST",
            "FMT",
            "BITstar",
            "ABITstar",
        ]
        if planner not in supported_planners:
            gs.raise_exception(f"Planner {planner} is not supported. Supported planners: {supported_planners}.")

        if self.robot._solver.n_envs > 0:
            gs.raise_exception("Motion planning is not supported for batched envs (yet).")

        if self.robot.n_qs != self.robot.n_dofs:
            gs.raise_exception("Motion planning is not yet supported for rigid entities with free joints.")

        qpos_cur = self.robot.get_qpos()

        if qpos_start is None:
            qpos_start = self.robot.get_qpos()
        qpos_start = tensor_to_array(qpos_start)
        qpos_goal = tensor_to_array(qpos_goal)

        if qpos_start.shape != (self.robot.n_qs,) or qpos_goal.shape != (self.robot.n_qs,):
            gs.raise_exception("Invalid shape for `qpos_start` or `qpos_goal`.")

        ######### process joint limit ##########

        # ensure we use numpy float64 for bounds
        q_limit_lower = np.asarray(self.robot.q_limit[0], dtype=float)
        q_limit_upper = np.asarray(self.robot.q_limit[1], dtype=float)

        ######### setup OMPL ##########
        space = ob.RealVectorStateSpace(self.robot.n_qs)
        bounds = ob.RealVectorBounds(self.robot.n_qs)

        for i_q in range(self.robot.n_qs):
            # pass native Python float (double) to OMPL to match C++ signature
            bounds.setLow(i_q, float(q_limit_lower[i_q]))
            bounds.setHigh(i_q, float(q_limit_upper[i_q]))
        space.setBounds(bounds)
        ss = og.SimpleSetup(space)

        self.attached_object = attached_object
        
        ss.setStateValidityChecker(ob.StateValidityCheckerFn(self._is_ompl_state_valid))
        ss.setPlanner(getattr(og, planner)(ss.getSpaceInformation()))

        state_start = ob.State(space)
        state_goal = ob.State(space)
        for i_q in range(self.robot.n_qs):
            state_start[i_q] = float(qpos_start[i_q])
            state_goal[i_q] = float(qpos_goal[i_q])
        # Diagnostic: check start/goal satisfy bounds and are valid according to the state validity checker
        si = ss.getSpaceInformation()
        start_in_bounds = bool(si.satisfiesBounds(state_start.get()))
        if not start_in_bounds:
            gs.logger.warning(f"OMPL start state out of bounds")
            self.diagnose_bounds_violation(si, state_start.get())

        goal_in_bounds = bool(si.satisfiesBounds(state_goal.get()))
        if not goal_in_bounds:
            gs.logger.warning(f"OMPL goal state out of bounds")
            self.diagnose_bounds_violation(si, state_goal)

        start_valid = bool(si.isValid(state_start.get()))
        if not start_valid:
            gs.logger.warning(f"OMPL start state invalid")
            self.diagnose_valid_violation(state_start)

        goal_valid = bool(si.isValid(state_goal.get()))
        if not goal_valid:
            gs.logger.warning(f"OMPL goal state invalid")
            self.diagnose_valid_violation(state_goal)

        # set start/goal in OMPL
        ss.setStartAndGoalStates(state_start, state_goal)
        ss.setup()

        ######### solve ##########
        solved = ss.solve(timeout)
        waypoints = []
        if solved:
            gs.logger.info("Path solution found successfully.")
            path = ss.getSolutionPath()
            if smooth_path:
                ss.simplifySolution()

            path.interpolate(num_waypoints)
            print("Number of waypoints in path:", path.getStateCount())
            waypoints = self._ompl_states_to_tensor_list(path.getStates())
        else:
            gs.logger.warning("Path planning failed. Returning empty path.")

        ########## restore original state #########
        self.robot.set_qpos(qpos_cur)

        return waypoints

    def _is_ompl_state_valid(self, state):
        self.robot.set_qpos(self._ompl_state_to_tensor(state))
        collision_pairs = self.robot.detect_collision()

        if not len(collision_pairs):
            return True

        if not self.attached_object:
            return False

        return self.collision_with_attached_object(collision_pairs)

    def collision_with_attached_object(self, collision_pairs):
        finger_names = {"left_finger", "right_finger", "hand"}
        for a, b in collision_pairs:
            name_a = self.scene.rigid_solver.geoms[a].link.name
            name_b = self.scene.rigid_solver.geoms[b].link.name
            if (name_a in finger_names and b == self.attached_object.idx) or \
                 (name_b in finger_names and a == self.attached_object.idx):
                continue
            return False
        return True

    def _ompl_states_to_tensor_list(self, states):
        tensor_list = []
        for state in states:
            tensor_list.append(self._ompl_state_to_tensor(state))
        return tensor_list

    def _ompl_state_to_tensor(self, state):
        tensor = torch.empty(self.robot.n_qs, dtype=gs.tc_float, device=gs.device)
        for i in range(self.robot.n_qs):
            tensor[i] = state[i]
        return tensor