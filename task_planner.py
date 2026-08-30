"""
Task planning: perceived predicates -> PDDL problem -> pyperplan -> action list.

  plan_symbolic(predicates, goal_id) -> [("PICKUP", "r"), ("STACK", "r", "g"), ...]

Goals 1-3 use A* (optimal, small problems); goal 4 uses enforced hill-climbing over its
position-aware domain. An empty plan means the goal already holds, which is success.
"""

from typing import List, Tuple, Set
from pathlib import Path

from pddl.problem_generator import make_problem_pddl, DOMAIN_FILE

# Domain file mapping for different goals
def get_domain_file(goal_id: int) -> str:
    """Return the appropriate domain file for the given goal_id."""
    if goal_id in [4, 41, 42]:
        return str(Path("pddl") / "domain_blocks_goal4.pddl")
    else:
        return DOMAIN_FILE


Action = Tuple[str, ...]  # e.g. ("STACK", "r", "g")


def run_pyperplan(domain_file: str, problem_file: str, goal_id: int = 1) -> str:
    """
    Call pyperplan using its Python API directly.

    Args:
        domain_file: Path to PDDL domain file
        problem_file: Path to PDDL problem file
        goal_id: Goal identifier (used to select appropriate planner)

    Returns:
        plan_text: the plan actions (one action per line).
    """
    try:
        # Import pyperplan modules
        from pyperplan import planner
        from pyperplan.heuristics.relaxation import hFFHeuristic

        # Use A* for optimal plans on small problems (Goals 1-3)
        # Use Enforced Hill-Climbing for faster planning on large problems (Goal 4)
        if goal_id in [4, 41, 42]:
            from pyperplan.search import enforced_hillclimbing_search
            print("[task_planner] Running pyperplan with Enforced Hill-Climbing (fast for Goal 4)...")
            plan = planner.search_plan(domain_file, problem_file, enforced_hillclimbing_search, hFFHeuristic)
        else:
            from pyperplan.search import astar_search
            print("[task_planner] Running pyperplan with A* (optimal for Goals 1-3)...")
            plan = planner.search_plan(domain_file, problem_file, astar_search, hFFHeuristic)
        
        if plan is None:
            raise RuntimeError("Pyperplan returned no plan (unsolvable problem?)")
        
        # Convert plan operators to string format
        # Each operator in plan is an Op object with a .name attribute like "(pickup r)"
        plan_actions = [str(op.name) for op in plan]
        
        # Write the plan to .soln file
        problem_path = Path(problem_file)
        soln_path = problem_path.with_suffix(".soln")
        plan_text = '\n'.join(plan_actions)
        soln_path.write_text(plan_text)
        
        print(f"[task_planner] Plan found with {len(plan_actions)} actions")
        print(f"[task_planner] Plan saved to {soln_path}")
        
        return plan_text
        
    except Exception as e:
        print(f"[task_planner] Error running pyperplan: {e}")
        raise


def parse_plan(plan_text: str) -> List[Action]:
    """
    Parse a .soln plan file into a list of action tuples.

    Typical .soln format (pyperplan):
        (pickup r)
        (stack r g)
        (pickup y)
        (stack y m)

        Goal 4 actions:
        (putdown-at y1 pos_r1_c2_bottom)
        (stack-at y3 y1 pos_r1_c2_bottom)

    We:
      - ignore empty lines and comment lines starting with ';'
      - expect each non-empty line to contain exactly one '(action ...)' term
      - action name -> upper-case (PUTDOWN-AT), args -> lower-case
      - supports variable-length arguments for Goal 4 position actions
    """
    plan: List[Action] = []

    for line in plan_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(";"):
            continue

        if "(" not in line or ")" not in line:
            continue

        try:
            start = line.index("(")
            end = line.rindex(")")
        except ValueError:
            continue

        inside = line[start + 1:end].strip()
        if not inside:
            continue

        tokens = inside.split()
        if not tokens:
            continue

        action_name = tokens[0].upper()
        args = [t.lower() for t in tokens[1:]]
        plan.append((action_name, *args))

    return plan


def plan_symbolic(current_predicates: Set[Tuple],
                  goal_id: int,
                  problem_name: str = "demo_goal") -> List[Action]:
    """
    Plan from the current perceived state toward `goal_id`.

    Input:
        current_predicates: set of tuples from abstract_state(...)
        goal_id: which goal to plan for (1 = two towers, etc.)
        problem_name: base name for the generated PDDL problem file

    Output:
        plan: list of actions as tuples, e.g.
              [
                ("UNSTACK", "r", "g"),
                ("PUTDOWN", "r"),
                ("PICKUP", "y"),
                ("STACK", "y", "m"),
                ...
              ]

        If no plan is found or parsing fails, we raise RuntimeError.
    """
    # 1) Generate PDDL problem file from current predicates + goal spec
    problem_file = make_problem_pddl(
        current_predicates=current_predicates,
        goal_id=goal_id,
        problem_name=problem_name,
    )

    # 2) Get the appropriate domain file for this goal
    domain_file = get_domain_file(goal_id)

    print(f"[task_planner] Using DOMAIN:  {domain_file}")
    print(f"[task_planner] Using PROBLEM: {problem_file}")

    # 3) Run pyperplan (extracts plan from stdout or .soln) and read plan text
    # Pass goal_id to select appropriate planner (A* for 1-3, EHC for 4)
    plan_text = run_pyperplan(domain_file, problem_file, goal_id=goal_id)

    # 4) Parse the plan text
    plan = parse_plan(plan_text)

    if not plan:
        # Empty plan = goal already holds. run_pyperplan() already raised for the
        # genuinely unsolvable case.
        print("[task_planner] Empty plan: goal already satisfied in the current state.")
        return []

    print("[task_planner] Parsed plan:")
    for i, act in enumerate(plan):
        print(f"  {i}: {act}")

    # Goal-4 plans are returned in planner order. Bottom-before-top is already enforced
    # by the domain: stack-at requires (at-position ?y ?p-bottom).
    return plan


# ---- Quick standalone test (optional) ----

if __name__ == "__main__":
    from symbolic_abstraction import abstract_state
    from scenes import create_scene_6blocks
    import genesis as gs
    import numpy as np

    print("\n[task_planner] Standalone test: plan from current 6-block scene\n")

    gs.init(backend=gs.cpu, logging_level="Warning", logger_verbose_time=False)
    scene, robot, blocks_state = create_scene_6blocks()
    robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    for _ in range(100):
        scene.step()

    current_predicates = abstract_state(scene, robot, blocks_state)
    print("[task_planner] Current predicates:", current_predicates)

    plan = plan_symbolic(current_predicates, goal_id=1, problem_name="test_goal1")
    print("\n[task_planner] Final plan returned by plan_symbolic():")
    for step, action in enumerate(plan):
        print(f"  {step}: {action}")