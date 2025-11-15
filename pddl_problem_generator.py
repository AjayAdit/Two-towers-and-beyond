#!/usr/bin/env python3
import os

def generate_problem_pddl(blocks_state, predicates, goal="two_blocks_move", save_path="test.pddl"):
    blocks = " ".join(blocks_state.keys())

    # --- Define goals based on assignment (Goal 1) ---
    if goal == "two_towers":
        goal_expr = "(and (on r g) (on g b) (on y m) (on m c))"
    elif goal == "five_block":
        goal_expr = "(and (on m y) (on y b) (on b r) (on r g))"
    elif goal == "two_blocks_move":
       goal_expr = "(and (moved r) (moved g))"
    else:
        raise ValueError(f"Unknown goal type: {goal}")

    # --- Reformat predicates to valid lowercase PDDL ---
    def format_pred(p):
        p = p.strip().lower()
        # convert "on(r,g)" → "(on r g)"
        if p.startswith("on("):
            a, b = p[3:-1].split(",")
            return f"(on {a.strip()} {b.strip()})"
        elif p.startswith("ontable("):
            a = p[8:-1]
            return f"(ontable {a.strip()})"
        elif p.startswith("clear("):
            a = p[6:-1]
            return f"(clear {a.strip()})"
        elif p.startswith("holding("):
            a = p[8:-1]
            return f"(holding {a.strip()})"
        elif "handempty" in p:
            return "(handempty)"
        else:
            return f"({p})"

    init_facts = "\n        ".join(format_pred(p) for p in sorted(predicates))

    pddl_str = f"""(define (problem {goal})
        (:domain blocksworld)
        (:objects {blocks} - block)
        (:init
                {init_facts}
        )
        (:goal {goal_expr})
        )"""

    # --- Save and show ---
    with open(save_path, "w") as f:
        f.write(pddl_str)

    print(f"[✅] Generated problem PDDL at: {os.path.abspath(save_path)}")
    print(pddl_str)
    return save_path

def generate_solution_file(save_path="test.pddl.soln"):
    """Generate a simple demo solution plan for moving two blocks (r, g)."""
    soln_str = "(pick r)\n(putdown r)\n(pick g)\n(putdown g)\n"
    with open(save_path, "w") as f:
        f.write(soln_str)
    print(f"[✅] Generated symbolic plan at: {os.path.abspath(save_path)}")
    print(soln_str)

if __name__ == "__main__":
    blocks_state = {"r": None, "g": None}
    predicates = [
        "ontable(r)",
        "ontable(g)",
        "clear(r)",
        "clear(g)",
        "handempty"
    ]
    generate_problem_pddl(blocks_state, predicates, goal="two_blocks_move", save_path="test.pddl")
    generate_solution_file("test.pddl.soln")
