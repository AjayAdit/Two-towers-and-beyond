"""Independent verification that a run actually achieved its goal.

The demo prints "GOAL REACHED" whenever the planner returns an empty plan. That is a
claim about the ABSTRACTED state, so it is only as trustworthy as the abstraction,
which is itself code under test. This module re-derives the structure from the raw block
positions in the run log and checks it against the goal geometrically, plus checks that
the video is real.

Usage:
    python verify_goal.py --goal 1 runs/verify/goal1_run1/run.log [more logs...]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

BLOCK = 0.04
XY_TOL = 0.010   # 10 mm: stricter than the abstraction's 15 mm, since a good stack is <1 mm
Z_TOL = 0.005

def goal_stacks(goal_id):
    """Derive bottom->top stacks from the goal spec itself.

    Hardcoding the expected structure would duplicate the goal definition and let the two
    drift apart, so read it from get_goal_predicates() instead.
    """
    from pddl.problem_generator import get_goal_predicates

    preds = get_goal_predicates(goal_id)
    on = {p[1]: p[2] for p in preds if p[0] == "on"}       # upper -> lower
    bases = [p[1] for p in preds if p[0] == "ontable"]

    above = {}
    for upper, lower in on.items():
        above[lower] = upper

    stacks = []
    for base in sorted(bases):
        stack, cur = [base], base
        while cur in above:
            cur = above[cur]
            stack.append(cur)
        if len(stack) > 1:          # a lone block on the table is not a stack
            stacks.append(stack)
    return stacks

POS_RE = re.compile(r"^\s*([A-Z][A-Z0-9]*)\s*@\s*\[([^\]]+)\]")


def final_positions(log_path):
    """Return {block: (x,y,z)} from the last '[BLOCK POS AFTER STEP n]' section."""
    text = Path(log_path).read_text(errors="replace")
    sections = text.split("[BLOCK POS AFTER STEP")
    if len(sections) < 2:
        return {}
    last = sections[-1]
    out = {}
    for line in last.splitlines()[1:]:
        if line.strip().startswith("["):  # next log section
            if out:
                break
            continue
        m = POS_RE.match(line)
        if m:
            name = m.group(1).lower()
            nums = [float(v) for v in m.group(2).split()]
            if len(nums) == 3:
                out[name] = tuple(nums)
    return out


def check_stack(pos, stack):
    """Verify one bottom->top stack. Returns (ok, [problems], max_xy_dev, max_z_dev)."""
    problems = []
    max_xy = 0.0
    max_z = 0.0

    missing = [b for b in stack if b not in pos]
    if missing:
        return False, [f"blocks absent from log: {missing}"], 0.0, 0.0

    bx, by, bz = pos[stack[0]]
    if abs(bz - BLOCK / 2) > 0.01:
        problems.append(f"base '{stack[0]}' not on table: z={bz:.4f}, expected {BLOCK/2:.4f}")

    for lower, upper in zip(stack, stack[1:]):
        lx, ly, lz = pos[lower]
        ux, uy, uz = pos[upper]
        xy = ((ux - lx) ** 2 + (uy - ly) ** 2) ** 0.5
        dz = uz - lz
        max_xy = max(max_xy, xy)
        max_z = max(max_z, abs(dz - BLOCK))
        if xy > XY_TOL:
            problems.append(f"'{upper}' on '{lower}': XY off by {xy*1000:.1f} mm (limit {XY_TOL*1000:.0f})")
        if abs(dz - BLOCK) > Z_TOL:
            problems.append(f"'{upper}' on '{lower}': Z gap {dz*1000:.1f} mm (expected {BLOCK*1000:.0f})")

    return not problems, problems, max_xy, max_z


def check_video(path):
    if not path or not Path(path).exists():
        return False, "no video file"
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=nb_frames", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        d = json.loads(out.stdout)
        dur = float(d.get("format", {}).get("duration", 0))
        frames = int(d.get("streams", [{}])[0].get("nb_frames", 0))
        if frames < 30 or dur < 1.0:
            return False, f"degenerate video: {frames} frames, {dur:.1f}s"
        return True, f"{frames} frames, {dur:.1f}s"
    except Exception as e:
        return False, f"ffprobe failed: {type(e).__name__}"


def verify(log_path, goal):
    log_path = Path(log_path)
    text = log_path.read_text(errors="replace")

    pos = final_positions(log_path)
    try:
        stacks = goal_stacks(goal)
    except Exception as e:
        print(f"{log_path}: cannot derive stacks for goal {goal}: {type(e).__name__}: {e}")
        return False
    if not stacks:
        print(f"{log_path}: goal {goal} specifies no stacks to verify")
        return False

    ok_all = True
    details, xy_devs, z_devs = [], [], []
    for stack in stacks:
        ok, problems, mxy, mz = check_stack(pos, stack)
        xy_devs.append(mxy)
        z_devs.append(mz)
        if not ok:
            ok_all = False
            details += problems

    claimed = "GOAL REACHED" in text
    vids = list(log_path.parent.glob("*.mp4"))
    vid_ok, vid_msg = check_video(vids[0] if vids else None)

    verdict = "PASS" if (ok_all and vid_ok) else "FAIL"
    print(f"\n{verdict}  {log_path}")
    print(f"  demo claimed GOAL REACHED : {claimed}")
    print(f"  geometric structure check : {'ok' if ok_all else 'FAILED'}"
          f"   max XY dev {max(xy_devs)*1000:.2f} mm, max Z dev {max(z_devs)*1000:.2f} mm")
    print(f"  video                     : {'ok' if vid_ok else 'FAILED'} ({vid_msg})")
    if claimed != ok_all:
        print("  !! demo's claim disagrees with the geometry; trust the geometry")
    for d in details:
        print(f"    - {d}")
    return verdict == "PASS"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", type=int, required=True)
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()

    results = [verify(p, a.goal) for p in a.logs]
    n_ok = sum(results)
    print(f"\n==== goal {a.goal}: {n_ok}/{len(results)} runs passed ====")
    sys.exit(0 if n_ok == len(results) else 1)
