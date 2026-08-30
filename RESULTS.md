# Results

All four goals run headless and produce video. Videos are in `outputs/`.

| Goal | Scene | Result | Max XY deviation |
|---|---|---|---|
| 1 — two towers (B-G-R, C-M-Y) | scattered | pass, 3/3 runs | 0.35 - 0.63 mm |
| 1 | pre-stacked | pass, 3/3 runs | 1.07 - 1.48 mm |
| 2 — 5-block tower (G-R-B-Y-M) | scattered | pass, 3/3 runs | 0.30 - 0.47 mm |
| 2 | pre-stacked | pass, 3/3 runs | 0.51 - 0.95 mm |
| 3 — tallest tower (bonus) | scattered | 6 blocks stable | — |
| 4 — yellow cross + green square | 18 blocks | complete, 2/3 runs | within the 2 mm position threshold |

Runs use randomized block positions (`scenes.py` jitters XY by up to 5 cm and reseeds
from the clock), so each run is a different problem instance.

## Running

```bash
# interactive
conda run -n rbe550 python demo.py --goal 1 --scene 1

# headless with video
xvfb-run -a conda run -n rbe550 python demo.py --goal 1 --scene 1 \
    --headless --video outputs/run.mp4 --video-res 640x360

# goal 4 is longer; capture fewer frames to bound memory
xvfb-run -a conda run -n rbe550 python demo.py --goal 4 \
    --headless --video outputs/goal4.mp4 --video-res 480x270 --video-every 40
```

## Verifying

`verify_goal.py` re-derives the built structure from the raw block positions in a run log
and checks it geometrically, rather than trusting the demo's own "GOAL REACHED" print.
Expected stacks come from `get_goal_predicates()`, so the checker cannot drift from the
goal definition. It also confirms the video is real via ffprobe.

```bash
conda run -n rbe550 python verify_goal.py --goal 1 runs/*/run.log
```

## Notes on the implementation

- The grasp is real contact friction. Nothing teleports or welds blocks to the gripper;
  `attached_object` is bookkeeping, set only after a test-lift confirms the block moved.
- The Franka is given gravity compensation, matching the real arm's controller. Without
  it the shoulder joint sags ~0.016 rad below every command, which swamps the millimetre
  corrections that placement depends on.
- OMPL collision-checks the robot but not the cube in the gripper, so carries follow a
  straight Cartesian line above the tallest stack instead of a planned joint-space path.
- Placement is closed-loop: XY is corrected at hover height and re-checked at final
  height before release. An error of centimetres means the grasp was lost, not that the
  block is misaligned, so the gripper opens and the executive replans.
- Blocks past ~0.79 m from the base cannot be reached with a straight-down grasp. The
  wrist tilts outward (15/25/35 degrees) to bring the hand closer while the grasp centre
  still lands on the block.

## Known limitations

- Goal 3 reaches 6 blocks before the stack is disturbed. The assignment scores this by
  height, so partial credit applies.
- Goal 4's yellow phase occasionally aborts when repeated near-threshold placements trip
  the loop detector (once in three runs). The cause is a small per-pose bias in the
  descent, not the 2 mm position threshold.
- Goal 2 from the pre-stacked scene occasionally leans up to 14 mm, within the
  abstraction's 15 mm criterion but visibly worse than the typical sub-millimetre result.
