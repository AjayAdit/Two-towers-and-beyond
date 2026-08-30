"""Offscreen video recording for headless verification runs.

The demos are normally watched in the Genesis viewer. For automated verification we
need to run without a viewer and still produce a video artifact, so this module owns
a single offscreen camera and the frame-capture cadence.

It lives in its own module so that both `scenes.py` (which creates the camera) and
`robot_adapter.py` (which drives most of the simulation stepping) can use it without
importing each other.

Usage:
    import recording
    recording.configure(headless=True, video_path="runs/goal1/run.mp4")
    ...                                  # scenes.py attaches the camera at build time
    recording.capture(scene)             # after every scene.step()
    recording.finish(scene)              # writes the mp4

Recording is entirely opt-in: with no `configure()` call every function here is a
no-op and the demos behave exactly as before.
"""

from typing import Optional, Tuple

# Module configuration, set once by the entry point before the scene is built.
_headless: bool = False
_video_path: Optional[str] = None
_res: Tuple[int, int] = (960, 540)
_fps: int = 30

# Simulation runs at dt=0.01, i.e. 100 steps per simulated second. Capturing one
# frame every `_every` steps gives 100/_every fps of simulated time; at the default
# of 4 that is 25 fps, which is smooth without making rendering the bottleneck.
_every: int = 4


def configure(
    headless: bool = False,
    video_path: Optional[str] = None,
    res: Tuple[int, int] = (960, 540),
    fps: int = 30,
    every: int = 4,
) -> None:
    """Enable headless mode and/or video capture. Call before building the scene."""
    global _headless, _video_path, _res, _fps, _every
    _headless = headless
    _video_path = video_path
    _res = res
    _fps = fps
    _every = max(1, int(every))


def is_headless() -> bool:
    """True if the viewer should be suppressed."""
    return _headless


def is_recording() -> bool:
    """True if frames should be captured."""
    return _video_path is not None


def attach_camera(scene, pos, lookat, fov: float = 30.0) -> None:
    """Add the offscreen camera to `scene`. Must be called BEFORE scene.build().

    The camera handle and frame counter are stored on the scene itself so that any
    module holding a scene reference can drive capture without extra plumbing.
    """
    scene.record_cam = None
    scene.record_counter = 0

    if not is_recording():
        return

    scene.record_cam = scene.add_camera(
        res=_res,
        pos=tuple(pos),
        lookat=tuple(lookat),
        fov=fov,
        GUI=False,
    )
    scene.record_started = False


def start(scene) -> None:
    """Begin recording. Must be called AFTER scene.build()."""
    cam = getattr(scene, "record_cam", None)
    if cam is None:
        return
    cam.start_recording()
    scene.record_started = True
    print(f"[RECORDING] started -> {_video_path} ({_res[0]}x{_res[1]}, every {_every} steps)")


def capture(scene) -> None:
    """Render one frame if it is time to. Safe to call after every scene.step()."""
    cam = getattr(scene, "record_cam", None)
    if cam is None or not getattr(scene, "record_started", False):
        return

    n = getattr(scene, "record_counter", 0)
    scene.record_counter = n + 1
    if n % _every == 0:
        cam.render()


def finish(scene) -> Optional[str]:
    """Encode and write the video. Returns the path written, or None."""
    cam = getattr(scene, "record_cam", None)
    if cam is None or not getattr(scene, "record_started", False):
        return None

    frames = getattr(scene, "record_counter", 0) // _every
    print(f"[RECORDING] encoding {frames} frames -> {_video_path}")
    cam.stop_recording(save_to_filename=_video_path, fps=_fps)
    scene.record_started = False
    print(f"[RECORDING] saved {_video_path}")
    return _video_path
