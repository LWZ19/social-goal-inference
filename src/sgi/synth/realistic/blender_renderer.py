"""The Blender renderer: a scene description drawn by Blender, one process for all the frames of a call.

BlenderRenderer honours the Renderer interface of spec.py. A call writes a job folder, runs
blender_side/job.py on it once, and reads back what that wrote. The two sides share that folder and nothing
else:

    job.json           {"spec": to_json(spec), "frames": [...], "settings": {...}, "prepared": path or null,
                        "heights": {avatar: metres}, "signs": {file: art}}
    sign_000.png ...   one picture for each distinct SignArt of the description, drawn here
    000.png ...        written by Blender: the picture of each frame that asked for one, 8-bit sRGB
    000.depth.npy ...  written by Blender: each frame's depth, float32, camera Z through the pixel centre in
                       metres, 0.0 where nothing is hit
    facts.json         written by Blender last: its version, the device it used, the seconds per frame

A frame of the job gives its name, the camera's fx, fy, cx, cy, width, height and T_world_camera, and the
frame's beauty and only. "signs" names, for each picture, the art it shows as to_json writes it, so the
Blender side finds the picture of a sign by the sign's own art. The pictures are drawn by signs.draw, or by
what the renderer was given in its place; once the assets are prepared signs.draw letters them in the asset
library's font SIGN_FONT, and before that in Pillow's own. "prepared" is the folder of the prepared
assets when the preparation covers the whole manifest, and null otherwise: without it Blender paints every
named material plainly and shows a plain sky, and a description that places a model or a person is refused
here, before Blender starts. "heights" gives each avatar that is cast its height as it was made: a person
is as wide as their height is to that one, times their build, and as tall as their height says (a PersonRef's
height is the top of the person as they stand in their pose, as the stub's figure has it).

The depth comes from a pass of its own and is the same, bit for bit, from run to run on one machine. The
pictures are path traced on the graphics card: between two runs about five pixels in a hundred differ, each
by one level of 255.
"""
from __future__ import annotations

import functools
import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from sgi.synth.realistic.blender import BLENDER_SIDE, blender_version, find_blender, run_script
from sgi.synth.realistic.library import Library, is_prepared, load_library
from sgi.synth.realistic.spec import Frame, SceneSpec, SignArt, to_json
from sgi.synth.render import Box

DEVICES = ("OPTIX", "CUDA", "CPU")                 # what Cycles may render on; a card that is not found: the CPU
LIGHTING = ("ceiling", "daylight", "evening")      # the presets of SceneSpec.lighting that blender_side/build.py knows
SIGN_FACES = ("front", "back", "left", "right", "top")     # the faces of a box that can carry a sign, as Box.label_face
PREPARE_COMMAND = ".venv/bin/python -m sgi.synth.realistic.assets prepare"
SIGN_FONT = "sans"                                 # the font of the asset library that signs are lettered in

_JOB_SCRIPT = BLENDER_SIDE / "job.py"
# What every picture is taken with. Cylinders and spheres are meshes of this many sides: with 128 their depth
# stays within 0.2 mm of the round shape away from its silhouette.
_SETTINGS = {"denoiser": "OPENIMAGEDENOISE", "view_transform": "AgX", "look": "AgX - Punchy", "segments": 128}
# How long a job may take: on the CPU a picture of a bare room took 16 s where the graphics card took 1 s.
_BASE_TIMEOUT_S, _TIMEOUT_PER_FRAME_S = 1800.0, 120.0


def _prepared_library() -> Library | None:
    """The asset library when its preparation covers the whole manifest; None before that, and with a manifest
    that does not read."""
    try:
        library = load_library()
        return library if is_prepared(library) else None
    except Exception:                              # a manifest that is being written: nothing is prepared yet
        return None


def _sign_arts(spec: SceneSpec) -> list[SignArt]:
    """Each distinct art of the description's signs, in the order they come. A sign that Blender could not put on
    its part is an error here."""
    arts: list[SignArt] = []
    for number, part in enumerate(spec.parts):
        for sign in part.signs:
            if not isinstance(part.solid, Box):
                raise ValueError(f"part {number} carries a sign and is a {type(part.solid).__name__}: "
                                 "a sign lies on a face of a box")
            if sign.face not in SIGN_FACES:
                raise ValueError(f"part {number} has a sign on its {sign.face!r} face: expected one of "
                                 f"{', '.join(SIGN_FACES)}")
            if sign.art not in arts:
                arts.append(sign.art)
    return arts


def _frame_json(name: str, frame: Frame) -> dict:
    camera = frame.camera
    return {"name": name, "fx": float(camera.fx), "fy": float(camera.fy), "cx": float(camera.cx),
            "cy": float(camera.cy), "width": int(camera.width), "height": int(camera.height),
            "T_world_camera": np.asarray(camera.T_world_camera, dtype=np.float64).tolist(),
            "beauty": bool(frame.beauty), "only": None if frame.only is None else [int(group) for group in frame.only]}


class BlenderRenderer:
    """Draws scene descriptions with Blender's path tracer. Making one asks the binary for its version, so it
    raises BlenderMissing where there is no Blender of the series."""
    name = "blender"

    def __init__(self, samples: int = 64, device: str = "OPTIX", work_dir: Path | None = None, keep: bool = False,
                 draw: Callable[[SignArt], np.ndarray] | None = None):
        """`samples` per pixel of a picture; `device` one of DEVICES; job folders are made under `work_dir`, or in
        the system's temporary folder, and removed after each call unless `keep` is set. `draw` gives the
        picture of a sign's art as H x W x 3 uint8; left out, it is signs.draw, in the asset library's font
        SIGN_FONT once the assets are prepared."""
        if device not in DEVICES:
            raise ValueError(f"unknown device {device!r}: expected one of {', '.join(DEVICES)}")
        if int(samples) < 1:
            raise ValueError(f"samples is {samples!r}: a picture needs one sample per pixel at the least")
        self.samples, self.device, self.keep = int(samples), device, bool(keep)
        self.work_dir = None if work_dir is None else Path(work_dir)
        self._draw = draw
        self.version: str | None = blender_version(find_blender())
        self.last_job: Path | None = None          # the folder of the last call's job, while it is kept
        self.last_facts: dict | None = None        # what Blender said of the last job it finished

    def render(self, spec: SceneSpec, frames: Sequence[Frame]) -> list[tuple[np.ndarray | None, np.ndarray]]:
        """Per frame: rgb uint8 (None when beauty is off) and depth float32, camera Z through each pixel centre,
        0.0 where nothing is hit.

        All frames are drawn by one Blender process. ValueError for a description Blender could not draw (an
        unknown lighting preset, a sign that is not on a face of a box, an avatar the library does not know)
        and RuntimeError for one that needs assets that are not prepared, both before Blender starts;
        RuntimeError when the run fails or leaves a frame unwritten."""
        frames = list(frames)
        if not frames:
            return []
        if spec.lighting not in LIGHTING:
            raise ValueError(f"unknown lighting preset {spec.lighting!r}: expected one of {', '.join(LIGHTING)}")
        arts = _sign_arts(spec)
        library = _prepared_library()
        if library is None and (spec.models or spec.people):
            raise RuntimeError(f"the description places {len(spec.models)} models and {len(spec.people)} people, "
                               "which come from the prepared assets, and the assets are not prepared. "
                               f"Prepare them with:\n    {PREPARE_COMMAND}")
        unknown = sorted({person.avatar for person in spec.people} - set(library.avatars if library else ()))
        if unknown:
            raise ValueError(f"the description casts avatars the asset library does not list: {', '.join(unknown)}")
        heights = {person.avatar: library.avatars[person.avatar].natural_height for person in spec.people}
        draw = self._draw if self._draw is not None or not arts else _own_drawing(library)
        pictures = {f"sign_{number:03d}.png": _drawn(draw, art) for number, art in enumerate(arts)}

        if self.work_dir is not None:
            self.work_dir.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="job_", dir=self.work_dir)).absolute()
        self.last_job, self.last_facts = (folder if self.keep else None), None
        try:
            for name, picture in pictures.items():
                Image.fromarray(picture).save(folder / name)
            names = [f"{number:03d}" for number in range(len(frames))]
            job = {"spec": to_json(spec), "frames": [_frame_json(name, frame) for name, frame in zip(names, frames)],
                   "settings": {"samples": self.samples, "device": self.device, **_SETTINGS},
                   "prepared": None if library is None else str(library.prepared), "heights": heights,
                   "signs": {name: to_json(art) for name, art in zip(pictures, arts)}}
            (folder / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
            run_script(_JOB_SCRIPT, ["--job", folder],
                       timeout_s=max(_BASE_TIMEOUT_S, _TIMEOUT_PER_FRAME_S * len(frames)))
            results = [_read(folder, name, frame) for name, frame in zip(names, frames)]
            self.last_facts = json.loads((folder / "facts.json").read_text(encoding="utf-8"))
            return results
        finally:
            if not self.keep:
                shutil.rmtree(folder, ignore_errors=True)


def _own_drawing(library: Library | None) -> Callable[[SignArt], np.ndarray]:
    """What draws a sign's art when the renderer was given nothing to draw it with: signs.draw, lettering in the
    asset library's font SIGN_FONT where the assets are prepared and that file is there, and in Pillow's own
    font otherwise."""
    from sgi.synth.realistic import signs          # asked for with the first sign: a scene without one needs no font
    font = None if library is None else library.fonts.get(SIGN_FONT)
    return functools.partial(signs.draw, font=font if font is not None and Path(font).is_file() else None)


def _drawn(draw: Callable[[SignArt], np.ndarray], art: SignArt) -> np.ndarray:
    """The picture of a sign's art as `draw` gives it, which has to be a picture."""
    picture = np.asarray(draw(art))
    if picture.dtype != np.uint8 or picture.ndim != 3 or picture.shape[2] != 3 or 0 in picture.shape:
        raise ValueError(f"the drawing of the {art.style!r} sign {art.text!r} is {picture.dtype} of shape "
                         f"{picture.shape}: expected H x W x 3 uint8")
    return picture


def _read(folder: Path, name: str, frame: Frame) -> tuple[np.ndarray | None, np.ndarray]:
    """What Blender wrote for a frame. A file that is not there, or is not of the frame's size, is an error."""
    shape = (frame.camera.height, frame.camera.width)
    depth_file, picture_file = folder / f"{name}.depth.npy", folder / f"{name}.png"
    if not depth_file.is_file() or (frame.beauty and not picture_file.is_file()):
        missing = depth_file if not depth_file.is_file() else picture_file
        raise RuntimeError(f"Blender finished without writing {missing.name} of the job in {folder}")
    depth = np.load(depth_file)
    if depth.shape != shape or depth.dtype != np.float32:
        raise RuntimeError(f"{depth_file.name} is {depth.dtype} of shape {depth.shape}: expected float32 of {shape}")
    if not frame.beauty:
        return None, depth
    with Image.open(picture_file) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape != (*shape, 3):
        raise RuntimeError(f"{picture_file.name} has the shape {rgb.shape}: expected {(*shape, 3)}")
    return rgb, depth
