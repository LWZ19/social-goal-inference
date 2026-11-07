"""Taking the frames of a job: the scene format's camera, the picture, and the exact depth. Runs inside Blender.

Conventions
  world     Z up, metres, the floor on z = 0: Blender's own world, nothing to convert.
  camera    the scene format's optical frame (+Z forward, +X right, +Y down) and Blender's camera (-Z forward,
            +X right, +Y up) differ by a half turn about X: T_world_blender = T_world_camera @ diag(1, -1, -1, 1).
  depth     the Cycles Depth pass, which for a perspective camera is the camera's Z (the distance along its
            axis, not the length of the ray), taken by one ray through the centre of each pixel. A ray that
            meets nothing comes back as 1e10 and is written as 0.0.
  arrays    row 0 is the top row of the image, as in the scene format.

The depth is a pass of its own for every frame: one sample, a box filter a hundredth of a pixel wide so that
the sample goes through the pixel centre, no bounce, no denoiser, written as 32-bit float and saved as .npy.
The Depth pass of a picture rendered with many samples is not used: on real meshes it differs from this one at
silhouette pixels. Glass is a surface in it: Cycles writes the first surface a camera ray meets, and a pane
refracts the ray, it does not let it through as a transparent surface would.

A frame of some object groups alone hides, for that frame, the room and everything of another group or of
none (hide_render), and shows it again afterwards; the lights stay, and write no depth.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

from build import GROUP

BLENDER_FROM_OPTICAL = np.diag([1.0, -1.0, -1.0, 1.0])
_MISS = 1.0e9                                      # the Depth pass holds 1e10 where a ray met nothing
_CLIP_M = (0.05, 200.0)                            # nothing nearer or further is drawn
_SENSOR_MM = 36.0
_PICTURE_FILTER = ("BLACKMAN_HARRIS", 1.5)         # Cycles' own default: what a picture is smoothed with
_CENTRE_FILTER = ("BOX", 0.01)                     # every sample within a hundredth of a pixel of the centre
_MAX_SEED = 2 ** 31 - 1


def set_camera(scene, frame: dict):
    """The scene's camera as the pinhole of the frame: fx, fy, cx, cy for an image of width x height, at the
    optical-frame pose T_world_camera.

    The sensor is fitted across, so fx = lens * width / sensor width whatever the image's shape, and fy follows
    from the pixel aspect: fy = fx * aspect_x / aspect_y. The principal point is the lens shift, which Blender
    counts in image widths along both axes. Pixel (i, j) has its centre at (i + 0.5, j + 0.5)."""
    camera = scene.camera
    if camera is None:
        camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
        scene.collection.objects.link(camera)
        scene.camera = camera
    width, height = int(frame["width"]), int(frame["height"])
    lens = camera.data
    lens.type, lens.sensor_fit, lens.sensor_width, lens.lens_unit = "PERSP", "HORIZONTAL", _SENSOR_MM, "MILLIMETERS"
    lens.lens = frame["fx"] * _SENSOR_MM / width
    taller = frame["fx"] / frame["fy"]             # aspect_y / aspect_x of a pixel
    scene.render.pixel_aspect_x = 1.0 if taller >= 1.0 else 1.0 / taller
    scene.render.pixel_aspect_y = taller if taller >= 1.0 else 1.0
    lens.shift_x = (width / 2.0 - frame["cx"]) / width
    lens.shift_y = (frame["cy"] - height / 2.0) * taller / width
    lens.clip_start, lens.clip_end = _CLIP_M
    lens.dof.use_dof = False
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = width, height, 100
    scene.render.use_border = False
    pose = np.asarray(frame["T_world_camera"], dtype=float) @ BLENDER_FROM_OPTICAL
    camera.matrix_world = Matrix(pose.tolist())
    return camera


def configure(scene, settings: dict, seed: int, exposure: float) -> str:
    """Cycles as every frame of the job uses it. Returns the device that renders: the one asked for, or "CPU"
    when Blender finds no such device on this machine."""
    scene.render.engine = "CYCLES"
    cycles = scene.cycles
    device = settings["device"]
    if device != "CPU":
        found = False
        try:
            preferences = bpy.context.preferences.addons["cycles"].preferences
            preferences.compute_device_type = device
            preferences.refresh_devices()
            for entry in preferences.devices:
                entry.use = entry.type == device
                found = found or entry.use
        except Exception as error:                 # no driver, no such kind of device in this build
            print(f"sgi: no {device} device ({error}); rendering on the CPU", flush=True)
        if not found:
            device = "CPU"
    cycles.device = "CPU" if device == "CPU" else "GPU"
    cycles.use_adaptive_sampling = False           # a fixed count, so that a frame costs and looks the same each time
    cycles.seed = int(seed) % _MAX_SEED
    cycles.use_animated_seed = False
    cycles.denoiser = settings["denoiser"]
    cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cycles.denoising_use_gpu = device != "CPU"
    cycles.caustics_reflective = cycles.caustics_refractive = False
    cycles.blur_glossy = 1.0
    scene.render.use_persistent_data = True        # the scene stays on the device from frame to frame
    scene.view_layers[0].use_pass_z = True
    scene.view_settings.view_transform = settings["view_transform"]
    scene.view_settings.look = settings["look"]
    scene.view_settings.exposure = exposure
    scene.view_settings.gamma = 1.0
    scene.display_settings.display_device = "sRGB"
    scene.render.use_file_extension = False
    for name in dir(scene.render):                 # no date, render time or host name in a file's metadata
        if name.startswith("use_stamp"):
            setattr(scene.render, name, False)
    return device


@contextlib.contextmanager
def quiet(log: Path | None):
    """Send what the process prints (Blender prints its progress from C) to `log` for the duration."""
    if log is None:
        yield
        return
    sys.stdout.flush()
    saved, target = os.dup(1), os.open(str(log), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(target, 1)
    os.close(target)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


@contextlib.contextmanager
def alone(scene, groups):
    """For the duration, only the things of these object groups are rendered: the room and everything of another
    group or of none is hidden. None hides nothing."""
    hidden = []
    if groups is not None:
        wanted = {int(group) for group in groups}
        for thing in scene.objects:
            if thing.type in ("LIGHT", "CAMERA") or thing.hide_render:
                continue
            if GROUP not in thing or int(thing[GROUP]) not in wanted:
                thing.hide_render = True
                hidden.append(thing)
    try:
        yield
    finally:
        for thing in hidden:
            thing.hide_render = False


def _render_to(scene, path: Path, log: Path | None) -> None:
    scene.render.filepath = str(Path(path).resolve())
    with quiet(log):
        bpy.ops.render.render(write_still=True)
    if not Path(path).is_file():
        raise RuntimeError(f"Blender rendered and did not write {path}")


def render_picture(scene, path: Path, samples: int, log: Path | None = None) -> None:
    """The picture as an 8-bit sRGB PNG: path traced with `samples` per pixel and denoised."""
    cycles, image = scene.cycles, scene.render.image_settings
    cycles.samples, cycles.use_denoising = int(samples), True
    cycles.pixel_filter_type, cycles.filter_width = _PICTURE_FILTER
    image.file_format, image.color_mode, image.color_depth, image.compression = "PNG", "RGB", "8", 15
    _render_to(scene, path, log)


def render_depth(scene, path: Path, log: Path | None = None) -> np.ndarray:
    """The exact depth of the frame, saved as float32 .npy at `path` and returned: camera Z in metres through
    each pixel centre, 0.0 where nothing is hit."""
    cycles, image = scene.cycles, scene.render.image_settings
    bounces = cycles.max_bounces
    cycles.samples, cycles.use_denoising, cycles.max_bounces = 1, False, 0
    cycles.pixel_filter_type, cycles.filter_width = _CENTRE_FILTER
    image.file_format, image.color_mode, image.color_depth, image.exr_codec = "OPEN_EXR_MULTILAYER", "RGBA", "32", "ZIP"
    layers = Path(path).with_suffix(".exr")
    try:
        _render_to(scene, layers, log)
        z = _layer(layers, "Depth.Z")
    finally:
        cycles.max_bounces = bounces
        layers.unlink(missing_ok=True)
    seen = np.isfinite(z) & (z > 0) & (z < _MISS)
    depth = np.where(seen, z, 0.0).astype(np.float32)
    np.save(str(path), depth)
    return depth


def _layer(path: Path, channel: str) -> np.ndarray:
    """One channel of a multilayer OpenEXR file as float32 (H, W), row 0 at the top. The file names it after its
    view layer, as in "ViewLayer.Depth.Z"."""
    import OpenImageIO as oiio                     # comes with Blender
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise OSError(f"cannot open {path}: {oiio.geterror()}")
    try:
        described = source.spec()
        names = list(described.channelnames)
        matching = [number for number, name in enumerate(names) if name == channel or name.endswith("." + channel)]
        if len(matching) != 1:
            raise RuntimeError(f"{path} holds the channels {names}: expected one that ends in {channel}")
        pixels = np.asarray(source.read_image("float")).reshape(described.height, described.width, described.nchannels)
    finally:
        source.close()
    return pixels[:, :, matching[0]].copy()


def render_frames(scene, frames: list[dict], out_dir: Path, samples: int, log: Path | None = None) -> list[float]:
    """Every frame of the job, in order: <name>.png where a picture is asked for, and <name>.depth.npy.
    Returns the seconds each frame took."""
    out_dir, seconds = Path(out_dir), []
    for frame in frames:
        began = time.perf_counter()
        set_camera(scene, frame)
        with alone(scene, frame["only"]):
            if frame["beauty"]:
                render_picture(scene, out_dir / f"{frame['name']}.png", samples, log)
            render_depth(scene, out_dir / f"{frame['name']}.depth.npy", log)
        seconds.append(time.perf_counter() - began)
        print(f"sgi: frame {frame['name']} in {seconds[-1]:.2f} s", flush=True)
    return seconds
