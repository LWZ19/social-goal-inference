"""The realistic generator with Blender (realistic-scenes spec 5.6 and 7): one whole scene, rendered by the
command as it is run by hand, through the scene check. Skipped where Blender or the prepared assets are absent.

What the generator does with a renderer's frames is tested through the stub in test_realistic_generate.py.
Here it is Blender's depth that has to agree with the ground truth, in the observation and round the tour.
"""
import json

import numpy as np
import pytest

from sgi.check_scene import check_scene, has_errors
from sgi.config import load_config
from sgi.scene import load_scene, read_scene_file
from sgi.synth.realistic import blender
from sgi.synth.realistic import generate as real
from sgi.synth.realistic.blender_renderer import BlenderRenderer
from sgi.synth.realistic.spec import Frame, from_json

pytestmark = pytest.mark.skipif(not blender.available(), reason="needs Blender and the prepared assets")

CFG = load_config()
LINE = "eval_07_waiting_line_2"                     # two people waiting at a desk with someone behind it


def test_one_scene_rendered_by_blender_passes_the_scene_check(tmp_path, capsys):
    assert real.main(["--out", str(tmp_path), "--variant", "real", "--only", LINE]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed == [f"{tmp_path}: 1 scenes rendered, 0 skipped as unchanged, 0 dressed again"]
    folder = tmp_path / "eval" / f"real_{LINE}"
    assert [str(path.relative_to(tmp_path)) for path in sorted(tmp_path.glob("*/*"))] == [f"eval/real_{LINE}"]
    results = check_scene(folder, CFG, write_png=False)
    faults = [result.message for result in results if result.level != "ok"]
    assert len(results) == 12 and not has_errors(results), faults
    assert {result.number for result in results if result.level == "warning"} <= {9}, faults

    scene = read_scene_file(folder)
    inputs, truth = load_scene(folder)
    assert scene.scene_id == f"real_{LINE}" and scene.source.simulator == "sgi-synth-blender/real"
    assert scene.source.version.startswith(blender.BLENDER_SERIES)
    job = json.loads((folder / "render_job.json").read_text(encoding="utf-8"))
    assert job["renderer"] == {"name": "blender", "version": scene.source.version} and job["variant"] == "real"
    # Two waiting, the member of staff behind the desk and the vendor at the second stall.
    assert [(person.id, person.role) for person in truth.people] == [
        ("p0", None), ("p1", None), ("s0", "staff"), ("s1", "staff")]
    assert len(job["spec"]["people"]) == 4 and len(job["spec"]["models"]) > 0
    # The dressing passed the check in all three variants, by Blender's depth of each.
    assert sorted(job) == ["attempt", "judged", "renderer", "seed", "spec", "thresholds", "variant"]
    assert job["attempt"] == 0 and list(job["judged"]) == ["real", "plain", "staffed"]

    obs = inputs.observation
    assert obs.depth.dtype == np.float32 and (obs.depth > 0).all()      # a closed room: every ray ends on something
    assert obs.rgb.std() > 20.0                                         # and a picture, not the grey it was judged with
    # The scene keeps the depth it was judged with. The generator held the call that drew the pictures to it, pixel
    # for pixel, and so does another call here: the depth Blender draws before a call's first picture is the same
    # in every call. (The depth that comes with a picture is not, at a few pixels of people's hair.)
    ((_, again),) = BlenderRenderer().render(from_json(job["spec"]), [Frame(obs.camera, beauty=False)])
    assert np.array_equal(again, obs.depth)
    assert len(inputs.tour) == 8 and all(frame.rgb.shape == obs.rgb.shape for frame in inputs.tour)
    assert np.array_equal(inputs.tour[3].rgb, obs.rgb)                  # the observation is among them, pixel for pixel
    assert len({frame.rgb.tobytes() for frame in inputs.tour}) == 8
    (target,) = truth.targets.values()
    assert 3 in target.tour_frames and 3 <= len(target.tour_frames) <= 8

    # Run again, the scene is found as it was drawn and is not drawn a second time.
    before = {path.name: path.stat().st_ino for path in folder.iterdir()}
    assert real.main(["--out", str(tmp_path), "--only", f"real_{LINE}"]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed == [f"{tmp_path}: 0 scenes rendered, 1 skipped as unchanged, 0 dressed again"]
    assert {path.name: path.stat().st_ino for path in folder.iterdir()} == before
