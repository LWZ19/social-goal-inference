import json
import re

from sgi import run as harness
from sgi.gt.derive import prepare
from sgi.scene import load_scene
from sgi.vlm.base import TextPart
from tests.test_run import config, placeholder, records, scene_dirs, subset, wire  # noqa: F401

def test_no_ground_truth_string_reaches_the_model(placeholder, tmp_path, monkeypatch):
    backend, cfg, scenes = wire(monkeypatch), config(placeholder), subset(tmp_path, placeholder / "eval")
    recs = records(harness.run(scenes, cfg, "leak", tmp_path / "runs"))
    secret = set()
    for folder in scene_dirs(scenes) + scene_dirs(placeholder / "examples"):
        inputs, truth = load_scene(folder)
        secret |= {inputs.scene_id, *truth.targets}
        for et in prepare(inputs, truth, cfg).values():
            secret |= {et.gt.situation} | {f.kind for f in et.gt.forbidden_regions or []}
    assert {"no_one", "waiting_line", "conversation_group", "ahead_in_line", "group_space"} <= secret
    sent = []
    for system, parts, schema in backend.calls:
        sent += [system or "", json.dumps(schema)] + [p.text for p in parts if isinstance(p, TextPart)]
    tokens = set(re.findall(r"[A-Za-z0-9_]+", "\n".join(sent)))
    fields = {f for _, _, schema in backend.calls for f in schema["properties"]}
    assert {"subtasks", "stop_elsewhere", "goal_point"} <= fields and recs[0]["example_ids"]   # every stage, with examples
    assert set(re.findall(r"[A-Za-z0-9_]+", recs[0]["request"])) <= tokens                    # data does get through
    assert secret & tokens == set()
