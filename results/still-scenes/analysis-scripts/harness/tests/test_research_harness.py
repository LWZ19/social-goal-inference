"""The offline harness of the research track (results/still-scenes/analysis-scripts/harness/research_harness.py) and the --whitelist of
replay_solver.py: what is held out or excluded is refused, by name and where a link leads, the written lists hold
only what is allowed, a run is replayed only against its listed scenes, and a record of a situation or scene that
is held out is skipped before any of its fields is used. No model call; the recorded runs are read only by the
tests marked as needing them, which skip where they are absent. The names of the tests that take tmp_path hold
nothing the guard refuses, since pytest names that folder after the test."""
import collections
import importlib.util
import json
import math
import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from sgi.config import REPO_ROOT, load_config

TOOLS = Path(__file__).resolve().parents[1]                 # the harness folder


def _module(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


rh = _module("research_harness")
replay_solver = _module("replay_solver")
LISTED_RUNS = [line for line in rh.RUNS_FILE.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
LISTED_SCENES = [line for line in rh.SCENES_FILE.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
DATA = (rh.main_checkout() / LISTED_RUNS[0].partition("=")[0] / "results.jsonl").is_file()
needs_data = pytest.mark.skipif(not DATA, reason="the recorded runs or the development scenes are absent here")
S19_REAL = next(line for line in LISTED_RUNS if "_s19_real_qwen38_low_con=" in line)
SCENE_JSON = REPO_ROOT / "data" / "placeholder" / "eval" / "eval_00_no_one_place" / "scene.json"

REFUSED = ["20261012-101010_s18_seed0_qwen38_low_con", "20261013-101010_s20_held_seed0_dsv41_con",
           "data/x/heldout/seed0/scenes", "/x/.cache/vlm", "20261017-101010_s21_replay_real_qwen38_low_con",
           "20261017-101010_s21_sonnet_cot_probe", ".env", "/x/y/.env", "s18_seed0_qwen38_low_con",
           "eval_16_group_photo_3_posing", "eval_17_group_photo_2_taker", "eval_18_lift_wait_2", "eval_19_lift_wait_4",
           "ph_eval_16_x", "eval_18", "data/placeholder/eval/eval_19_lift_wait_4/rgb.png",
           # held out whatever the case, inside a component, or a stage's sub-run (found by review in the repository)
           "data/dev_scenes/compare/stage18a_heldout_marked.json",
           "data/dev_scenes/compare/stage18b_heldout_con.json",
           "docs/results/pictures/heldout_picture_cases.png",
           "old/s22_old/docs/results/pictures/heldout_picture_cases.png",
           "heldout_x", "x_heldout", "HeldOut", "held_out", "Eval_16_x", "eval_16.json", "eval_16-x", "s18b_x", "S18_x",
           "s20b_held_x", "x/.Cache/y", ".ENV", "x/.env.local", "keys/prod.env",
           # excluded by the plan, not held out: the s22 replays and the pre-flight runs
           "20261017-113116_s22_replay2_real_qwen38_low_cot", "s22_replay_plain_qwen38_low_fsc",
           "20261017-120256_pf_claude_sonnet55_low_cot", "pf_gemini_fast_con", "runs/20261017-121040_PF_gemini_fast_con"]
# The runs of step 0 by name, as step 0 lists them, with the later additions: the list must
# hold these and no other.
ARMS3, DEV = ("real", "staffed", "plain"), ("dev_seed0", "dev_seed1", "dev_seed2")
STEP0_RUNS = {
    "s8": {f"s8_{kind}_{m}" for kind in ("none", "convlm", "conorc")
           for m in ("glm53", "ornith15", "qwen36_35", "qwen38", "qwen3vl")},
    "s10": {f"s10_{m}_{effort}_{k}" for m in ("glm53", "qwen38") for effort in ("none", "low", "medium")
            for k in ("cot", "fsc", "con")} - {f"s10_glm53_medium_{k}" for k in ("cot", "fsc", "con")},  # no folder
    "s16b": {f"s16b_{a}_{m}_{k}" for a in (*ARMS3, "ph") for m in ("qwen38_low", "glm53") for k in ("cot", "fsc")},
    "s17": {f"s17_seed{n}_{m}_{k}" for n in (0, 1, 2) for m in ("qwen38_low", "glm53") for k in ("cot", "fsc3")},
    "s19": {f"s19_{a}_{m}_con" for a in (*ARMS3, "ph", *DEV) for m in ("qwen38_low", "glm53")},
    "s20": {f"s20_{a}_dsv41_{k}" for a in (*ARMS3, "ph", *DEV) for k in ("cot", "fsc", "con")},
    "s21": {f"s21_{a}_qwen38_low_{k}" for a in ARMS3 for k in ("cot", "fsc", "con")},
    "s22": {f"s22_{a}_qwen38_low_{k}" for a in ARMS3 for k in ("cot", "fsc")},
    "s23": {f"s23_{a}_{m}_{k}" for a in ARMS3 for m in ("glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low")
            for k in ("cot", "fsc", "con")},
}
ADMITTED = ["eval_20_no_one_lift", "eval_21_no_one_screen", "real_eval_13_x", "eval_15_audience_5", "eval_160",
            "20261011-160334_s19_real_qwen38_low_con", "runs/20261013-190939_s20_ph_dsv41_con", ".venv/bin/python",
            "data/placeholder/eval/eval_14_audience_3", "20261017-022122_s21_real_qwen38_low_con", "s180_x",
            "environment", "results/still-scenes/analysis-scripts/harness/research_harness.py", "20261017-114056_s22_real_qwen38_low_cot",
            "s23_real_haiku55_con", "s17_seed0_glm53_fsc3", "x_pf_y", "data/placeholder/examples/examples_07_audience_4",
            "real_examples_03_waiting_line_3"]


@pytest.mark.parametrize("name", REFUSED)
def test_the_guard_refuses_every_held_out_or_excluded_name(name):
    assert rh.forbidden(name) is not None
    with pytest.raises(SystemExit, match="refused"):
        rh.guard(name)


@pytest.mark.parametrize("name", ADMITTED)
def test_the_guard_admits_the_development_names(name):
    assert rh.forbidden(name) is None and rh.guard(name) == name


def test_the_written_lists_hold_only_the_runs_and_scenes_of_the_plan():
    folders = [line.partition("=")[0].removeprefix("runs/") for line in LISTED_RUNS]
    assert all("=" in line and line.startswith("runs/") for line in LISTED_RUNS)
    assert [line for line in LISTED_RUNS + LISTED_SCENES if rh.forbidden(line) is not None] == []
    names = [folder.partition("_")[2] for folder in folders]
    assert len(set(names)) == len(names)                              # one folder per name
    assert set(names) == set().union(*STEP0_RUNS.values())            # these runs and no other
    stages = collections.Counter(name.partition("_")[0] for name in names)
    assert stages == {stage: len(some) for stage, some in STEP0_RUNS.items()}
    assert not [name for name in names if re.search(r"s18|s20_held|s21_replay|s21_sonnet|s20_.*_low_|s22_replay|pf_", name)]
    scene_ids = [line.rpartition("/")[2] for line in LISTED_SCENES]
    assert len(LISTED_SCENES) == len(set(LISTED_SCENES)) == 111
    assert not [s for s in scene_ids if re.match(r"(?:\w+_)?eval_1[6-9]_", s)]
    assert not [s for s in scene_ids if re.search(r"group_photo|lift_wait", s)]
    assert {"eval_20_no_one_lift", "eval_21_no_one_screen", "eval_14_audience_3"} <= set(scene_ids)
    examples = [line for line in LISTED_SCENES if line.rpartition("/")[0].endswith("/examples")]
    evaluated = [line for line in LISTED_SCENES if line not in examples]
    assert len(evaluated) == 82 and len(examples) == 29
    scenes_dirs = {line.partition("=")[2] for line in LISTED_RUNS}
    assert scenes_dirs == {line.rpartition("/")[0] for line in evaluated}           # the scenes the runs use, no other
    keep = "data/dev_scenes/realistic_final"
    assert {line.rpartition("/")[0] for line in examples} == {
        "data/placeholder/examples", *(f"{keep}/{arm}/examples" for arm in ("real", "staffed", "plain"))}
    wl = rh.whitelist()                                               # paths only: nothing is opened
    assert len(wl.scenes) == 82 and len(wl.examples) == 29 and not {s for _, s in wl.scenes} & {s for _, s in wl.examples}
    # the copies of the dev_new/seed0 scenes in the placeholder folder, which no listed run was evaluated on there
    assert not [line for line in evaluated if re.match(r"data/placeholder/eval/eval_(?:1[4-9]|2[0-9])_", line)]
    moved = rh.moved_known()                                          # the known moves name listed runs only
    assert moved and {run for run, *_ in moved} <= set(names) and all(moved.values())


def test_the_runs_are_described_and_filtered_by_their_names(tmp_path):
    wl = rh.whitelist(repo=tmp_path)                                  # paths only: nothing is opened
    assert rh.describe("s10_glm53_none_cot") == ("s10", "ph", "glm53", "cot")
    assert rh.describe("s10_qwen38_medium_con") == ("s10", "ph", "qwen38_medium", "con")
    assert rh.describe("s8_conorc_qwen36_35") == ("s8", "ph", "qwen36_35", "conorc")
    assert rh.describe("s20_dev_seed2_dsv41_fsc") == ("s20", "dev_seed2", "dsv41", "fsc")
    assert rh.describe("s16b_plain_qwen38_low_cot") == ("s16b", "plain", "qwen38_low", "cot")
    # stages the list may come to hold: the list decides, the names are read the same way
    assert rh.describe("s17_seed1_glm53_fsc3") == ("s17", "dev_seed1", "glm53", "fsc3")
    assert rh.describe("s17_seed2_qwen38_low_cot") == ("s17", "dev_seed2", "qwen38_low", "cot")
    assert rh.describe("s22_staffed_qwen38_low_fsc") == ("s22", "staffed", "qwen38_low", "fsc")
    assert rh.describe("s23_real_sonnet55_low_con") == ("s23", "real", "sonnet55_low", "con")
    assert rh.describe("s23_plain_haiku55_cot") == ("s23", "plain", "haiku55", "cot")
    for name in ("pf_claude_haiku55_cot", "s22_replay2_real_qwen38_low_cot", "notes_x_y", "s21_x", "s18_seed0_x_con"):
        with pytest.raises(SystemExit, match="refused"):
            rh.describe(name)
    assert len(rh.runs(stage="s21", wl=wl)) == 9
    assert {ref.name for ref in rh.runs(stage="s19", arm="real", wl=wl)} == {"s19_real_qwen38_low_con", "s19_real_glm53_con"}
    assert [ref.name for ref in rh.runs(stage="s20", arm="ph", method="con", wl=wl)] == ["s20_ph_dsv41_con"]
    assert all(ref.path.parent == tmp_path / "runs" and rh.forbidden(ref.path) is None for ref in wl.runs)
    assert rh.run_ref("s20_real_dsv41_cot", wl).folder == "20261013-200416_s20_real_dsv41_cot"        # the latest
    with pytest.raises(SystemExit, match="not on the list"):
        rh.run_ref("20261013-185203_s20_real_dsv41_cot", wl)                                       # superseded
    with pytest.raises(SystemExit, match="not on the list"):
        rh.run_ref(tmp_path / "runs" / "20261001-000000_s19_real_other_con", wl)
    with pytest.raises(SystemExit, match="not on the list"):
        rh.records(tmp_path / "runs" / "20261001-000000_s19_real_other_con", wl=wl)          # before any record
    with pytest.raises(SystemExit, match="s18 run"):
        rh.records("20261012-101010_s18_seed0_qwen38_low_con", wl=wl)
    with pytest.raises(SystemExit, match="s21_replay"):
        rh.run_ref("20261017-010941_s21_replay_real_qwen38_low_con", wl)


def _fake_run(tmp_path, records, scenes_dir=None):
    """A run folder, its scenes folder (empty) and two written lists naming them, under tmp_path."""
    run = tmp_path / "runs" / "20261001-000000_s19_real_fake_con"
    run.mkdir(parents=True)
    (tmp_path / "scenes").mkdir()
    (tmp_path / "elsewhere" / "real_eval_00_x").mkdir(parents=True)          # a scene folder that is a link
    shutil.copy(SCENE_JSON, tmp_path / "elsewhere" / "real_eval_00_x" / "scene.json")
    (tmp_path / "scenes" / "real_eval_00_x").symlink_to(tmp_path / "elsewhere" / "real_eval_00_x")
    doc = {"name": "s19_real_fake_con", "scenes_dir": str(scenes_dir or tmp_path / "scenes"),
           "config": load_config().model_dump(mode="json")}
    (run / "config.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (tmp_path / "runs.txt").write_text(f"# a fake list\nruns/{run.name}=scenes\n", encoding="utf-8")
    (tmp_path / "scenes.txt").write_text("scenes/real_eval_00_x\nscenes/eval_20_no_one_lift\n", encoding="utf-8")
    return run, rh.whitelist(repo=tmp_path, runs_file=tmp_path / "runs.txt", scenes_file=tmp_path / "scenes.txt")


def _record(scene_id, situation, **result):
    calls = [{"stage": "plan", "latency_s": 2.0, "latency_total_s": 2.5, "cached": True, "transient_retries": 1,
              "validation_retries": 0, "usage": {"input_tokens": 100, "output_tokens": 10, "reasoning_tokens": 4}},
             {"stage": "plan", "latency_s": 1.0, "latency_total_s": 1.0, "cached": False, "transient_retries": 0,
              "validation_retries": 1, "usage": {"input_tokens": 50, "output_tokens": 5}},
             {"stage": "people", "latency_s": 4.0, "latency_total_s": 9.0, "cached": False, "transient_retries": 2,
              "validation_retries": 0, "usage": {"output_tokens": None}}]
    return {"scene_id": scene_id, "episode_id": "e0", "trial": 0, "labels": {"situation": situation},
            "result": {"status": "ok", "calls": calls, "timings_s": {"plan": 1.25, "geometry": 0.5}, **result},
            "score": {"success": True}, "measures": {"latency_s": 1.75}}


def _bare(scene_id, situation):
    """A record whose fields past its situation and scene are unusable: using any of them would fail."""
    return {"scene_id": scene_id, "episode_id": None, "labels": {"situation": situation}, "result": None,
            "score": None, "measures": None}


def test_records_skip_records_of_a_situation_or_scene_not_allowed_before_using_them(tmp_path):
    held = load_config().held_out_situations[0]
    run, wl = _fake_run(tmp_path, [_record("real_eval_00_x", "no_one", source="constraints", pose_world=[1, 2, 0],
                                           people=[], constraints=[]),
                                   _record("real_eval_00_x", held), _record("real_eval_00_x", None),
                                   _record("eval_17_group_photo_2_taker", "no_one"), _record("real_eval_09_x", "no_one"),
                                   _record("eval_20_no_one_lift", "no_one"),
                                   _bare("real_eval_00_x", held), _bare("eval_18_lift_wait_2", "no_one")])
    tally = collections.Counter()
    found = list(rh.records(run, wl=wl, tally=tally))
    assert [(rec.scene_id, rec.solver_step) for rec in found] == [("real_eval_00_x", True), ("eval_20_no_one_lift", False)]
    assert tally == {"allowed": 2, "skipped: held-out or unknown situation": 3, "skipped: held-out scene": 2,
                     "skipped: scene not listed": 1}
    rec = found[0]
    assert (rec.run, rec.arm, rec.model, rec.method, rec.pose_world) == (run.name, "real", "fake", "con", [1, 2, 0])
    assert (rec.timings_s, rec.latency_s) == ({"plan": 1.25, "geometry": 0.5}, 1.75)
    plan, people = rec.timings["plan"], rec.timings["people"]
    assert plan == {"calls": 2, "cached": 1, "latency_s": 3.0, "latency_total_s": 3.5, "transient_retries": 1,
                    "validation_retries": 1, "input_tokens": 150, "output_tokens": 15, "reasoning_tokens": 4,
                    "fresh_latency_s": 1.0, "fresh_latency_total_s": 1.0, "fresh_input_tokens": 50,
                    "fresh_output_tokens": 5, "fresh_reasoning_tokens": 0}
    assert (people["latency_total_s"], people["fresh_latency_total_s"], people["transient_retries"]) == (9.0, 9.0, 2)
    with pytest.raises(SystemExit, match="held-out scene"):
        rh.scene(run, "eval_18_lift_wait_2", wl=wl)
    with pytest.raises(SystemExit, match="not on the list of scenes"):
        rh.scene(run, "real_eval_09_x", wl=wl)


def test_a_written_list_with_a_line_the_guard_refuses_is_refused_whole(tmp_path):
    (tmp_path / "runs.txt").write_text("runs/20261011-160334_s19_real_qwen38_low_con=scenes\n"
                                       "runs/20261012-101010_s18_seed0_qwen38_low_con=scenes\n", encoding="utf-8")
    (tmp_path / "scenes.txt").write_text("scenes/real_eval_00_x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="s18 run"):
        rh.whitelist(repo=tmp_path, runs_file=tmp_path / "runs.txt", scenes_file=tmp_path / "scenes.txt")
    (tmp_path / "runs2.txt").write_text("runs/20261011-160334_s19_real_qwen38_low_con=scenes\n", encoding="utf-8")
    (tmp_path / "scenes2.txt").write_text("scenes/real_eval_00_x\ndata/placeholder/eval/eval_17_x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="eval_16 to eval_19"):
        rh.whitelist(repo=tmp_path, runs_file=tmp_path / "runs2.txt", scenes_file=tmp_path / "scenes2.txt")


def test_links_with_innocent_names_that_lead_into_a_refused_folder_are_refused(tmp_path):
    hidden = tmp_path / "keep" / "heldout"
    (hidden / "run").mkdir(parents=True)
    (hidden / "scene").mkdir()
    (hidden / "results.jsonl").write_text(json.dumps(_record("real_eval_00_x", "no_one")) + "\n", encoding="utf-8")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "20261001-000000_s19_real_fake_con").symlink_to(hidden / "run")
    (tmp_path / "scenes").mkdir()
    (tmp_path / "scenes" / "real_eval_00_x").symlink_to(hidden / "scene")
    (tmp_path / "runs.txt").write_text("runs/20261001-000000_s19_real_fake_con=scenes\n", encoding="utf-8")
    (tmp_path / "scenes.txt").write_text("scenes/real_eval_00_x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="held out"):                 # the run folder is a link into heldout
        rh.whitelist(repo=tmp_path, runs_file=tmp_path / "runs.txt", scenes_file=tmp_path / "scenes.txt")
    (tmp_path / "runs.txt").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="held out"):                 # the scene folder is a link into heldout
        rh.whitelist(repo=tmp_path, runs_file=tmp_path / "runs.txt", scenes_file=tmp_path / "scenes.txt")


def test_a_linked_file_inside_a_listed_run_or_scene_that_leads_into_a_refused_folder_is_refused(tmp_path):
    run, wl = _fake_run(tmp_path, [_record("real_eval_00_x", "no_one")])
    hidden = tmp_path / "keep" / "heldout"
    hidden.mkdir(parents=True)
    (hidden / "scene.json").write_text("{}", encoding="utf-8")
    (tmp_path / "elsewhere" / "real_eval_00_x" / "scene.json").unlink()
    (tmp_path / "elsewhere" / "real_eval_00_x" / "scene.json").symlink_to(hidden / "scene.json")
    with pytest.raises(SystemExit, match="held out"):                 # before load_scene opens anything
        rh.scene(run, "real_eval_00_x", wl=wl)
    (hidden / "results.jsonl").write_text((run / "results.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    (run / "results.jsonl").unlink()
    (run / "results.jsonl").symlink_to(hidden / "results.jsonl")
    with pytest.raises(SystemExit, match="held out"):
        list(rh.records(run, wl=wl))
    with pytest.raises(SystemExit, match="held out"):
        replay_solver._whitelisted([f"runs/{run.name}"], tmp_path, wl)


def test_replay_solver_refuses_a_linked_config_whether_or_not_the_scenes_folder_is_given(tmp_path):
    run, wl = _fake_run(tmp_path, [_record("real_eval_00_x", "no_one")])
    hidden = tmp_path / "fake1" / "keep" / "heldout"
    hidden.mkdir(parents=True)
    (hidden / "config.yaml").write_text((run / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (run / "config.yaml").unlink()
    (run / "config.yaml").symlink_to(hidden / "config.yaml")
    for given in (f"runs/{run.name}=scenes", f"runs/{run.name}"):
        with pytest.raises(SystemExit, match="held out"):
            replay_solver._whitelisted([given], tmp_path, wl)
    with pytest.raises(SystemExit, match="held out"):                 # and the harness, before it reads it
        rh.recorded_config(rh.run_ref(run, wl))


def test_a_scene_whose_scene_json_names_a_file_outside_its_folder_is_refused_before_it_is_loaded(tmp_path):
    run, wl = _fake_run(tmp_path, [_record("real_eval_00_x", "no_one")])
    assert rh.check_scene(rh.run_ref(run, wl), "real_eval_00_x", wl) == tmp_path / "scenes" / "real_eval_00_x"
    scene_json = tmp_path / "elsewhere" / "real_eval_00_x" / "scene.json"
    doc = json.loads(SCENE_JSON.read_text(encoding="utf-8"))
    for named, why in (("../../data/heldout/seed0/scenes/x/rgb.png", "held out"),
                       ("../other_scene/rgb.png", "outside its folder"), ("/x/eval_17_y/rgb.png", "eval_16 to eval_19")):
        doc["observation"]["rgb"] = named
        scene_json.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(SystemExit, match=why):
            rh.check_scene(rh.run_ref(run, wl), "real_eval_00_x", wl)
    doc["observation"]["rgb"] = "rgb.png"
    doc["tour"][0]["rgb"] = "tour/../../sibling/000.png"
    scene_json.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit, match="outside its folder"):
        rh.scene(run, "real_eval_00_x", wl=wl)                        # before load_scene reads an image


def test_worked_example_scenes_are_listed_apart_and_refused_unless_listed(tmp_path):
    (tmp_path / "examples" / "examples_00_x").mkdir(parents=True)
    shutil.copy(SCENE_JSON, tmp_path / "examples" / "examples_00_x" / "scene.json")
    (tmp_path / "runs.txt").write_text("", encoding="utf-8")
    (tmp_path / "scenes.txt").write_text("scenes/real_eval_00_x\nexamples/examples_00_x\n", encoding="utf-8")
    wl = rh.whitelist(repo=tmp_path, runs_file=tmp_path / "runs.txt", scenes_file=tmp_path / "scenes.txt")
    assert wl.scenes == {(tmp_path / "scenes", "real_eval_00_x")}
    assert wl.examples == {(tmp_path / "examples", "examples_00_x")}
    assert rh.examples(wl=wl) == rh.examples(tmp_path / "examples", wl=wl) == [(tmp_path / "examples", "examples_00_x")]
    assert rh.examples(tmp_path / "scenes", wl=wl) == []
    assert rh.check_example("examples_00_x", wl=wl) == tmp_path / "examples" / "examples_00_x"
    for scene_id, why in (("examples_01_x", "not on the list of example scenes"), ("real_eval_00_x", "not on the list"),
                          ("eval_18_lift_wait_2", "eval_16 to eval_19")):
        with pytest.raises(SystemExit, match=why):
            rh.example(scene_id, wl=wl)
    with pytest.raises(SystemExit, match="not on the list of example scenes of"):
        rh.check_example("examples_00_x", tmp_path / "scenes", wl=wl)


def test_distinct_scenes_take_a_scene_reached_through_a_link_once_and_fingerprints_compare_contents(tmp_path):
    for name in ("real_eval_00_x", "real_eval_01_x"):
        (tmp_path / "own" / name).mkdir(parents=True)
        shutil.copy(SCENE_JSON, tmp_path / "own" / name / "scene.json")
    (tmp_path / "links").mkdir()
    (tmp_path / "links" / "real_eval_00_x").symlink_to(tmp_path / "own" / "real_eval_00_x")
    (tmp_path / "runs.txt").write_text("runs/20261001-000000_s19_real_fake_con=links\n"
                                       "runs/20261001-000001_s19_plain_fake_con=own\n", encoding="utf-8")
    (tmp_path / "scenes.txt").write_text("links/real_eval_00_x\nown/real_eval_00_x\nown/real_eval_01_x\n", encoding="utf-8")
    wl = rh.whitelist(repo=tmp_path, runs_file=tmp_path / "runs.txt", scenes_file=tmp_path / "scenes.txt")
    first, second = rh.distinct_scenes(wl=wl)
    assert (first.folder, first.scenes_dir, first.scene_id) == (tmp_path / "own" / "real_eval_00_x", tmp_path / "own",
                                                                "real_eval_00_x")    # known by the folder it lies in
    assert set(first.listed) == {(tmp_path / "links", "real_eval_00_x"), (tmp_path / "own", "real_eval_00_x")}
    assert [ref.name for ref in first.runs] == ["s19_real_fake_con", "s19_plain_fake_con"]
    assert (second.scene_id, second.listed, [ref.name for ref in second.runs]) == (
        "real_eval_01_x", ((tmp_path / "own", "real_eval_01_x"),), ["s19_plain_fake_con"])
    one, other = (rh.fingerprint(tmp_path / "own" / name) for name in ("real_eval_00_x", "real_eval_01_x"))
    assert one == other == rh.fingerprint(tmp_path / "links" / "real_eval_00_x")          # the same bytes
    (tmp_path / "own" / "real_eval_01_x" / "notes.txt").write_text("x", encoding="utf-8")
    assert rh.fingerprint(tmp_path / "own" / "real_eval_01_x") != one
    (tmp_path / "links" / "real_eval_09_y").symlink_to(tmp_path / "own" / "real_eval_01_x")
    (tmp_path / "scenes3.txt").write_text("links/real_eval_09_y\nown/real_eval_01_x\n", encoding="utf-8")
    renamed = rh.whitelist(repo=tmp_path, runs_file=tmp_path / "runs.txt", scenes_file=tmp_path / "scenes3.txt")
    with pytest.raises(SystemExit, match="under more than one name"):
        rh.distinct_scenes(wl=renamed)
    hidden = tmp_path / "keep" / "heldout"
    hidden.mkdir(parents=True)
    (tmp_path / "own" / "real_eval_01_x" / "extra.png").symlink_to(hidden / "x.png")
    with pytest.raises(SystemExit, match="held out"):                 # guarded before anything is read
        rh.fingerprint(tmp_path / "own" / "real_eval_01_x")


def test_replay_solver_with_whitelist_refuses_a_run_not_listed_before_opening_anything(tmp_path):
    missing = "runs/20261001-000000_s19_real_fake_con"                 # not there: opening it would fail otherwise
    with pytest.raises(SystemExit, match=f"not on research_runs.txt: .*{missing}"):
        replay_solver.main(["--whitelist", "--root", str(tmp_path), missing])
    with pytest.raises(SystemExit, match="s18 run"):
        replay_solver.main(["--whitelist", "--root", str(tmp_path), "runs/20261012-101010_s18_seed0_qwen38_low_con"])
    listed = LISTED_RUNS[0].partition("=")[0]
    with pytest.raises(SystemExit, match="heldout"):
        replay_solver.main(["--whitelist", f"{listed}=x/data/heldout/seed0/scenes"])
    (tmp_path / "innocent").symlink_to(tmp_path / "keep" / "heldout", target_is_directory=True)
    with pytest.raises(SystemExit, match="held out"):                 # a link with an innocent name
        replay_solver.main(["--whitelist", f"{listed}={tmp_path / 'innocent'}"])


def test_replay_solver_with_whitelist_wants_the_scenes_folder_the_list_gives(tmp_path):
    run_dir, _, scenes_dir = S19_REAL.partition("=")
    for other in (REPO_ROOT / "data" / "realistic" / "eval", REPO_ROOT / "data" / "placeholder" / "eval",
                  scenes_dir.replace("/real/", "/plain/"), scenes_dir.replace("realistic_final", "realistic_firstpass")):
        with pytest.raises(SystemExit, match="is not the scenes folder"):
            replay_solver._whitelisted([f"{run_dir}={other}"], rh.main_checkout())
    run, wl = _fake_run(tmp_path, [_record("real_eval_00_x", "no_one")])
    with pytest.raises(SystemExit, match="is not the scenes folder"):
        replay_solver._whitelisted([f"runs/{run.name}=elsewhere"], tmp_path, wl)
    [(path, folder, allow)] = replay_solver._whitelisted([f"runs/{run.name}"], tmp_path, wl)   # as config.yaml records
    assert (path, folder) == (run, tmp_path / "scenes")
    assert allow("real_eval_00_x") == tmp_path / "scenes" / "real_eval_00_x"
    with pytest.raises(SystemExit, match="not on the list of scenes"):
        allow("real_eval_09_x")
    with pytest.raises(SystemExit, match="eval_16 to eval_19"):
        allow("eval_18_lift_wait_2")
    (run / "config.yaml").write_text(yaml.safe_dump({"scenes_dir": str(tmp_path / "elsewhere"), "config": {}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="is not the scenes folder"):
        replay_solver._whitelisted([f"runs/{run.name}"], tmp_path, wl)


def test_replay_solver_with_whitelist_refuses_rows_inside_the_main_checkout(tmp_path):
    inside = rh.main_checkout() / "replay_rows.json"
    with pytest.raises(SystemExit, match="--rows inside the repository"):
        replay_solver.main(["--whitelist", "--rows", str(inside), S19_REAL])
    assert not inside.exists()


def test_replay_solver_leaves_out_a_scene_not_allowed_whatever_its_situation(tmp_path):
    run, _ = _fake_run(tmp_path, [_record("eval_17_group_photo_2_taker", "no_one"), _record("ph_eval_19_x", "no_one")])
    assert replay_solver.replay(run, tmp_path / "nowhere") == ([], 2)   # the scenes folder is never opened
    assert (replay_solver.HELD_OUT_SCENE.pattern, replay_solver.HELD_OUT_SCENE.flags) == (
        rh.HELD_OUT_SCENE.pattern, rh.HELD_OUT_SCENE.flags)


def test_replay_solver_stands_alone_without_the_harness_beside_it(tmp_path):
    (tmp_path / "alone").mkdir()
    shutil.copy(TOOLS / "replay_solver.py", tmp_path / "alone" / "replay_solver.py")
    spec = importlib.util.spec_from_file_location("replay_solver_alone", tmp_path / "alone" / "replay_solver.py")
    alone = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alone)
    run, _ = _fake_run(tmp_path, [_record("eval_17_group_photo_2_taker", "no_one"), _record("real_eval_00_x", "lift_wait")])
    assert alone.replay(run, tmp_path / "nowhere") == ([], 2)


def test_the_file_of_known_moves_wants_a_reason_on_every_place(tmp_path):
    (tmp_path / "moved.txt").write_text("# why\n\ns8_none_x eval_08_y e1 1  # a later rule\n", encoding="utf-8")
    assert rh.moved_known(tmp_path / "moved.txt") == {("s8_none_x", "eval_08_y", "e1", "1"): "a later rule"}
    (tmp_path / "bare.txt").write_text("s8_none_x eval_08_y e1 1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="a reason"):
        rh.moved_known(tmp_path / "bare.txt")


@needs_data
def test_the_listed_runs_and_scenes_are_there_and_a_recorded_pose_scores_as_it_is_rescored():
    assert all((ref.path / "results.jsonl").is_file() and ref.scenes_dir.is_dir() for ref in rh.runs())
    assert all((rh.main_checkout() / line / "scene.json").is_file() for line in LISTED_SCENES)
    every = collections.Counter()
    for ref in rh.runs():
        collections.deque(rh.records(ref, tally=every), maxlen=0)
    assert every["allowed"] > 5000 and "skipped: scene not listed" not in every       # every scene a run used is listed
    ref = rh.runs(stage="s19", arm="real", model="qwen38_low")[0]
    tally = collections.Counter()
    some = [rec for rec in rh.records(ref, tally=tally) if rec.pose_world is not None][:6]
    assert some and set(tally) == {"allowed"}
    for rec in some:
        assert rh.score_pose(rec, rec.pose_world).model_dump() == rh.rescore(rec).model_dump()
        moved = (rec.pose_world[0] + 30.0, rec.pose_world[1], rec.pose_world[2])
        assert not rh.score_pose(rec, moved).success


@needs_data
def test_the_listed_scenes_are_68_once_links_are_resolved_and_no_two_are_one_by_content():
    distinct = rh.distinct_scenes()
    assert len(distinct) == 68
    twice = [one for one in distinct if len(one.listed) > 1]
    assert len(twice) == 14 and all(len(one.listed) == 2 for one in twice)
    placeholder = rh.main_checkout() / "data" / "placeholder" / "eval"
    assert {one.scenes_dir for one in twice} == {placeholder}             # known by the folder they lie in
    assert all({d.name for d, _ in one.listed} == {"eval", "placeholder_first14"} for one in twice)
    assert all(one.runs for one in distinct)                              # each loads under a listed run
    assert {stage for one in twice for stage in (ref.stage for ref in one.runs)} == {"s8", "s10", "s16b", "s19", "s20"}
    one = twice[0]
    assert rh.scene(one.runs[-1], one.scene_id).inputs.scene_id == one.scene_id
    prints = [rh.fingerprint(one.folder) for one in distinct] + [rh.fingerprint(f / s) for f, s in rh.examples()]
    assert len(set(prints)) == len(prints) == 68 + 29


@needs_data
def test_many_poses_scored_with_the_freespace_cached_score_as_the_package_scores_each():
    from sgi.evaluate import score
    from sgi.geometry import make_poses
    from sgi.types import GoalResult
    checked = collections.Counter()
    for ref in rh.runs(stage="s19", model="glm53")[:2]:
        for rec in [rec for rec in rh.records(ref) if rec.pose_world is not None][::9][:3]:
            sc, gt, th, _ = rh._against(rec)
            x, y, theta = rec.pose_world
            poses = [(x + dx, y + dy, theta + dt) for dx in (-1.5, 0.0, 2.0) for dy in (0.0, 0.7) for dt in (0.0, 1.2)]
            fast = rh.score_poses(rec, poses)
            result = GoalResult.model_validate(rec.result)
            for pose, got in zip(poses, fast):
                pose_world, pose_robot = make_poses(sc.observation.robot, pose[:2], pose[2])
                want = score(result.model_copy(update={"pose_world": pose_world, "pose_robot": pose_robot}),
                             sc.observation, sc.truth, gt, th)
                assert got.model_dump() == want.model_dump()
                checked[(want.reachable, want.success)] += 1
    assert len(checked) >= 3                                          # reachable and not, success and not


@needs_data
def test_a_target_put_in_is_scored_and_one_grounded_as_the_pipeline_grounds_it_matches_the_record():
    ref = rh.runs(stage="s19", arm="real", model="qwen38_low")[0]
    recs = [rec for rec in rh.records(ref) if rec.target is not None][:5]
    assert recs
    for rec in recs:
        recorded = rec.target
        again = rh.target_at(rec, point=(recorded["point"]["x"], recorded["point"]["y"]))
        assert again.model_dump(mode="json") == recorded                       # the pipeline's grounding, again
        assert rh.score_pose(rec, rec.pose_world, target=again) == rh.rescore(rec)
        from_world = rh.target_at(rec, world=recorded["world"])
        assert math.dist(from_world.pixel, recorded["pixel"]) < 1.0 and from_world.world == tuple(recorded["world"])
        far = from_world.model_copy(update={"world": (recorded["world"][0] + 30.0, *recorded["world"][1:])})
        [same, off] = rh.score_poses(rec, [rec.pose_world] * 2, target=[None, far])
        assert same == rh.rescore(rec) and not off.target_correct and not off.success


@needs_data
def test_resolve_gives_the_recorded_place_again_and_takes_other_inputs():
    ref = rh.runs(stage="s19", arm="real", model="qwen38_low")[0]
    steps = [rec for rec in rh.records(ref) if rec.solver_step][:4]
    assert steps
    for rec in steps:
        again = rh.resolve(rec)
        assert again.reproduced and again.score == rh.rescore(rec)
        assert list(again.result.pose_world) == list(rec.pose_world)
    rec = next(rec for rec in steps if rec.constraints)
    alone = rh.resolve(rec, constraints=[])                           # the solver on its own rules alone
    assert alone.uses == [] and alone.moved_m is not None
    nobody = rh.resolve(rec, people=[], constraints=[])
    assert nobody.result.people == [] and nobody.score.target_correct == rh.rescore(rec).target_correct


@needs_data
def test_every_worked_example_a_listed_run_showed_is_listed_and_loads_without_a_run():
    shown = {ref.folder: rh.run_examples(ref) for ref in rh.runs()}
    assert all(bool(some) == ref.method.startswith("fsc") for ref, some in zip(rh.runs(), shown.values()))
    used = {folder for some in shown.values() for folder, _, _ in some}
    assert used == {folder for folder, _ in rh.examples()}            # every listed examples folder was used
    ref = rh.runs(stage="s16b", arm="real", model="qwen38_low", method="fsc")[0]
    for folder, scene_id, episode in rh.run_examples(ref):
        sc = rh.example(scene_id, run=ref)
        assert sc.inputs.scene_id == scene_id and episode in sc.truths
        assert rh.example(scene_id, folder).truths.keys() == sc.truths.keys()

