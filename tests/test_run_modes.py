import dataclasses
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from sgi import run as harness
from sgi.geometry import bearing, wrap_angle
from sgi.gt.derive import oracle_path, oracle_point, oracle_target, prepare
from sgi.scene import Heading, SceneError, load_scene, read_scene_file
from sgi.synth.render import Box
from sgi.types import Subtask
from sgi.vlm.base import BackendError, ImagePart, TransientError
from sgi.vlm.env import require_key
from tests.helpers import rendered_observation, scene_dict, write_scene
from tests.test_constraint_runs import Scripted, run_method
from tests.test_run import (CONTEXT_ONLY, METHOD_COLUMNS, ORACLE, STAGES, config, context_answers, episodes,  # noqa: F401
                            placeholder, records, scene_dirs, subset, summary, wire)

def floor_plan(parts, json_schema):                     # marks the floor just ahead of the robot, never the target
    step = {"type": "navigate", "description": "Go there", "target": "it", "target_point": {"x": 500, "y": 950}}
    return {key: "reasons" if key == "reasoning" else [step] for key in json_schema["properties"]}

def plan_script(answer, target=None):
    """Responder for the plan call. A list is a plan with one code per subtask: "act", or the point of a navigate step
    (None, "floor" or "target", the Point given). Anything else is returned as it is."""
    points = {None: None, "floor": {"x": 500, "y": 950}, "target": target and {"x": target.x, "y": target.y}}
    def responder(parts, json_schema):
        if not isinstance(answer, list):
            return answer
        steps = [{"type": "act", "description": "Do it", "target": None, "target_point": None} if code == "act" else
                 {"type": "navigate", "description": "Go there", "target": "it", "target_point": points[code]} for code in answer]
        return {key: "reasons" if key == "reasoning" else steps for key in json_schema["properties"]}
    return responder

def edited(tmp_path, folder, change):                   # a copy of every scene with each episode's ground truth changed
    for d in scene_dirs(folder):
        shutil.copytree(d, tmp_path / "edited" / d.name)
        doc = json.loads((d / "scene.json").read_text())
        for episode in doc["episodes"]:
            change(episode["ground_truth"])
        (tmp_path / "edited" / d.name / "scene.json").write_text(json.dumps(doc))
    return tmp_path / "edited"

def strip(gt):
    gt.pop("acceptable_regions", None), gt.pop("forbidden_regions", None)

def test_oracle_inferrer_marks_the_representative_point_and_takes_the_region_heading(placeholder):
    cfg = config(placeholder)
    inputs, truth = load_scene(scene_dirs(placeholder / "eval")[0])
    et, obs, th = next(iter(prepare(inputs, truth, cfg).values())), inputs.observation, cfg.thresholds
    region, target = et.gt.acceptable_regions[0], oracle_target(obs, truth.targets[et.gt.target_ids[0]], th)
    step = Subtask(type="navigate", description="Go to it", target="it")
    res = harness.OracleGoalInferrer(region, th).infer(obs, "a request", step, target, None)
    assert (res.status, res.source, res.calls, res.target) == ("ok", "oracle", [], target)
    assert res.goal_mark.goal_point == oracle_point(obs, (*region.representative_point, 0.0))
    lifted = tuple(oracle_path(obs, region.representative_point, th)[:2])               # project, round to a Point, lift
    assert res.pose_world[:2] == pytest.approx(lifted) and lifted == pytest.approx(region.representative_point, abs=0.05)
    assert res.pose_world[2] == pytest.approx(bearing(res.pose_world[:2], region.heading.face_point))
    turned = region.model_copy(update={"heading": Heading(yaw=-2.5)})
    res = harness.OracleGoalInferrer(turned, th).infer(obs, "a request", step, target, None)
    assert res.pose_world[2] == pytest.approx(-2.5) and res.pose_robot[2] == pytest.approx(wrap_angle(-2.5 - obs.robot.yaw))

@pytest.mark.parametrize("split, stripped, check", [("eval", False, "always_yes"), ("examples", False, "always_yes"),
                                                    ("eval", True, "always_yes"), ("examples", False, "oracle")])
def test_oracle_run_makes_no_call_and_scores_100_percent(placeholder, tmp_path, monkeypatch, split, stripped, check):
    backend = wire(monkeypatch, check=True)             # the real scene check: warnings must not stop a run
    scenes = edited(tmp_path, placeholder / split, strip) if stripped else placeholder / split
    cfg = config(placeholder)
    held_out = {sit for _, _, sit in episodes(scenes)} & set(cfg.held_out_situations)
    assert bool(held_out) == (split == "eval")          # eval/ has scenes of the held-out situations and is run here as the final
    run = {**ORACLE, "context_check": check, "final_test": split == "eval"}     # test; examples/ has none and is run by default
    out = harness.run(scenes, config(placeholder, run=run), "oracle", tmp_path / "runs")
    recs = records(out)
    assert backend.calls == [] and len(recs) == len(episodes(scenes)) > 0
    assert {r["labels"]["situation"] for r in recs} == {sit for _, _, sit in episodes(scenes)}      # every situation the folder holds, none skipped
    assert yaml.safe_load((out / "config.yaml").read_text())["skipped_held_out"] == 0
    assert [r for r in recs if not r["score"]["success"]] == [] and float(summary(out)[0]["success_rate"]) == 1.0
    assert {r["result"]["source"] for r in recs} == ({"oracle"} if check == "always_yes" else {"oracle", "standard"})
    assert all(r["plan"]["subtasks"][0]["description"] == "Go to the " + r["plan"]["subtasks"][0]["target"] for r in recs)
    assert all(r["example_ids"] == [] and r["result"]["calls"] == [] and r["measures"]["plan_exact"] is None for r in recs)
    assert yaml.safe_load((out / "config.yaml").read_text())["examples_missing"] == []       # no example is shown, none is missed

def test_a_default_oracle_run_over_the_split_folder_is_the_development_episodes_and_counts_the_held_out_ones(placeholder, tmp_path, monkeypatch):
    backend = wire(monkeypatch, check=True)
    cfg, scenes = config(placeholder, run=ORACLE), placeholder / "eval"
    assert cfg.run.final_test is False                  # the default: what is not the final test leaves the held-out situations alone
    everything = episodes(scenes)                       # both numbers from the folder and the config, whatever the generator writes
    development = [(s, e) for s, e, sit in everything if sit not in cfg.held_out_situations]
    held_out = [(s, e) for s, e, sit in everything if sit in cfg.held_out_situations]
    assert development and held_out and len(development) + len(held_out) == len(everything)
    assert {sit for _, _, sit in everything} == set(cfg.dev_situations) | set(cfg.held_out_situations)     # the folder has every situation of both lists
    out = harness.run(scenes, cfg, "default", tmp_path / "runs")
    recs = records(out)
    assert backend.calls == [] and [(r["scene_id"], r["episode_id"]) for r in recs] == development
    assert [r for r in recs if not r["score"]["success"]] == [] and float(summary(out)[0]["success_rate"]) == 1.0
    assert yaml.safe_load((out / "config.yaml").read_text())["skipped_held_out"] == len(held_out)

def test_always_no_is_the_standard_goal_everywhere(placeholder, tmp_path, monkeypatch):
    backend = wire(monkeypatch)
    cfg = config(placeholder, run={"grounding": "oracle", "context_check": "always_no",
                                   "examples_dir": str(placeholder / "eval")})   # no usable example, and none needed
    recs = records(harness.run(placeholder / "eval", cfg, "unaware", tmp_path / "runs"))
    assert backend.calls == [] and {r["result"]["source"] for r in recs} == {"standard"}
    alone = [r for r in recs if r["labels"]["situation"] == "no_one"]
    frontal = [r for r in alone if (a := r["labels"]["factors"]["view_angle_deg"]) is not None and a <= 30.001]
    assert len(alone) > len(frontal) > 0 and all(r["score"]["reachable"] for r in alone)
    assert [r for r in frontal if not r["score"]["success"]] == []

def test_goal_oracle_computes_no_goal_when_the_marked_target_is_missed(placeholder, tmp_path, monkeypatch):
    backend = wire(monkeypatch, floor_plan)
    cfg = config(placeholder, run={"context_check": "always_yes", "goal": "oracle"})
    recs = records(harness.run(subset(tmp_path, placeholder / "eval"), cfg, "missed", tmp_path / "runs"))
    assert len(backend.calls) == len(recs) and all("subtasks" in schema["properties"] for _, _, schema in backend.calls)
    assert {(r["result"]["status"], r["result"]["source"], r["result"]["pose_robot"], r["score"]["success"],
             r["measures"]["later_subtask_hit_target"]) for r in recs} == {("ok", "oracle", None, False, False)}

@pytest.mark.parametrize("answer, status, evaluated, scored, measured", [
    # scored is (target_correct, success); measured is (plan_exact, navigate_recall, later_subtask_hit_target)
    ("not json", "parse_failed", None, (False, False), (False, 0.0, False)),        # the model's own answer: a wrong plan
    (TransientError("503"), "vlm_error", None, (False, False), (None, None, None)), # no answer at all: not measured
    ([None, "act"], "no_target", None, (False, False), (False, 0.0, False)),
    ([None, "target", "act"], "ok", 1, (True, True), (False, 1.0, False)),          # a deferred step, then the target
    (["floor", "target", "act"], "ok", 0, (False, False), (False, 1.0, True)),      # a miss, then the target
    (["floor", None, "floor"], "ok", 0, (False, False), (False, 0.0, False)),       # a miss, and no later step on the target
    (["target", "act", None, "act"], "ok", 0, (True, True), (True, 1.0, False)),    # the reference plan, optional steps too
], ids=["parse_failed", "vlm_error", "no_target", "deferred", "later_hit", "later_miss", "exact"])
def test_view_mode_plan_outcomes_reach_the_record(placeholder, monkeypatch, answer, status, evaluated, scored, measured):
    cfg = config(placeholder, run={"context_check": "always_yes", "goal": "oracle"})    # the plan is the only model call
    inputs, truth = load_scene(scene_dirs(placeholder / "eval")[0])
    et, obs = prepare(inputs, truth, cfg)["e1"], inputs.observation     # e1: navigate t0, act, then an optional way back
    backend = wire(monkeypatch, plan_script(answer, oracle_point(obs, truth.targets["t0"].anchor)))
    record = harness.run_episode(harness.build_pipeline(cfg), inputs, truth, et, inputs.episodes[1], cfg, 0)
    result, measures = record["result"], record["measures"]
    (call,) = result["calls"]                                           # never a locate call
    assert (call["stage"], result["status"], record["evaluated_subtask"]) == ("plan", status, evaluated)
    assert len(backend.calls) == 1 + call["validation_retries"] + call["transient_retries"]
    assert call["status"] == (status if record["plan"] is None else "ok")    # the call's own outcome, not the episode's
    assert (call["validation_retries"], call["transient_retries"] > 0) == (int(status == "parse_failed"), status == "vlm_error")
    assert result["timings_s"]["plan"] >= 0.0 and (record["plan"] is None) == (status in ("parse_failed", "vlm_error"))
    assert (record["score"]["target_correct"], record["score"]["success"]) == scored
    assert (measures["plan_exact"], measures["navigate_recall"], measures["later_subtask_hit_target"]) == measured

def test_two_valid_targets_use_the_matched_target(placeholder, tmp_path, monkeypatch):
    doc, second = scene_dict(), [[2.1, 6.0], [3.1, 6.0], [3.1, 7.0], [2.1, 7.0]]    # t0 is the desk at x = 1; t1 one at x = 2.6
    doc["ground_truth"]["targets"].append({"id": "t1", "kind": "place", "label": "ticket desk", "anchor": [2.6, 6.0, 1.0], "footprint": second})
    doc["ground_truth"]["obstacles"].append({"polygon": second})
    doc["episodes"][0]["ground_truth"].update(target_ids=["t0", "t1"], acceptable_regions=[
        {"target_id": t, "polygon": [[x - 0.6, 4.4], [x + 0.6, 4.4], [x + 0.6, 5.4], [x - 0.6, 5.4]],
         "heading": {"face_point": [x, 6.0]}} for t, x in (("t0", 1.0), ("t1", 2.6))])
    desks = [Box(cx=x, cy=6.5, yaw=-math.pi / 2, sx=1.0, sy=1.0, sz=1.2) for x in (1.0, 2.6)]
    inputs, truth = load_scene(write_scene(tmp_path, doc, rendered_observation(desks)))
    marks_t1 = plan_script(["target"], oracle_point(inputs.observation, truth.targets["t1"].anchor))
    def responder(parts, json_schema):                   # the context check says people matter; a goal request gets no JSON
        if "subtasks" in json_schema["properties"]:
            return marks_t1(parts, json_schema)
        return {"relevant_people": "two people", "stop_elsewhere": True} if "stop_elsewhere" in json_schema["properties"] else "not json"
    backend = wire(monkeypatch, responder)
    def go(et, run, **top):
        cfg = config(placeholder, run={"goal": "oracle", **run}, **top)
        r = harness.run_episode(harness.build_pipeline(cfg), inputs, truth, et, inputs.episodes[0], cfg, 0)
        return r, r["result"], r["score"], r["measures"]
    et = prepare(inputs, truth, config(placeholder))["e0"]
    assert et.problems == () and et.standard_goal_ok == {"t0": True, "t1": True}
    _, result, score, _ = go(et, {"context_check": "always_yes"})                    # goal: oracle marks t1's region
    assert (score["matched_target_id"], score["success"]) == ("t1", True) and 2.0 <= result["pose_world"][0] <= 3.2
    two = dataclasses.replace(et, standard_goal_ok={"t0": False, "t1": True})        # as if people mattered at t0 only
    _, result, score, measures = go(two, {"context_check": "oracle"})
    assert (result["source"], measures["reference_people_matter"], score["success"]) == ("standard", False, True)
    _, result, score, measures = go(two, {"context_check": "vlm", "goal": "direct_marking"}, stages={"goal": {"strategy": "zero_shot"}})
    assert (result["status"], score["matched_target_id"]) == ("parse_failed", None)  # the scorer matches nothing here
    assert (measures["reference_people_matter"], measures["context_unnecessary"]) == (False, True)   # still t1's reference
    sent = len(backend.calls)
    record, result, score, measures = go(two, {"context_check": "oracle", "grounding": "oracle"})     # the first id, t0
    assert (result["source"], measures["reference_people_matter"], score["matched_target_id"]) == ("oracle", True, "t0")
    assert record["plan"]["subtasks"][0]["target"] == "info desk" and 0.4 <= result["pose_world"][0] <= 1.6
    assert score["success"] and len(backend.calls) == sent

def test_held_out_episodes_are_skipped_unless_final_test(placeholder, tmp_path, monkeypatch, caplog):
    wire(monkeypatch)
    scenes = subset(tmp_path, placeholder / "eval")
    lists = {"dev_situations": ["no_one", "waiting_line", "audience"],
             "held_out_situations": ["group_photo", "lift_wait", "conversation_group"]}
    kept = [(s, e) for s, e, sit in episodes(scenes) if sit != "conversation_group"]
    with caplog.at_level("INFO", logger="sgi.run"):
        run = {"leave_out_situation": "waiting_line", "context_check": "always_yes"}      # always_yes: every example is eligible
        out = harness.run(scenes, config(placeholder, run=run, **lists), "dev", tmp_path / "runs")
    recs = records(out)
    assert [(r["scene_id"], r["episode_id"]) for r in recs] == kept      # the left-out situation is still evaluated
    assert sorted(p.stem for p in (out / "viz").iterdir()) == sorted(f"{s}_{e}" for s, e in kept)
    assert f"skipped {len(episodes(scenes)) - len(kept)} held-out episodes" in caplog.text
    shown = {i.split("/")[0] for i in recs[0]["example_ids"]}            # of the lists' three development situations one is left
    left = {"no_one", "audience"}                                        # out: examples of the other two are shown, and of none else
    assert {sit for s, _, sit in episodes(placeholder / "examples") if s in shown} == left
    assert not shown & {s for s, _, sit in episodes(placeholder / "examples") if sit not in left}
    final = records(harness.run(scenes, config(placeholder, run={**ORACLE, "final_test": True}, **lists), "final", tmp_path / "runs"))
    assert len(final) == len(episodes(scenes)) and all(r["score"]["success"] for r in final)

def test_final_test_refuses_a_held_out_episode_that_has_no_region(placeholder, tmp_path, monkeypatch, caplog):
    wire(monkeypatch, check=True)                       # the real scene check: there such an episode is only a warning
    # Every held-out situation the default config names has a reference rule, so a made-up one stands for one without.
    lists = {"held_out_situations": ["group_photo", "lift_wait", "parade"]}
    def parade(gt):                                     # a held-out situation with no rule, exported without polygons
        strip(gt), gt.update(situation="parade")
    scenes = edited(tmp_path, subset(tmp_path, placeholder / "eval"), parade)
    (scene_id, episode_id, _), n = episodes(scenes)[0], len(episodes(scenes))
    final = config(placeholder, run={**ORACLE, "final_test": True}, **lists)
    with pytest.raises(SceneError) as refusal:          # it could never succeed, and would sit in every denominator
        harness.run(scenes, final, "final", tmp_path / "runs")
    assert str(refusal.value) == (f"{scene_id}: episode {episode_id} has no acceptable region and cannot succeed: "
                                  f"{episode_id}: no region (no rule for parade)")
    assert not (tmp_path / "runs").exists()             # refused before anything is written
    out = harness.run(scenes, config(placeholder, run=ORACLE, **lists), "dev", tmp_path / "runs")     # without it: skipped as before
    assert records(out) == [] and yaml.safe_load((out / "config.yaml").read_text())["skipped_held_out"] == n
    with caplog.at_level("WARNING", logger="sgi.run"):  # as --final-test --skip-scene-check: run, and said once an episode
        forced = records(harness._run(scenes, final, "forced", tmp_path / "runs", skip_scene_check=True))
    assert len(forced) == n and not any(r["score"]["success"] for r in forced)
    assert caplog.text.count("has no acceptable region and cannot succeed: ") == n
    assert f"{scene_id}: episode {episode_id} has no acceptable region" in caplog.text

def test_failed_scene_check_or_unknown_label_stops_the_run_and_names_the_scene(placeholder, tmp_path, monkeypatch):
    cfg, scenes = config(placeholder, run=ORACLE), subset(tmp_path, placeholder / "eval")
    relabelled = edited(tmp_path, scenes, lambda gt: gt.update(situation="picnic"))
    broken = scene_dirs(scenes)[1]
    np.save(broken / "depth.npy", np.load(broken / "depth.npy") * 2.0)   # wrong units: fails the floor-depth check
    wire(monkeypatch, check=True)
    with pytest.raises(SceneError, match=read_scene_file(broken).scene_id):
        harness.run(scenes, cfg, "bad_depth", tmp_path / "refused")
    saved = config(placeholder, run={**ORACLE, "use_cache": True}, stages={s: {"model": "gemini_fast"} for s in STAGES})
    (tmp_path / "oracle.yaml").write_text(yaml.safe_dump(saved.model_dump(mode="json")))
    monkeypatch.chdir(tmp_path)                                          # main writes under ./runs
    flags = ["--config", "oracle.yaml", "--model", "mock", "--final-test", "--no-cache", "--skip-scene-check"]
    assert harness.main(["--scenes", str(scenes), "--name", "forced", *flags]) == 0
    (out,) = (tmp_path / "runs").iterdir()
    used = yaml.safe_load((out / "config.yaml").read_text())["config"]
    assert out.name.endswith("_forced") and len(records(out)) == len(episodes(scenes))
    assert (used["run"]["final_test"], used["run"]["use_cache"], {s["model"] for s in used["stages"].values()}) == (True, False, {"mock"})
    wire(monkeypatch)                                                    # scene check off: the harness checks labels itself
    with pytest.raises(SceneError, match=episodes(relabelled)[0][0]):
        harness.run(relabelled, cfg, "bad_label", tmp_path / "refused")
    assert harness.main(["--scenes", str(relabelled), "--name", "bad_label", *flags]) == 2
    assert not (tmp_path / "refused").exists() and len(list((tmp_path / "runs").iterdir())) == 1    # nothing was written

def test_broken_example_scene_stops_the_run(placeholder, tmp_path, monkeypatch):
    shutil.copytree(placeholder / "examples", tmp_path / "examples")
    broken = scene_dirs(tmp_path / "examples")[2]
    np.save(broken / "depth.npy", np.load(broken / "depth.npy") * 2.0)   # a mis-exported example: wrong depth units
    wire(monkeypatch, check=True)
    cfg = config(placeholder, run={"examples_dir": str(tmp_path / "examples"), "k": 50})
    scenes = subset(tmp_path, placeholder / "eval")
    with pytest.raises(SceneError, match=read_scene_file(broken).scene_id):
        harness.run(scenes, cfg, "bad_example", tmp_path / "runs")
    assert not (tmp_path / "runs").exists()
    out = harness._run(scenes, cfg, "unchecked", tmp_path / "runs", skip_scene_check=True)     # as --skip-scene-check does
    shown = {i.split("/")[0] for i in records(out)[0]["example_ids"]}    # its points fail the oracle path: still never shown
    assert len(shown) == 5 and read_scene_file(broken).scene_id not in shown       # in two of the other seven the standard goal passes

def goal_images(backend):                               # the images of each goal request: the worked examples and the case
    return [sum(isinstance(part, ImagePart) for part in parts)
            for _, parts, schema in backend.calls if "goal_point" in schema["properties"]]

def missing_warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "sgi.run" and r.getMessage().startswith("no in-context example")]

def test_a_situation_without_an_eligible_example_is_warned_of_recorded_and_still_run(placeholder, tmp_path, monkeypatch, caplog):
    backend, scenes = wire(monkeypatch), subset(tmp_path, placeholder / "eval")        # one scene of each situation
    lines = [d for d in scene_dirs(placeholder / "examples") if read_scene_file(d).episodes[0].ground_truth.situation == "waiting_line"]
    for d in lines:                                     # an examples folder with the waiting lines alone
        shutil.copytree(d, tmp_path / "examples" / d.name)
    cfg = config(placeholder, run={"examples_dir": str(tmp_path / "examples")})
    with caplog.at_level("WARNING", logger="sgi.run"):
        out = harness.run(scenes, cfg, "gap", tmp_path / "runs")
    (said,) = missing_warnings(caplog)                  # one warning, naming the situation
    assert said.startswith("no in-context example for conversation_group: ")
    doc, recs = yaml.safe_load((out / "config.yaml").read_text()), records(out)
    assert doc["examples_missing"] == ["conversation_group"] and (doc["incomplete"], doc["vlm_errors"]) == (False, 0)
    (shown,) = doc["example_ids"]                       # the example there is, of the other situation that needs one
    assert shown.split("/")[0] in {read_scene_file(d).scene_id for d in lines}
    assert len(recs) == len(episodes(scenes)) and all(r["example_ids"] == [shown] for r in recs)
    assert {r["labels"]["situation"] for r in recs if r["result"]["status"] == "ok"} == {"no_one", "waiting_line", "conversation_group"}
    assert goal_images(backend) == [2] * len(recs)      # every episode, the group's too, with that example and its own image
    assert (out / "summary.csv").exists()

def test_a_few_shot_run_with_no_eligible_example_at_all_goes_on_without_examples(placeholder, tmp_path, monkeypatch, caplog):
    backend, scenes = wire(monkeypatch), subset(tmp_path, placeholder / "examples")
    cfg = config(placeholder, run={"examples_dir": str(scenes)})        # the example scenes are the ones evaluated: none is left
    (tmp_path / "self.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    monkeypatch.chdir(tmp_path)                         # main writes under ./runs
    with caplog.at_level("WARNING", logger="sgi.run"):
        assert harness.main(["--scenes", str(scenes), "--config", "self.yaml", "--name", "self"]) == 0
    (out,) = (tmp_path / "runs").iterdir()
    (said,) = missing_warnings(caplog)                  # the same warning; nobody about needs no example, the standard goal passes
    assert said.startswith("no in-context example for waiting_line, conversation_group: ")
    doc, recs = yaml.safe_load((out / "config.yaml").read_text()), records(out)
    assert (doc["example_ids"], doc["examples_missing"]) == ([], ["waiting_line", "conversation_group"])
    assert (doc["config"]["stages"]["goal"]["strategy"], doc["incomplete"]) == ("few_shot", False)      # the strategy as it was asked for
    assert len(recs) == len(episodes(scenes)) and all(r["example_ids"] == [] and r["result"]["status"] == "ok" for r in recs)
    assert goal_images(backend) == [1] * len(recs)      # each goal request with its own image alone

def refused(capsys, *argv):                             # main stops with exit code 2; returns its one line on stderr
    assert harness.main(list(argv)) == 2
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("error: ") and err.count("\n") == 1 and "Traceback" not in err
    return err

def test_provider_error_stops_the_run_cleanly(placeholder, tmp_path, monkeypatch, capsys):
    scenes = subset(tmp_path, placeholder / "eval")
    (tmp_path / "context.yaml").write_text(yaml.safe_dump(config(placeholder, run=CONTEXT_ONLY).model_dump(mode="json")))
    monkeypatch.chdir(tmp_path)                                          # main writes under ./runs
    flags = ["--scenes", str(scenes), "--config", "context.yaml", "--model", "openai_fast", "--skip-scene-check"]
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=key-of-the-other-provider\n")         # .env.example copied, one key filled in
    monkeypatch.setattr("sgi.vlm.factory.require_key", lambda name: require_key(name, env))
    said = refused(capsys, *flags, "--name", "no_key")                   # the real build_pipeline: no client without a key
    assert str(env) in said and "OPENAI_API_KEY" in said and "key-of-the-other-provider" not in said
    assert not (tmp_path / "runs").exists()                              # nothing was written
    answer, sent = context_answers(), []
    def responder(parts, json_schema):                                   # the provider refuses the third request outright
        sent.append(parts)
        return BackendError("openai 404: model not found") if len(sent) == 3 else answer(parts, json_schema)
    backend = wire(monkeypatch, responder)
    assert refused(capsys, *flags, "--name", "retired") == "error: openai 404: model not found\n"
    (out,) = (tmp_path / "runs").iterdir()
    assert len(backend.calls) == 3                                       # not retried, and no later episode is tried
    done = records(out)                                                  # what was written before the error stays
    assert [(r["scene_id"], r["episode_id"], r["result"]["status"]) for r in done] == [
        (s, e, "ok") for s, e, _ in episodes(scenes)[:2]]
    doc = yaml.safe_load((out / "config.yaml").read_text())
    assert doc["incomplete"] is True and not (out / "summary.csv").exists()
    assert "stopped_by" not in doc and "resumes" not in said             # a refusal: no allowance to wait for, nothing to resume

def test_missing_scene_or_example_folder_is_a_config_error_naming_the_path(placeholder, tmp_path, monkeypatch, capsys):
    wire(monkeypatch)
    clone, work, scenes = tmp_path / "clone", tmp_path / "work", subset(tmp_path, placeholder / "eval")
    clone.mkdir(), work.mkdir()
    monkeypatch.setattr("sgi.run.REPO_ROOT", clone)      # a fresh clone: data/ is git-ignored, so it has none
    monkeypatch.chdir(work)                              # --scenes is read from here, run.examples_dir from the clone
    def check(setting, path, problem, *argv):            # refused before anything is written, in a line that says what to do
        line = refused(capsys, *argv, "--no-cache", "--name", "early")
        assert line.startswith(f"error: {setting}: {path.resolve()}: {problem}") and not (work / "runs").exists()
        assert line.endswith("python -m sgi.synth.generate --out data/placeholder\n")
    eval_here, examples = work / "data" / "placeholder" / "eval", clone / "data" / "placeholder" / "examples"
    check("--scenes", eval_here, "no such folder", "--scenes", "data/placeholder/eval")
    check("--scenes", placeholder, "holds no scene", "--scenes", str(placeholder))        # the parent of eval/ and examples/
    check("run.examples_dir", examples, "no such folder", "--scenes", str(scenes))        # the default examples_dir
    examples.mkdir(parents=True)
    check("run.examples_dir", examples, "holds no scene", "--scenes", str(scenes))
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: {goal: guess}\n")               # pydantic reports this on several lines; main prints one
    assert str(bad) in refused(capsys, "--scenes", str(scenes), "--config", str(bad), "--name", "bad")
    absent = {"examples_dir": str(tmp_path / "absent")}  # never read when no example is shown
    for name, cfg in (("oracle", config(placeholder, run={**ORACLE, **absent})),
                      ("zero_shot", config(placeholder, run=absent, stages={"goal": {"strategy": "zero_shot"}}))):
        recs = records(harness.run(scenes, cfg, name, tmp_path / "runs"))
        assert len(recs) == len(episodes(scenes)) and all(r["example_ids"] == [] for r in recs)

def test_main_reports_an_undefined_success_rate(placeholder, tmp_path, monkeypatch, capsys, caplog):
    backend = wire(monkeypatch, lambda parts, json_schema: TransientError("503"))    # no request is ever answered
    scenes = subset(tmp_path, placeholder / "eval")
    for name, cfg in (("default", config(placeholder)), ("oracle", config(placeholder, run=ORACLE))):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    monkeypatch.chdir(tmp_path)
    def go(folder, name, saved="default"):               # exit code, what is printed after "<run folder>: ", the outputs
        code = harness.main(["--scenes", str(folder), "--config", f"{saved}.yaml", "--name", name])
        (out,) = (tmp_path / "runs").glob(f"*_{name}")
        shown, _, said = capsys.readouterr().out.partition(": ")
        assert shown == str(Path("runs", out.name))
        return code, said, out, summary(out), yaml.safe_load((out / "config.yaml").read_text())
    n = len(episodes(scenes))
    no_call = "0 calls, 0 from the cache, about $0.00"   # nothing was asked, so nothing was spent
    assert go(scenes, "defined", "oracle")[:2] == (0, f"success rate 1.000 ({n} records, 0 ended in vlm_error; {no_call})\n")
    code, said, out, table, doc = go(scenes, "outage")
    assert (code, said) == (1, f"success rate n/a ({n} records, {n} ended in vlm_error; {n} calls, 0 from the cache)\n")
    assert len(backend.calls) == 3 * n                   # three attempts at each record's plan call, and nothing after it
    assert {c["status"] for r in records(out) for c in r["result"]["calls"]} == {"vlm_error"}
    assert {r["success_rate"] for r in table} == {""} and (int(table[0]["records"]), int(table[0]["vlm_errors"])) == (n, n)
    assert float(table[0]["transient_retry_share"]) == 1.0 and (doc["incomplete"], doc["vlm_errors"]) == (True, n)
    held_out = edited(tmp_path, scenes, lambda gt: gt.update(situation="lift_wait"))    # nothing but held-out episodes
    with caplog.at_level("WARNING", logger="sgi.run"):
        code, said, out, table, doc = go(held_out, "held_out")
    assert (code, said) == (0, f"success rate n/a (0 records, 0 ended in vlm_error; {no_call})\n")
    assert (out / "results.jsonl").read_text() == "" and len(backend.calls) == 3 * n     # written, and empty; no further call
    assert [(r["group_by"], r["group"], r["episodes"], r["records"], r["vlm_errors"], r["success_rate"]) for r in table] == [
        ("overall", "all", "0", "0", "0", "")]
    assert (doc["incomplete"], doc["vlm_errors"], doc["skipped_held_out"]) == (False, 0, n)
    assert f"no episode to evaluate ({n} held-out episodes skipped" in caplog.text

def test_duplicate_scene_id_is_refused(placeholder, tmp_path, monkeypatch):
    wire(monkeypatch)
    scenes = subset(tmp_path, placeholder / "eval")
    export = scene_dirs(scenes)[1].rename(scenes / "export_a")           # a folder's name says nothing about its scene_id
    shared = read_scene_file(export).scene_id
    shutil.copytree(placeholder / "examples", tmp_path / "examples")
    shutil.copytree(export, tmp_path / "examples" / "export_a")          # an evaluated scene, offered as an example too
    cfg = config(placeholder, run={"examples_dir": str(tmp_path / "examples"), "context_check": "always_yes", "k": 50})
    ids = records(harness.run(scenes, cfg, "shared", tmp_path / "runs"))[0]["example_ids"]
    assert len(ids) == len(episodes(placeholder / "examples"))           # every other example: it is left out, no error
    assert shared not in {i.split("/")[0] for i in ids}
    shutil.copytree(export, scenes / "export_b")                         # copied without editing scene.json
    with pytest.raises(SceneError) as refusal:
        harness.run(scenes, cfg, "copied", tmp_path / "runs")
    assert str(refusal.value).startswith(f"{shared}: ") and all(name in str(refusal.value) for name in ("export_a", "export_b"))
    assert len(list((tmp_path / "runs").iterdir())) == 1                 # nothing was written for the refused run

def test_an_episode_without_a_target_is_refused_naming_the_scene_and_the_episode(placeholder, tmp_path, monkeypatch):
    wire(monkeypatch)                                    # scene check off, as --skip-scene-check leaves it
    def untarget(gt):                                    # load_scene accepts an episode that names no target
        gt.update(target_ids=[]), gt.pop("reference_plan", None)
    scenes = edited(tmp_path, subset(tmp_path, placeholder / "eval"), untarget)
    scene_id, episode_id, _ = episodes(scenes)[0]
    assert load_scene(scene_dirs(scenes)[0])[1].episodes[episode_id].target_ids == []
    with pytest.raises(SceneError, match=f"^{scene_id}: episode {episode_id}: target_ids is empty"):
        harness.run(scenes, config(placeholder, run=ORACLE), "untargeted", tmp_path / "runs")
    assert not (tmp_path / "runs").exists()              # refused before anything is written

def method_measures(recs):                              # by situation: the constraint method's measures of a record, in the order of their columns
    by_situation = {}
    for r in recs:
        by_situation.setdefault(r["labels"]["situation"], []).append([r["measures"][name] for name in METHOD_COLUMNS])
    assert all(len(rows) == 2 and rows[0] == rows[1] for rows in by_situation.values())     # the two requests of a scene measure alike
    return {situation: rows[0] for situation, rows in by_situation.items()}

def test_a_run_of_the_method_measures_who_was_found_who_was_listed_and_what_the_solver_made_of_the_answer(placeholder, tmp_path, monkeypatch):
    # The scripted model points at everyone in view, lists every person it is then shown as coming after, and states one
    # constraint about a number nobody has. In one scene nobody is in view, in one a person waits, in one four people stand about.
    out, recs, _, _ = run_method(placeholder, tmp_path, monkeypatch, "measured")
    assert method_measures(recs) == {"no_one": [None, 0, None, None, 2, 0],         # nobody to find, and nobody to come after: unusable too
                                     "waiting_line": [1.0, 0, 1.0, 1.0, 1, 0],      # found, and rightly listed
                                     "conversation_group": [1.0, 0, None, None, 1, 0]}      # all four found; in its reference nobody waits
    table = {(row["group_by"], row["group"]): [row[name] for name in METHOD_COLUMNS] for row in summary(out)}
    assert [float(value) for value in table["overall", "all"]] == pytest.approx([1.0, 0.0, 1.0, 1.0, 8 / 6, 0.0])
    assert table["situation", "no_one"] == ["", "0.0", "", "", "2.0", "0.0"]        # a mean over the records that measured it, or empty
    assert table["situation", "conversation_group"] == ["1.0", "0.0", "", "", "1.0", "0.0"]
    scenes = tmp_path / "subset"
    blind = Scripted(scenes, people={"people": []})     # a model that sees nobody: nobody is found, so nobody can be listed
    _, recs, _, _ = run_method(placeholder, tmp_path, monkeypatch, "blind", scenes=scenes, script=blind)
    assert method_measures(recs) == {"no_one": [None, 0, None, None, 2, 0], "waiting_line": [0.0, 0, None, 0.0, 2, 0],
                                     "conversation_group": [0.0, 0, None, None, 2, 0]}

@pytest.mark.parametrize("run, waiting, others", [
    ({"people": "oracle"}, [None, None, 1.0, 1.0, 1, 0], [None, None, None, None]),         # the people found are measured for a model's alone
    ({"constraints": "none"}, [1.0, 0, None, None, 0, 0], [1.0, 0, None, None]),            # and the members listed for a model's answer alone
], ids=["people_given", "solver_alone"])
def test_what_no_model_was_asked_for_is_not_measured_against_ground_truth(placeholder, tmp_path, monkeypatch, run, waiting, others):
    out, recs, _, _ = run_method(placeholder, tmp_path, monkeypatch, "partly", **run)
    measures = method_measures(recs)
    assert measures["waiting_line"] == waiting and measures["conversation_group"][:4] == others
    empty = [name for name, value in zip(METHOD_COLUMNS, waiting) if value is None]
    assert empty and {row[name] for row in summary(out) for name in empty} == {""}          # those columns stay empty in every row

def test_members_right_is_taken_from_the_answer_before_the_solver_adds_anyone(placeholder, tmp_path, monkeypatch):
    shutil.copytree(placeholder / "eval" / "eval_13_waiting_line_3", tmp_path / "three" / "eval_13_waiting_line_3")     # three wait their turn
    answer = {"analysis": {"arrangement": "a", "expectations": "b", "placement": "c"},      # the middle one, and a number nobody has
              "constraints": [{"because": "They were there first.", "kind": "come_after", "people": [2, 99], "range_m": None, "target": False}]}
    _, recs, _, _ = run_method(placeholder, tmp_path, monkeypatch, "partial", scenes=tmp_path / "three",
                               script=Scripted(tmp_path / "three", constraints=answer), people="oracle")
    assert len(recs) == 2
    for r in recs:
        assert r["result"]["solver"]["joined_waiting"] == [1]                               # the solver takes the one behind them as waiting,
        assert [r["measures"][name] for name in METHOD_COLUMNS] == [None, None, 0.5, pytest.approx(1 / 3), 0, 0]      # and the answer still missed two

def test_someone_of_ground_truth_whom_the_picture_does_not_show_is_nobody_to_find_and_nobody_to_list(placeholder, tmp_path, monkeypatch):
    source = placeholder / "eval" / "eval_06_waiting_line_1"                                # one person waits, and is in view
    shutil.copytree(source, tmp_path / "behind" / source.name)
    doc = json.loads((source / "scene.json").read_text())
    pose = doc["observation"]["robot_pose_world"]       # a second member of that group, 2 m behind the robot and its camera
    doc["ground_truth"]["people"].append({"id": "unseen", "x": pose["x"] - 2.0 * math.cos(pose["yaw"]),
                                          "y": pose["y"] - 2.0 * math.sin(pose["yaw"]), "yaw": pose["yaw"], "group": "g0"})
    (tmp_path / "behind" / source.name / "scene.json").write_text(json.dumps(doc))
    _, recs, _, _ = run_method(placeholder, tmp_path, monkeypatch, "behind", scenes=tmp_path / "behind")
    assert len(recs) == 2 and {r["hidden_relevant_people"] for r in recs} == {1}
    for r in recs:                                      # the one in view was found and listed: all there was to find and to list
        assert [r["measures"][name] for name in METHOD_COLUMNS] == [1.0, 0, 1.0, 1.0, 1, 0]

def test_the_solver_alone_run_of_a_pair_replays_the_calls_of_the_run_with_constraints_and_asks_for_nothing_new(placeholder, tmp_path, monkeypatch):
    pair = {"grounding": "vlm", "context_check": "vlm", "use_cache": True, "cache_dir": str(tmp_path / "cache")}
    _, first, _, asked = run_method(placeholder, tmp_path, monkeypatch, "stated", **pair)
    _, alone, _, replayed = run_method(placeholder, tmp_path, monkeypatch, "alone", scenes=tmp_path / "subset", constraints="none", **pair)
    assert len(asked.calls) > len(first) and replayed.calls == []       # one setting apart: every request it has was the first run's
    assert [[call["stage"] for call in r["result"]["calls"]] for r in alone] == [["plan", "context", "people"]] * len(first)
    assert all(call["cached"] for r in alone for call in r["result"]["calls"])
    assert [r["result"]["source"] for r in alone] == [r["result"]["source"] for r in first] == ["constraints"] * len(first)
    assert [r["result"]["people"] for r in alone] == [r["result"]["people"] for r in first]      # the same episodes, from the same people
    assert {r["result"]["flags"]["no_usable_constraint"] for r in alone} == {True} and all(r["result"]["constraints"] for r in first)
