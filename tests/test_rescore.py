import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from sgi import rescore as tool
from sgi import run as harness
from sgi.config import Config
from sgi.geometry import make_poses
from sgi.gt.derive import prepare
from sgi.scene import load_scene
from sgi.vlm.base import TransientError, VLMClient
from sgi.vlm.mock import centre_responder
from tests.test_run import CONTEXT_ONLY, ORACLE, config, context_answers, placeholder, records, subset, summary, wire  # noqa: F401

LINE = "eval_06_waiting_line_1"          # one person waits at the place: the standard goal stops 0.52 m from them, too near to be reached
GROUP = "eval_10_conversation_group_2"   # two people talk: the standard goal stops 1.21 m from the nearer one, just outside the region

def no_client(monkeypatch):               # from here on no client can be built
    def refuse(*args, **kwargs):
        raise AssertionError("rescoring built a client")
    monkeypatch.setattr(VLMClient, "__init__", refuse)

def finish(placeholder, tmp_path, monkeypatch, run, responder, **top):
    """A finished mock run on three scenes: its folder, its scenes and its back end. From here on no client can be built."""
    backend, scenes = wire(monkeypatch, responder), subset(tmp_path, placeholder / "eval")
    out = harness.run(scenes, config(placeholder, run=run, **top), "done", tmp_path / "runs")
    no_client(monkeypatch)
    return out, scenes, backend

@pytest.fixture
def finished(placeholder, tmp_path, monkeypatch):     # six records, each with one model call and the standard goal
    return finish(placeholder, tmp_path, monkeypatch, CONTEXT_ONLY, context_answers(people_matter=False))

def edit(out, change):                   # change(the records of the run), written back as the run wrote them
    recs = records(out)
    change(recs)
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

def last_line(capsys, out, recs, changed):               # what the command printed, against the form sgi.run prints
    rate = sum(r["score"]["success"] for r in recs) / len(recs)
    calls = sum(len(r["result"]["calls"]) for r in recs)
    assert capsys.readouterr().out == (f"{out / 'rescored'}: success rate {rate:.3f} ({len(recs)} records, 0 ended in vlm_error; "
                                       f"{calls} calls, 0 from the cache); success changed in {changed} of {len(recs)} records\n")

@pytest.mark.parametrize("run, responder", [
    (CONTEXT_ONLY, context_answers(people_matter=False)),                # oracle grounding, the standard goal everywhere
    ({}, centre_responder),                                              # every stage answered by the mock, on the view
    ({"grounding_source": "tour"}, centre_responder),                    # the plan made on the tour, then locate
])
def test_rescoring_with_unchanged_code_reproduces_the_run(placeholder, tmp_path, monkeypatch, capsys, run, responder):
    # personal_gap_m is not the default: scored with the default thresholds, nobody would stand too near in any record.
    out, _, backend = finish(placeholder, tmp_path, monkeypatch, run, responder, thresholds={"personal_gap_m": 5.0})
    made = len(backend.calls)
    assert made == sum(len(r["result"]["calls"]) for r in records(out)) >= 6 and tool.rescore(out) == out / "rescored"
    assert len(backend.calls) == made                    # no model call, and finish() lets no client be built
    assert sorted(p.name for p in (out / "rescored").iterdir()) == ["note.txt", "results.jsonl", "summary.csv"]
    assert records(out / "rescored") == records(out) and len(records(out)) == 6
    assert (out / "rescored" / "summary.csv").read_bytes() == (out / "summary.csv").read_bytes()
    assert True in {r["score"]["personal_space_intrusion"] for r in records(out)}
    assert tool.main([str(out)]) == 0                    # a second time: the folder is written over
    last_line(capsys, out, records(out), 0)

@pytest.mark.parametrize("run, responder, code, rate, cost, measured", [
    (ORACLE, centre_responder, 0, True, True, False),                    # no call is made: a rate, a cost of nothing, no context answer
    (CONTEXT_ONLY, context_answers(people_matter=False), 0, True, False, True),      # an entry without prices, asked about the context
    ({}, lambda parts, json_schema: TransientError("503"), 1, False, False, False),  # no request is answered: no rate
])
def test_the_last_line_the_table_and_the_context_measures_are_those_sgi_run_itself_gives(placeholder, tmp_path, monkeypatch, capsys,
                                                                                         run, responder, code, rate, cost, measured):
    # sgi.run prints its last line, writes summary.csv and works out the context measures inline, so sgi.rescore has all three
    # written again. Here each is held against what sgi.run's own command gave: a change there that is not made here fails.
    wire(monkeypatch, responder)
    scenes = subset(tmp_path, placeholder / "eval")
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(config(placeholder, run=run).model_dump(mode="json")))
    monkeypatch.chdir(tmp_path)                          # sgi.run's command writes under ./runs
    assert harness.main(["--scenes", str(scenes), "--config", "run.yaml", "--name", "done"]) == code
    no_client(monkeypatch)
    (out,) = Path("runs").glob("*_done")
    shown, _, said = capsys.readouterr().out.partition(": ")
    assert shown == str(out) and said.endswith(")\n") and ("rate n/a" not in said, "about $" in said) == (rate, cost)
    assert tool.main([str(out)]) == 0
    assert capsys.readouterr().out == f"{out / 'rescored'}: {said[:-1]}; success changed in 0 of 6 records\n"
    assert (out / "rescored" / "summary.csv").read_bytes() == (out / "summary.csv").read_bytes()
    assert records(out / "rescored") == records(out) and len(records(out)) == 6
    unmeasured = {(r["measures"]["context_missed"], r["measures"]["context_unnecessary"]) == (None, None) for r in records(out)}
    assert unmeasured == {not measured}                  # with no context answer of a model there is nothing to hold against the reference

def test_a_pose_moved_into_the_region_changes_that_record_and_is_counted(finished, capsys):
    out, scenes, _ = finished
    inputs, truth = load_scene(scenes / LINE)
    cfg = Config.model_validate(yaml.safe_load((out / "config.yaml").read_text())["config"])
    region = prepare(inputs, truth, cfg)["e0"].gt.acceptable_regions[0]
    _, pose = make_poses(inputs.observation.robot, region.representative_point, region.heading.yaw)   # behind the one who waits, facing as they do
    def move(recs):
        (rec,) = [r for r in recs if (r["scene_id"], r["episode_id"]) == (LINE, "e0")]
        # In the region, facing as the line does, and no success: 0.52 m from a person, where the region has begun (at 0.5 m)
        # and the scorer does not yet let the robot stand (0.55 m between centres). A count by the distances alone takes it.
        assert [rec["score"][k] for k in ("in_region", "heading_ok", "reachable", "success")] == [True, True, False, False]
        rec["result"]["pose_robot"] = list(pose)
    edit(out, move)
    assert tool.main([str(out)]) == 0
    old, new = records(out), records(out / "rescored")
    (at,) = [i for i, (o, n) in enumerate(zip(old, new)) if o != n]
    assert (new[at]["scene_id"], new[at]["episode_id"]) == (LINE, "e0")
    score = new[at]["score"]
    assert [score[k] for k in ("success", "target_correct", "reachable", "in_region", "heading_ok")] == [True] * 5
    assert score["distance_to_region_m"] == 0.0 and score["forbidden"] == {"ahead_in_line": False}
    assert {k: v for k, v in new[at].items() if k != "score"} == {k: v for k, v in old[at].items() if k != "score"}
    last_line(capsys, out, new, 1)
    rows = {(r["group_by"], r["group"]): float(r["success_rate"]) for r in summary(out / "rescored")}
    assert (rows["overall", "all"], rows["situation", "waiting_line"]) == (0.5, 0.5)       # they were a third and nought
    assert "success changed in 1 of 6 records (1 now a success, 0 now a failure)" in (out / "rescored" / "note.txt").read_text()

def test_the_reference_of_the_context_check_follows_the_regions_derived_now(finished, tmp_path):
    out, scenes, _ = finished
    shutil.copytree(scenes, tmp_path / "wider")
    (rec,) = [r for r in records(out) if (r["scene_id"], r["episode_id"]) == (GROUP, "e0")]
    assert (rec["measures"]["reference_people_matter"], rec["measures"]["context_missed"]) == (True, True)
    x, y, theta = rec["result"]["pose_world"]            # the standard goal: as if a rule now took the square round it, faced that way
    square = {"polygon": [[x - 0.3, y - 0.3], [x + 0.3, y - 0.3], [x + 0.3, y + 0.3], [x - 0.3, y + 0.3]], "heading": {"yaw": theta}}
    doc = json.loads((tmp_path / "wider" / GROUP / "scene.json").read_text())
    for episode in doc["episodes"]:
        episode["ground_truth"]["acceptable_regions"] = [square]
    (tmp_path / "wider" / GROUP / "scene.json").write_text(json.dumps(doc))
    tool.rescore(out, tmp_path / "wider")
    old, new = records(out), records(out / "rescored")
    changed = [n for o, n in zip(old, new) if o != n]
    assert [(n["scene_id"], n["score"]["success"]) for n in changed] == [(GROUP, True)] * 2    # both requests of the scene
    assert [(n["measures"]["reference_people_matter"], n["measures"]["context_missed"], n["measures"]["context_unnecessary"])
            for n in changed] == [(False, False, False)] * 2     # "nobody matters" was the answer, and now it is right
    missed = {r["group"]: float(r["context_missed_rate"]) for r in summary(out / "rescored") if r["group_by"] == "situation"}
    assert missed == {"no_one": 0.0, "waiting_line": 1.0, "conversation_group": 0.0}
    assert "success changed in 2 of 6 records (2 now a success, 0 now a failure)" in (out / "rescored" / "note.txt").read_text()

def test_a_record_without_its_scene_or_episode_is_an_error_and_nothing_is_written(finished, tmp_path, capsys):
    out, scenes, _ = finished
    fewer = tmp_path / "fewer"
    shutil.copytree(scenes, fewer)
    shutil.rmtree(fewer / LINE)
    assert tool.main([str(out), "--scenes", str(fewer)]) == 2
    said = capsys.readouterr()
    assert said.out == "" and said.err.count("\n") == 1 and said.err.startswith(f"error: {LINE}: ") and str(fewer) in said.err
    edit(out, lambda recs: recs[-1].update(episode_id="e9"))
    assert tool.main([str(out)]) == 2
    said = capsys.readouterr()
    assert said.out == "" and said.err.count("\n") == 1 and re.match(r"error: eval_\w+: episode e9: ", said.err)
    with pytest.raises(harness.SceneError, match="episode e9"):
        tool.rescore(out)
    assert not (out / "rescored").exists()

def test_scenes_takes_the_place_of_a_recorded_folder_that_is_gone(finished, tmp_path, capsys):
    out, scenes, _ = finished
    moved = scenes.rename(tmp_path / "moved")
    assert tool.main([str(out)]) == 2                    # the folder config.yaml records, and the way out
    said = capsys.readouterr().err
    assert said.count("\n") == 1 and said.startswith("error: ") and str(scenes) in said and "--scenes" in said
    assert not (out / "rescored").exists()
    assert tool.main([str(out), "--scenes", str(moved)]) == 0 and records(out / "rescored") == records(out)
    capsys.readouterr()
    for folder, word in ((tmp_path, "config.yaml"), (tmp_path / "nowhere", "config.yaml")):       # not a run folder
        assert tool.main([str(folder)]) == 2
        said = capsys.readouterr().err
        assert said.count("\n") == 1 and said.startswith("error: ") and word in said
    (out / "results.jsonl").write_text((out / "results.jsonl").read_text() + '{"scene_id": "cut off\n')
    assert tool.main([str(out), "--scenes", str(moved)]) == 2 and "results.jsonl: line 7" in capsys.readouterr().err

def test_an_unknown_field_is_carried_over_and_a_missing_one_is_read_as_its_default(finished):
    out, _, _ = finished
    def age(recs):
        for r in recs:                   # three fields that no current type knows, and two that an older version did not write
            r["operator"], r["result"]["battery"], r["score"]["old_measure"] = "night shift", {"volts": 24.1}, 1.5
            del r["result"]["flags"]["standard_goal_frontal"], r["measures"]["cost_usd"]
    edit(out, age)
    tool.rescore(out)
    assert records(out / "rescored") == records(out)     # all five as they were: nothing dropped, nothing made up
    overall = summary(out / "rescored")[0]
    assert (overall["flag_standard_goal_frontal_rate"], overall["cost_usd"], overall["success_rate"]) == ("", "", summary(out)[0]["success_rate"])

def test_the_note_says_when_with_which_commit_and_how_many_records_changed(finished, monkeypatch):
    out, scenes, _ = finished
    def git(answer):                     # what `git rev-parse --short HEAD` gives, or how it fails
        def run(command, **kwargs):
            assert command == ["git", "rev-parse", "--short", "HEAD"]
            if isinstance(answer, Exception):
                raise answer
            return subprocess.CompletedProcess(command, 0, stdout=answer)
        monkeypatch.setattr(tool.subprocess, "run", run)
    git("abc1234\n")
    tool.rescore(out)
    assert re.fullmatch(r"rescored \d{4}-\d\d-\d\d \d\d:\d\d:\d\d [+-]\d{4} with the code at commit abc1234\n"
                        rf"scenes: {re.escape(str(scenes))}\n"
                        r"success changed in 0 of 6 records \(0 now a success, 0 now a failure\)\n", (out / "rescored" / "note.txt").read_text())
    for failure in (FileNotFoundError("git"), subprocess.CalledProcessError(128, "git")):      # no git, or no repository
        git(failure)
        tool.rescore(out)
        assert " at commit unknown\n" in (out / "rescored" / "note.txt").read_text()
