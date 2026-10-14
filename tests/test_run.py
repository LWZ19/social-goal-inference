import csv
import importlib.util
import json
import math
import re
import shutil
import typing
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from sgi import run as harness
from sgi.config import REPO_ROOT, Config, ConfigError, RunConfig, Thresholds, load_config
from sgi.freespace import static_grid
from sgi.geometry import bearing, wrap_angle
from sgi.gt.derive import oracle_point, prepare
from sgi.imaging import encode_png
from sgi.pipeline import Pipeline
from sgi.prompts import stages_of
from sgi.scene import EpisodeGT, GroundTruth, Person, load_scene, read_scene_file
from sgi.synth.generate import generate
from sgi.types import (Constraint, ConstraintKind, ConstraintUse, DetectedPerson, Flags, GoalResult, Keys, SolverNote,
                       Usage)
from sgi.vlm.base import BackendResult, ImagePart, TransientError, VLMClient
from sgi.vlm.cache import Cache
from sgi.vlm.mock import MockBackend, centre_responder
from tests.solver_layouts import HERE, ask, layout
from tests.test_docs import commands

ALL = {"no_one", "waiting_line", "conversation_group"}
SOCIAL = ALL - {"no_one"}                # with nobody about the standard goal passes in all but one scene (a lone person), so that holds the only no_one example eligible by default
SHOWN = sorted(ALL | {"audience"})       # the situations with an eligible example in examples/: the three of ALL, and the audience of the generator's second pass
STAGES = ("plan", "locate", "context", "goal")
ORACLE = {"grounding": "oracle", "context_check": "always_yes", "goal": "oracle"}
CONTEXT_ONLY = {"grounding": "oracle", "context_check": "vlm", "goal": "oracle"}     # one model call per record
METHOD_COLUMNS = ["people_found_share", "people_unmatched", "members_precision", "members_recall", "constraints_unusable",
                  "exclusions_broken"]   # the measures of the constraint method, in the order its design's section 9 lists them
SHARED, VIEW, AFTER, WITHIN, VISIBLE, FACE = ("keep_out_of_shared_space", "keep_out_of_view", "come_after", "within_distance",
                                              "visible_to", "face")
_GENERATED = {}

@pytest.fixture(scope="session")
def placeholder(tmp_path_factory):       # eval/ and examples/, generated once however many modules import this fixture
    if not _GENERATED:
        _GENERATED["root"] = tmp_path_factory.mktemp("placeholder")
        generate(_GENERATED["root"], load_config(), seed=0)
    return _GENERATED["root"]

def config(placeholder, run=None, **top):
    base = {"examples_dir": str(placeholder / "examples"), "use_cache": False}
    return load_config(overrides={"run": {**base, **(run or {})}, **top})

def wire(monkeypatch, responder=centre_responder, check=False):
    """Give sgi.run a pipeline whose stages share one recording mock back end. check=False also skips the scene check."""
    backend = MockBackend(responder)
    def build(cfg, examples=()):
        cache = Cache(REPO_ROOT / cfg.run.cache_dir) if cfg.run.use_cache else None      # the folder make_client uses
        client = VLMClient(cfg.models[cfg.stages["goal"].model], backend, cache, sleep=lambda _: None)
        return Pipeline(cfg, dict.fromkeys(stages_of(cfg.run), client), examples)     # a client for each stage of the run
    monkeypatch.setattr("sgi.run.build_pipeline", build)
    if not check:
        monkeypatch.setattr("sgi.run.check_scene", lambda *args, **kwargs: [])
    return backend

def context_answers(people_matter=True, outage=False):
    """Responder for CONTEXT_ONLY runs. outage: the first request gets no answer, however often it is tried."""
    seen = []
    def responder(parts, json_schema):
        seen.append(parts)
        if outage and parts == seen[0]:
            return TransientError("503")
        return {key: people_matter if key == "stop_elsewhere" else "the people nearby" for key in json_schema["properties"]}
    return responder

def scene_dirs(folder):
    return sorted(d for d in Path(folder).iterdir() if (d / "scene.json").is_file())

def episodes(folder):                    # (scene id, episode id, situation) in run order
    return [(f.scene_id, e.episode_id, e.ground_truth.situation)
            for f in map(read_scene_file, scene_dirs(folder)) for e in f.episodes]

def subset(tmp_path, folder):            # a copy of the first scene of each of the three situations of ALL: the generator's
    first = {}                           # first pass. Its second pass (a screen, a picture, a lift) is run whole where it is run
    for d in scene_dirs(folder):
        situation = read_scene_file(d).episodes[0].ground_truth.situation
        if situation in ALL:
            first.setdefault(situation, d)
    for d in first.values():
        shutil.copytree(d, tmp_path / "subset" / d.name)
    return tmp_path / "subset"

def records(out):
    return [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]

def summary(out):
    return list(csv.DictReader((out / "summary.csv").read_text().splitlines()))

def rec(scene, trial, success, status="ok", situation="waiting_line", forbidden=None, result=None, measures=None, **factors):
    return {"scene_id": scene, "episode_id": "e0", "trial": trial, "result": {"status": status, **(result or {})},
            "labels": {"request_type": "high_level", "situation": situation, "factors": factors},
            "score": {"success": success, "forbidden": forbidden or {}}, "measures": measures or {}}

def found(*spots):                       # detected people, numbered from 1 in the order given, each standing at its (x, y)
    return [DetectedPerson(number=n, xy=xy, pixel=(0.0, 0.0)) for n, xy in enumerate(spots, 1)]

def standing(*spots):                    # the people of ground truth that are in view, p0 on: each (x, y) or (x, y, group)
    return [Person(id=f"p{i}", x=x, y=y, yaw=0.0, group=group[0] if group else None) for i, (x, y, *group) in enumerate(spots)]

def stated(kind, *people, status="used", broken_m=0.0):       # one constraint of an answer, and what the solver made of it
    asked = Constraint(because="what the people do", kind=kind, people=list(people))
    return ConstraintUse(constraint=asked, status=status, broken_m=broken_m)

def placed(people, *uses, own=None):     # a result of the method: the people found, each constraint, the own spaces the place is inside
    keys = Keys(excluded_m=0.0, excluded_total_m=0.0, preference_m=0.0, standard_m=0.0)
    note = SolverNote(keys=keys, own_space_broken_m=own or {}, target_distance_m=1.0)
    return GoalResult(status="ok", source="constraints", people=people, constraints=list(uses), solver=note)

def measured(result, visible, group=None, **run):             # the method's measures of one result; group: whose members wait their turn
    return harness._method_measures(result, visible, group, RunConfig(goal="constraints", **run), Thresholds())

def test_summarise_bins_trials_and_denominators():
    slow = {"timings_s": {"goal": 4.0}, "flags": {"heading_fallback": True},
            "calls": [{"transient_retries": 0}, {"transient_retries": 1}]}
    fast = {"timings_s": {"goal": 2.0}, "flags": {"heading_fallback": False}}
    rows = {(r["group_by"], r["group"]): r for r in harness.summarise([
        rec("a", 0, True, group_size=0, view_angle_deg=30.0000004),
        rec("a", 1, False, group_size=0, view_angle_deg=30.0000004),
        rec("b", 0, True, group_size=2, view_angle_deg=60.0, forbidden={"ahead_in_line": False}, result=slow),
        rec("b", 1, True, group_size=2, view_angle_deg=60.0, forbidden={"ahead_in_line": True}, result=fast),
        rec("c", 0, False, status="vlm_error", group_size=4,
            result={"timings_s": {"goal": 40.0}, "calls": [{"transient_retries": 2}]}),
        rec("c", 1, True, group_size=4, forbidden={"ahead_in_line": False, "group_space": False}),
        rec("d", 0, False, status="parse_failed", situation="no_one", group_size=7),
        rec("d", 1, False, situation="no_one", group_size=7)])}
    overall = rows["overall", "all"]
    assert (overall["episodes"], overall["records"], overall["vlm_errors"]) == (4, 8, 1)
    assert [overall[k] for k in ("success_rate", "success_rate_min", "success_rate_max")] == pytest.approx(
        [0.5833, 0.5, 0.6667], abs=1e-4)                 # trial 0: 2 of the 3 with an answer; trial 1: 2 of 4
    assert [overall["forbidden_ahead_in_line_rate" + end] for end in ("", "_min", "_max")] == [0.25, 0.0, 0.5]
    assert overall["forbidden_group_space_rate"] == 0.0 and overall["reachable_rate"] is None
    assert [overall[k] for k in ("latency_goal_s", "latency_goal_s_max", "flag_heading_fallback_rate")] == [3.0, 4.0, 0.5]
    assert [overall[f"status_{s}_rate"] for s in ("parse_failed", "no_target")] == pytest.approx([0.1667, 0.0], abs=1e-4)
    assert overall["latency_plan_s"] is None and overall["hidden_relevant_people"] is None
    assert rows["situation", "no_one"]["forbidden_ahead_in_line_rate"] is None      # no such region: not measured
    assert [g for b, g in rows if b == "group_size"] == ["0", "1-2", "3-5", "6-10"]
    assert [g for b, g in rows if b == "view_angle_deg"] == ["<=30", ">30", "unlabelled"]
    assert [g for b, g in rows if b == "density"] == ["unlabelled"]
    late = rows["group_size", "3-5"]
    assert (late["records"], late["vlm_errors"], late["success_rate"], late["success_rate_min"]) == (2, 1, 1.0, 1.0)
    shares = [rows[key]["transient_retry_share"] for key in (("overall", "all"), ("group_size", "3-5"), ("situation", "no_one"))]
    assert shares == [pytest.approx(0.6667, abs=1e-4), 1.0, None]   # 2 of 3 calls; the vlm_error record's call counts; no call
    assert "transient_retry_share_min" not in overall and list(overall)[:6] == [
        "group_by", "group", "episodes", "records", "vlm_errors", "transient_retry_share"]
    only = harness.summarise([rec("c", 0, False, status="vlm_error")])[0]
    assert (only["group_by"], only["success_rate"], only["vlm_errors"]) == ("overall", None, 1)
    assert "success_rate_min" not in only                # one trial: no minimum and maximum columns
    (empty,) = harness.summarise([])                     # no record at all: the overall row alone, nothing divided
    assert (empty["group_by"], empty["episodes"], empty["records"], empty["vlm_errors"]) == ("overall", 0, 0, 0)
    assert empty["success_rate"] is None and empty["transient_retry_share"] is None and set(empty) == set(only)

def test_summarise_counts_the_calls_and_adds_up_what_they_cost():
    def made(*cached):                                   # a record's calls, each fresh or answered from the cache
        return {"calls": [{"transient_retries": 0, "cached": hit} for hit in cached]}
    rows = {(r["group_by"], r["group"]): r for r in harness.summarise([
        rec("a", 0, True, result=made(False, True), measures={"cost_usd": 0.25}),
        rec("a", 1, True, result=made(True, True), measures={"cost_usd": 0.25}),          # replayed: it cost what it cost
        rec("b", 0, False, status="vlm_error", situation="no_one", result=made(False), measures={"cost_usd": 0.0}),
        rec("c", 0, True, situation="audience", measures={"cost_usd": 0.0}),              # no call at all
        rec("d", 0, True, situation="audience", result=made(False), measures={"cost_usd": None})])}     # no prices known
    def counted(key):
        return [rows[key][name] for name in ("calls", "cached_calls", "cost_usd")]
    assert counted(("situation", "waiting_line")) == [4, 3, 0.5] and counted(("situation", "no_one")) == [1, 0, 0.0]
    assert counted(("situation", "audience")) == [1, 0, None]         # one call of unknown cost leaves the sum unknown
    assert counted(("overall", "all")) == [6, 3, None] == counted(("request_type", "high_level"))
    assert list(rows["overall", "all"])[-3:] == ["calls", "cached_calls", "cost_usd"]     # after every other column,
    assert "success_rate_max" in rows["overall", "all"] and "calls_max" not in rows["overall", "all"]   # with no spread
    (empty,) = harness.summarise([])
    assert [empty[name] for name in ("calls", "cached_calls", "cost_usd")] == [0, 0, 0.0]

def test_people_found_is_the_share_of_the_visible_people_with_a_detection_a_bodys_width_away_each_matched_once():
    visible = standing((0.0, 0.0), (0.6, 0.0), (5.0, 5.0))
    # Nearest first: detection 2 is 0.10 m from p0 and is theirs, so detection 1, nearer p0 than p1, is p1's. Nobody found p2.
    people = found((0.25, 0.0), (-0.1, 0.0))
    assert harness._matched_people(people, visible, Thresholds()) == {2: "p0", 1: "p1"}
    got = measured(placed(people), visible)
    assert list(got) == METHOD_COLUMNS and (got["people_found_share"], got["people_unmatched"]) == (pytest.approx(2 / 3), 0)
    # Each once: a third detection, 0.45 m from p1 who is taken, matches nobody, and so does one that stands where nobody does.
    got = measured(placed(found((0.25, 0.0), (-0.1, 0.0), (1.05, 0.0), (9.0, 9.0))), visible)
    assert (got["people_found_share"], got["people_unmatched"]) == (pytest.approx(2 / 3), 2)
    # A body's width is twice the radius of one, 0.5 m, and "within" takes the limit itself.
    alone = standing((0.0, 0.0))
    at_width, beyond = measured(placed(found((0.5, 0.0))), alone), measured(placed(found((0.51, 0.0))), alone)
    assert (at_width["people_found_share"], at_width["people_unmatched"]) == (1.0, 0)
    assert (beyond["people_found_share"], beyond["people_unmatched"]) == (0.0, 1)
    wider = Thresholds(person_radius_m=0.3)              # the radius is the configuration's
    assert harness._method_measures(placed(found((0.55, 0.0))), alone, None, RunConfig(goal="constraints"), wider)["people_found_share"] == 1.0
    # With nobody of ground truth in view there is no share, and every detection matches nobody; nobody found is a share of 0.
    nobody, missed = measured(placed(people), []), measured(placed([]), visible)
    assert (nobody["people_found_share"], nobody["people_unmatched"]) == (None, 2)
    assert (missed["people_found_share"], missed["people_unmatched"]) == (0.0, 0)
    # The two are of people a model found: with the people of ground truth they are not measured, and the rest is.
    given = measured(placed(people), visible, people="oracle")
    assert (given["people_found_share"], given["people_unmatched"], given["constraints_unusable"]) == (None, None, 0)
    # A result that is not of the method, and one whose people call brought no valid answer, measure none of the six.
    nothing = dict.fromkeys(METHOD_COLUMNS)
    assert measured(GoalResult(status="ok", source="direct_marking"), visible) == nothing
    assert measured(GoalResult(status="parse_failed", source="constraints"), visible) == nothing
    # A constraints call without a valid answer leaves the people known: they are measured, and nothing of the solver is.
    unsolved = measured(GoalResult(status="parse_failed", source="constraints", people=people), visible)
    assert unsolved == {**nothing, "people_found_share": pytest.approx(2 / 3), "people_unmatched": 0}

def test_members_right_is_the_precision_and_recall_of_the_people_listed_as_coming_after_against_the_members_in_view():
    # Three wait their turn (g0), one stands by, one belongs to another group. The first four are found where they stand,
    # the fifth detection stands where nobody does, and nobody found p4.
    visible = standing((0.0, 0.0, "g0"), (0.0, 1.0, "g0"), (0.0, 2.0, "g0"), (3.0, 0.0), (4.0, 4.0, "g1"))
    people = found((0.0, 0.1), (0.0, 1.1), (0.0, 2.1), (3.0, 0.1), (7.0, 7.0))
    def right(*uses, among=visible, group="g0", by=people, **run):
        got = measured(placed(by, *uses), among, group, **run)
        return got["members_precision"], got["members_recall"]
    assert right(stated(AFTER, 1, 2, 3)) == (1.0, 1.0)
    assert right(stated(AFTER, 3, 1)) == (1.0, pytest.approx(2 / 3))                 # a member left out
    assert right(stated(AFTER, 1, 2, 3, 4)) == (0.75, 1.0)                           # someone who stands by, listed
    # A detection where nobody stands and a number nobody has are listed all the same, and are wrong.
    assert right(stated(AFTER, 1, 4, 5, 9)) == (0.25, pytest.approx(1 / 3))
    # Several are one, a number counts once, and what the solver made of a constraint does not matter: it is the model's list.
    assert right(stated(AFTER, 2, 2, 1), stated(AFTER, 1, 3, status="cut"), stated(AFTER, 9, status="unusable")) == (0.75, 1.0)
    # No other kind lists who waits. With nobody listed there is no precision, and every member is missed.
    assert right(stated(VIEW, 1, 2, 3), stated(SHARED, 1, 2), stated(FACE, 3)) == (None, 0.0) == right()
    # Against the members in view: one the picture hides is nobody's to list, and the detection on them matches nobody.
    assert right(stated(AFTER, 1, 2, 3), among=visible[:2] + visible[3:]) == (pytest.approx(2 / 3), 1.0)
    # A member in view whom the people step did not find is missed, whatever the answer lists.
    assert right(stated(AFTER, 1, 2), by=people[:2]) == (1.0, pytest.approx(2 / 3))
    assert right(stated(AFTER, 1, 2, 3), group="g1") == (0.0, 0.0)                   # the members are those of the group named
    assert right(stated(AFTER, 4), among=visible[3:4], by=found((3.0, 0.1))) == (0.0, None)      # no member in view: no recall
    # Not measured: a reference in which nobody waits their turn, and a run in which no model stated the constraints.
    assert right(stated(AFTER, 1, 2, 3), group=None) == (None, None) == right(stated(AFTER, 1, 2, 3), constraints="none")
    assert right(stated(AFTER, 1, 2, 3), people="oracle") == (1.0, 1.0)              # measured with the people of ground truth too

def test_who_waits_their_turn_is_the_group_at_the_chosen_target_where_the_reference_has_people_waiting():
    def group(situation, at, chosen, targets=("t0", "t1")):
        ep = EpisodeGT(request_type="grounded", target_ids=list(targets), situation=situation, group_at_target=at)
        truth = GroundTruth(scene_id="s", people=(), targets={}, bounds=None, obstacles=(), occupancy=None, occupancy_image=None,
                            episodes={"e0": ep})
        return harness._turn_group(truth, "e0", ep, chosen)
    both = {"t0": "g0", "t1": "g1"}
    assert group("waiting_line", both, "t1") == "g1" and group("waiting_line", both, None) == "g0"     # else the first target's
    assert group("waiting_line", "g0", None, targets=("t0",)) == "g0"                # a bare id: the group at the one target
    assert group("waiting_line", {"t0": "g0", "t1": None}, "t1") is None and group("waiting_line", None, None) is None
    assert group("conversation_group", both, "t0") is None and group("no_one", None, None) is None     # there nobody waits

def test_unusable_constraints_and_exclusions_broken_are_counted_from_what_the_solver_recorded():
    people = found((0.0, 0.0), (1.0, 0.0))
    def counted(*uses, own=None, **run):
        got = measured(placed(people, *uses, own=own), [], **run)
        return got["constraints_unusable"], got["exclusions_broken"]
    uses = [stated(AFTER, 1, broken_m=0.2), stated(AFTER, 2, broken_m=0.2),          # several are one exclusion
            stated(VIEW, 1, broken_m=0.1), stated(SHARED, 1, 2),                     # one broken, one kept
            stated(WITHIN, 1, broken_m=0.5), stated(VISIBLE, 1, broken_m=0.3),       # preferences: missed, and no exclusion
            stated(VISIBLE, 9, status="unusable"), stated(FACE, 1), stated(FACE, 2, status="superseded"),
            stated(VIEW, 2, status="cut")]
    # Unusable as the flag has it, superseded included; a constraint cut is not one. The own space of two people, the view
    # and the turn are four exclusions broken.
    assert counted(*uses, own={1: 0.3, 2: 0.05}) == (2, 4)
    assert counted(stated(AFTER, 1), stated(VIEW, 1), stated(SHARED, 1, 2)) == (0, 0)
    assert counted(stated(SHARED, 1, 2, broken_m=0.001), stated(VIEW, 1, broken_m=0.3)) == (0, 2)
    assert counted(own={2: 0.4}, constraints="none") == (0, 1) and counted(constraints="none") == (0, 0)     # the solver alone
    assert counted(stated(VISIBLE, 9, status="unusable"), people="oracle") == (1, 0)

def test_exclusions_broken_counts_what_the_solver_itself_records_and_is_above_zero_exactly_where_its_flag_is_set():
    def counted(lay, *answer):           # the solver's own record of a layout, counted
        solved = lay.solve(*answer)
        result = GoalResult(status="ok", source="constraints", people=lay.people, constraints=solved.uses, solver=solved.note,
                            flags=solved.flags)
        broken = measured(result, [])["exclusions_broken"]
        assert (broken > 0) is solved.flags.exclusion_broken
        return broken
    # Between two people 1.5 m apart, with no floor but a strip across the line between them: inside the own space of both.
    strip = lambda ahead, left: (np.abs(ahead) <= 0.21) & (np.abs(left) <= 0.75)
    assert counted(layout(HERE, [(0.0, 0.75), (0.0, -0.75)], thing=(3.0, 0.0), floor=strip)) == 2
    # Someone 2 m behind the robot looks along a lane that is all the floor there is: their view is broken, and nothing else.
    lane = lambda ahead, left: (np.abs(left) <= 0.21) & (ahead >= -0.31) & (ahead <= 3.0)
    assert counted(layout(HERE, [(-2.0, 0.0, 0.0)], thing=(3.0, 0.0), floor=lane), ask(VIEW, 1), ask(FACE, target=True)) == 1
    # On an open floor the view holds. A range that own space does not let the robot keep is a preference missed, and no exclusion.
    assert counted(layout(HERE, [(-2.0, 0.0, 0.0)], thing=(3.0, 0.0)), ask(VIEW, 1), ask(WITHIN, 1, between=(0.0, 0.5))) == 0

def test_the_measures_of_the_method_are_columns_after_the_others_and_empty_where_no_record_measured_them():
    def of(*values):
        return dict(zip(METHOD_COLUMNS, values))
    rows = {(r["group_by"], r["group"]): r for r in harness.summarise([
        rec("a", 0, True, measures=of(1.0, 0, 1.0, 0.5, 1, 0)),
        rec("a", 1, True, measures=of(0.5, 2, None, 0.0, 3, 1)),
        rec("b", 0, False, situation="conversation_group", measures=of(0.5, 1, None, None, 0, 2)),
        rec("b", 1, False, situation="conversation_group", measures=of(0.0, 0, None, None, 1, 0)),
        rec("c", 0, True, situation="no_one", measures=of(*[None] * 6)),             # the standard goal: nothing of the method
        rec("c", 1, True, situation="no_one")])}                                     # a record written before there were any
    overall = rows["overall", "all"]
    def values(row, name):
        return [row[name + end] for end in ("", "_min", "_max")]
    assert values(overall, "people_found_share") == [0.5, 0.25, 0.75] and values(overall, "people_unmatched") == [0.75, 0.5, 1.0]
    assert values(overall, "members_precision") == [1.0, 1.0, 1.0]                   # one trial measured it, in one record
    assert values(overall, "members_recall") == [0.25, 0.0, 0.5]
    assert values(overall, "constraints_unusable") == [1.25, 0.5, 2.0] and values(overall, "exclusions_broken") == [0.75, 0.5, 1.0]
    group = rows["situation", "conversation_group"]
    assert values(group, "members_recall") == [None] * 3 and values(group, "exclusions_broken") == [1.0, 0.0, 2.0]
    assert [values(rows["situation", "no_one"], name) for name in METHOD_COLUMNS] == [[None] * 3] * 6
    # They follow every measure there was, and the three totals stay the last columns.
    names = [name for name in overall if not name.endswith(("_min", "_max"))]
    assert names[-10:] == ["hidden_relevant_people", *METHOD_COLUMNS, "calls", "cached_calls", "cost_usd"]
    assert list(overall)[-21:-3] == [name + end for name in METHOD_COLUMNS for end in ("", "_min", "_max")]
    baseline = harness.summarise([rec("a", 0, True)])[0]                             # one record of a baseline run
    assert [baseline[name] for name in METHOD_COLUMNS] == [None] * 6 and list(baseline)[-9:-3] == METHOD_COLUMNS

def test_the_readme_and_the_page_of_the_method_name_its_measures_and_the_commands_of_the_page_exist():
    readme, page = ((REPO_ROOT / name).read_text(encoding="utf-8") for name in ("README.md", "docs/constraint_method.md"))
    reading = readme.split("\n## Reading a run folder\n")[1].split("\n## ")[0]
    assert [name for name in METHOD_COLUMNS if f"`{name}`" not in reading or f"`{name}`" not in page] == []
    assert "(docs/constraint_method.md)" in readme.split("\n## Setup\n")[0]            # the introduction points to the page
    # The pair of runs a result of the method is read from: two configs one setting apart, the second the solver alone.
    pair = readme.split("\n### Read a constraint run beside the solver alone\n")[1].split("\n### ")[0]
    assert [args[args.index("--config") + 1] for module, args in commands(pair)] == ["constraints.yaml", "solver_alone.yaml"]
    assert all(said in pair for said in ("`constraints: none`", "the difference between the two", "asks for nothing new"))
    # The page: its commands name real modules, options and model entries, and its links lead somewhere.
    given, cfg = commands(page), load_config()
    assert [module for module, _ in given] == ["sgi.run", "sgi.smoke", "sgi.run", "sgi.run", "sgi.run", "sgi.run"]
    for module, args in given:
        source = Path(importlib.util.find_spec(module).origin).read_text(encoding="utf-8")
        assert [a for a in args if a.startswith("--") and not re.search(rf"""['"]{a}['"]""", source)] == [], module
        assert args[args.index("--model") + 1] in cfg.models
    assert [args[args.index("--config") + 1] for _, args in given if "--config" in args] == [
        "constraints.yaml", "constraints.yaml", "solver_alone.yaml", "given_people.yaml"]
    links = re.findall(r"\]\(([^)#]+)\)", page)
    assert len(links) >= 4 and [link for link in links if not (REPO_ROOT / "docs" / link).exists()] == []
    words = [*typing.get_args(ConstraintKind), "goal: constraints", "people: vlm", "people: oracle", "constraints: none",
             *list(Flags.model_fields)[-6:], "joined_waiting", "tests/test_reference_constraints.py", "tests/test_solver.py"]
    assert len(words) == 19 and [word for word in words if f"`{word}`" not in page] == []
    assert all((REPO_ROOT / word).is_file() for word in words[-2:])                   # the tests the page cites for what it says

def chosen(placeholder, needed, taken=(), run=None, **top):       # the selection, and the situation of each example in it
    situation = {f"{s}/{e}": sit for s, e, sit in episodes(placeholder / "examples")}
    picked = harness.select_examples(scene_dirs(placeholder / "examples"), set(taken), set(needed),
                                     config(placeholder, run, **top))
    return picked, [situation[i] for i, _ in picked]

def test_selection_eligibility_coverage_and_round_robin(placeholder):
    def kinds(*args, **kwargs):
        return chosen(placeholder, *args, **kwargs)[1]
    everything = [sit for _, _, sit in episodes(placeholder / "examples")]
    plain, one_each = chosen(placeholder, SOCIAL)
    assert sorted(one_each) == SHOWN                                              # default: one per situation that has one, needed or not:
    assert all(ex.reasoning is None and ex.look_at_point is None for _, ex in plain)    # so one of an audience too, which no evaluated episode here is
    held_out = dict(dev_situations=["no_one", "waiting_line", "audience"],
                    held_out_situations=["group_photo", "lift_wait", "conversation_group"])
    assert kinds(SOCIAL, run={"leave_out_situation": "waiting_line"}) == ["conversation_group", "no_one", "audience"]   # needed first
    assert kinds(SOCIAL, **held_out) == ["waiting_line", "no_one", "audience"]
    every = {"context_check": "always_yes"}                                       # drops the people_matter filter
    assert sorted(kinds(ALL, run={**every, "leave_out_situation": "waiting_line"})) == ["audience", "conversation_group", "no_one"]
    assert sorted(kinds(ALL, run=every, **held_out)) == ["audience", "no_one", "waiting_line"]
    assert sorted(kinds(set(), run={**every, "k": 50})) == sorted(everything)
    # The standard goal is fine in some: those with nobody about, but for the lone person's scene (two episodes).
    assert sorted(kinds(set(), run={"k": 50})) == ["audience"] * 2 + ["conversation_group"] * 4 + ["no_one"] * 2 + ["waiting_line"] * 4
    assert kinds({"waiting_line"}, run={**every, "k": 1}) == ["waiting_line"]     # k: round-robin, needed situations first
    assert kinds({"conversation_group"}, run={**every, "k": 4}) == ["conversation_group", "no_one", "waiting_line", "audience"]
    assert kinds({"conversation_group"}, run={**every, "k": 5})[4] == "conversation_group"     # then round again, from the needed one
    with pytest.raises(ConfigError, match=r"selected for conversation_group: run\.k is 2, "):
        kinds(ALL, run={**every, "k": 2})                                         # k too small to cover three

def test_a_needed_situation_without_an_eligible_example_is_warned_of_and_the_rest_is_selected(placeholder, caplog):
    def kinds(needed, **kwargs):                                                  # the situations selected, and what was logged
        caplog.clear()
        with caplog.at_level("WARNING", logger="sgi.run"):
            picked = chosen(placeholder, needed, **kwargs)[1]
        return sorted(picked), [r.getMessage() for r in caplog.records if r.name == "sgi.run"]
    others = sorted(SOCIAL | {"audience"})                                        # what is shown beside an example of nobody about
    assert kinds(SOCIAL) == (SHOWN, [])                                           # every needed situation has one: nothing is said
    assert kinds(ALL) == (SHOWN, [])                                              # the lone person's scene is the one no_one example
    shown, shown_kinds = chosen(placeholder, ALL)
    lone = {i.split("/")[0] for (i, _), sit in zip(shown, shown_kinds) if sit == "no_one"}
    picked, said = kinds(ALL, taken=lone)                                         # never from an evaluated scene: with that scene
    assert picked == others and len(said) == 1                                    # taken no no_one example is eligible: needed,
    assert said[0].startswith("no in-context example for no_one: ")               # such an example is missed
    lines = {s for s, _, sit in episodes(placeholder / "examples") if sit == "waiting_line"}
    picked, said = kinds({"waiting_line"}, taken=lines)                           # never from an evaluated scene, so a needed
    assert picked == ["audience", "conversation_group", "no_one"] and len(said) == 1     # situation is left with none
    assert said[0].startswith("no in-context example for waiting_line: ")
    everything = {s for s, _, _ in episodes(placeholder / "examples")}
    picked, said = kinds(ALL, taken=everything)                                   # no example at all: one warning for the three
    assert picked == [] and len(said) == 1                                        # that are needed, in the order of the config's dev_situations
    assert said[0].startswith("no in-context example for no_one, waiting_line, conversation_group: ")
    assert kinds(ALL, run={"leave_out_situation": "no_one"}) == (others, [])      # left out: no example of it is asked for
    picked, said = kinds(ALL, taken=lone, run={"k": 3})                           # with run.k: the same warning, k of the others,
    assert picked == ["audience", "conversation_group", "waiting_line"] and len(said) == 1    # one of each in the first round
    assert said[0].startswith("no in-context example for no_one: ")
    # A needed situation that has an eligible example which run.k is too small to reach still stops the run, as it did.
    with pytest.raises(ConfigError, match=r"selected for conversation_group: run\.k is 1, "):
        kinds(ALL, taken=lone, run={"k": 1})

@pytest.mark.parametrize("check, needed, count", [("vlm", SOCIAL, 4), ("always_yes", ALL, 4)])    # always_yes: nobody about, too. One of each situation of SHOWN
def test_examples_are_marked_images_with_projected_answers(placeholder, check, needed, count):
    cfg = config(placeholder, run={"heading_mode": "vlm_look_at", "context_check": check}, stages={"goal": {"strategy": "few_shot_cot"}})
    chosen = harness.select_examples(scene_dirs(placeholder / "examples"), set(), needed, cfg)
    assert len(chosen) == count
    for example_id, ex in chosen:
        scene_id, episode_id = example_id.split("/")
        folder = next(d for d in scene_dirs(placeholder / "examples") if read_scene_file(d).scene_id == scene_id)
        inputs, truth = load_scene(folder)
        et, obs = prepare(inputs, truth, cfg)[episode_id], inputs.observation
        region, target = et.gt.acceptable_regions[0], truth.targets[et.gt.target_ids[0]]
        asked = next(e.request for e in inputs.episodes if e.episode_id == episode_id)
        assert (ex.request, ex.target_text, ex.reasoning.model_dump()) == (asked, target.label, et.gt.rationale.model_dump())
        assert ex.goal_point == oracle_point(obs, (*region.representative_point, 0.0))
        (x, y), heading = region.representative_point, region.heading      # a waiting line's heading is a yaw: 1 m along it
        faced = heading.face_point or (x + math.cos(heading.yaw), y + math.sin(heading.yaw))
        assert ex.look_at_point == oracle_point(obs, (*faced, 0.0))
        assert (ex.image.width, ex.image.height) == (640, 480) and ex.image.png != encode_png(obs.rgb).png

def test_yaw_look_at_point_and_the_three_filters_on_an_edited_example(placeholder, tmp_path, caplog):
    cfg = config(placeholder, run={"heading_mode": "vlm_look_at"}, stages={"goal": {"strategy": "few_shot_cot"}})
    source = placeholder / "examples" / "examples_03_waiting_line_3"      # its region lies near the edge of the image
    inputs, truth = load_scene(source)
    region = truth.episodes["e0"].acceptable_regions[0]
    (x, y), desk = region.representative_point, truth.targets["t0"].anchor[:2]
    last = max(((p.x, p.y) for p in truth.people), key=lambda p: math.dist(p, desk))      # the last person in the line
    def offer(name, change, needed=()):      # the scene, with every episode's ground truth changed, as the only example scene
        shutil.copytree(source, tmp_path / name)
        doc = json.loads((source / "scene.json").read_text())
        for episode in doc["episodes"]:
            change(episode["ground_truth"])
        (tmp_path / name / "scene.json").write_text(json.dumps(doc))
        return harness.select_examples([tmp_path / name], set(), set(needed), cfg)
    def facing(yaw):
        return lambda gt: gt["acceptable_regions"][0].update(heading={"yaw": yaw})
    ahead = bearing((x, y), last)            # towards the last person in the line: 1 m on is still in view
    ((_, ex),) = offer("ahead", facing(ahead), {"waiting_line"})
    assert ex.look_at_point == oracle_point(inputs.observation, (x + math.cos(ahead), y + math.sin(ahead), 0.0))
    assert ex.look_at_point not in (ex.goal_point, oracle_point(inputs.observation, (*last, 0.0)))
    ((_, back),) = offer("away", facing(wrap_angle(ahead + math.pi)))     # 1 m the other way is in view as well: the point is 0.89 m from the line's end
    assert back.look_at_point == oracle_point(inputs.observation, (x - math.cos(ahead), y - math.sin(ahead), 0.0))
    beyond = [x - 3.0 * math.cos(ahead), y - 3.0 * math.sin(ahead)]      # 3 m that way is outside the image
    assert offer("out_of_view", lambda gt: gt["acceptable_regions"][0].update(heading={"face_point": beyond})) == []
    assert offer("on_a_person", lambda gt: gt["acceptable_regions"][0].update(representative_point=last)) == []   # fails the oracle path
    assert offer("bare", lambda gt: gt.pop("rationale")) == []            # few_shot_cot needs the rationale
    with caplog.at_level("WARNING", logger="sgi.run"):                    # needed, it is missed: said, and no example is given
        assert offer("needed", lambda gt: gt.pop("rationale"), {"waiting_line"}) == []
    said = [r.getMessage() for r in caplog.records if r.name == "sgi.run"]
    assert len(said) == 1 and said[0].startswith("no in-context example for waiting_line: ")

def test_default_mock_run_writes_every_output(placeholder, tmp_path, monkeypatch, caplog):
    backend, cfg, scenes = wire(monkeypatch), config(placeholder), subset(tmp_path, placeholder / "eval")
    with caplog.at_level("WARNING", logger="sgi.run"):
        out = harness.run(scenes, cfg, "plumbing", tmp_path / "runs")
    assert "no in-context example" not in caplog.text    # every situation that needs a worked example has one
    assert out.parent == tmp_path / "runs" and re.fullmatch(r"\d{8}-\d{6}_plumbing", out.name)
    eps, recs = episodes(scenes), records(out)
    assert [(r["scene_id"], r["episode_id"], r["trial"]) for r in recs] == [(s, e, 0) for s, e, _ in eps]
    assert sorted(p.name for p in (out / "viz").iterdir()) == sorted(f"{s}_{e}.png" for s, e, _ in eps)
    doc = yaml.safe_load((out / "config.yaml").read_text())
    assert Config.model_validate(doc["config"]) == cfg and (doc["incomplete"], doc["vlm_errors"]) == (False, 0)
    first = recs[0]
    assert set(first) == {"scene_id", "episode_id", "trial", "request", "config", "labels", "hidden_relevant_people",
                          "plan", "evaluated_subtask", "result", "score", "measures", "example_ids"}
    assert (first["config"]["grounding"], first["config"]["stages"]["goal"]["strategy"]) == ("vlm", "few_shot")
    assert set(first["config"]) == {"grounding", "context_check", "goal", "people", "constraints", "grounding_source",
                                    "heading_mode", "stages"}      # the run modes, the method's two among them
    assert (first["config"]["people"], first["config"]["constraints"]) == ("vlm", "vlm")
    assert list(first["config"]["stages"]) == [*STAGES, "people", "constraints"]     # all six, of which this run calls three
    assert sorted(p.name for p in (out / "prompts").iterdir()) == sorted(f"{stage}.md" for stage in STAGES)
    assert set(first["labels"]) == {"request_type", "situation", "factors"} and first["evaluated_subtask"] == 0
    assert [c["stage"] for c in first["result"]["calls"]] == ["plan", "context", "goal"]
    assert all(c["status"] == "ok" and c["raw_text"] for c in first["result"]["calls"])
    assert set(first["result"]["timings_s"]) == {"plan", "context", "goal", "geometry"}
    assert first["measures"]["latency_s"] == pytest.approx(sum(first["result"]["timings_s"].values()))
    ids = doc["example_ids"]             # chosen once: every record and every goal request carries the same examples
    assert ids and all(r["example_ids"] == ids for r in recs) and doc["examples_missing"] == []
    shown = [{p.png for p in parts if isinstance(p, ImagePart)}
             for _, parts, schema in backend.calls if "goal_point" in schema["properties"]]
    assert len(shown) == len(recs) and all(len(images) == len(ids) + 1 for images in shown)
    assert len(set.intersection(*shown)) == len(ids)
    table = summary(out)
    rows, overall = [(r["group_by"], r["group"]) for r in table], table[0]
    assert rows[0] == ("overall", "all") and {g for b, g in rows if b == "situation"} == {sit for _, _, sit in eps}
    assert {b for b, _ in rows} == {"overall", "request_type", "situation", "density", "group_size", "bystanders",
                                    "other_groups", "view_angle_deg"}
    assert float(overall["latency_plan_s"]) == pytest.approx(sum(r["result"]["timings_s"]["plan"] for r in recs) / len(recs))
    assert (overall["latency_locate_s"], float(overall["status_parse_failed_rate"])) == ("", 0.0)   # locate never ran
    assert (float(overall["flag_heading_degenerate_rate"]), float(overall["hidden_relevant_people"])) == (1.0, 0.0)
    flags = {name: overall[name] for name in overall if name.startswith("flag_")}    # every flag is a rate: the six of the
    assert set(flags) == {f"flag_{name}_rate" for name in Flags.model_fields} and len(flags) == 11      # constraint method too
    assert {float(flags[f"flag_{name}_rate"]) for name in ("no_usable_constraint", "constraint_unusable", "constraints_cut",
                                                           "exclusion_broken", "people_cut", "far_from_target")} == {0.0}
    assert list(overall)[-9:-3] == METHOD_COLUMNS and [overall[name] for name in METHOD_COLUMNS] == [""] * 6    # of a baseline run the
    assert all(list(r["measures"])[-6:] == METHOD_COLUMNS for r in recs)             # method's measures are there, and measure nothing
    assert {r["measures"][name] for r in recs for name in METHOD_COLUMNS} == {None}
    shutil.copytree(placeholder / "examples", tmp_path / "repo" / "data" / "placeholder" / "examples")
    monkeypatch.setattr("sgi.run.REPO_ROOT", tmp_path / "repo")
    monkeypatch.chdir(tmp_path / "runs")                 # a relative examples_dir is read under the repository, not here
    default = config(placeholder, run={"examples_dir": "data/placeholder/examples"})
    again = harness.run(scenes, default, "relative", tmp_path / "runs")
    assert yaml.safe_load((again / "config.yaml").read_text())["example_ids"] == ids

def test_a_scene_check_warning_is_logged_and_does_not_stop_the_run(placeholder, tmp_path, monkeypatch, caplog):
    source = placeholder / "eval" / "eval_07_waiting_line_2"
    folder = tmp_path / "scenes" / source.name
    shutil.copytree(source, folder)
    grid = static_grid(load_scene(source)[1], 0.05)      # its bounds and obstacles as a map: free[0] is the row of the smallest y
    Image.fromarray(np.where(grid.free, 255, 0).astype(np.uint8)).save(folder / "occupancy.png")    # stored bottom row first:
    doc = json.loads((folder / "scene.json").read_text())                                           # the top row has to come first
    doc["ground_truth"].update(bounds=None, obstacles=[], occupancy={"image": "occupancy.png", "resolution": 0.05,
                                                                      "origin": [grid.x0, grid.y0]})
    (folder / "scene.json").write_text(json.dumps(doc))
    wire(monkeypatch, check=True)                        # the real scene check: no error here, and warning 11 alone
    with caplog.at_level("WARNING", logger="sgi.run"):   # says that every goal is about to be scored against a flipped map
        out = harness.run(tmp_path / "scenes", config(placeholder, run=ORACLE), "flipped", tmp_path / "runs")
    assert "eval_07_waiting_line_2: check 11 (free-space agreement): " in caplog.text and "flipped map?" in caplog.text
    assert (out / "summary.csv").exists() and len(records(out)) == 2

def test_two_trials_are_two_records_with_their_own_cache_entries(placeholder, tmp_path, monkeypatch):
    cfg = config(placeholder, run={"trials": 2, "use_cache": True, "cache_dir": str(tmp_path / "cache")})
    scenes, fresh = subset(tmp_path, placeholder / "eval"), wire(monkeypatch)
    recs = records(harness.run(scenes, cfg, "first", tmp_path / "runs"))
    assert [(r["scene_id"], r["episode_id"], r["trial"]) for r in recs] == [
        (s, e, t) for s, e, _ in episodes(scenes) for t in (0, 1)]
    calls = [c for r in recs for c in r["result"]["calls"]]
    assert len(fresh.calls) == len(calls) and not any(c["cached"] for c in calls)   # trial 1 never reuses trial 0
    replay = wire(monkeypatch)
    out = harness.run(scenes, cfg, "again", tmp_path / "runs")
    assert replay.calls == [] and all(c["cached"] for r in records(out) for c in r["result"]["calls"])
    assert [r["result"]["pose_robot"] for r in records(out)] == [r["result"]["pose_robot"] for r in recs]
    assert len(list((out / "viz").iterdir())) == len(episodes(scenes))              # one picture per episode
    assert {"success_rate", "success_rate_min", "success_rate_max"} <= set(summary(out)[0])

def test_a_run_counts_its_calls_and_what_they_cost_also_when_replayed_from_the_cache(placeholder, tmp_path, monkeypatch, capsys):
    def metered(parts, json_schema):                     # the centre answer, with the token counts a provider reports
        goal = "goal_point" in json_schema["properties"]     # one stage's provider reports no thinking tokens
        usage = Usage(input_tokens=3000, output_tokens=100) if goal else Usage(input_tokens=1000, output_tokens=200, reasoning_tokens=50)
        return BackendResult(json.dumps(centre_responder(parts, json_schema)), "ok", usage)
    per_call = {"goal": (3000 * 2.0 + 100 * 10.0) / 1e6, "other": (1000 * 2.0 + (200 + 50) * 10.0) / 1e6}
    priced = {"provider": "mock", "model": "mock-priced", "price_input_per_mtok": 2.0, "price_output_per_mtok": 10.0}
    cfg = config(placeholder, run={"use_cache": True, "cache_dir": str(tmp_path / "cache")}, models={"priced": priced})
    (tmp_path / "priced.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    scenes = subset(tmp_path, placeholder / "eval")
    monkeypatch.chdir(tmp_path)                          # main writes under ./runs
    def go(name, entry):                                 # what main prints in its brackets, the records, the summary
        wire(monkeypatch, metered)
        assert harness.main(["--scenes", str(scenes), "--config", "priced.yaml", "--model", entry, "--name", name]) == 0
        (out,) = (tmp_path / "runs").glob(f"*_{name}")
        return capsys.readouterr().out.strip().partition("; ")[2], records(out), summary(out)
    said, recs, table = go("first", "priced")
    stages = [c["stage"] for r in recs for c in r["result"]["calls"]]
    n, cost = len(stages), sum(per_call["goal" if stage == "goal" else "other"] for stage in stages)
    assert n > stages.count("goal") > 0 and said == f"{n} calls, 0 from the cache, about ${cost:.2f})"
    overall = table[0]
    assert (int(overall["calls"]), int(overall["cached_calls"]), float(overall["cost_usd"])) == (n, 0, pytest.approx(cost))
    assert list(overall)[-3:] == ["calls", "cached_calls", "cost_usd"]
    for r in recs:                                       # each record carries the cost of its own calls
        assert r["measures"]["cost_usd"] == pytest.approx(sum(
            per_call["goal" if c["stage"] == "goal" else "other"] for c in r["result"]["calls"]))
    groups = [row for row in table if row["group_by"] == "situation"]
    assert len(groups) == 3 and sum(int(row["calls"]) for row in groups) == n
    assert sum(float(row["cost_usd"]) for row in groups) == pytest.approx(cost)
    said, recs, table = go("again", "priced")            # every answer from the cache, at what it cost when it was made
    assert said == f"{n} calls, {n} from the cache, about ${cost:.2f})"
    assert (int(table[0]["calls"]), int(table[0]["cached_calls"]), float(table[0]["cost_usd"])) == (n, n, pytest.approx(cost))
    said, recs, table = go("unpriced", "mock")           # an entry without prices: no cost (one price alone is refused: test_config)
    assert said == f"{n} calls, 0 from the cache)" and {row["cost_usd"] for row in table} == {""}
    assert int(table[0]["calls"]) == n and {r["measures"]["cost_usd"] for r in recs} == {None}

def test_a_call_with_no_answer_is_counted_apart_and_marks_the_run_incomplete(placeholder, tmp_path, monkeypatch, caplog):
    backend, cfg = wire(monkeypatch, context_answers(outage=True)), config(placeholder, run=CONTEXT_ONLY)
    with caplog.at_level("INFO", logger="sgi.run"):
        out = harness.run(subset(tmp_path, placeholder / "eval"), cfg, "outage", tmp_path / "runs")
    recs = records(out)
    assert [r["result"]["status"] for r in recs] == ["vlm_error"] + ["ok"] * (len(recs) - 1)
    (failed,) = recs[0]["result"]["calls"]                # the context call: every attempt at it reached the back end
    assert (failed["stage"], failed["status"]) == ("context", "vlm_error")
    assert len(backend.calls) - len(recs) == failed["transient_retries"] > 0 and recs[0]["score"]["success"] is False
    assert {c["status"] for r in recs[1:] for c in r["result"]["calls"]} == {"ok"}
    assert all(r["score"]["success"] and r["measures"]["context_missed"] is False for r in recs[1:])
    assert recs[1]["measures"]["context_unnecessary"] is True     # nobody at this target, seen from the front
    overall = summary(out)[0]
    assert (int(overall["records"]), int(overall["vlm_errors"]), float(overall["success_rate"])) == (len(recs), 1, 1.0)
    assert float(overall["transient_retry_share"]) == pytest.approx(1 / len(recs))
    doc = yaml.safe_load((out / "config.yaml").read_text())
    assert (doc["incomplete"], doc["vlm_errors"]) == (True, 1) and "run incomplete: 1 record" in caplog.text

def test_rerun_after_an_outage_makes_only_the_missing_calls(placeholder, tmp_path, monkeypatch):
    cfg = config(placeholder, run={**CONTEXT_ONLY, "use_cache": True, "cache_dir": str(tmp_path / "cache")})
    scenes = subset(tmp_path, placeholder / "eval")
    wire(monkeypatch, context_answers(outage=True))
    first = harness.run(scenes, cfg, "outage", tmp_path / "runs")
    healthy = wire(monkeypatch, context_answers())
    second = harness.run(scenes, cfg, "resumed", tmp_path / "runs")
    recs = records(second)
    assert len(healthy.calls) == 1 and [r["result"]["status"] for r in recs] == ["ok"] * len(recs)
    assert [c["cached"] for r in recs for c in r["result"]["calls"]] == [False] + [True] * (len(recs) - 1)
    docs = [yaml.safe_load((out / "config.yaml").read_text()) for out in (first, second)]
    assert [(d["incomplete"], d["vlm_errors"]) for d in docs] == [(True, 1), (False, 0)]    # the first folder is left as it was

def test_context_check_answered_false_is_a_miss_where_the_standard_goal_fails(placeholder, tmp_path, monkeypatch):
    backend, scenes = wire(monkeypatch, context_answers(people_matter=False)), subset(tmp_path, placeholder / "eval")
    out = harness.run(scenes, config(placeholder, run=CONTEXT_ONLY), "unaware", tmp_path / "runs")
    # Of the three scenes the standard goal passes in one, the one with nobody about. It fails behind the one who stands
    # in line, and beside the member of a group of two that the request names: seen in profile, they are no face.
    recs, fails = records(out), [sit != "no_one" for _, _, sit in episodes(scenes)]
    assert len(backend.calls) == len(recs) and {r["result"]["source"] for r in recs} == {"standard"}
    assert [r["measures"]["context_missed"] for r in recs] == fails == [not r["score"]["success"] for r in recs]
    assert not any(r["measures"]["context_unnecessary"] for r in recs)
    missed = {row["group"]: float(row["context_missed_rate"]) for row in summary(out) if row["group_by"] in ("overall", "situation")}
    assert missed == {"all": pytest.approx(2 / 3), "no_one": 0.0, "waiting_line": 1.0, "conversation_group": 1.0}
