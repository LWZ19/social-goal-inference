"""Zero-call dry runs of driver.py (FROZEN.md section 8). No key is read, no real back end is built, nothing is sent.

    dryrun.py a      the mock model end to end on 10 episodes (lines, groups, nobody near; all three arms) with the
                     recorded entries: no people call; the constraints request carries the detector's numbered people
                     (rebuilt independently and compared byte for byte; two saved for viewing); the leakage rule of
                     tests/test_leakage.py on every request built; the run folders read back by the package and by the
                     harness (a whitelist of the builder's own).
    dryrun.py b      count-only against an empty cache of the builder's own, both models, all 84 t0 records each:
                     every call a miss, nothing sent, nothing written to the cache.
    dryrun.py b2     count-only against a cache filled by a fake gateway back end (provider openai_compatible, mock
                     answers, keys exactly the live ones) with its constraints entries removed: plan and context hit,
                     every constraints call counted as a miss; a fake live re-run of the full cache sends nothing.
    dryrun.py c      the hard stop at the cap with fake clients: per model, in total, across invocations sharing one
                     ledger, a validation retry counted as a call, caps above 120/60 refused, and run_arm stopping
                     with its folder marked incomplete.
Writes into dryrun/ beside this file; prints one PASS/FAIL line per check and writes dryrun/<part>_result.json.
"""
from __future__ import annotations

import collections
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import driver as D                                          # noqa: E402  (puts src_202c165 first on sys.path)

from pydantic import BaseModel                              # noqa: E402

from sgi.gt.derive import prepare                           # noqa: E402
from sgi.imaging import encode_png                          # noqa: E402
from sgi.prompts import people_lines                        # noqa: E402
from sgi.rescore import _recorded_config, _records          # noqa: E402
from sgi.scene import load_scene                            # noqa: E402
from sgi.stages.constraints import numbered_image           # noqa: E402
from sgi.stages.people import number_people                 # noqa: E402
from sgi.types import DetectedPerson, GroundedTarget, Usage # noqa: E402
from sgi.vlm.base import BackendResult, ImagePart, TextPart # noqa: E402
from sgi.vlm.mock import centre_responder                   # noqa: E402

H = D.H
OUT = HERE / "dryrun"
SCENES_A = ["real_eval_01_no_one_place", "real_eval_08_waiting_line_4", "real_eval_11_conversation_group_3",
            "plain_eval_09_waiting_line_6", "staffed_eval_12_conversation_group_4"]
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, evidence: str) -> bool:
    RESULTS.append((name, bool(ok), evidence))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {evidence}", flush=True)
    return bool(ok)


def setup() -> None:
    D._forbid_keys("dry run")
    D.sgi_run._given_inferrer = D._given_inferrer_detector


def write_result(part: str, extra: dict | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {"part": part, "time": D._now(), "driver_sha256": D._sha256(HERE / "driver.py"),
           "frozen_sha256": D._sha256(HERE / "FROZEN.md"),
           "checks": [{"check": n, "pass": ok, "evidence": e} for n, ok, e in RESULTS], **(extra or {})}
    (OUT / f"{part}_result.json").write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------------
# (a) mock end to end

def part_a() -> None:
    setup()
    out = OUT / "a_mock"
    out.mkdir(parents=True, exist_ok=True)
    table, tally, entries = D.detector_table(), D.Tally(), []
    for arm in D.ARMS:
        only = [s for s in SCENES_A if s.startswith(arm + "_")]
        entries.append(D.run_arm("glm53", arm, "mock", out, None, None, table, tally, only=only, capture=True))
    n_records = sum(e["records"] for e in entries)
    situations = collections.Counter()
    records = {}
    for e in entries:
        for rec in _records(Path(e["run_folder"]) / "results.jsonl"):
            records[(e["arm"], rec["scene_id"], rec["episode_id"])] = rec
            situations[rec["labels"]["situation"]] += 1
    check("a1 episodes across lines, groups and nobody near", n_records >= 6 and len(situations) == 3,
          f"{n_records} episodes in {len(entries)} arms; by situation {dict(situations)}")

    sent = tally.sent_requests
    by_stage = collections.Counter(stage for stage, *_ in sent)
    people_in_records = sum(c["stage"] == "people" for r in records.values() for c in r["result"]["calls"] or [])
    asked = sum(e["people_asked"] for e in entries)
    goal_steps = sum(any(c["stage"] == "constraints" for c in r["result"]["calls"] or []) for r in records.values())
    check("a2 the people stage makes no call", by_stage["people"] == 0 and people_in_records == 0 and asked == goal_steps > 0,
          f"requests by stage {dict(by_stage)}; people calls in records {people_in_records}; DetectorPeople asked "
          f"{asked} times for {goal_steps} goal steps")

    # The constraints request carries the detector's numbered people: rebuilt independently, compared byte for byte.
    rebuilt_ok, details, saved = 0, [], []
    for stage, where, system, parts, schema in sent:
        if stage != "constraints":
            continue
        arm, scene_id, episode = where["arm"], where["scene"], where["episode"]
        rec = records[(arm, scene_id, episode)]
        ref = D.recorded_run("glm53", arm)
        cfg = D.arm_config(ref, "mock", None)
        th = cfg.thresholds
        folder = H.check_scene(ref, scene_id)
        inputs, truth = load_scene(folder)
        frozen = table[D.fingerprint(folder)]
        source = D.DetectorPeople(scene_id, D.fingerprint(folder), inputs.observation.sha256(), frozen["people"], th)
        target = GroundedTarget.model_validate(rec["result"]["target"])
        people, cut = number_people(source(inputs.observation).found, target, inputs.observation, th, th.max_people)
        lines = people_lines(people, target, inputs.observation, th)
        image = encode_png(numbered_image(inputs.observation, target, people), cfg.run.max_image_side)
        texts = [p.text for p in parts if isinstance(p, TextPart)]
        images = [p for p in parts if isinstance(p, ImagePart)]
        recorded_people = [DetectedPerson.model_validate(p) for p in rec["result"]["people"]]
        same_people = [p.model_dump() for p in recorded_people] == [p.model_dump() for p in people]
        frozen_xy = sorted((round(p["x"], 9), round(p["y"], 9)) for p in frozen["people"])
        listed_xy = sorted((round(p.xy[0], 9), round(p.xy[1], 9)) for p in people)
        n_lines = len(re.findall(r"^Person \d+( \(the target\))?:", texts[0], flags=re.M)) if texts else -1
        ok = (len(texts) == 1 and len(images) == 1 and (lines in texts[0] if lines else "Person 1:" not in texts[0])
              and images[0].png == image.png and same_people and frozen_xy == listed_xy and n_lines == len(people)
              and not cut)
        rebuilt_ok += ok
        details.append(f"{scene_id}/{episode}: {len(people)} people, {n_lines} lines, image equal "
                       f"{images[0].png == image.png}, people equal {same_people}")
        if len(saved) < 2 and len(people) >= 3 and scene_id not in {s for s, _ in saved}:
            stem = OUT / f"a_numbered_{scene_id}_{episode}"
            stem.with_suffix(".png").write_bytes(images[0].png)
            stem.with_suffix(".txt").write_text(
                f"people lines (as sent):\n{lines}\n\n--- full text part of the constraints request ---\n{texts[0]}\n",
                encoding="utf-8")
            saved.append((scene_id, str(stem.with_suffix(".png"))))
    n_con = by_stage["constraints"]
    check("a3 the constraints request carries the detector's numbered people", n_con > 0 and rebuilt_ok == n_con,
          f"{rebuilt_ok} of {n_con} constraints requests: numbered image byte-equal to one rebuilt from the frozen "
          f"detector people, people lines present, one 'Person k' line per person, record people = detector "
          f"people; saved {[s for s, _ in saved]}")

    # Leakage: the rule of tests/test_leakage.py on every request built.
    secret = set()
    for e in entries:
        ref = D.recorded_run("glm53", e["arm"])
        cfg = D.arm_config(ref, "mock", None)
        for scene_id in e["scenes"]:
            inputs, truth = load_scene(H.check_scene(ref, scene_id))
            secret |= {inputs.scene_id, *truth.targets}
            for et in prepare(inputs, truth, cfg).values():
                secret |= {et.gt.situation} | {f.kind for f in et.gt.forbidden_regions or []}
    texts = []
    for stage, where, system, parts, schema in sent:
        texts += [system or "", json.dumps(schema)] + [p.text for p in parts if isinstance(p, TextPart)]
    tokens = set(re.findall(r"[A-Za-z0-9_]+", "\n".join(texts)))
    fields = {f for *_, schema in sent for f in schema.get("properties", {})}
    leaked = secret & tokens
    requests_through = all(set(re.findall(r"[A-Za-z0-9_]+", r["request"])) <= tokens for r in records.values())
    kinds = {k for k in secret if k not in {e for e in secret if e.startswith(("real_", "plain_", "staffed_"))}}
    check("a4 nothing sent holds a scene id, target id, situation or forbidden-region kind",
          not leaked and requests_through and {"subtasks", "stop_elsewhere"} <= fields
          and {"no_one", "waiting_line", "conversation_group"} <= secret,
          f"{len(sent)} requests, {len(tokens)} distinct tokens; {len(secret)} secret strings (e.g. "
          f"{sorted(kinds)[:8]}); leaked {sorted(leaked)}; every request's own words got through: {requests_through}; "
          f"schema fields include subtasks and stop_elsewhere: {({'subtasks', 'stop_elsewhere'} <= fields)}")

    # The run folders read back: by the package (config.yaml, results.jsonl) and by the harness with its own list.
    ok_pkg, ok_h, notes = True, True, []
    runs_file, scenes_file = OUT / "a_whitelist_runs.txt", OUT / "a_whitelist_scenes.txt"
    runs_lines, scene_lines = [], []
    for e in entries:
        folder = Path(e["run_folder"])
        doc, cfg = _recorded_config(folder / "config.yaml")
        ok_pkg &= (doc["people_source"]["tag"] == D.PEOPLE_SOURCE and cfg.run.trials == 1 and not doc["incomplete"]
                   and (folder / "summary.csv").is_file() and len(list((folder / "viz").glob("*.png"))) == e["records"]
                   and sorted(p.name for p in (folder / "prompts").iterdir()) ==
                   ["constraints.md", "context.md", "goal.md", "locate.md", "people.md", "plan.md"]
                   and all(r["config"]["people_source"] == D.PEOPLE_SOURCE for r in _records(folder / "results.jsonl")))
        runs_lines.append(f"{folder}={doc['scenes_dir']}")
        scene_lines += [f"{doc['scenes_dir']}/{s}" for s in e["scenes"]]
    runs_file.write_text("\n".join(runs_lines) + "\n", encoding="utf-8")
    scenes_file.write_text("\n".join(scene_lines) + "\n", encoding="utf-8")
    wl = H.whitelist(H.main_checkout(), runs_file, scenes_file)
    n_read = n_agree = n_solver = 0
    for ref in wl.runs:
        for rec in H.records(ref, wl=wl):
            n_read += 1
            n_solver += rec.solver_step
            n_agree += bool(H.rescore(rec).success) == bool(rec.score.get("success"))
        notes.append(f"{ref.folder}: harness reads stage {ref.stage}, arm {ref.arm}, model {ref.model}, method {ref.method}")
    ok_h = n_read == n_records and n_agree == n_read
    check("a5 the run folders read back by the package and by the harness", ok_pkg and ok_h,
          f"config.yaml, prompts/, results.jsonl, summary.csv, viz/ present and marked; harness reads {n_read} of "
          f"{n_records} records ({n_solver} solver steps), rescore agrees on {n_agree}; name parse: {notes[0]}")
    write_result("a", {"entries": entries, "saved_numbered": saved, "details": details})


# ----------------------------------------------------------------------------------------------------------------
# (b) count-only against an empty cache

def part_b() -> None:
    setup()
    out = OUT / "b_count_only_empty"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cache = Path(tempfile.mkdtemp(prefix="i09_empty_cache_", dir=str(OUT)))
    totals = {}
    for model in ("glm53", "dsv41"):
        code = D.main(["--model", model, "--trial", "0", "--out", str(out), "--cache-dir", str(cache), "--count-only"])
        report = json.loads(sorted(out.glob(f"count_only_{model}_*.json"))[-1].read_text())
        rows = [json.loads(l) for l in sorted(out.glob(f"count_only_{model}_*.jsonl"))[-1].read_text().splitlines()]
        n_rec = sum(a["records"] for a in report["arms"])
        hits = sum(r["outcome"] == "hit" for r in rows)
        misses = sum(r["outcome"] == "miss" for r in rows)
        sent = sum(r["outcome"] == "sent" for r in rows)
        totals[model] = {"records": n_rec, "misses": misses, "hits": hits, "sent": sent, "bounds": report["bounds"],
                         "code": code}
        check(f"b1 {model}: count-only on an empty cache counts every call as a miss and sends nothing",
              code == 0 and n_rec == 84 and hits == 0 and sent == 0 and misses == n_rec
              and report["bounds"]["misses_by_stage"] == {"plan": 84},
              f"{n_rec} records, {misses} misses (all plan: {report['bounds']['misses_by_stage']}), {hits} hits, "
              f"{sent} sent; bounds lower {report['bounds']['lower']}, upper {report['bounds']['upper']}; exit {code}")
    files = list(cache.rglob("*"))
    check("b2 nothing written to the cache and no run folder", not files and not [p for p in out.iterdir() if p.is_dir()],
          f"{len(files)} files in {cache}; outputs {sorted(p.name for p in out.iterdir())}")
    cache.rmdir()
    write_result("b", {"totals": totals})


# ----------------------------------------------------------------------------------------------------------------
# (b2) count-only against a cache filled by a fake gateway back end

class FakeGateway:
    """A back end that claims provider openai_compatible (so the cache keys are the live ones) and answers like the
    package's mock; it never leaves the process."""
    provider = "openai_compatible"

    def __init__(self, bad_first: int = 0):
        self.calls = 0
        self.bad_first = bad_first

    def call(self, entry, system, parts, json_schema):
        self.calls += 1
        if self.calls <= self.bad_first:
            return BackendResult("not json", "ok", Usage())
        return BackendResult(json.dumps(centre_responder(parts, json_schema)), "ok", Usage())


def part_b2() -> None:
    setup()
    out = OUT / "b2_count_only_filled"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cache = out / "cache"
    cache.mkdir()
    table = D.detector_table()
    only = ["real_eval_08_waiting_line_4", "real_eval_11_conversation_group_3", "real_eval_01_no_one_place"]
    fake = FakeGateway()
    tally = D.Tally()
    ledger = D.Ledger(out / "ledger.jsonl", "glm53", 120, 60)
    entry = D.run_arm("glm53", "real", "live", out, cache, ledger, table, tally, only=only,
                      backend_factory=lambda e: fake)
    keys = collections.defaultdict(set)
    for r in tally.rows:
        keys[r["stage"]].add(r["key"])
    sent_fill = sum(r["outcome"] == "sent" for r in tally.rows)
    check("b2a a fake gateway fill sends through the ledger only", fake.calls == sent_fill == ledger.total,
          f"fake back end called {fake.calls} times; {sent_fill} requests tallied as sent; ledger {ledger.total}; "
          f"by stage {tally.counts()}")
    for key in keys["constraints"]:
        (cache / f"{key}.json").unlink()
    t2 = D.Tally()
    e2 = D.run_arm("glm53", "real", "count-only", out, cache, None, table, t2, only=only)
    c2 = t2.counts()
    b = D.count_bounds(t2.rows)
    miss_keys = {r["key"] for r in t2.rows if r["outcome"] == "miss"}
    check("b2b count-only: plan and context hit, every constraints call a miss, nothing sent",
          b["upstream_misses"] == 0 and miss_keys == keys["constraints"] and fake.calls == sent_fill
          and c2.get("constraints", {}).get("miss") == len(keys["constraints"]) == entry["people_asked"],
          f"count-only tallies {c2}; misses = the removed constraints keys: {miss_keys == keys['constraints']} "
          f"({len(miss_keys)}); goal steps of the fill {entry['people_asked']}; fake back end still at {fake.calls}")
    # Restore the constraints answers by a fake live run, then a full re-run must send nothing (all hits).
    t3 = D.Tally()
    D.run_arm("glm53", "real", "live", out, cache, ledger, table, t3, only=only, backend_factory=lambda e: fake)
    t4 = D.Tally()
    before = fake.calls
    e4 = D.run_arm("glm53", "real", "live", out, cache, ledger, table, t4, only=only, backend_factory=lambda e: fake)
    check("b2c a re-run on a full cache sends nothing and the ledger counts no hit",
          fake.calls == before and all(r["outcome"] == "hit" for r in t4.rows)
          and all(u["all_equal"] is not None for u in e4["upstream"]),
          f"re-run tallies {t4.counts()}; fake back end calls before {before}, after {fake.calls}; ledger {ledger.total}")
    write_result("b2", {"fill": entry["cache"], "count_only": c2, "bounds": b})


# ----------------------------------------------------------------------------------------------------------------
# (c) the hard stop

class Answer(BaseModel):
    n: int


class Counting:
    provider = "openai_compatible"

    def __init__(self, bad_first: int = 0):
        self.calls = 0
        self.bad_first = bad_first

    def call(self, entry, system, parts, json_schema):
        self.calls += 1
        if self.calls <= self.bad_first:
            return BackendResult("{}", "ok", Usage())          # fails validation: the client asks once more
        return BackendResult(json.dumps({"n": self.calls}), "ok", Usage())


def _client(ledger, backend, cache_dir, model_entry, where, tally):
    from sgi.vlm.cache import Cache
    return D.DriverClient(model_entry, D.GuardedBackend(backend, ledger, "constraints", where), Cache(cache_dir),
                          stage="constraints", mode="live", where=where, tally=tally)


def part_c() -> None:
    setup()
    out = OUT / "c_cap"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    from sgi.config import ModelEntry
    entry = ModelEntry(provider="openai_compatible", model="fake", base_url="http://127.0.0.1:9/", api_key_env="NONE")
    where, tally = D.Where(arm="real", scene="s", episode="e0"), D.Tally()

    # c1: lowered caps, per model then in total, across two invocations sharing one ledger.
    path = out / "c1_ledger.jsonl"
    backend = Counting()
    led_a = D.Ledger(path, "glm53", 5, 3)
    cl = _client(led_a, backend, out / "c1_cache", entry, where, tally)
    got = [cl.generate([TextPart(f"q{i}")], Answer).status for i in range(3)]
    try:
        cl.generate([TextPart("q3")], Answer)
        refused_a = False
    except D.CapReached:
        refused_a = True
    calls_after_a = backend.calls
    hit = cl.generate([TextPart("q0")], Answer)               # a cache hit costs nothing, even at the cap
    led_b = D.Ledger(path, "dsv41", 5, 3)                     # a second invocation reads the first one's sends
    cl_b = _client(led_b, backend, out / "c1_cache", entry, where, tally)
    got_b = [cl_b.generate([TextPart(f"r{i}")], Answer).status for i in range(2)]
    try:
        cl_b.generate([TextPart("r2")], Answer)
        refused_b = False
    except D.CapReached:
        refused_b = True
    sent_lines = [json.loads(l) for l in path.read_text().splitlines() if '"sent"' in l]
    check("c1 per-model and total caps hold across invocations sharing a ledger",
          got == ["ok"] * 3 and refused_a and calls_after_a == 3 and hit.cached and led_b.at_start == (3, 0)
          and got_b == ["ok"] * 2 and refused_b and backend.calls == 5 and len(sent_lines) == 5,
          f"caps 5/3: glm53 sent 3, 4th refused before the back end ({calls_after_a} calls); a hit at the cap answered "
          f"from the cache; dsv41 started at {led_b.at_start}, sent 2, 3rd refused (total cap); back end calls "
          f"{backend.calls}; ledger 'sent' lines {len(sent_lines)}")

    # c2: a validation retry is a second request and is counted.
    path2 = out / "c2_ledger.jsonl"
    b2 = Counting(bad_first=1)
    led = D.Ledger(path2, "glm53", 120, 60)
    resp = _client(led, b2, out / "c2_cache", entry, where, tally).generate([TextPart("v")], Answer)
    check("c2 a validation retry counts as a call", resp.status == "ok" and resp.validation_retries == 1
          and led.total == 2 == b2.calls, f"one generate, {b2.calls} back-end requests, ledger {led.total}")

    path3 = out / "c3_ledger.jsonl"
    b3 = Counting()
    led_g = D.Ledger(path3, "glm53", 120, 60)
    cg = _client(led_g, b3, out / "c3_cache", entry, where, tally)
    n_g = 0
    try:
        for i in range(100):
            cg.generate([TextPart(f"g{i}")], Answer)
            n_g += 1
    except D.CapReached:
        pass
    led_d = D.Ledger(path3, "dsv41", 120, 60)
    cd = _client(led_d, b3, out / "c3_cache", entry, where, tally)
    n_d = 0
    try:
        for i in range(100):
            cd.generate([TextPart(f"d{i}")], Answer)
            n_d += 1
    except D.CapReached:
        pass
    led_x = D.Ledger(path3, "glm53", 120, 60)
    try:
        _client(led_x, b3, out / "c3_cache", entry, where, tally).generate([TextPart("x")], Answer)
        refused_x = False
    except D.CapReached:
        refused_x = True
    check("c3 the 61st call of a model and the 121st in all are refused before they are sent",
          n_g == 60 and n_d == 60 and refused_x and b3.calls == 120,
          f"glm53 sent {n_g}, dsv41 sent {n_d}, a further glm53 request refused: {refused_x}; back end calls {b3.calls}")

    refused = []
    for total, per in ((121, 60), (120, 61)):
        try:
            D.Ledger(out / "c4_ledger.jsonl", "glm53", total, per)
            refused.append(False)
        except SystemExit:
            refused.append(True)
    try:
        D.main(["--model", "glm53", "--out", str(out), "--cache-dir", str(out), "--count-only", "--max-calls", "121"])
        refused.append(False)
    except SystemExit:
        refused.append(True)
    check("c4 caps above 120 in total or 60 per model are refused", all(refused), f"refused: {refused}")

    # c5: run_arm stops at the cap and keeps what it finished, marked incomplete.
    table = D.detector_table()
    fake = FakeGateway()
    led5 = D.Ledger(out / "c5_ledger.jsonl", "glm53", 5, 5)
    t5 = D.Tally()
    try:
        D.run_arm("glm53", "real", "live", out, out / "c5_cache", led5, table, t5,
                  only=["real_eval_08_waiting_line_4", "real_eval_11_conversation_group_3"],
                  backend_factory=lambda e: fake)
        stopped = False
    except D.CapReached:
        stopped = True
    folder = sorted(out.glob("*_s24_i09_real_glm53_con"))[-1]
    doc = json.loads(json.dumps(__import__("yaml").safe_load((folder / "config.yaml").read_text())))
    n_lines = len((folder / "results.jsonl").read_text().splitlines())
    check("c5 run_arm stops at the cap, keeps its records and marks the folder incomplete",
          stopped and fake.calls == 5 and doc["incomplete"] is True and "CapReached" in doc.get("stopped_by", "")
          and (folder / "summary.csv").is_file() and n_lines >= 1,
          f"stopped {stopped}; fake back end calls {fake.calls}; {n_lines} records kept; incomplete "
          f"{doc['incomplete']}; stopped_by {doc.get('stopped_by')!r}")

    # c6: the command's own live path (lock, ledger, exit codes, check file), with the real back end replaced by the
    # fake gateway. --env-file names a file that does not exist and is never opened: the fake needs no key.
    fake6 = FakeGateway()
    D.sgi_factory._backend = lambda entry: fake6
    out6, out7 = out / "c6_main", out / "c6_main_full"
    for folder in (out6, out7):
        (folder / "cache").mkdir(parents=True)

    def args(folder, *extra):
        return ["--trial", "0", "--out", str(folder), "--cache-dir", str(folder / "cache"), "--arms", "real",
                "--env-file", str(folder / "no_key_file_here"), *extra,
                "--only", "real_eval_08_waiting_line_4", "real_eval_11_conversation_group_3"]
    code_g = D.main(["--model", "glm53", *args(out6, "--max-calls", "5", "--max-calls-per-model", "5")])
    calls_g = fake6.calls
    code_d = D.main(["--model", "dsv41", *args(out6, "--max-calls", "5", "--max-calls-per-model", "5")])
    rows6 = [json.loads(l) for l in (out6 / "ledger.jsonl").read_text().splitlines()]
    reports = [json.loads(p.read_text()) for p in sorted(out6.glob("check_*_live_*.json"))]
    code_ok = D.main(["--model", "glm53", *args(out7)])
    sent7 = [json.loads(l) for l in (out7 / "ledger.jsonl").read_text().splitlines() if '"sent"' in l]
    rep7 = json.loads(sorted(out7.glob("check_glm53_live_*.json"))[-1].read_text())
    n_sent6 = sum(r["event"] == "sent" for r in rows6)
    n_refused6 = sum(r["event"] == "refused" for r in rows6)
    check("c6 the command's live path: cap stop exit 3 across invocations, lock released, full pass exit 0",
          code_g == 3 and calls_g == 5 and code_d == 3 and n_sent6 == 5 and n_refused6 == 2
          and not (out6 / "ledger.lock").exists() and len(reports) == 2 and all(r["stopped_by"] for r in reports)
          and code_ok == 0 and len(sent7) == rep7["ledger_at_end"]["total"] == fake6.calls - 5
          and not (out7 / "ledger.lock").exists(),
          f"glm53 caps 5/5: exit {code_g} after {calls_g} sends; dsv41 on the same ledger: exit {code_d}, "
          f"{sum(r['event'] == 'sent' and r['model'] == 'dsv41' for r in rows6)} dsv41 sends; ledger sent {n_sent6}, "
          f"refused {n_refused6}; "
          f"{len(reports)} check files with stopped_by; full pass exit {code_ok}, {len(sent7)} sends = ledger "
          f"{rep7['ledger_at_end']['total']}; locks released")
    write_result("c")


if __name__ == "__main__":
    parts = {"a": part_a, "b": part_b, "b2": part_b2, "c": part_c}
    started, every = time.time(), []
    for name in sys.argv[1:] or ["c", "b2", "a", "b"]:
        RESULTS.clear()
        parts[name]()
        every += RESULTS
    print(f"{time.time() - started:.1f} s; {sum(ok for _, ok, _ in every)} of {len(every)} checks pass")
    sys.exit(0 if all(ok for _, ok, _ in every) else 1)
