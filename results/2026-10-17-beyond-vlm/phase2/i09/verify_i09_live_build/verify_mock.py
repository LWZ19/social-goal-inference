"""Verifier: check the mock run made through wrap_driver.py (no calls). Independent of the builder's dryrun.py.

    verify_mock.py TAG MOCK_OUT_DIR MODEL

- people calls: none in the captured requests and none in any record;
- each constraints request: its image equals my own rebuild from eval_people.json (score order, pixel = projection of
  (x, y, person_mark_height_m) as OraclePeople builds it, then the package's number_people / numbered_image /
  encode_png), its text holds my own people_lines string, and the record's people equal my numbered people;
- leakage, the rule of tests/test_leakage.py: no scene id, target id, situation name or forbidden-region kind is a
  token of any request's system text, schema or text parts; every request's own words do get through;
- writes: every write-mode open of the wrapped process lies in this verifier folder or the user's home caches.
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

V = Path(__file__).resolve().parent
RD = V.parents[2]
LIVE = V.parent / "live"
sys.path.insert(0, str(V / "archive_202c165" / "src"))
sys.path.insert(1, "docs/model_notes/tools")
import sgi  # noqa: E402
from sgi.geometry import project  # noqa: E402
from sgi.imaging import encode_png  # noqa: E402
from sgi.prompts import people_lines  # noqa: E402
from sgi.stages.constraints import numbered_image  # noqa: E402
from sgi.stages.people import FoundPerson, number_people  # noqa: E402
from sgi.types import GroundedTarget  # noqa: E402
import research_harness as H  # noqa: E402

TAG, OUT, MODEL = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
wrap = json.loads((V / f"wrap_{TAG}.json").read_text())
EP = json.loads((RD / "phase1c/i09/a1/eval_people.json").read_text())
by_id = {e["scene_id"]: e for e in EP["scenes"].values() if e["domain"] == "E-real"}
stage_of = {"glm53": "s19", "dsv41": "s20"}[MODEL]

captured = wrap["captured"]
kinds = collections.Counter()
for c in captured:
    title = (c["schema"].get("title") or "")
    kinds[title] += 1

folders = sorted(p for p in OUT.iterdir() if p.is_dir() and "_s24_i09_" in p.name)
records = []
for f in folders:
    arm = f.name.split("_s24_i09_")[1].split("_")[0]
    for line in (f / "results.jsonl").read_text().splitlines():
        records.append((arm, json.loads(line)))

people_calls_in_records = sum(1 for _, r in records for c in r["result"].get("calls") or [] if c["stage"] == "people")
con_requests = [c for c in captured if any(t.startswith("Person 1") or "\nPerson 1" in t for t in c["texts"])
                or "constraint" in json.dumps(c["schema"]).lower() and "stop_elsewhere" not in json.dumps(c["schema"])]
# Identify request kinds by schema property names instead (robust)
def props(c):
    return tuple(sorted(c["schema"].get("properties", {})))
by_props = collections.Counter(props(c) for c in captured)

# order: per record, its calls in order; the captured list is in the same order
calls_seq = [c["stage"] for _, r in records for c in r["result"]["calls"]]
assert len(calls_seq) == len(captured), (len(calls_seq), len(captured))
checks = []
secret, words_ok = set(), True
sent_tokens = set()
image_eq = lines_eq = people_eq = 0
n_con = 0
detail = []
k = 0
for arm, r in records:
    ref = H.runs(stage=stage_of, arm=arm, model=MODEL, method="con")[0]
    sc = H.scene(ref, r["scene_id"])
    th = H.recorded_config(ref)[1].thresholds
    secret |= {sc.inputs.scene_id, *sc.truth.targets}
    for et in sc.truths.values():
        secret |= {et.gt.situation} | {f.kind for f in et.gt.forbidden_regions or []}
    for call in r["result"]["calls"]:
        req = captured[k]
        k += 1
        text = "\n".join([req["system"] or "", json.dumps(req["schema"])] + req["texts"])
        toks = set(re.findall(r"[A-Za-z0-9_]+", text))
        sent_tokens |= toks
        if not set(re.findall(r"[A-Za-z0-9_]+", r["request"])) <= toks:
            words_ok = False
        if call["stage"] != "constraints":
            continue
        n_con += 1
        e = by_id[r["scene_id"]]
        obs = sc.observation
        found = []
        for p in sorted(e["people"], key=lambda p: -p["score"]):
            m = project(obs.camera, (p["x"], p["y"], th.person_mark_height_m))
            if m is not None:
                found.append(FoundPerson(xy=(float(p["x"]), float(p["y"])),
                                         yaw=None if p["yaw"] is None else float(p["yaw"]), pixel=m[:2], label=None))
        target = GroundedTarget.model_validate(r["result"]["target"])
        people, cut = number_people(found, target, obs, th, th.max_people)
        cfg = H.recorded_config(ref)[1]
        png = encode_png(numbered_image(obs, target, people), cfg.run.max_image_side)
        mine = hashlib.sha256(png.png).hexdigest()
        ie = req["images"] and req["images"][-1] == mine
        pl = people_lines(people, target, obs, th)
        le = any(pl in t for t in req["texts"]) if pl else True
        rec_people = r["result"].get("people") or []
        pe = [(q["number"], tuple(q["xy"]), q["yaw"], tuple(q["pixel"]), q["label"]) for q in rec_people] == \
             [(q.number, tuple(q.xy), q.yaw, tuple(q.pixel), q.label) for q in people]
        image_eq += bool(ie)
        lines_eq += bool(le)
        people_eq += bool(pe)
        detail.append({"arm": arm, "scene": r["scene_id"], "episode": r["episode_id"], "n_people": len(people),
                       "image_equal": bool(ie), "people_lines_in_request": bool(le), "record_people_equal": bool(pe),
                       "person_k_lines": sum(t.count("Person ") for t in req["texts"])})
assert k == len(captured)
leaked = sorted(secret & sent_tokens)
fields = {f for c in captured for f in c["schema"].get("properties", {})}
writes = wrap["audit"]["writes"]
home = str(Path.home())
outside = [w for w in writes if not (w.startswith(str(V)) or w.startswith(home + "/.cache/") or
                                     w.startswith(home + "/.config/matplotlib") or w.startswith("/tmp/")
                                     or "/__pycache__/" in w)]
res = {"tag": TAG, "exit": wrap["exit"], "records": len(records),
       "situations": dict(collections.Counter(r["labels"]["situation"] for _, r in records)),
       "requests": len(captured), "requests_by_stage": dict(collections.Counter(calls_seq)),
       "people_calls_in_records": people_calls_in_records,
       "constraints_requests": n_con, "image_equal": image_eq, "people_lines_in_request": lines_eq,
       "record_people_equal": people_eq, "secret_n": len(secret), "secret_sample": sorted(secret)[:12],
       "sent_distinct_tokens": len(sent_tokens), "leaked": leaked, "request_words_through": words_ok,
       "schema_fields_have_subtasks_stop_elsewhere": {"subtasks", "stop_elsewhere"} <= fields,
       "blocked": wrap["audit"]["blocked"], "writes_outside_allowed": outside,
       "pycache_writes": [w for w in writes if "/__pycache__/" in w][:5],
       "subprocess": wrap["audit"]["subprocess"], "detail": detail}
(V / f"verify_mock_{TAG}.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "detail"}, indent=1))
for d in detail:
    print(d)
