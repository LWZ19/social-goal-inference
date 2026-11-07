"""Independent verifier, pass 1: every signal of FROZEN.md section 4 and every check field, recomputed from the
recorded runs (through the step-0 harness), the frozen detector people, the package solver and A2's replay_d.py
(data source and the definition of setting D; the D people list is also rebuilt here by own code and compared).
Shares no code with code/confidence.py or code/tables.py.

    v_pass1.py GROUP      GROUP: oldwatch | s23      -> verify/vrows_<GROUP>.jsonl
Zero model or network calls. No .env. CPU only.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONF = HERE.parent
RD = CONF.parents[1]
assert RD.name == "2026-10-17-beyond-vlm", RD

# FROZEN.md hash first
_exp = (CONF / "FROZEN.sha256").read_text().split()[0]
_got = hashlib.sha256((CONF / "FROZEN.md").read_bytes()).hexdigest()
assert _exp == _got == "2a3a890a3b23ff466e8b72c63e6dbef9143d7738f68badc34a4fad6a17ccb4b3", (_exp, _got)

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RDD = _load("replay_d_v", RD / "phase1c" / "i09" / "a2" / "replay_d.py")
I14 = _load("i14_v", RD / "phase1" / "i14_v0" / "i14.py")
REP = sys.modules["repairs"]                       # loaded by replay_d (sys.path insert of phase1/i01_repairs)

from sgi.solver import target_person  # noqa: E402
from sgi.types import DetectedPerson, GroundedTarget  # noqa: E402

SHORT = {"qwen38_low": "qwen", "glm53": "glm", "dsv41": "deepseek", "haiku55": "haiku", "sonnet55_low": "sonnet",
         "gemini38_low": "gemini"}
S23 = ("glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low")
REAL = ("real", "staffed", "plain")


def plan(group):
    out = []
    if group == "oldwatch":
        for label, ref in I14.plan("oldwatch"):
            out.append((label, ref))
    elif group == "s23":
        for m in S23:
            for a in REAL:
                refs = H.runs(stage="s23", arm=a, model=m, method="con")
                assert len(refs) == 1, (m, a, refs)
                out.append(("new", refs[0]))
    else:
        raise SystemExit(group)
    return out


def layout_of(ref, scene_id):
    if ref.arm.startswith("dev_seed"):
        return f"{ref.arm}/{scene_id}"
    m = re.findall(r"eval_\d\d", scene_id)
    assert len(m) == 1, scene_id
    return m[0]


def my_match(vlm, dets, width=0.5):
    """Own one-to-one greedy: sorted by (distance, VLM number, detector index); gap <= width."""
    cand = []
    for p in vlm:
        for j, q in enumerate(dets):
            cand.append((math.hypot(p["xy"][0] - q["x"], p["xy"][1] - q["y"]), p["number"], j))
    cand.sort()
    used_v, used_d, out = set(), set(), {}
    for g, n, j in cand:
        if g <= width and n not in used_v and j not in used_d:
            out[n] = j
            used_v.add(n)
            used_d.add(j)
    return out


def wrapdeg(a):
    d = math.degrees(a) % 360.0
    return d - 360.0 if d > 180.0 else d


def main():
    group = sys.argv[1]
    t0 = time.monotonic()
    hashes = RDD.verify_hashes()
    det_of = RDD.load_detector_people()
    a2 = {}
    for wording in ("old", "new"):
        for m in SHORT:
            p = RD / "phase1c" / "i09" / "a2" / f"rows_{wording}_{m}.jsonl"
            if not p.exists():
                continue
            for line in p.open():
                r = json.loads(line)
                a2[(r["run"], r["scene"], r["episode"], r["trial"])] = r
    out = open(HERE / f"vrows_{group}.jsonl", "w")
    log = open(HERE / f"vpass1_{group}.log", "w")
    print(f"start {time.strftime('%F %T %z')} group {group} hashes {hashes}", file=log, flush=True)
    n_all = 0
    for label, ref in plan(group):
        th = H.recorded_config(ref)[1].thresholds
        assert th.person_radius_m == 0.25 and th.match_tolerance_m == 0.3, ref.folder
        allrecs = list(H.records(ref))
        # trial groups over every allowed record of the run
        groups = collections.defaultdict(list)
        for rec in allrecs:
            groups[(rec.scene_id, rec.episode_id)].append(rec)
        cellname = ("old_" if label == "old" else "watch_" if label == "watch" else "new_") + SHORT[ref.model]
        n = 0
        for rec in allrecs:
            if label == "watch" and rec.situation != "audience":
                continue
            if not rec.solver_step:
                continue
            key = f"{ref.folder}/{rec.scene_id}/{rec.episode_id}/{rec.trial}"
            row = dict(key=key, cell=cellname, label=label, run=ref.folder, scene=rec.scene_id, ep=rec.episode_id,
                       trial=rec.trial, arm=ref.arm, model=ref.model, layout=layout_of(ref, rec.scene_id),
                       situation=rec.situation, success=bool(rec.score["success"]),
                       target_correct=bool(rec.score.get("target_correct")),
                       flags={k: bool(rec.result["flags"][k]) for k in
                              ("exclusion_broken", "far_from_target", "constraint_unusable")})
            # R2: rescore
            row["rescore_success"] = bool(H.rescore(rec).success)
            # detector people
            folder, dets = det_of[(str(ref.scenes_dir), rec.scene_id)]
            row["det_folder"] = folder
            vlm = list(rec.people)
            nums = [p["number"] for p in vlm]
            assert len(set(nums)) == len(nums)
            cons = [u["constraint"] for u in rec.constraints]
            # S2
            pairs_mine = my_match(vlm, dets)
            det = [DetectedPerson.model_validate(p) for p in vlm]
            pairs_rd = RDD.match_xy(det, dets)
            row["match_same"] = pairs_mine == pairs_rd
            n_v, n_d, m = len(vlm), len(dets), len(pairs_mine)
            row["S2_parts"] = [n_v, n_d, m]
            # S3 (literal support: any detector within 0.5 m inclusive)
            named = sorted({x for c in cons for x in (c.get("people") or [])})
            byn = {p["number"]: p for p in vlm}
            unsup = []
            for x in named:
                p = byn.get(x)
                if p is None:
                    unsup.append(x)
                    continue
                if not any(math.hypot(p["xy"][0] - q["x"], p["xy"][1] - q["y"]) <= 0.5 for q in dets):
                    unsup.append(x)
            row["S3_parts"] = [len(unsup), len(named)]
            # S4 parts
            g = groups[(rec.scene_id, rec.episode_id)]
            row["S4_trials"] = sorted(r.trial for r in g)
            places = [r.pose_world for r in g]
            tgts = [None if r.target is None else r.target.get("world") for r in g]
            row["S4_missing_place"] = any(p is None for p in places)
            row["S4_missing_target"] = any(t is None for t in tgts)
            if not row["S4_missing_place"]:
                row["S4_pose_spread"] = max((math.dist(a[:2], b[:2]) for i, a in enumerate(places)
                                             for b in places[i + 1:]), default=0.0)
            if not row["S4_missing_target"]:
                row["S4_target_spread"] = max((math.dist(a[:2], b[:2]) for i, a in enumerate(tgts)
                                               for b in tgts[i + 1:]), default=0.0)
            # P1, realistic only
            if label in ("old", "new"):
                sc, th2, _vis, _ = REP.scene_info(rec)      # only observation and thresholds are used below
                obs = sc.observation
                cnums = [x for c in cons for x in (c.get("people") or [])]
                M = max(nums + cnums, default=0)
                people, out_cons, dropped, add_nums, info = RDD.build(det, cons, dets, pairs_rd, False, M, obs, th2)
                # own rebuild of D's people (numbers, xy, yaw) and constraints for comparison
                mine_people = []
                for p in vlm:
                    if p["number"] in pairs_mine:
                        q = dets[pairs_mine[p["number"]]]
                        mine_people.append((p["number"], (q["x"], q["y"]), q["yaw"] if q["yaw"] is not None else p["yaw"]))
                used = set(pairs_mine.values())
                k = M
                for j, q in enumerate(dets):
                    if j not in used:
                        k += 1
                        mine_people.append((k, (q["x"], q["y"]), q["yaw"]))
                drop = {p["number"] for p in vlm if p["number"] not in pairs_mine}
                mine_cons = [{**c, "people": [x for x in (c.get("people") or []) if x not in drop]} for c in cons]
                theirs = [(p.number, tuple(p.xy), p.yaw) for p in people]
                row["D_build_same"] = (theirs == [(a, tuple(b), c) for a, b, c in mine_people]) and (mine_cons == out_cons)
                try:
                    st0 = H.resolve(rec)
                    row["step0_moved_m"] = st0.moved_m
                    row["step0_success"] = bool(st0.score.success)
                except Exception as e:  # noqa: BLE001
                    row["step0_error"] = repr(e)
                try:
                    res = H.resolve(rec, people=[p.model_dump() for p in people], constraints=out_cons)
                    if res.pose_world is None:
                        row["D_error"] = "no pose_world"
                    else:
                        row["D_pose"] = list(res.pose_world)
                        row["D_success"] = bool(res.score.success)
                        row["P1_raw"] = math.dist(rec.pose_world[:2], res.pose_world[:2])
                        row["P1_from_step0"] = math.dist(st0.pose_world[:2], res.pose_world[:2])
                        row["D_heading_diff_deg"] = abs(wrapdeg(res.pose_world[2] - rec.pose_world[2]))
                except Exception as e:  # noqa: BLE001
                    row["D_error"] = repr(e) + traceback.format_exc(limit=2)
                gt_ = GroundedTarget.model_validate(rec.target)
                row["tnum_C"] = target_person(obs, gt_, det, th2)
                row["tnum_D"] = target_person(obs, gt_, people, th2)
                ar = a2.get((ref.folder, rec.scene_id, rec.episode_id, rec.trial))
                if ar is None:
                    row["a2"] = None
                else:
                    row["a2"] = dict(solver_step=ar.get("solver_step"), D=ar.get("D"),
                                     moved=(ar.get("D_d") or {}).get("moved_from_C_m"),
                                     tnum_C=ar.get("tnum_C"), tnum_D=(ar.get("D_d") or {}).get("tnum"),
                                     C_rec=ar.get("C_rec"))
            out.write(json.dumps(row) + "\n")
            n += 1
        out.flush()
        n_all += n
        print(f"{time.monotonic() - t0:7.1f} s {label} {ref.folder} goal steps {n}", file=log, flush=True)
    print(f"done {n_all} rows {time.monotonic() - t0:.1f} s", file=log, flush=True)


if __name__ == "__main__":
    main()
