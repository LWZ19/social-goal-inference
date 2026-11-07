"""Independent verifier for I03 Stage B: my own implementation of the four arms of FROZEN.md section 7, solved with
the package's solver through research_harness.resolve, on my own re-implemented Stage A blobs (v_blobs.json).

Does NOT import the compute agent's replay.py / tables.py / posthoc.py. Imports: the package (sgi), the harness, and
phase1/i01_repairs/repairs.py ONLY for the placeholder runs' P (the note defines P there as "computed with the
unchanged repairs.solver_row").

Usage: v_replay.py old|new|ph -> v_rows_<wording>_<model>.jsonl. No model, no network.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402

HERE = Path(__file__).resolve().parent
RD = HERE.parents[2]
I01 = RD / "phase1" / "i01_repairs"

NEAR = 0.5
PX = 40.0
ADD_Z = 1.1
REL_Z = 1.0
EX_Z = [round(0.9 + 0.1 * k, 1) for k in range(9)]
assert EX_Z == [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
REAL = ("real", "staffed", "plain")
PH = ("ph", "dev_seed0", "dev_seed1", "dev_seed2")
PLAN = {
    "old": [("s19", "qwen38_low", REAL), ("s19", "glm53", REAL), ("s20", "dsv41", REAL)],
    "new": [("s21", "qwen38_low", REAL), ("s23", "glm53", REAL), ("s23", "dsv41", REAL), ("s23", "haiku55", REAL),
            ("s23", "sonnet55_low", REAL), ("s23", "gemini38_low", REAL)],
    "ph": [("s19", "qwen38_low", PH), ("s19", "glm53", PH), ("s20", "dsv41", PH)],
}


def load_blobs():
    v = json.loads((HERE / "v_blobs.json").read_text())
    out = {}
    for folder, e in v.items():
        for d, i in e["listed"]:
            out[(d, i)] = e["blobs"]
    return out


def bxy(b):
    return (b["x"], b["y"])


def seen_samples(obs, xy, tol):
    from sgi.geometry import in_image, project
    n = 0
    for z in EX_Z:
        pr = project(obs.camera, (xy[0], xy[1], z))
        if pr is None or not in_image(obs.camera, pr[0], pr[1]):
            continue
        d = float(obs.depth[int(math.floor(pr[1])), int(math.floor(pr[0]))])
        if math.isfinite(d) and d > 0 and d >= pr[2] - tol:
            n += 1
    return n


def make_added(obs, blobs, idxs, start):
    """DetectedPerson dicts for blob indices idxs (blob order), numbered start, start+1, ..."""
    from sgi.geometry import in_image, project
    out, noproj, off = [], 0, 0
    num = start
    for i in sorted(idxs):
        b = blobs[i]
        pr = project(obs.camera, (b["x"], b["y"], ADD_Z))
        if pr is None:
            noproj += 1
            continue
        if not in_image(obs.camera, pr[0], pr[1]):
            off += 1
        out.append({"number": num, "xy": [b["x"], b["y"]], "yaw": None, "pixel": [pr[0], pr[1]], "label": None})
        num += 1
    return out, noproj, off


def strip_numbers(cons, gone):
    return [{**c, "people": [n for n in (c.get("people") or []) if n not in gone]} for c in cons]


def run_one(rec, blobs, th, vis, obs):
    det = [dict(p) for p in rec.people]
    cons = [dict(u["constraint"]) for u in rec.constraints]
    tpt = (rec.target["world"][0], rec.target["world"][1])
    tblob = {i for i, b in enumerate(blobs) if math.dist(tpt, bxy(b)) <= NEAR}

    def supported(xy):
        return any(math.dist(xy, bxy(b)) <= NEAR for b in blobs)

    def unexplained(people):
        return [i for i, b in enumerate(blobs) if all(math.dist(p["xy"], bxy(b)) > NEAR for p in people)]

    start = max((p["number"] for p in det), default=0) + 1
    tol = th.visibility_depth_tolerance_m
    assert tol == 0.3

    def prune(people):
        kept, gone, exempt = [], [], []
        for p in people:
            if supported(p["xy"]):
                kept.append(p)
            elif seen_samples(obs, p["xy"], tol) < 3:
                exempt.append(p["number"])
                kept.append(p)
            else:
                gone.append(p["number"])
        return kept, gone, exempt

    arms = {}
    unexp0 = unexplained(det)
    add0, np0, off0 = make_added(obs, blobs, [i for i in unexp0 if i not in tblob], start)
    arms["Badd"] = dict(people=det + add0, cons=cons, added=add0, pruned=[], exempt=[], moved={}, noproj=np0, off=off0)
    kept, gone, exempt = prune(det)
    arms["Bprune"] = dict(people=kept + add0, cons=strip_numbers(cons, set(gone)), added=add0, pruned=gone,
                          exempt=exempt, moved={}, noproj=np0, off=off0)
    # relocate
    from sgi.geometry import project
    unsup = [p for p in det if not supported(p["xy"])]
    cands = []
    for p in unsup:
        for i in unexp0:
            pr = project(obs.camera, (blobs[i]["x"], blobs[i]["y"], REL_Z))
            if pr is None:
                continue
            dpx = math.hypot(pr[0] - p["pixel"][0], pr[1] - p["pixel"][1])
            if dpx <= PX:
                cands.append((dpx, p["number"], i))
    cands.sort()
    pairs, usedb = {}, set()
    for dpx, n, i in cands:
        if n in pairs or i in usedb:
            continue
        pairs[n] = i
        usedb.add(i)
    moved = [({**p, "xy": [blobs[pairs[p["number"]]]["x"], blobs[pairs[p["number"]]]["y"]]}
              if p["number"] in pairs else p) for p in det]
    still = unexplained(moved)
    unpaired = [i for i in unexp0 if i not in usedb]
    addr, npr, offr = make_added(obs, blobs, [i for i in still if i not in tblob], start)
    arms["Breloc"] = dict(people=moved + addr, cons=cons, added=addr, pruned=[], exempt=[], moved=pairs,
                          noproj=npr, off=offr)
    keptr, goner, exemptr = prune(moved)
    arms["BrelocP"] = dict(people=keptr + addr, cons=strip_numbers(cons, set(goner)), added=addr, pruned=goner,
                           exempt=exemptr, moved=pairs, noproj=npr, off=offr)
    # sensitivity: B-relocate with B-add on the unpaired blobs instead of the recomputed unexplained ones
    addu, _, _ = make_added(obs, blobs, [i for i in unpaired if i not in tblob], start)
    arms["Breloc_unpaired"] = dict(people=moved + addu, cons=cons, added=addu, pruned=[], exempt=[], moved=pairs,
                                   noproj=0, off=0)
    info = dict(n_det=len(det), n_blobs=len(blobs), n_tblob=len(tblob), n_unsup=len(unsup), n_unexp=len(unexp0),
                n_cands=len(cands), reloc_pairs=len(pairs), reloc_differs=sorted(still) != sorted(unpaired),
                reloc_moved_m=[round(math.dist(p["xy"], bxy(blobs[pairs[p["number"]]])), 4) for p in det
                               if p["number"] in pairs],
                cons_numbers=sorted({n for c in cons for n in (c.get("people") or [])}),
                det_numbers=sorted(p["number"] for p in det), add_start=start)
    return arms, info, cons, det


def matched(people, vis, th):
    from sgi.run import _matched_people
    from sgi.types import DetectedPerson
    return _matched_people([DetectedPerson.model_validate(p) for p in people], vis, th)


def line_class(rec, res, people, blobs, sc, th, vis):
    numbers = {p["number"]: p for p in people}
    waiting = set()
    for u in res.uses:
        if u.status == "used" and u.constraint.kind == "come_after":
            waiting |= {n for n in u.constraint.people if n in numbers}
    joined = set(res.note.joined_waiting) if res.note is not None else set()
    gt = sc.truths[rec.episode_id].gt
    tid = rec.score.get("matched_target_id") if rec.score.get("target_correct") else gt.target_ids[0]
    group = sc.truth.group_for(rec.episode_id, tid)
    vis_ids = {p.id for p in vis}
    members = [] if group is None else [p.id for p in sc.truth.group(group) if p.id in vis_ids]
    m = matched(people, vis, th)
    inline = waiting | joined
    out = [mid for mid in members if not any(t == mid and n in inline for n, t in m.items())]
    unsup = sorted(n for n in waiting if not any(math.dist(numbers[n]["xy"], bxy(b)) <= NEAR for b in blobs))
    if not waiting:
        c = 1
    elif unsup:
        c = 2
    elif out:
        c = 3
    else:
        c = 4
    return c, dict(waiting=sorted(waiting), joined=sorted(joined), members=len(members), out=len(out),
                   unsup=unsup)


def main(wording):
    from sgi.gt.derive import is_visible
    blobs_of = load_blobs()
    rep = None
    if wording == "ph":
        sys.path.insert(0, str(I01))
        import repairs as rep  # noqa: F811  (unchanged I01 code; P for placeholder runs only)
    t0 = time.monotonic()
    for stage, model, arms in PLAN[wording]:
        rows = []
        for arm in arms:
            found = H.runs(stage=stage, arm=arm, model=model, method="con")
            assert len(found) == 1, (stage, arm, model, found)
            ref = found[0]
            th = H.recorded_config(ref)[1].thresholds
            assert th.person_radius_m == 0.25
            recs = list(H.records(ref))
            for rec in recs:
                s = rec.score
                row = dict(wording=wording, model=model, run=ref.folder, arm=arm, scene=rec.scene_id,
                           episode=rec.episode_id, trial=rec.trial, situation=rec.situation,
                           solver_step=rec.solver_step, C_rec=bool(s.get("success")),
                           tc=bool(s.get("target_correct")), in_region=bool(s.get("in_region")),
                           reachable=bool(s.get("reachable")), has_people=rec.people is not None,
                           people_json=None if rec.people is None else json.dumps(rec.people, sort_keys=True))
                if not rec.solver_step:
                    rows.append(row)
                    continue
                sc = H.scene(ref, rec.scene_id)
                obs = sc.observation
                vis = [p for p in sc.truth.people if is_visible(p, obs, th)]
                blobs = blobs_of[(str(ref.scenes_dir), rec.scene_id)]
                c0 = H.resolve(rec)
                row.update(C_moved=c0.moved_m, C_repro=c0.reproduced, C_now=bool(c0.score.success))
                arms_d, info, cons, det = run_one(rec, blobs, th, vis, obs)
                row.update(info)
                m_rec = matched(det, vis, th)
                row["n_vis"] = len(vis)
                row["n_matched_rec"] = len(m_rec)
                rec_ids = set(m_rec.values())
                for name, a in arms_d.items():
                    res = H.resolve(rec, people=a["people"], constraints=a["cons"])
                    sco = res.score
                    m = matched(a["people"], vis, th)
                    addn = {p["number"] for p in a["added"]}
                    joined = list(res.note.joined_waiting) if res.note is not None else []
                    special = addn | set(a["moved"])
                    close = sum(1 for i in range(len(a["people"])) for j in range(i + 1, len(a["people"]))
                                if (a["people"][i]["number"] in special or a["people"][j]["number"] in special)
                                and math.dist(a["people"][i]["xy"], a["people"][j]["xy"]) < 0.25)
                    d = dict(ok=bool(sco.success), tc=bool(sco.target_correct), in_region=bool(sco.in_region),
                             reachable=bool(sco.reachable), heading_ok=bool(sco.heading_ok),
                             added=len(addn), added_missed=sum(n in m and m[n] not in rec_ids for n in addn),
                             added_reassigned=sum(n in m and m[n] in rec_ids for n in addn),
                             added_false=sum(n not in m for n in addn), pruned=len(a["pruned"]),
                             pruned_matched=sum(n in m_rec for n in a["pruned"]), exempt=len(a["exempt"]),
                             moved=len(a["moved"]), moved_was_matched=sum(n in m_rec for n in a["moved"]),
                             joined=joined, joined_added=sum(n in addn for n in joined), close=close,
                             noproj=a["noproj"], off=a["off"], uses=[u.status for u in res.uses])
                    if rec.situation == "waiting_line" and not sco.success:
                        d["cls"], d["cls_info"] = line_class(rec, res, a["people"], blobs, sc, th, vis)
                    if a["pruned"]:
                        d["pruned_numbers"] = a["pruned"]
                    row[name] = d
                if wording == "ph":
                    tmp = {"C_rec": row["C_rec"]}
                    rep.solver_row(rec, tmp)
                    row["P_ph"] = tmp["P"]
                    row["R3_ph"] = tmp["R3"]
                    row["R4_ph"] = tmp["R4"]
                rows.append(row)
            print(f"{wording} {ref.folder}: {len(recs)} records, "
                  f"{sum(r['solver_step'] for r in rows if r['run'] == ref.folder)} steps, "
                  f"{time.monotonic() - t0:.0f} s", flush=True)
        with open(HERE / f"v_rows_{wording}_{model}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    print("done", wording, f"{time.monotonic() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    for w in sys.argv[1:]:
        main(w)
