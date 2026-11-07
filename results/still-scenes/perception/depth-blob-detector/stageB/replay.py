"""I03 Stage B: solver replays of every recorded C goal step with the frozen Stage A blobs (FROZEN.md section 7,
CLARIFICATIONS.md). Zero calls; runs and scenes are read only through research_harness; the solver is the package's,
unchanged, through H.resolve exactly as phase1/i01_repairs/repairs.py runs it (its scene_info, strip and, for the
placeholder runs, solver_row are imported and used unchanged).

Arms (each re-solved with the recorded target; records that are not solver steps keep their recorded verdict):
  C        the recorded inputs (step 0: reproduced when the place moves <= 1e-6 m; verdict vs recorded).
  Badd     every unexplained blob that is not a target blob is added as a person (number max listed + 1, ... in
           blob order; xy = blob position; yaw None; pixel = projection of (x, y, 1.1 m); label None).
  Bprune   Badd, plus every unsupported, non-exempt detection dropped and its number stripped from every constraint.
  Breloc   unsupported detections paired one to one with unexplained blobs whose projection at 1.0 m lies within
           40 px of the detection's recorded pixel (greedy by pixel distance; ties lower number, then blob order); a
           paired detection moves to the blob (number, yaw, pixel, label kept); then Badd on the blobs still
           unexplained (recomputed against the relocated list).
  BrelocP  Breloc, then prune the detections still unsupported (non-exempt).
Terms: supported = a blob within 0.5 m; unexplained = no listed detection within 0.5 m; target blob = within 0.5 m of
the recorded target's world xy; exempt = fewer than 3 of the 9 axis samples (z = 0.9..1.7 m) project into the image
onto finite positive depth >= sample Z - 0.3 m.

Usage: replay.py step0|arms old|new|ph [MODEL ...] -> step0_<w>_<model>.jsonl or rows_<w>_<model>.jsonl here.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
I01 = paths.STILL / "replay-harness" / "repairs"
sys.path.insert(0, str(I01))
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402


def here(stored) -> str:
    """A scene folder as a stored file names it (the recorded absolute path without its local root), as it lies
    here: through the harness's relocated() and from the repository root."""
    return str((H.main_checkout() / H.relocated(stored)).resolve())
import repairs as R  # noqa: E402  (replay-harness/repairs/repairs.py, unchanged)

NEAR_M = 0.5
RELOC_PX = 40.0
CLOSE_M = 0.25
ADD_Z = 1.1
RELOC_Z = 1.0
EXEMPT_Z = (0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7)
EXEMPT_MIN = 3
EXEMPT_TOL = 0.3
REAL = ("real", "staffed", "plain")
PH = ("ph", "dev_seed0", "dev_seed1", "dev_seed2")
PLAN = {
    "old": [("s19", "qwen38_low", REAL), ("s19", "glm53", REAL), ("s20", "dsv41", REAL)],
    "new": [("s21", "qwen38_low", REAL), ("s23", "glm53", REAL), ("s23", "dsv41", REAL), ("s23", "haiku55", REAL),
            ("s23", "sonnet55_low", REAL), ("s23", "gemini38_low", REAL)],
    "ph": [("s19", "qwen38_low", PH), ("s19", "glm53", PH), ("s20", "dsv41", PH)],
}
ARMS = ("Badd", "Bprune", "Breloc", "BrelocP")


# ------------------------------------------------------------------------------------------------ inputs
def load_cache():
    cache = json.loads((HERE / "blob_cache.json").read_text())
    blobs_of = {}
    for folder, v in cache["scenes"].items():
        for scenes_dir, scene_id in v["listed"]:
            blobs_of[(here(scenes_dir), scene_id)] = v["blobs"]
    return cache["meta"], blobs_of


def i01_rows(wording, model):
    path = I01 / f"rows_{wording}_{model}.jsonl"
    if not path.exists():
        return None
    return {(r["arm"], r["scene"], r["episode"], r["trial"]): r for r in map(json.loads, path.open())}


def layout_of(scene_id: str) -> str:
    for prefix in ("real_", "staffed_", "plain_"):
        if scene_id.startswith(prefix):
            return scene_id[len(prefix):]
    return scene_id


def people_hash(people) -> str | None:
    if people is None:
        return None
    return hashlib.sha1(json.dumps(people, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ geometry helpers
def seen_count(obs, xy) -> int:
    from sgi.geometry import in_image, project
    cam, n = obs.camera, 0
    for z in EXEMPT_Z:
        p = project(cam, (xy[0], xy[1], z))
        if p is None or not in_image(cam, p[0], p[1]):
            continue
        d = float(obs.depth[math.floor(p[1]), math.floor(p[0])])
        if math.isfinite(d) and d > 0 and d >= p[2] - EXEMPT_TOL:
            n += 1
    return n


def near(a, b, d=NEAR_M) -> bool:
    return math.dist(a, b) <= d


def supported(xy, blobs) -> bool:
    return any(near(xy, (b["x"], b["y"])) for b in blobs)


def unexplained_idx(people, blobs) -> list[int]:
    return [i for i, b in enumerate(blobs) if not any(near(p.xy, (b["x"], b["y"])) for p in people)]


def added_people(people, blobs, idx, target_idx, obs):
    """Badd: DetectedPerson for every blob index in idx that is not a target blob, in blob order."""
    from sgi.geometry import in_image, project
    from sgi.types import DetectedPerson
    nxt = max((p.number for p in people), default=0) + 1
    out, no_proj, off_img = [], 0, 0
    for i in idx:
        if i in target_idx:
            continue
        b = blobs[i]
        pr = project(obs.camera, (b["x"], b["y"], ADD_Z))
        if pr is None:
            no_proj += 1
            continue
        off_img += not in_image(obs.camera, pr[0], pr[1])
        out.append(DetectedPerson(number=nxt, xy=(b["x"], b["y"]), yaw=None, pixel=(pr[0], pr[1]), label=None))
        nxt += 1
    return out, no_proj, off_img


def close_pairs(people, special: set[int]) -> int:
    n = 0
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            a, b = people[i], people[j]
            if (a.number in special or b.number in special) and math.dist(a.xy, b.xy) < CLOSE_M:
                n += 1
    return n


def prune(people, blobs, obs, keep: set[int]):
    """(kept people, pruned numbers, exempt unsupported numbers): drop unsupported detections that the depth can see;
    numbers in `keep` (added people) are never pruned."""
    kept, pruned, exempt = [], set(), set()
    for p in people:
        if p.number in keep or supported(p.xy, blobs):
            kept.append(p)
        elif seen_count(obs, p.xy) < EXEMPT_MIN:
            exempt.add(p.number)
            kept.append(p)
        else:
            pruned.add(p.number)
    return kept, pruned, exempt


def relocate(det, blobs, obs):
    """(relocated list, {number: blob index}, candidate count): Breloc's one-to-one pairing."""
    from sgi.geometry import project
    unsup = [d for d in det if not supported(d.xy, blobs)]
    unexp = unexplained_idx(det, blobs)
    cands = []
    for d in unsup:
        for i in unexp:
            b = blobs[i]
            pr = project(obs.camera, (b["x"], b["y"], RELOC_Z))
            if pr is None:
                continue
            dist = math.hypot(pr[0] - d.pixel[0], pr[1] - d.pixel[1])
            if dist <= RELOC_PX:
                cands.append((dist, d.number, i))
    cands.sort()
    used_d, used_b, pairs = set(), set(), {}
    for dist, n, i in cands:
        if n in used_d or i in used_b:
            continue
        used_d.add(n)
        used_b.add(i)
        pairs[n] = i
    moved = [d.model_copy(update={"xy": (blobs[pairs[d.number]]["x"], blobs[pairs[d.number]]["y"])})
             if d.number in pairs else d for d in det]
    return moved, pairs, len(cands), unexp


# ------------------------------------------------------------------------------------------------ one solver step
def score_fields(res) -> dict:
    s = res.score
    return {"success": bool(s.success), "tc": bool(s.target_correct), "in_region": bool(s.in_region),
            "reachable": bool(s.reachable), "heading_ok": bool(s.heading_ok)}


def line_class(rec, res, people, blobs, sc, th, vis):
    """CLARIFICATIONS.md point 18: why a line step still fails under an arm (first class that applies)."""
    from sgi.run import _matched_people
    from sgi.solver import target_person
    from sgi.types import GroundedTarget
    numbers = {p.number: p for p in people}
    tnum = target_person(sc.observation, GroundedTarget.model_validate(rec.target), people, th)
    waiting = set()
    for u in res.uses:
        if u.status == "used" and u.constraint.kind == "come_after":
            waiting |= {n for n in u.constraint.people if n in numbers and n != tnum}
    joined = set(res.note.joined_waiting) if res.note is not None else set()
    gt = sc.truths[rec.episode_id].gt
    tid = rec.score.get("matched_target_id") if rec.score.get("target_correct") else gt.target_ids[0]
    group = sc.truth.group_for(rec.episode_id, tid)
    vis_ids = {p.id for p in vis}
    members = [] if group is None else [p.id for p in sc.truth.group(group) if p.id in vis_ids]
    matched = _matched_people(people, vis, th)
    in_line = waiting | joined
    out_unmatched = [m for m in members if m not in matched.values()]
    out_not_waiting = [m for m in members if m in matched.values()
                       and not any(n in in_line for n, t in matched.items() if t == m)]
    info = {"waiting": sorted(waiting), "joined": sorted(joined), "members": len(members),
            "members_unmatched": len(out_unmatched), "members_matched_not_in_line": len(out_not_waiting),
            "waiting_unsupported": sorted(n for n in waiting if not supported(numbers[n].xy, blobs))}
    if not waiting:
        return "1 no come_after used", info
    if info["waiting_unsupported"]:
        return "2 a waiting person is unsupported", info
    if out_unmatched or out_not_waiting:
        return "3 a visible line member is not in the line", info
    return "4 line complete but still fails", info


def solver_step(rec, row, blobs):
    from sgi.run import _matched_people
    from sgi.types import DetectedPerson
    sc, th, vis, _ = R.scene_info(rec)
    assert th.visibility_depth_tolerance_m == EXEMPT_TOL and th.person_radius_m == 0.25
    obs = sc.observation
    det = [DetectedPerson.model_validate(p) for p in rec.people]
    cons = [u["constraint"] for u in rec.constraints]
    txy = tuple(rec.target["world"][:2])
    target_idx = {i for i, b in enumerate(blobs) if near(txy, (b["x"], b["y"]))}
    m_rec = _matched_people(det, vis, th)
    unsup = [d for d in det if not supported(d.xy, blobs)]
    unexp = unexplained_idx(det, blobs)
    row.update(n_det=len(det), n_vis=len(vis), n_matched=len(m_rec), n_blobs=len(blobs), n_target_blobs=len(target_idx),
               n_supported=len(det) - len(unsup), n_unsupported=len(unsup), n_unexplained=len(unexp),
               n_unexplained_nontarget=sum(i not in target_idx for i in unexp),
               n_unsupported_exempt=sum(seen_count(obs, d.xy) < EXEMPT_MIN for d in unsup),
               n_unmatched_det=len(det) - len(m_rec),
               n_unsupported_unmatched=sum(d.number not in m_rec for d in unsup),
               kinds=[c["kind"] for c in cons], has_come_after=any(c["kind"] == "come_after" for c in cons))

    variants = {}
    # Badd
    add, no_proj, off_img = added_people(det, blobs, unexp, target_idx, obs)
    variants["Badd"] = dict(people=det + add, cons=None, added={a.number for a in add}, moved=set(), pruned=set(),
                            exempt=set(), no_proj=no_proj, off_img=off_img)
    # Bprune
    kept, pruned, exempt = prune(det, blobs, obs, keep=set())
    variants["Bprune"] = dict(people=kept + add, cons=R.strip(cons, pruned)[0], added={a.number for a in add},
                              moved=set(), pruned=pruned, exempt=exempt, no_proj=no_proj, off_img=off_img)
    # Breloc
    moved, pairs, n_cands, unexp0 = relocate(det, blobs, obs)
    still = unexplained_idx(moved, blobs)
    unpaired = [i for i in unexp0 if i not in set(pairs.values())]
    add_r, no_proj_r, off_img_r = added_people(moved, blobs, still, target_idx, obs)
    variants["Breloc"] = dict(people=moved + add_r, cons=None, added={a.number for a in add_r}, moved=set(pairs),
                              pruned=set(), exempt=set(), no_proj=no_proj_r, off_img=off_img_r)
    kept_r, pruned_r, exempt_r = prune(moved, blobs, obs, keep=set())
    variants["BrelocP"] = dict(people=kept_r + add_r, cons=R.strip(cons, pruned_r)[0], added={a.number for a in add_r},
                               moved=set(pairs), pruned=pruned_r, exempt=exempt_r, no_proj=no_proj_r, off_img=off_img_r)
    row.update(reloc_candidates=n_cands, reloc_pairs=len(pairs),
               reloc_pairs_to_target_blob=sum(i in target_idx for i in pairs.values()),
               reloc_recompute_differs=sorted(still) != sorted(unpaired),
               reloc_moved_m=[round(math.dist(d.xy, (blobs[pairs[d.number]]["x"], blobs[pairs[d.number]]["y"])), 3)
                              for d in det if d.number in pairs],
               reloc_moved_was_matched=sum(d.number in m_rec for d in det if d.number in pairs))

    rr = H.resolve(rec)
    row.update(C=bool(rr.score.success), C_moved_m=rr.moved_m, C_reproduced=rr.reproduced,
               C_verdict_ok=bool(rr.score.success) == row["C_rec"])
    for arm, v in variants.items():
        res = H.resolve(rec, people=[p.model_dump() for p in v["people"]], constraints=v["cons"])
        f = score_fields(res)
        m = _matched_people(v["people"], vis, th)
        rec_ids = set(m_rec.values())
        d = {**f, "n_people": len(v["people"]), "added": len(v["added"]), "moved": len(v["moved"]),
             "pruned": len(v["pruned"]), "exempt": len(v["exempt"]), "no_proj": v["no_proj"],
             "added_off_image": v["off_img"], "close_pairs": close_pairs(v["people"], v["added"] | v["moved"]),
             "added_matched_missed": sum(n in m and m[n] not in rec_ids for n in v["added"]),
             "added_matched_reassigned": sum(n in m and m[n] in rec_ids for n in v["added"]),
             "added_false": sum(n not in m for n in v["added"]),
             "moved_matched": sum(n in m for n in v["moved"]),
             "pruned_were_matched": sum(n in m_rec for n in v["pruned"]),
             "n_matched": len(m), "joined": list(res.note.joined_waiting) if res.note is not None else [],
             "joined_added": sum(n in v["added"] for n in (res.note.joined_waiting if res.note is not None else [])),
             "uses": [u.status for u in res.uses], "moved_from_C_m": round(math.dist(res.pose_world[:2], rr.pose_world[:2]), 4)}
        if rec.situation == "waiting_line" and not f["success"]:
            d["class"], d["class_info"] = line_class(rec, res, v["people"], blobs, sc, th, vis)
        row[arm] = f["success"]
        row[arm + "_d"] = d
    # recorded C's own line class, for reference (recorded inputs)
    if rec.situation == "waiting_line" and not row["C_rec"]:
        row["C_class"], row["C_class_info"] = line_class(rec, rr, det, blobs, sc, th, vis)
    return row


# ------------------------------------------------------------------------------------------------ main
def base_row(wording, stage, model, ref, rec):
    s = rec.score
    return dict(wording=wording, stage=stage, arm=ref.arm, model=model, run=ref.folder, scene=rec.scene_id,
                layout=layout_of(rec.scene_id), episode=rec.episode_id, trial=rec.trial, situation=rec.situation,
                source=rec.result.get("source"), status=rec.result.get("status"), solver_step=rec.solver_step,
                C_rec=bool(s.get("success")), target_correct=bool(s.get("target_correct")),
                line_failure=bool(rec.situation == "waiting_line" and rec.solver_step and s.get("target_correct")
                                  and not s.get("success") and (not s.get("in_region") or not s.get("reachable"))),
                answer=f"{ref.folder}|{rec.scene_id}|{rec.trial}", people_hash=people_hash(rec.people),
                has_people=rec.people is not None)


def main(argv):
    mode, wording, only = argv[0], argv[1], set(argv[2:])
    assert mode in ("step0", "arms") and wording in PLAN
    meta, blobs_of = load_cache()
    t_all = time.monotonic()
    for stage, model, arms in PLAN[wording]:
        if only and model not in only:
            continue
        i01 = i01_rows(wording, model) if wording in ("old", "new") else None
        rows = []
        for arm in arms:
            ref = R.one_run(stage, arm, model)
            t0 = time.monotonic()
            recs = sorted(H.records(ref), key=lambda r: (r.scene_id, r.episode_id, r.trial))
            for rec in recs:
                row = base_row(wording, stage, model, ref, rec)
                blobs = blobs_of[(str(ref.scenes_dir), rec.scene_id)]
                if i01 is not None:
                    o = i01[(ref.arm, rec.scene_id, rec.episode_id, rec.trial)]
                    assert o["C_rec"] == row["C_rec"] and o["line_failure"] == row["line_failure"]
                    assert o["solver_step"] == row["solver_step"]
                    row.update(P_i01=o["P"], R4_i01=o["R4"], R3_i01=o["R3"], i01_source="phase1 rows")
                if mode == "step0":
                    if rec.solver_step:
                        rr = H.resolve(rec)
                        row.update(C=bool(rr.score.success), C_moved_m=rr.moved_m, C_reproduced=rr.reproduced,
                                   C_verdict_ok=bool(rr.score.success) == row["C_rec"])
                    rows.append(row)
                    continue
                if rec.solver_step:
                    if i01 is None:                    # placeholder runs: P with the unchanged repairs.solver_row
                        tmp = {"C_rec": row["C_rec"]}
                        R.solver_row(rec, tmp)
                        row.update(P_i01=tmp["P"], R4_i01=tmp["R4"], R3_i01=tmp["R3"], i01_source="repairs.solver_row")
                    solver_step(rec, row, blobs)
                else:
                    for k in ("C",) + ARMS:
                        row[k] = row["C_rec"]
                    if i01 is None:
                        row.update(P_i01=row["C_rec"], R4_i01=row["C_rec"], R3_i01=row["C_rec"], i01_source="kept")
                rows.append(row)
            st = [r for r in rows if r["arm"] == arm and r["solver_step"]]
            print(f"{mode} {wording} {ref.folder}: {len(recs)} records, {len(st)} solver steps, "
                  f"reproduced {sum(r.get('C_reproduced', False) for r in st)}, verdict ok "
                  f"{sum(r.get('C_verdict_ok', False) for r in st)}, {time.monotonic() - t0:.0f} s", flush=True)
        name = f"{'step0' if mode == 'step0' else 'rows'}_{wording}_{model}.jsonl"
        with open(HERE / name, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    print(f"done {mode} {wording} in {time.monotonic() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
