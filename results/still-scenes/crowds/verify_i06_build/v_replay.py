"""Independent verifier, I06 build: replay every planned variant of one seed from its own stream with my own
pre-checks (vlib.py), render each attempt through the package's scene path, decide acceptance with the package's
generate._rejection, and compare with the build's manifest and scene folders.

    v_replay.py SEED

Per variant: the replayed outcome of each attempt (no_spot / the scene check's reason / accepted) against the
manifest's reasons; for kept variants the replayed people against scene.json (ids, x, y, yaw, group, role) and the
shirts against the manifest, and the sha256 of rgb.png, depth.npy and scene.json of my re-rendered scene (regions
filled by generate._with_regions) against the build's files; then the family-definition geometry and checks c1-c8 on
the stored people against MY reproduced base. Rows go to out/replay_seed<SEED>.jsonl. Re-rendered folders are written
under work/ and removed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from sgi.config import load_config
from sgi.geometry import wrap_angle
from sgi.scene import load_scene, save_scene
from sgi.synth import generate as G

import vlib as V

BUILD = V.I06 / "build"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def variant_layout(layout, pairs):
    gt = layout.scene_gt
    return dataclasses.replace(layout, figures=(*layout.figures, *(f for _, f in pairs)),
                               scene_gt=gt.model_copy(update={"people": [*gt.people, *(p for p, _ in pairs)]}))


def geometry(b, fam, lv, truth, inputs, info):
    """Family-definition measures from the STORED scene against my reproduced base."""
    g = {}
    ppl = list(truth.people)
    n0 = len(b.people)
    g["base_people_unchanged"] = [(p.id, p.x, p.y, p.yaw, p.group, p.role) for p in ppl[:n0]] == \
                                 [(p.id, p.x, p.y, p.yaw, p.group, p.role) for p in b.people]
    added = ppl[n0:]
    g["n_added"] = len(added)
    g["n_people"] = len(ppl)
    g["ids_continue"] = [p.id for p in added] == [f"p{n0 + i}" for i in range(len(added))]
    g["roles_none"] = all(p.role is None for p in added)
    g["non_members"] = all(p.group != b.gid for p in added)
    if fam == "C":
        gid = added[0].group if added else None
        g["groups_ok"] = (all(p.group == gid for p in added) and gid not in b.groups and gid is not None
                          and gid == next(f"g{k}" for k in range(1, 100) if f"g{k}" not in b.groups))
    else:
        g["groups_ok"] = all(p.group is None for p in added)
    T = np.asarray(inputs.observation.camera.T_world_camera)
    g["camera_same"] = bool(np.array_equal(T, np.asarray(b.inputs.observation.camera.T_world_camera)))
    g["robot_same"] = inputs.observation.robot == b.robot
    g["targets_same"] = all(truth.targets[t].model_dump(exclude={"tour_frames"}) ==
                            b.truth.targets[t].model_dump(exclude={"tour_frames"}) for t in b.truth.targets) and \
        set(truth.targets) == set(b.truth.targets)
    g["bounds_same"] = list(map(tuple, truth.bounds)) == list(map(tuple, b.truth.bounds))
    g["obstacles_same"] = [list(map(tuple, o)) for o in truth.obstacles] == [list(map(tuple, o)) for o in b.truth.obstacles]
    # regions: polygons and headings equal the base's (L1 invariance), episodes' situations/group unchanged
    def regs(t, eid):
        return [(list(map(tuple, r.polygon)), r.heading.model_dump()) for r in t.episodes[eid].acceptable_regions or []]
    g["l1_invariant"] = all(regs(truth, e) == regs(b.truth, e) for e in b.truth.episodes) and \
        set(truth.episodes) == set(b.truth.episodes)
    g["group_at_target_same"] = all(truth.episodes[e].group_at_target == b.truth.episodes[e].group_at_target
                                    and truth.episodes[e].situation == b.truth.episodes[e].situation
                                    for e in b.truth.episodes)
    g["situations"] = sorted({ep.situation for ep in truth.episodes.values()})
    g["rep_moved_m"] = math.dist(truth.episodes["e0"].acceptable_regions[0].representative_point, b.rep)
    # nearest distances
    g["nearest_member_m"] = [min(math.hypot(p.x - m.x, p.y - m.y) for m in b.members) for p in added]
    g["nearest_other_m"] = [min(math.hypot(p.x - o.x, p.y - o.y) for o in ppl if o is not p) for p in added]
    # checks on the final configuration (others = everyone else in the variant)
    g["check_fails"] = {p.id: b.checks(fam, p, ppl) for p in added}
    if fam == "A":
        gap, phi = V.A_GAP_PHI[lv]
        line = sorted(b.members, key=lambda m: math.hypot(m.x - b.anchor[0], m.y - b.anchor[1]))
        L = np.array([line[-1].x, line[-1].y])
        B = np.array([line[-2].x, line[-2].y]) if len(line) > 1 else np.array(b.anchor)
        u = (L - B) / np.linalg.norm(L - B)
        face = math.atan2(-u[1], -u[0])
        nom = L + gap * u
        p = added[0]
        g["jitter_xy_max"] = max(abs(p.x - nom[0]), abs(p.y - nom[1]))
        g["yaw_dev"] = min(abs(wrap_angle(p.yaw - (face + s * math.radians(phi)))) for s in (1, -1))
        g["gap_along_axis_m"] = float(np.dot(np.array([p.x, p.y]) - L, u))
        g["facing_off_line_deg"] = math.degrees(abs(wrap_angle(p.yaw - face)))
    elif fam in ("B", "C"):
        d = [math.hypot(a.x - c.x, a.y - c.y) for i, a in enumerate(added) for c in added[i + 1:]]
        g["pair_m"] = d
        g["nearest_pair_to_member_m"] = min(g["nearest_member_m"])
        g["spacing_nonmembers_min_m"] = min(math.hypot(p.x - o.x, p.y - o.y) for p in added for o in b.nonmembers) \
            if b.nonmembers else None
        if info:
            cx, cy = info["centre"]
            turns = [info["psi"] + 2 * math.pi * k / 3 for k in range(3)]
            noms = [(cx + V.TRI_R * math.cos(t), cy + V.TRI_R * math.sin(t)) for t in turns]
            g["jitter_xy_max"] = max(max(abs(p.x - n[0]), abs(p.y - n[1])) for p, n in zip(added, noms))
            gap = 1.2 if fam == "B" else 1.5
            near = min(math.hypot(n[0] - m.x, n[1] - m.y) for n in noms for m in b.members)
            g["nominal_nearest_pair_m"] = near
            g["nominal_gap_err"] = abs(near - gap)
            # outermost root: farther along theta the cluster stays at or past the gap
            th = info["theta"]
            mx0 = np.mean([m.x for m in b.members]); my0 = np.mean([m.y for m in b.members])
            ok = True
            for dd in np.arange(info["d"] + 0.01, 30.0, 0.01):
                ccx, ccy = mx0 + dd * math.cos(th), my0 + dd * math.sin(th)
                if min(math.hypot(ccx + V.TRI_R * math.cos(t) - m.x, ccy + V.TRI_R * math.sin(t) - m.y)
                       for t in turns for m in b.members) < gap - 1e-9:
                    ok = False
                    break
            g["outermost_root_ok"] = ok
            g["on_bearing_err"] = abs(wrap_angle(math.atan2(cy - my0, cx - mx0) - th)) if info["d"] > 1e-9 else 0.0
            if fam == "C":
                g["yaw_dev_face_centre"] = max(abs(wrap_angle(p.yaw - (t + math.pi))) for p, t in zip(added, turns))
                g["radius_nominal"] = V.TRI_R
        # facing centre measured on the stored people alone (jittered centroid)
        if fam == "C":
            mxc = np.mean([p.x for p in added]); myc = np.mean([p.y for p in added])
            g["facing_centroid_dev_max"] = max(abs(wrap_angle(p.yaw - math.atan2(myc - p.y, mxc - p.x))) for p in added)
            g["radius_stored"] = [math.hypot(p.x - mxc, p.y - myc) for p in added]
    elif fam == "D":
        g["pairwise_added_min_m"] = min((math.hypot(a.x - c.x, a.y - c.y) for i, a in enumerate(added)
                                         for c in added[i + 1:]), default=None)
        g["to_base_min_m"] = min(math.hypot(p.x - o.x, p.y - o.y) for p in added for o in b.people)
    elif fam == "E":
        cam, rep = np.array(b.cam), np.array(b.rep)
        v = (rep - cam) / np.linalg.norm(rep - cam)
        nom = (rep - 1.5 * v) if lv == 1 else (cam + 2.0 * v)
        p = added[0]
        g["jitter_xy_max"] = max(abs(p.x - nom[0]), abs(p.y - nom[1]))
        w = np.array([p.x, p.y]) - cam
        g["perp_off_line_m"] = float(abs(w[0] * v[1] - w[1] * v[0]))
        g["along_from_cam_m"] = float(np.dot(w, v))
        g["cam_to_rep_m"] = float(np.linalg.norm(rep - cam))
    return g


def main(seed: int) -> None:
    cfg = load_config()
    bases = V.load_bases(seed)
    entries = {}
    for line in (BUILD / "manifest.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["kind"] == "variant" and e["seed"] == seed:
            entries[(e["row"], e["family"], e["level"])] = e
    out = V.HERE / "out" / f"replay_seed{seed}.jsonl"
    work = Path(tempfile.mkdtemp(prefix=f"replay{seed}_", dir=V.HERE / "work"))
    t0 = time.perf_counter()
    plan = [p for p in V.planned() if p[0] == seed]
    try:
        with open(out, "w") as f:
            for (_, row, fam, lv) in plan:
                e = entries.get((row, fam, lv))
                b = bases[row]
                sid = f"i06_s{seed}_{fam}{lv}_eval_{row:02d}_{b.layout.name}"
                rec = {"scene_id": sid, "seed": seed, "row": row, "family": fam, "level": lv,
                       "in_manifest": e is not None, "manifest_reasons": e["reasons"] if e else None,
                       "manifest_stream_ok": bool(e) and e["stream"] == [seed, 2, V.FAM_CODE[fam], row, lv],
                       "manifest_scene_id_ok": bool(e) and e["scene_id"] == sid}
                rng = np.random.default_rng([seed, 2, V.FAM_CODE[fam], row, lv])
                reasons, kept_pairs, info, draws = [], None, None, []
                for attempt in range(6):
                    try:
                        pairs, inf, n = V.place(rng, b, fam, lv)
                    except V.NoSpot:
                        reasons.append("no_spot")
                        break
                    draws.append(n)
                    lay = variant_layout(b.layout, pairs)
                    scene, rgb, depth = G.build_scene(lay, sid)
                    folder = work / sid
                    if folder.exists():
                        shutil.rmtree(folder)
                    save_scene(folder, scene, rgb, depth)
                    why = G._rejection(folder, cfg)
                    if why is None:
                        scene_r = G._with_regions(folder, scene, cfg)
                        save_scene(folder, scene_r, rgb, depth)
                        reasons.append("accepted")
                        kept_pairs, info = pairs, inf
                        rec["my_sha256"] = {n_: sha(folder / n_) for n_ in ("rgb.png", "depth.npy", "scene.json")}
                        shutil.rmtree(folder)
                        break
                    reasons.append(why)
                    rec["last_attempt_people"] = [(p.id, p.x, p.y, p.yaw, p.group) for p, _ in pairs]
                    shutil.rmtree(folder)
                rec["replay_reasons"] = reasons
                rec["replay_draws"] = draws
                rec["reasons_match"] = e is not None and reasons == e["reasons"]
                if e is not None and not e["accepted"] and "last_attempt_added" in e and "last_attempt_people" in rec:
                    la = e["last_attempt_added"]
                    rec["last_attempt_match"] = max(max(abs(a["x"] - p[1]), abs(a["y"] - p[2]),
                                                        abs(wrap_angle(a["yaw"] - p[3])))
                                                    for a, p in zip(la, rec["last_attempt_people"])) < 1e-9 and \
                        len(la) == len(rec["last_attempt_people"])
                if kept_pairs is not None and e is not None and e["accepted"]:
                    folder = V.I06 / e["folder"]
                    inputs, truth = load_scene(folder)
                    stored = list(truth.people)[len(b.people):]
                    rec["people_max_diff"] = max(max(abs(p.x - q.x), abs(p.y - q.y), abs(wrap_angle(p.yaw - q.yaw)))
                                                 for (p, _), q in zip(kept_pairs, stored)) if stored else None
                    rec["people_ids_groups_match"] = [(p.id, p.group, p.role) for p, _ in kept_pairs] == \
                        [(q.id, q.group, q.role) for q in stored] and len(stored) == len(kept_pairs)
                    rec["shirts_match"] = [list(fg.color) for _, fg in kept_pairs] == [a["shirt"] for a in e["added"]]
                    theirs = {n_: sha(folder / n_) for n_ in ("rgb.png", "depth.npy", "scene.json")}
                    rec["bytes_equal"] = {n_: rec["my_sha256"][n_] == theirs[n_] for n_ in theirs}
                    rec["manifest_sha_ok"] = e["sha256"] == {n_: sha(folder / n_) for n_ in e["sha256"]}
                    rec["geometry"] = geometry(b, fam, lv, truth, inputs, info)
                    rec["info"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in info.items()}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"{sid}: replay {reasons} manifest {e['reasons'] if e else None} match {rec['reasons_match']} "
                      f"bytes {rec.get('bytes_equal')} [{time.perf_counter() - t0:.0f} s]", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"seed {seed}: done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
