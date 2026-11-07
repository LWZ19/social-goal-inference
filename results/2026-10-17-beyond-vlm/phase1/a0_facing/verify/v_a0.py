"""Independent verifier of A0 (VLM facing accuracy). Written without reading the first agent's scripts.

Unit: one people answer per (arm, scene, trial) of a C run; its people are the union of the records' result.people
(deduplicated by xy). Each listed person is tied to its raw entry by re-placing the raw people answer with the
package's _place. Truth: visible true people (is_visible under the run's thresholds), matched with sgi.run._matched_people.
"""
import collections
import json
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402
from sgi.gt.derive import is_visible  # noqa: E402
from sgi.run import _matched_people  # noqa: E402
from sgi.schemas import FACINGS, people_from_wire  # noqa: E402
from sgi.stages.people import _place, _TURN_DEG, _lift_point, _centre  # noqa: E402

OUT = Path(__file__).parent
REAL = ("real", "staffed", "plain")

SOURCES = {
    "qwen_s19": [("s19", a, "qwen38_low", "con") for a in REAL],
    "qwen_s21": [("s21", a, "qwen38_low", "con") for a in REAL],
    "glm_s19": [("s19", a, "glm53", "con") for a in REAL],
    "glm_s23": [("s23", a, "glm53", "con") for a in REAL],
    "dsv_s20": [("s20", a, "dsv41", "con") for a in REAL],
    "dsv_s23": [("s23", a, "dsv41", "con") for a in REAL],
    "haiku": [("s23", a, "haiku55", "con") for a in REAL],
    "sonnet": [("s23", a, "sonnet55_low", "con") for a in REAL],
    "gemini": [("s23", a, "gemini38_low", "con") for a in REAL],
}
for m in ("glm53", "ornith15", "qwen36_35", "qwen38", "qwen3vl"):
    SOURCES[f"s8_{m}"] = [("s8", "ph", m, "convlm")]


def wrap(a):
    return math.pi - ((math.pi - a) % (2 * math.pi))


def bearing(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


class DP:
    def __init__(self, number, xy, yaw):
        self.number, self.xy, self.yaw = number, tuple(xy), yaw


def groups_by_situation(truth):
    out = collections.defaultdict(set)
    for ep in truth.episodes.values():
        g = ep.group_at_target
        ids = list(g.values()) if isinstance(g, dict) else [g]
        for gid in ids:
            if gid is not None:
                out[gid].add(ep.situation)
    return out


def role_of(person, gsits):
    if person.role:
        return "staff"
    if person.group is not None:
        s = gsits.get(person.group, set())
        if "waiting_line" in s:
            return "line"
        if s & {"conversation_group", "audience", "group_photo"}:
            return "group"
    return "other"


def layout_of(scene_id, arm):
    if arm in REAL and scene_id.startswith(arm + "_"):
        return scene_id[len(arm) + 1:]
    return scene_id


def main():
    pairs = []          # one per matched pair
    keys_out = {}       # per source: list of key dicts
    for src, specs in SOURCES.items():
        keyinfo = []
        for (stage, arm, model, method) in specs:
            refs = H.runs(stage=stage, arm=arm, model=model, method=method)
            assert len(refs) == 1, (src, stage, arm, refs)
            ref = refs[0]
            th = H.recorded_config(ref)[1].thresholds
            groups = collections.defaultdict(list)
            for rec in H.records(ref):
                groups[(rec.scene_id, rec.trial)].append(rec)
            for (scene_id, trial), recs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
                raws = set()
                pcall = None
                people = {}
                sits = {r.situation for r in recs}
                for r in recs:
                    for c in r.result.get("calls") or []:
                        if c.get("stage") == "people":
                            pcall = c
                            raws.add((c.get("status"), c.get("raw_text")))
                    for p in r.people or []:
                        k = (round(p["xy"][0], 9), round(p["xy"][1], 9))
                        people.setdefault(k, p)
                if pcall is None:
                    continue
                sc = H.scene(ref, scene_id)
                obs = sc.observation
                cam = obs.camera.T_world_camera[:2, 3]
                camxy = (float(cam[0]), float(cam[1]))
                visible = [p for p in sc.truth.people if is_visible(p, obs, th)]
                gsits = groups_by_situation(sc.truth)
                # re-place the raw answer
                entries = {}
                n_raw = n_unlifted = 0
                ok_raw = [t for s, t in raws if s == "ok"]
                if len(ok_raw) == 1:
                    wire = json.loads(ok_raw[0])
                    pf = pcall["point_format"]
                    size = tuple(json.loads(pcall["image_size"])) if isinstance(pcall["image_size"], str) else tuple(pcall["image_size"])
                    ans = people_from_wire(wire, pf, size)
                    for e in ans.people:
                        n_raw += 1
                        fp = _place(e, obs, th)
                        if fp is None:
                            n_unlifted += 1
                            continue
                        # faced point used?
                        faced = None if e.faced_point is None else _lift_point(obs, e.faced_point, th)[1]
                        used_fp = faced is not None and math.dist(fp.xy, faced[:2]) >= th.person_radius_m - 1e-9
                        k = (round(fp.xy[0], 9), round(fp.xy[1], 9))
                        entries.setdefault(k, (e.facing, fp.yaw, used_fp))
                listed = list(people.values())
                dps = [DP(i + 1, p["xy"], p["yaw"]) for i, p in enumerate(listed)]
                matched = _matched_people(dps, visible, th)
                byid = {p.id: p for p in visible}
                tied = 0
                for d, p in zip(dps, listed):
                    k = (round(p["xy"][0], 9), round(p["xy"][1], 9))
                    ent = entries.get(k)
                    if ent is not None and (ent[1] == p["yaw"] or (ent[1] is not None and p["yaw"] is not None
                                                                     and abs(ent[1] - p["yaw"]) < 1e-9)):
                        tied += 1
                    if d.number not in matched:
                        continue
                    t = byid[matched[d.number]]
                    ang = abs(math.degrees(wrap(bearing((t.x, t.y), camxy) - t.yaw)))
                    rel_true = wrap(t.yaw - bearing((t.x, t.y), camxy))
                    true_bin = min(FACINGS[:8], key=lambda f: abs(wrap(rel_true - math.radians(_TURN_DEG[f]))))
                    label = ent[0] if ent else None
                    label_yaw = None
                    if label in _TURN_DEG:
                        label_yaw = wrap(bearing(p["xy"], camxy) + math.radians(_TURN_DEG[label]))
                    err = None if p["yaw"] is None else abs(math.degrees(wrap(p["yaw"] - t.yaw)))
                    lerr = None if label_yaw is None else abs(math.degrees(wrap(label_yaw - t.yaw)))
                    pairs.append(dict(src=src, arm=arm, scene=scene_id, layout=layout_of(scene_id, arm), trial=trial,
                                      sit=sorted(sits)[0], nsits=len(sits), pid=t.id, role=role_of(t, gsits),
                                      ang=ang, err=err, true_bin=true_bin, label=label, lerr=lerr,
                                      used_fp=(ent[2] if ent else None),
                                      gap=math.dist(d.xy, (t.x, t.y)),
                                      yaw_rep=p["yaw"], yaw_true=t.yaw, b_true=bearing((t.x, t.y), camxy),
                                      b_rep=bearing(p["xy"], camxy), label_yaw=label_yaw))
                keyinfo.append(dict(arm=arm, scene=scene_id, trial=trial, nraw=len(raws), raw=sorted(raws, key=str),
                                    nlisted=len(listed), nvisible=len(visible), nmatched=len(matched),
                                    n_entries_raw=n_raw, n_unlifted=n_unlifted, tied=tied, sits=sorted(sits),
                                    nrec=len(recs)))
        keys_out[src] = keyinfo
        print(src, len(keyinfo), "answers", flush=True)
    with open(OUT / "v_pairs.pkl", "wb") as f:
        pickle.dump(dict(pairs=pairs, keys=keys_out), f)


if __name__ == "__main__":
    main()
