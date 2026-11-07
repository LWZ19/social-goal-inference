"""Independent re-computation of R1, R2, R3, R4 and P for the old-wording runs (s19 Qwen/GLM, s20 DeepSeek), with
its own matching code (not sgi.run._matched_people) and its own people assembly from sgi.run.OraclePeople, compared
record by record with rows_old_*.jsonl. Writes verify.txt beside this file."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402

OUT = Path(__file__).resolve().parent
PLAN = [("s19", "qwen38_low"), ("s19", "glm53"), ("s20", "dsv41")]


def greedy(det, vis, width):
    """One-to-one, nearest pair first; ties by lower number, then earlier true person."""
    cand = []
    for d in det:
        for i, t in enumerate(vis):
            g = math.hypot(d["xy"][0] - t.x, d["xy"][1] - t.y)
            if g <= width:
                cand.append((g, d["number"], i))
    cand.sort()
    out, used_d, used_t = {}, set(), set()
    for g, n, i in cand:
        if n in used_d or i in used_t:
            continue
        out[n] = vis[i]
        used_d.add(n)
        used_t.add(i)
    return out


def main():
    from sgi.gt.derive import is_visible
    from sgi.run import OraclePeople
    lines, bad, total = [], 0, 0
    for stage, model in PLAN:
        mine = {}
        for line in (OUT / f"rows_old_{model}.jsonl").open():
            r = json.loads(line)
            mine[(r["arm"], r["scene"], r["episode"], r["trial"])] = r
        agree = {k: 0 for k in ("R1", "R2", "R3", "R4", "P")}
        n = 0
        for arm in ("real", "staffed", "plain"):
            ref = H.runs(stage=stage, arm=arm, model=model, method="con")[0]
            th = H.recorded_config(ref)[1].thresholds
            for rec in H.records(ref):
                if not rec.solver_step:
                    continue
                sc = H.scene(ref, rec.scene_id)
                vis = [p for p in sc.truth.people if is_visible(p, sc.observation, th)]
                orc = OraclePeople(sc.truth, th)(sc.observation).found
                orc_by_xy = {tuple(f.xy): f for f in orc}
                det = [dict(p) for p in rec.people]
                m = greedy(det, vis, 2 * th.person_radius_m)
                hit = {t.id for t in m.values()}
                top = max([d["number"] for d in det], default=0)
                added, k = [], 1
                for t in vis:
                    f = orc_by_xy.get((t.x, t.y))
                    if t.id in hit or f is None:
                        continue
                    added.append(dict(number=top + k, xy=list(f.xy), yaw=f.yaw, pixel=list(f.pixel), label=None))
                    k += 1
                r1 = [dict(d, xy=[m[d["number"]].x, m[d["number"]].y]) if d["number"] in m else d for d in det]
                r2 = [dict(d, yaw=m[d["number"]].yaw) if d["number"] in m else d for d in det]
                gone = {d["number"] for d in det if d["number"] not in m}
                cons = [dict(u["constraint"], people=[x for x in u["constraint"]["people"] if x not in gone])
                        for u in rec.constraints]
                keep = [d for d in det if d["number"] in m]
                pp = [dict(d, xy=[m[d["number"]].x, m[d["number"]].y], yaw=m[d["number"]].yaw) for d in keep] + added
                got = {"R1": H.resolve(rec, people=r1).score.success,
                       "R2": H.resolve(rec, people=r2).score.success,
                       "R3": H.resolve(rec, people=det + added).score.success,
                       "R4": H.resolve(rec, people=keep, constraints=cons).score.success,
                       "P": H.resolve(rec, people=pp, constraints=cons).score.success}
                row = mine[(arm, rec.scene_id, rec.episode_id, rec.trial)]
                n += 1
                for key, val in got.items():
                    if bool(val) == row[key]:
                        agree[key] += 1
                    else:
                        bad += 1
                        lines.append(f"  DIFFER {stage} {model} {arm} {rec.scene_id} {rec.episode_id} t{rec.trial} {key}: "
                                     f"verify {val}, repairs.py {row[key]}")
                total += 1
        lines.append(f"{stage} {model}: {n} solver steps; agree " + ", ".join(f"{k} {v}/{n}" for k, v in agree.items()))
        print(lines[-1], flush=True)
    lines.append(f"total disagreements: {bad} over {total} records x 5 arms")
    (OUT / "verify.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
