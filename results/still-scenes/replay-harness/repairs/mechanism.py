"""Why the repairs flip what they flip (zero calls): for the old-wording line failures (s19 GLM, s20 DeepSeek; also
s23 GLM/DeepSeek) and for every record that P or R2 breaks, the come_after numbers the model stated and which of them
are spurious (unmatched at 0.5 m), how far each spurious detection lies from the nearest visible true person and
from the nearest MISSED true person, and the verdicts from rows_*.jsonl. Writes mechanism.txt beside this file."""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402

OUT = Path(__file__).resolve().parent
PLAN = [("old", "s19", "glm53"), ("old", "s20", "dsv41"), ("new", "s23", "glm53"), ("new", "s23", "dsv41")]


def main():
    from sgi.gt.derive import is_visible
    from sgi.run import _matched_people
    from sgi.types import DetectedPerson
    out = []
    for wording, stage, model in PLAN:
        rows = {}
        for line in (OUT / f"rows_{wording}_{model}.jsonl").open():
            r = json.loads(line)
            rows[(r["arm"], r["scene"], r["episode"], r["trial"])] = r
        want = {k for k, r in rows.items() if r["line_failure"] or (r["solver_step"] and r["C_rec"]
                                                                     and (not r["P"] or not r["R2"]))}
        tally = collections.Counter()
        out.append(f"== {wording} {model} ({stage}) ==")
        for arm in ("real", "staffed", "plain"):
            ref = H.runs(stage=stage, arm=arm, model=model, method="con")[0]
            th = H.recorded_config(ref)[1].thresholds
            for rec in H.records(ref):
                key = (arm, rec.scene_id, rec.episode_id, rec.trial)
                if key not in want:
                    continue
                r = rows[key]
                sc = H.scene(ref, rec.scene_id)
                vis = [p for p in sc.truth.people if is_visible(p, sc.observation, th)]
                det = [DetectedPerson.model_validate(p) for p in rec.people]
                m = _matched_people(det, vis, th)
                hit = set(m.values())
                missed = [p for p in vis if p.id not in hit]
                ca = sorted({n for u in rec.constraints if u["constraint"]["kind"] == "come_after"
                             for n in u["constraint"]["people"]})
                spur = [d for d in det if d.number not in m]
                spur_ca = [d.number for d in spur if d.number in ca]
                desc = []
                for d in spur:
                    near_v = min((math.dist(d.xy, (p.x, p.y)) for p in vis), default=float("inf"))
                    near_m = min((math.dist(d.xy, (p.x, p.y)) for p in missed), default=float("inf"))
                    desc.append(f"#{d.number}{'(ca)' if d.number in ca else ''} vis {near_v:.2f} m, missed {near_m:.2f} m")
                if r["line_failure"]:
                    tally["line failures"] += 1
                    tally["with a spurious number in come_after"] += bool(spur_ca)
                    tally["R4 flips"] += r["R4"]
                    tally["R4 flips with a spurious number in come_after"] += r["R4"] and bool(spur_ca)
                    tally["P flips"] += r["P"]
                    tally["P flips with a spurious number in come_after"] += r["P"] and bool(spur_ca)
                    tally["no come_after stated"] += not ca
                    tag = "LF"
                else:
                    tag = "BREAK" + ("" if r["P"] else " P") + ("" if r["R2"] else " R2")
                out.append(f"  {tag:10s} {arm:8s}{rec.scene_id:36s}{rec.episode_id} t{rec.trial} {rec.situation:19s} "
                           f"C {r['C_rec']:d} R1 {r['R1']:d} R2 {r['R2']:d} R3 {r['R3']:d} R4 {r['R4']:d} P {r['P']:d} "
                           f"R5 {r['R5']:d} P1m {r['P1m']:d} | det {len(det)} vis {len(vis)} missed {len(missed)} "
                           f"come_after {ca} spurious: {'; '.join(desc) or '-'} | kinds {r['kinds']} "
                           f"uses C {r['uses_C']} P {r['uses_P']} facing err {r['facing_err_deg']}")
        out.append("  summary: " + "; ".join(f"{k} {v}" for k, v in tally.items()))
        out.append("")
        print("\n".join(out[-3:]), flush=True)
    (OUT / "mechanism.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
