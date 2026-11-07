"""I03 Stage B, POST HOC diagnostics (written after the arms were solved and tables.txt was read; nothing here changes
an arm or a criterion). No solving: the arms' people lists are rebuilt with replay.py's own functions and compared
with the records and the truth.

1. Number collisions: solver steps whose recorded constraints name numbers that are not in the recorded people list
   (dangling numbers), and where an added person (B-add / B-relocate numbering, max listed + 1, ...) takes such a
   number, so a constraint the solver judged unusable in C now binds to a blob. Verdict changes among them.
2. Who pruning removes: each pruned detection that is matched to a visible true person, by that person's role class
   (people_scorer.role_classes: staff / line / group / other).
3. Every break of every arm, and every B-add flip, with what changed (old, new and placeholder runs).
Writes posthoc.txt.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import replay as RP  # noqa: E402
import people_scorer as PS  # noqa: E402

H, R = RP.H, RP.R


def main():
    from sgi.run import _matched_people
    from sgi.types import DetectedPerson
    _, blobs_of = RP.load_cache()
    L = []
    p = L.append
    p("I03 Stage B post hoc diagnostics (labelled post hoc: designed after reading tables.txt).")
    coll = collections.Counter()
    coll_rows = []
    pruned_roles = collections.defaultdict(collections.Counter)
    events = []
    for wording, plan in RP.PLAN.items():
        for stage, model, arms in plan:
            rows = {(r["arm"], r["scene"], r["episode"], r["trial"]): r
                    for r in map(json.loads, (HERE / f"rows_{wording}_{model}.jsonl").open())}
            for arm in arms:
                ref = R.one_run(stage, arm, model)
                for rec in H.records(ref):
                    if not rec.solver_step:
                        continue
                    row = rows[(arm, rec.scene_id, rec.episode_id, rec.trial)]
                    blobs = blobs_of[(str(ref.scenes_dir), rec.scene_id)]
                    sc, th, vis, _ = R.scene_info(rec)
                    obs = sc.observation
                    det = [DetectedPerson.model_validate(x) for x in rec.people]
                    listed = {d.number for d in det}
                    named = {n for u in rec.constraints for n in (u["constraint"].get("people") or [])}
                    dangling = named - listed
                    txy = tuple(rec.target["world"][:2])
                    tidx = {i for i, b in enumerate(blobs) if RP.near(txy, (b["x"], b["y"]))}
                    add, _, _ = RP.added_people(det, blobs, RP.unexplained_idx(det, blobs), tidx, obs)
                    taken = dangling & {a.number for a in add}
                    key = (wording, model)
                    coll[key + ("steps",)] += 1
                    coll[key + ("dangling",)] += bool(dangling)
                    coll[key + ("taken",)] += bool(taken)
                    if taken:
                        ch = [k for k in RP.ARMS if row[k] != row["C_rec"]]
                        coll[key + ("taken_changed_Badd",)] += row["Badd"] != row["C_rec"]
                        coll_rows.append(f"  {wording} {model} {rec.scene_id} {rec.episode_id} t{rec.trial}: listed "
                                         f"{sorted(listed)}, dangling {sorted(dangling)}, taken by added {sorted(taken)}; "
                                         f"C_rec {row['C_rec']}, Badd {row['Badd']}, arms changed {ch}")
                    # who pruning removes (B-prune)
                    m = _matched_people(det, vis, th)
                    roles = PS.role_classes(sc.truth)
                    _, pruned, _ = RP.prune(det, blobs, obs, keep=set())
                    for n in pruned:
                        pruned_roles[key][roles[m[n]] if n in m else "unmatched (no visible person)"] += 1
                    for k in RP.ARMS:
                        if row["C_rec"] and not row[k]:
                            d = row[k + "_d"]
                            events.append(f"  BREAK {k:8s} {wording} {model} {rec.scene_id} {rec.episode_id} t{rec.trial} "
                                          f"({rec.situation}): added {d['added']} (false {d['added_false']}), moved "
                                          f"{d['moved']}, pruned {d['pruned']} (were matched {d['pruned_were_matched']}); "
                                          f"region/reach/heading {d['in_region']}/{d['reachable']}/{d['heading_ok']}; "
                                          f"dangling numbers taken by added {sorted(taken)}")
                    if row["Badd"] and not row["C_rec"]:
                        d = row["Badd_d"]
                        events.append(f"  FLIP  Badd     {wording} {model} {rec.scene_id} {rec.episode_id} t{rec.trial} "
                                      f"({rec.situation}): det {row['n_det']} (unsupported {row['n_unsupported']}, "
                                      f"matched {row['n_matched']} of {row['n_vis']} visible), added {d['added']} "
                                      f"(matched missed {d['added_matched_missed']}), joined {d['joined']} (added among "
                                      f"them {d['joined_added']}); I01 R3 {row['R3_i01']}, P {row['P_i01']}")
    p("")
    p("1. Number collisions (dangling constraint numbers taken by added people), per cell: solver steps; steps with a "
      "dangling number; steps where an added person takes one; of those, B-add verdict changed.")
    for wording, plan in RP.PLAN.items():
        for _, model, _ in plan:
            k = (wording, model)
            p(f"  {wording:4s} {model:13s} steps {coll[k + ('steps',)]}, dangling "
              f"{coll[k + ('dangling',)]}, taken {coll[k + ('taken',)]}, B-add changed {coll[k + ('taken_changed_Badd',)]}")
    L.extend(coll_rows)
    p("")
    p("2. Who B-prune removes: pruned detections by the role class of the visible true person they are matched to.")
    for k, c in pruned_roles.items():
        p(f"  {k[0]:4s} {k[1]:13s} {sum(c.values())} pruned: {sorted(c.items())}")
    p("")
    p("3. Every break of every arm, and every B-add flip.")
    L.extend(events)
    (HERE / "posthoc.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
