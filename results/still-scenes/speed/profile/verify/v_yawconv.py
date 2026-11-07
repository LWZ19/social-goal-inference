"""Yaw convention check: do line members face along the line toward their target (counter)? Plain arm only."""
import sys, math
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
def wrap(a): return math.pi - ((math.pi - a) % (2 * math.pi))
out = []
for sref in H.distinct_scenes():
    rr = [r for r in sref.runs if r.arm == "plain"]
    if not rr: continue
    sc = H.scene(rr[0], sref.scene_id)
    for eid, ep in sc.truth.episodes.items():
        if ep.situation != "waiting_line": continue
        for tid in ep.target_ids:
            g = sc.truth.group_for(eid, tid)
            if g is None: continue
            t = sc.truth.targets[tid]
            tx, ty = t.anchor[0], t.anchor[1]
            offs = [abs(math.degrees(wrap(math.atan2(ty - m.y, tx - m.x) - m.yaw))) for m in sc.truth.group(g)] if tx is not None else []
            s = f"{sref.scene_id} {eid} target={tid} group={g} off-from-facing-target(deg)={[round(o,1) for o in offs]}"
            print(s); out.append(s)
open('results/still-scenes/speed/profile/verify/v_yawconv.txt', 'w').write("\n".join(out) + "\n")
