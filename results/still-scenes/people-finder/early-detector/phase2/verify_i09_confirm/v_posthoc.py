"""Verifier, POST HOC (not pre-registered; checks the compute agent's post hoc false-people classes; changes no
verdict). Classes, first that applies: A within 0.8 m of a visible staff member with the lower body hidden; B within
0.5 m of a true person who fails is_visible; C within 1.5 m of any true person; D otherwise. Project venv."""
import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "results/still-scenes/analysis-scripts/harness")
sys.path.insert(0, str(HERE))
import research_harness as H  # noqa: E402
import v_pipe as VP  # noqa: E402
import v_score as VS  # noqa: E402
from sgi.config import load_config  # noqa: E402
from sgi.gt.derive import is_visible  # noqa: E402
from sgi.scene import load_scene  # noqa: E402

CONFIRM = VS.CONFIRM
th = load_config().thresholds
man = json.loads((CONFIRM / "manifest.json").read_text())["entries"]
raws = [json.loads(x) for x in (HERE / "raw/v_rtdetr_v2_r50vd.jsonl").read_text().splitlines()]
cls = collections.Counter()
rows_ = collections.defaultdict(collections.Counter)
info = collections.defaultdict(list)
for e, raw in zip(man, raws):
    f = Path(e["folder"])
    VS.VG.check(str(f / "scene.json"))
    H.guard_scene(f)
    inputs, truth = load_scene(f)
    obs = inputs.observation
    ppl = VP.people(raw, obs.depth, obs.camera)
    dets = [(p["xy"][0], p["xy"][1], p["yaw"], p["score"]) for p in ppl]
    vis = [p for p in truth.people if is_visible(p, obs, th)]
    hid = [p for p in truth.people if not is_visible(p, obs, th)]
    m = VS.match([(d[0], d[1]) for d in dets], vis, 0.5)
    ex = [(t.anchor[0], t.anchor[1]) for t in truth.targets.values() if t.person_id is None]
    for i, d in enumerate(dets):
        if i in m or any(math.dist(d[:2], q) <= 0.5 for q in ex):
            continue
        xy = d[:2]
        lhs = [p for p in vis if p.role and VS.lower_hidden(p, obs) and math.dist(xy, (p.x, p.y)) <= 0.8]
        nh = min((math.dist(xy, (p.x, p.y)) for p in hid), default=math.inf)
        na = min((math.dist(xy, (p.x, p.y)) for p in truth.people), default=math.inf)
        if lhs:
            c = "A"
            staff_matched = any(vis.index(p) in m.values() for p in lhs)
            info["A_matched_staff"].append(staff_matched)
        elif nh <= 0.5:
            c = "B"
            hp = min(hid, key=lambda p: math.dist(xy, (p.x, p.y)))
            q = VS.project(obs.camera, (hp.x, hp.y, 1.1))
            info["B_hidden_in_picture"].append(bool(q and VS.inimg(obs.camera, q[0], q[1])))
            info["B_role"].append(hp.role)
        elif na <= 1.5:
            c = "C"
            info["C_hidden_dist"].append(round(nh, 3))
        else:
            c = "D"
            info["D_dist_all"].append(na)
        cls[c] += 1
        info[c + "_score"].append(d[3])
        rows_[c][e["scene_id"][5:12]] += 1
L = [f"POST HOC verifier: classes {dict(sorted(cls.items()))}; total {sum(cls.values())}",
     f"A: second box on a matched staff member {sum(info['A_matched_staff'])}, paired with a staff miss "
     f"{len(info['A_matched_staff']) - sum(info['A_matched_staff'])}; scores {min(info['A_score']):.2f}-{max(info['A_score']):.2f}",
     f"B: hidden person projects into the picture {sum(info['B_hidden_in_picture'])}/{len(info['B_hidden_in_picture'])}; "
     f"roles {collections.Counter(info['B_role'])}",
     f"C: distance to nearest hidden person {sorted(info['C_hidden_dist'])}",
     f"D: distance to every true person {min(info['D_dist_all']):.2f}-{max(info['D_dist_all']):.2f} m; scores "
     f"{min(info['D_score']):.2f}-{max(info['D_score']):.2f}, median {statistics.median(info['D_score']):.2f}; rows "
     f"{dict(rows_['D'])}",
     f"sensitivity: without A {(44 - cls['A']) / 182:.3f}; without A,B {(44 - cls['A'] - cls['B']) / 182:.3f}; "
     f"without A,B,C {cls['D'] / 182:.3f} per image"]
print("\n".join(L))
(HERE / "v_posthoc.txt").write_text("\n".join(L) + "\n")
