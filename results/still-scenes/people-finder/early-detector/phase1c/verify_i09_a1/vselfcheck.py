"""Verifier: the scorer with the true visible people as detections on the 289 fit scenes (R3 claim), and the
section 4 synthetic sign checks on the verifier's own facing code."""
import math, sys
sys.path.insert(0, 'results/still-scenes/people-finder/early-detector/phase1c/verify_i09_a1')
sys.path.insert(0, 'results/still-scenes/perception/depth-blob-detector/stageA')
sys.path.insert(0, 'results/still-scenes/perception/depth-blob-detector')
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import fitdata as FD, people_scorer as PS, vpipe as VP
ok = 0; n = 0
for dom, sset, sid, loader in FD.fit_refs():
    sc, th = FD.load(loader)
    vis = PS.truth_info(sc, th)["visible"]
    r = PS.score_people([(p.x, p.y, p.yaw) for p in vis], sc, th, exempt_xy=PS.exempt_points(sc.truth))
    n += 1
    ok += (r["recall"] in (None, 1.0)) and r["false"] == 0 and r["facing"]["within45_full"] == r["facing"]["known"] == len(vis)
print(f"true visible people as detections: {ok}/{n} scenes with recall 1 (or none visible), 0 false, all facing within 45")
# synthetic: L at +y, R at -y -> yaw 0
b = {"vecs": {"sh": (0.0, 0.4)}, "face": False, "prof_du": None}
print("L +y / R -y gives F1", VP.facing(b, (0.0, 3.0), (0.0, 0.0))["F1"])
b = {"vecs": {}, "face": False, "prof_du": 5.0}
print("camera at origin looking +y, person (0,3), nose image-right: F4", VP.facing(b, (0.0, 3.0), (0.0, 0.0))["F4"], "F5", VP.facing(b, (0.0, 3.0), (0.0, 0.0))["F5"])
