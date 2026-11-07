"""Synthetic checks of replay_d's matching and list building (no run, no scene, no record is read)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_d as A  # noqa: E402
import numpy as np  # noqa: E402
from sgi.types import DetectedPerson  # noqa: E402


class Cam:
    fx = fy = 300.0
    cx, cy = 320.0, 240.0
    width, height = 640, 480
    # camera at origin looking along +y (camera z = world y, camera x = world x, camera y = -world z)
    T_world_camera = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 1.2], [0, 0, 0, 1]], dtype=float)


class Obs:
    camera = Cam()


class Th:
    person_mark_height_m = 1.1
    person_radius_m = 0.25


def P(n, x, y, yaw=None, pix=(0.0, 0.0)):
    return DetectedPerson(number=n, xy=(x, y), yaw=yaw, pixel=pix, label=f"p{n}")


def Q(x, y, yaw, box):
    return dict(x=x, y=y, yaw=yaw, box=box, box_centre=[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


ok = []
# 1. xy matching: nearest first, one to one, within 0.5 m, tie -> lower number then detector order
det = [P(3, 0.0, 3.0), P(1, 0.1, 3.0), P(7, 5.0, 5.0)]
dets = [Q(0.05, 3.0, 0.5, [0, 0, 10, 10]), Q(0.6, 3.0, None, [0, 0, 10, 10]), Q(-2, 2, 1.0, [0, 0, 10, 10])]
m = A.match_xy(det, dets)
# pairs: (3,0)=0.05 (1,0)=0.05 tie -> number 1 first gets 0; then 3 -> 1 at 0.6 (<=0.5? no) -> 3 unmatched
ok.append(("xy tie lower number", m == {1: 0}))
det2 = [P(1, 0.0, 3.0), P(2, 1.0, 3.0)]
dets2 = [Q(0.5, 3.0, 0.0, [0, 0, 1, 1])]
ok.append(("xy tie distance lower number", A.match_xy(det2, dets2) == {1: 0}))
ok.append(("xy boundary 0.5 included", A.match_xy([P(1, 0.0, 3.0)], [Q(0.5, 3.0, 0, [0, 0, 1, 1])]) == {1: 0}))
ok.append(("xy beyond 0.5 excluded", A.match_xy([P(1, 0.0, 3.0)], [Q(0.5001, 3.0, 0, [0, 0, 1, 1])]) == {}))
# 2. pixel matching: box contains pixel (edges in), nearest box centre, one to one
detp = [P(2, 0, 0, pix=(10.0, 10.0)), P(1, 0, 0, pix=(12.0, 10.0)), P(5, 0, 0, pix=(100.0, 100.0))]
detsp = [Q(0, 0, 0, [0, 0, 20, 20]), Q(1, 1, 0, [10, 0, 30, 20])]
mp = A.match_pix(detp, detsp)
# number 1: d to c0 (10,10) = 2, to c1 (20,10) = 8; number 2: to c0 = 0 -> 2 takes 0; 1 takes 1 (8)
ok.append(("pix greedy", mp == {2: 0, 1: 1}))
ok.append(("pix edge", A.match_pix([P(1, 0, 0, pix=(20.0, 20.0))], [Q(0, 0, 0, [0, 0, 20, 20])]) == {1: 0}))
# 3. build: numbering above every listed or constraint number, strip, yaw fallback, keep
det3 = [P(1, 0.0, 3.0, yaw=0.3), P(2, 3.0, 3.0, yaw=0.7), P(4, 9.0, 9.0, yaw=1.0)]
cons = [dict(because="x", kind="come_after", people=[2, 4, 9], range_m=None, target=False)]
dets3 = [Q(0.1, 3.0, None, [0, 0, 1, 1]), Q(3.2, 3.0, 2.0, [0, 0, 1, 1]), Q(-1.0, 4.0, 1.5, [0, 0, 1, 1])]
pairs = A.match_xy(det3, dets3)
people, oc, dropped, add, info = A.build(det3, cons, dets3, pairs, False, 9, Obs(), Th())
byn = {p.number: p for p in people}
ok.append(("pairs", pairs == {1: 0, 2: 1}))
ok.append(("dropped 4", dropped == {4}))
ok.append(("stripped", oc[0]["people"] == [2, 9]))
ok.append(("added numbered M+1", add == {10} and byn[10].xy == (-1.0, 4.0) and byn[10].yaw == 1.5))
ok.append(("yaw fallback", byn[1].yaw == 0.3 and byn[1].xy == (0.1, 3.0)))
ok.append(("detector yaw", byn[2].yaw == 2.0 and byn[2].xy == (3.2, 3.0) and byn[2].label == "p2"))
pk, ock, dk, addk, _ = A.build(det3, cons, dets3, pairs, True, 9, Obs(), Th())
ok.append(("keep", ock is None and dk == set() and {p.number for p in pk} == {1, 2, 4, 10}
           and [p for p in pk if p.number == 4][0].xy == (9.0, 9.0)))
# added pixel = projection of (x, y, 1.1)
px = byn[10].pixel
exp = (300 * (-1.0) / 4.0 + 320, 300 * (1.2 - 1.1) / 4.0 + 240)
ok.append(("added pixel projection", math.isclose(px[0], exp[0]) and math.isclose(px[1], exp[1])))
# empty list -> all detector people added above the constraint numbers
pe, oce, de, adde, _ = A.build([], cons, dets3, {}, False, 9, Obs(), Th())
ok.append(("empty list", adde == {10, 11, 12} and len(pe) == 3 and oce[0]["people"] == [2, 4, 9]))
# a collision is refused
try:
    A.build([], cons, dets3, {}, False, 3, Obs(), Th())
    ok.append(("collision refused", False))
except AssertionError:
    ok.append(("collision refused", True))
for name, v in ok:
    print(("PASS" if v else "FAIL"), name)
assert all(v for _, v in ok)
print("all", len(ok), "checks pass")
