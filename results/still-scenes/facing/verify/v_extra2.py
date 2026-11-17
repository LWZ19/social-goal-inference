import pickle, collections, statistics, math, hashlib
from pathlib import Path
from sgi.schemas import FACINGS
from sgi.stages.people import _TURN_DEG
OUT = Path(__file__).parent
D = pickle.load(open(OUT / "v_pairs.pkl", "rb")); P = D["pairs"]; K = D["keys"]
PRIMARY = ["qwen_s19", "glm_s19", "dsv_s20", "haiku", "sonnet", "gemini"]
def wrap(a): return math.pi - ((math.pi - a) % (2 * math.pi))
def ad(a, b): return abs(math.degrees(wrap(a - b)))
def view(a): return "frontal" if a < 45 else "oblique" if a < 75 else "profile" if a <= 105 else "back"
def nb(rel): return min(FACINGS[:8], key=lambda f: abs(wrap(rel - math.radians(_TURN_DEG[f]))))
out = []
for s in PRIMARY:
    rows = [r for r in P if r["src"] == s]
    fp = [r for r in rows if r["used_fp"] and r["label_yaw"] is not None]
    # to_cam: true angle to camera minus reported angle to camera (reported yaw against the true bearing, as theirs; and against own bearing)
    tc_true_b = [ad(r["b_true"], r["yaw_true"]) - ad(r["b_true"], r["yaw_rep"]) for r in fp]
    tc_own_b = [ad(r["b_true"], r["yaw_true"]) - ad(r["b_rep"], r["yaw_rep"]) for r in fp]
    tc_lab = [ad(r["b_true"], r["yaw_true"]) - ad(r["b_true"], r["label_yaw"]) for r in fp]
    rb = collections.Counter()
    for r in rows:
        if r["yaw_rep"] is None: continue
        rb[(r["true_bin"], nb(wrap(r["yaw_rep"] - r["b_true"])))] += 1
    tr_line_prof = sum(1 for r in rows if r["yaw_rep"] is not None and r["true_bin"] == "right" and nb(wrap(r["yaw_rep"] - r["b_true"])) == "towards_right" and r["role"] == "line" and view(r["ang"]) == "profile")
    tr_any = rb[("right", "towards_right")]
    out.append(f"{s}: to_cam faced (true bearing) {statistics.median(tc_true_b):+.1f}, (own bearing) {statistics.median(tc_own_b):+.1f}, label {statistics.median(tc_lab):+.1f}; "
               f"pipeline-yaw bin true right->towards_right: all {tr_any}, profile line {tr_line_prof}; diag right {rb[('right','right')]}, left {rb[('left','left')]}")
# trials identical within (arm, scene)?
for s in PRIMARY:
    by = collections.defaultdict(set); cnt = collections.Counter()
    for k in K[s]:
        by[(k["arm"], k["scene"])].add(hashlib.md5(str(k["raw"]).encode()).hexdigest())
        cnt[(k["arm"], k["scene"])] += 1
    multi = [a for a in by if cnt[a] > 1]
    ident = sum(len(by[a]) == 1 for a in multi)
    out.append(f"{s}: (arm,scene) with >1 trial {len(multi)}; all trials identical text {ident}; total distinct texts {sum(len(v) for v in by.values())} of {len(K[s])}")
# pairs per (arm, scene, trial, person) unique?
for s in PRIMARY:
    rows = [r for r in P if r["src"] == s]
    keys = collections.Counter((r["arm"], r["scene"], r["trial"], r["pid"]) for r in rows)
    out.append(f"{s}: duplicate pair keys {sum(v > 1 for v in keys.values())}")
print("\n".join(out))
(OUT / "v_extra2.txt").write_text("\n".join(out) + "\n")
