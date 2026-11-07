import pickle, collections, math, sys
from pathlib import Path
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
from sgi.schemas import FACINGS
from sgi.stages.people import _TURN_DEG
OUT = Path(__file__).parent
D = pickle.load(open(OUT / "v_pairs.pkl", "rb")); P = D["pairs"]; K = D["keys"]
def wrap(a): return math.pi - ((math.pi - a) % (2 * math.pi))
def view(a): return "frontal" if a < 45 else "oblique" if a < 75 else "profile" if a <= 105 else "back"
def nb(rel): return min(FACINGS[:8], key=lambda f: abs(wrap(rel - math.radians(_TURN_DEG[f]))))
for s in ["qwen_s19", "glm_s19", "dsv_s20", "haiku", "sonnet", "gemini"]:
    rows = [r for r in P if r["src"] == s and r["role"] == "line" and view(r["ang"]) == "profile"]
    a = sum(r["true_bin"] == "right" and r["label"] == "right" and r["yaw_rep"] is not None and nb(wrap(r["yaw_rep"] - r["b_true"])) == "towards_right" for r in rows)
    b = sum(r["label"] in ("right", "left") and r["yaw_rep"] is not None and nb(wrap(r["yaw_rep"] - r["b_true"])) in ("towards_right", "towards_left") for r in rows)
    c = sum(r["true_bin"] in ("right", "left") and r["yaw_rep"] is not None and nb(wrap(r["yaw_rep"] - r["b_true"])) in ("towards_right", "towards_left") for r in rows)
    print(s, "profile line n", len(rows), "| true right, label right, yaw bin tr:", a, "| label side, yaw bin toward-side:", b, "| true side, yaw bin toward-side:", c)
# sonnet duplicate texts across trials: cached flags
by = collections.defaultdict(list)
for k in K["sonnet"]:
    by[(k["arm"], k["scene"], str(k["raw"]))].append(k["trial"])
dups = {k: v for k, v in by.items() if len(v) > 1}
print("sonnet dup texts:", [(k[0], k[1], v) for k, v in dups.items()])
for (arm, scene, _), trials in dups.items():
    ref = H.runs(stage="s23", arm=arm, model="sonnet55_low", method="con")[0]
    for rec in H.records(ref):
        if rec.scene_id == scene and rec.trial in trials:
            for c in rec.result["calls"]:
                if c["stage"] == "people":
                    print(arm, scene, rec.episode_id, rec.trial, "cached", c.get("cached"), "lat", round(c.get("latency_s") or 0, 2), "tokens", c.get("usage"))
