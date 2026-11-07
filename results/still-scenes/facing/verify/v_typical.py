import pickle, math, collections
from sgi.schemas import FACINGS
from sgi.stages.people import _TURN_DEG
P = pickle.load(open("results/still-scenes/facing/verify/v_pairs.pkl", "rb"))["pairs"]
def wrap(a): return math.pi - ((math.pi - a) % (2 * math.pi))
def nb(rel): return min(FACINGS[:8], key=lambda f: abs(wrap(rel - math.radians(_TURN_DEG[f]))))
for s in ["qwen_s19", "glm_s19", "dsv_s20", "haiku", "sonnet", "gemini"]:
    rows = [r for r in P if r["src"] == s and r["role"] == "line" and r["yaw_rep"] is not None]
    a = sum(r["true_bin"] == "right" and nb(wrap(r["yaw_rep"] - r["b_true"])) == "towards_right" for r in rows)
    b = sum(r["true_bin"] == "right" and r["label"] == "towards_right" for r in rows)
    c = sum(r["true_bin"] == "right" and r["used_fp"] and nb(wrap(r["yaw_rep"] - r["b_true"])) == "towards_right" for r in rows)
    print(s, "line: true right & yaw bin tr", a, "| label tr", b, "| via fp", c)
