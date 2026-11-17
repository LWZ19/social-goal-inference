import pickle, collections, statistics
from pathlib import Path
OUT = Path(__file__).parent
D = pickle.load(open(OUT / "v_pairs.pkl", "rb")); P = D["pairs"]
PRIMARY = ["qwen_s19", "glm_s19", "dsv_s20", "haiku", "sonnet", "gemini"]
def view(a):
    return "frontal" if a < 45 else "oblique" if a < 75 else "profile" if a <= 105 else "back"
for s in PRIMARY:
    rows = [r for r in P if r["src"] == s]
    print(s, "err>135:", sum(r["err"] is not None and r["err"] > 135 for r in rows),
          "| true right & label towards_right in profile line:", sum(r["true_bin"]=="right" and r["label"]=="towards_right" and r["role"]=="line" and view(r["ang"])=="profile" for r in rows),
          "| any true right label towards_right:", sum(r["true_bin"]=="right" and r["label"]=="towards_right" for r in rows))
for s in [k for k in sorted({r['src'] for r in P}) if k.startswith('s8_')]:
    pr = [r for r in P if r["src"] == s and view(r["ang"]) == "profile" and r["err"] is not None]
    print(s, "profile w45", round(sum(r["err"] <= 45 for r in pr)/len(pr), 3), len(pr))
rows = [r for r in P if r["src"] == "s8_qwen36_35"]
print("qwen36_35 label-only", sum(r["lerr"] <= 45 for r in rows if r["lerr"] is not None), len(rows),
      sorted(round(r["lerr"],3) for r in rows if r["lerr"] is not None and 40 < r["lerr"] < 50))
