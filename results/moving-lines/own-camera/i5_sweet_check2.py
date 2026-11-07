"""Item 5, POST HOC follow-up: re-solve outcomes by scene and arm (status counts; ok re-solves with no person to come
after), over the 4 models' paired cells in runs_i5. Zero calls; read-only."""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mcompare as C  # noqa: E402

MODELS = {"sonnet55_low": range(10, 20), "dsv41": range(10, 15), "claude_haiku55": range(10, 15), "gemini_fast": range(10, 15)}
st = defaultdict(Counter)
for d, seeds in MODELS.items():
    eps = C.load(f"runs_i5/{d}", C.EVAL5, seeds, ("CTd", "CT2"))
    for (sc, s, arm), e in eps.items():
        if e is None:
            continue
        for r in e.get("resolves", []):
            k = r.get("status")
            if k == "ok" and r.get("last") is None and not r.get("vacated"):
                k = "ok, nobody to follow"
            st[(sc, arm)][k] += 1
for sc in C.EVAL5:
    for arm in ("CT2", "CTd"):
        c = st[(sc, arm)]
        n = sum(c.values())
        print(f"{sc:26s} {arm}: {n} re-solves; " + ", ".join(f"{k} {v} ({v / max(n, 1):.2f})" for k, v in c.most_common()))
