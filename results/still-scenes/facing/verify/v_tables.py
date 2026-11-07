import collections
import pickle
import random
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).parent
D = pickle.load(open(OUT / "v_pairs.pkl", "rb"))
P, K = D["pairs"], D["keys"]
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


PRIMARY = ["qwen_s19", "glm_s19", "dsv_s20", "haiku", "sonnet", "gemini"]


def stats(rows, field="err"):
    known = [r[field] for r in rows if r[field] is not None]
    n = len(rows)
    if not n:
        return None
    w = sum(e <= 45 for e in known)
    return dict(n=n, known=len(known), w45=w / len(known) if known else float("nan"),
                w45_miss=w / n, med=statistics.median(known) if known else float("nan"),
                unk=(n - len(known)) / n,
                people=len({(r["layout"], r["pid"]) for r in rows}), layouts=len({r["layout"] for r in rows}))


def fmt(s):
    if s is None:
        return "-"
    return (f"w45 {s['w45']:.3f} (miss {s['w45_miss']:.3f}) med {s['med']:.1f} unk {s['unk']:.3f} n {s['n']} "
            f"(ppl {s['people']}, lay {s['layouts']})")


say("== integrity per source ==")
for src, ks in K.items():
    nraw = collections.Counter(k["nraw"] for k in ks)
    unl = sum(k["n_unlifted"] for k in ks)
    ent = sum(k["n_entries_raw"] for k in ks)
    listed = sum(k["nlisted"] for k in ks)
    tied = sum(k["tied"] for k in ks)
    vis = sum(k["nvisible"] for k in ks)
    mat = sum(k["nmatched"] for k in ks)
    status = collections.Counter(s for k in ks for s, _ in k["raw"])
    nsits = collections.Counter(len(k["sits"]) for k in ks)
    over40 = sum(k["nlisted"] >= 40 for k in ks)
    say(f"{src}: answers {len(ks)} distinct-raw-per-key {dict(nraw)} status {dict(status)} raw entries {ent} "
        f"unlifted {unl} listed {listed} tied {tied} visible {vis} matched {mat} recall {mat / vis if vis else 0:.3f} "
        f"unmatched {listed - mat} sits/key {dict(nsits)} keys>=40 {over40}")

say("\n== copy checks ==")


def rawmap(src):
    return {(k["arm"], k["scene"], k["trial"]): tuple(k["raw"]) for k in K[src]}


for a, b in [("qwen_s21", "qwen_s19"), ("glm_s23", "glm_s19"), ("dsv_s23", "dsv_s20")]:
    ma, mb = rawmap(a), rawmap(b)
    shared = set(ma) & set(mb)
    same = sum(ma[k] == mb[k] for k in shared)
    say(f"{a} vs {b}: {a} {len(ma)}, {b} {len(mb)}, shared {len(shared)}, identical {same}, only {a} {len(set(ma) - set(mb))}, only {b} {len(set(mb) - set(ma))}")

say("\n== overall ==")
for src in PRIMARY + ["qwen_s21", "glm_s23", "dsv_s23"] + [s for s in K if s.startswith("s8_")]:
    rows = [r for r in P if r["src"] == src]
    gaps = [r["gap"] for r in rows]
    say(f"{src}: {fmt(stats(rows))} median gap {statistics.median(gaps):.3f}")

say("\n== bootstrap over layouts (2000, seed 0), within 45 of known ==")
for src in PRIMARY:
    rows = [r for r in P if r["src"] == src and r["err"] is not None]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["layout"]].append(r["err"] <= 45)
    lays = sorted(by)
    rng = random.Random(0)
    vals = []
    for _ in range(2000):
        pick = [rng.choice(lays) for _ in lays]
        hits = sum(sum(by[l]) for l in pick)
        n = sum(len(by[l]) for l in pick)
        vals.append(hits / n)
    vals.sort()
    say(f"{src}: [{vals[49]:.3f}, {vals[1949]:.3f}] layouts {len(lays)}")

say("\n== per situation ==")
for src in PRIMARY:
    for sit in ["waiting_line", "conversation_group", "no_one"]:
        rows = [r for r in P if r["src"] == src and r["sit"] == sit]
        say(f"{src} {sit}: {fmt(stats(rows))}")

say("\n== per role ==")
for src in PRIMARY:
    for role in ["line", "group", "staff", "other"]:
        rows = [r for r in P if r["src"] == src and r["role"] == role]
        say(f"{src} {role}: {fmt(stats(rows))}")


def view(a):
    if a < 45:
        return "frontal"
    if a < 75:
        return "oblique"
    if a <= 105:
        return "profile"
    return "back"


say("\n== per view ==")
for src in PRIMARY:
    for v in ["frontal", "oblique", "profile", "back"]:
        rows = [r for r in P if r["src"] == src and view(r["ang"]) == v]
        say(f"{src} {v}: {fmt(stats(rows))}")

say("\n== I09 bars applied to VLMs ==")
for src in PRIMARY:
    np_rows = [r for r in P if r["src"] == src and view(r["ang"]) != "profile"]
    pr = [r for r in P if r["src"] == src and view(r["ang"]) == "profile" and r["err"] is not None]
    fb = sum(r["err"] < 90 for r in pr) / len(pr)
    say(f"{src}: non-profile {fmt(stats(np_rows))}; profile front/back (err<90) {fb:.3f} n {len(pr)}")

say("\n== common support ==")
inst = collections.defaultdict(set)
for r in P:
    if r["src"] in PRIMARY:
        inst[(r["arm"], r["scene"], r["pid"])].add(r["src"])
common = {k for k, v in inst.items() if len(v) == len(PRIMARY)}
say(f"common instances {len(common)} distinct people {len({(k[2], k[1].split('_', 1)[1]) for k in common})}")
for src in PRIMARY:
    rows = [r for r in P if r["src"] == src and (r["arm"], r["scene"], r["pid"]) in common]
    say(f"{src}: {fmt(stats(rows))}")

say("\n== label-only and faced point ==")
for src in PRIMARY + [s for s in K if s.startswith("s8_")]:
    rows = [r for r in P if r["src"] == src]
    fp = [r for r in rows if r["used_fp"]]
    s_fp = stats(fp)
    s_fp_l = stats(fp, "lerr")
    s_l = stats(rows, "lerr")
    say(f"{src}: fp share {len(fp) / len(rows):.3f}; on fp pairs pipeline w45 {s_fp['w45']:.3f} label {s_fp_l['w45']:.3f} "
        f"(n {len(fp)}); label-only overall {fmt(s_l)}; no label tied {sum(r['label'] is None for r in rows)}")

say("\n== confusion: true bin vs label ==")
from sgi.schemas import FACINGS  # noqa: E402
for src in PRIMARY:
    rows = [r for r in P if r["src"] == src]
    say(f"-- {src}")
    for tb in FACINGS[:8]:
        rr = [r for r in rows if r["true_bin"] == tb]
        if not rr:
            continue
        c = collections.Counter(r["label"] for r in rr)
        say(f"  {tb:17s} n {len(rr):3d} diag {c[tb] / len(rr):.2f} " + ", ".join(f"{k}:{v}" for k, v in c.most_common()))

say("\n== I09 bar ==")
best = max(PRIMARY, key=lambda s: stats([r for r in P if r["src"] == s])["w45"])
bw = stats([r for r in P if r["src"] == best])["w45"]
say(f"best {best} {bw:.3f}; bar = max(0.70, best-0.10) = {max(0.70, bw - 0.10):.3f}")
school = ["qwen_s19", "glm_s19", "dsv_s20"]
bs = max(school, key=lambda s: stats([r for r in P if r["src"] == s])["w45"])
sw = stats([r for r in P if r["src"] == bs])["w45"]
say(f"school best {bs} {sw:.3f}; bar {max(0.70, sw - 0.10):.3f}")

(OUT / "v_tables.txt").write_text("\n".join(lines) + "\n")
