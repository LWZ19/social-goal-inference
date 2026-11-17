"""Verifier: compare every number printed in followup/exposure.txt with the verifier's own v_followup.json, at the
printed precision (the verifier's value formatted the same way must give the same string). Parses text only;
imports nothing from exposure.py."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TXT = (HERE.parent / "exposure.txt").read_text().splitlines()
V = json.load(open(HERE / "v_followup.json"))
CELLS = ["old_qwen", "old_glm", "old_deepseek", "pooled"]
LAB = {"P1 re-solve shift (judged)": "P1", "S2 people 1-F1": "S2", "S3 named unsupported": "S3",
       "S4 trial disagreement (2 runs)": "S4", "S6z frozen (fit on 441)": "S6z_frozen",
       "S6z refit (fit on 404)": "S6z_refit", "S5a any flag (ref.)": "S5a"}
NUM = r"[+-]?\d+\.\d+"
n_cmp = 0
diffs = []


def fmt(x, d, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s


def cmp(where, theirs, mine, d=3, sign=False):
    global n_cmp
    n_cmp += 1
    if theirs == "—":
        ok = mine is None
        m = "—" if mine is None else fmt(mine, d, sign)
    else:
        m = fmt(mine, d, sign) if mine is not None else "—"
        ok = (m == theirs) or (m.lstrip("+") == theirs.lstrip("+"))
    if not ok:
        diffs.append((where, theirs, m, mine))


def section(title):
    i = next(k for k, l in enumerate(TXT) if l.startswith(title))
    out = []
    for l in TXT[i + 1:]:
        if l.startswith("Table ") or l.startswith("Reading"):
            break
        out.append(l)
    return out


def label_of(line):
    for lab in sorted(LAB, key=len, reverse=True):
        if line.strip().startswith(lab):
            return lab
    return None


# Tables A and B
for tab, k in (("Table A.", "a"), ("Table B.", "b")):
    for l in section(tab):
        lab = label_of(l)
        if lab:
            s = LAB[lab]
            body = l.strip()[len(lab):]
            cells = re.findall(rf"({NUM}) \[({NUM}),({NUM})\] d(\d+)", body)
            assert len(cells) == 4, l
            for c, (auc, lo, hi, d) in zip(CELLS, cells):
                cmp(f"{tab}{s}/{c}/auc", auc, V["auc"][k][s][c])
                cmp(f"{tab}{s}/{c}/lo", lo, V["boot"][k][s][c]["ci"][0])
                cmp(f"{tab}{s}/{c}/hi", hi, V["boot"][k][s][c]["ci"][1])
                if int(d) != V["boot"][k][s][c]["dropped"]:
                    diffs.append((f"{tab}{s}/{c}/dropped", d, V["boot"][k][s][c]["dropped"], None))
                n_cmp += 1
            strat = re.findall(NUM, body)[-1]
            cmp(f"{tab}{s}/stratified", strat, V["auc"][k][s]["stratified"])
        elif l.strip().startswith("model identity (ref.)"):
            nums = re.findall(NUM, l)
            cmp(f"{tab}model_identity/pooled", nums[3], V["model_identity"][k])

# Table C
for l in section("Table C."):
    lab = label_of(l)
    if lab:
        s = LAB[lab]
        toks = l.strip()[len(lab):].split()
        assert len(toks) == 5, l
        for c, t in zip(CELLS, toks[:4]):
            cmp(f"C.{s}/{c}", t, V["auc"]["c"][s][c])
        cmp(f"C.{s}/stratified", toks[4], V["auc"]["c"][s]["stratified"])

# Table D
for l in section("Table D."):
    lab = label_of(l)
    if lab:
        s = LAB[lab]
        nums = re.findall(NUM, l.strip()[len(lab):])
        a, b, dd, sa, sb, sd = nums
        cmp(f"D.{s}/pooled_a", a, V["auc"]["a"][s]["pooled"])
        cmp(f"D.{s}/pooled_b", b, V["auc"]["b"][s]["pooled"])
        cmp(f"D.{s}/pooled_delta", dd, V["auc"]["b"][s]["pooled"] - V["auc"]["a"][s]["pooled"], sign=True)
        cmp(f"D.{s}/strat_a", sa, V["auc"]["a"][s]["stratified"])
        cmp(f"D.{s}/strat_b", sb, V["auc"]["b"][s]["stratified"])
        cmp(f"D.{s}/strat_delta", sd, V["auc"]["b"][s]["stratified"] - V["auc"]["a"][s]["stratified"], sign=True)

# Table E
for l in section("Table E."):
    lab = label_of(l)
    if lab:
        s = LAB[lab]
        nums = re.findall(NUM, l.strip()[len(lab):])
        for c, t in zip(CELLS, nums):
            cmp(f"E.{s}/{c}", t, V["share_gt_half"]["b"][s][c])

# Table F
for l in section("Table F."):
    if " - " not in l:
        continue
    left, right = l.strip().split(" - ", 1)
    la = label_of(left)
    lb = label_of(right)
    if not (la and lb):
        continue
    key = f"{LAB[la]}-{LAB[lb]}"
    nums = re.findall(NUM, right[len(lb):])
    pt, lo, hi, sh = nums
    e = V["paired"]["b"][key]
    cmp(f"F.{key}/point", pt, e["point"], sign=True)
    cmp(f"F.{key}/lo", lo, e["ci"][0], sign=True)
    cmp(f"F.{key}/hi", hi, e["ci"][1], sign=True)
    cmp(f"F.{key}/share", sh, e["share_gt0"])

# Table G
for l in section("Table G."):
    lab = label_of(l)
    st = l.strip()
    if lab:
        s = LAB[lab]
        ents = re.findall(rf"({NUM}) \(({NUM})\) \| ({NUM})", st[len(lab):])
        assert len(ents) == 4, l
        for c, (acc, d, cau) in zip(CELLS, ents):
            e = V["selective80"]["b"][s][c]
            cmp(f"G.{s}/{c}/acc", acc, e["acc"])
            cmp(f"G.{s}/{c}/diff", d, e["acc"] - e["base"], sign=True)
            cmp(f"G.{s}/{c}/caught", cau, e["caught"], d=1)
    elif st.startswith("base rate"):
        ents = re.findall(rf"({NUM}) \| ({NUM})/(\d+)", st)
        for c, (b, r, f) in zip(CELLS, ents):
            e = V["selective80"]["b"]["P1"][c]
            cmp(f"G.base/{c}", b, e["base"])
            cmp(f"G.random/{c}", r, e["random"], d=1)
            if int(f) != e["failures"]:
                diffs.append((f"G.failures/{c}", f, e["failures"], None))
            n_cmp += 1

# Table H
for l in section("Table H."):
    lab = label_of(l)
    if lab:
        s = LAB[lab]
        nums = re.findall(NUM, l.strip()[len(lab):])
        assert len(nums) == 8, l
        for c, t in zip(CELLS, nums[:4]):
            cmp(f"H.{s}/rt/{c}", t, V["targetsplit_b"][s][f"rt/{c}"])
        for c, t in zip(CELLS, nums[4:]):
            cmp(f"H.{s}/wt/{c}", t, V["targetsplit_b"][s][f"wt/{c}"])

# Table I
for l in section("Table I."):
    st = l.strip()
    if st.startswith("fit on 441") or st.startswith("fit on 404"):
        name = "frozen_441" if "441" in st[:10] else "refit_404"
        nums = re.findall(NUM, st)
        b, w1, w2, w3, w4 = nums[:5]
        e = V["coef"][name]
        cmp(f"I.{name}/b", b, e["b"], sign=True)
        for j, t in enumerate((w1, w2, w3, w4)):
            cmp(f"I.{name}/w{j + 1}", t, e["w"][j], sign=True)

print(f"compared {n_cmp} printed numbers of exposure.txt against the verifier's own values")
print(f"differences at printed precision: {len(diffs)}")
for d in diffs:
    print("  ", d)
