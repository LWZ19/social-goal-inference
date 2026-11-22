"""Zero-call split of family-A crowd failures (our method, AI answers): absorbed by the solver, listed by the model, or other."""
import json, re, math, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir() and (p / "src").is_dir())
man = {}
for l in open(ROOT / "results/still-scenes/crowds/build/manifest.jsonl"):
    x = json.loads(l)
    if x["accepted"]:
        man[(x["seed"], x["scene_id"])] = x
latest = {}
for p in sorted((ROOT / "runs").glob("*_crowd_vlm_seed[0-2]_*_con")):
    m = re.match(r"^\d{8}-\d{6}_crowd_vlm_seed(\d)_(.+)_con$", p.name)
    if m and (p / "results.jsonl").exists():
        latest[(m.group(2), int(m.group(1)))] = p
models = ["claude_haiku55", "claude_sonnet55_low", "gemini_fast", "soclaas_dsv41", "soclaas_glm53"]
tab = defaultdict(Counter)
for (model, seed), p in latest.items():
    if model not in models:
        continue
    for line in open(p / "results.jsonl"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        x = man[(seed, r["scene_id"])]
        if x["family"] not in ("A", "base") or not x["base_scene_id"].split("_", 2)[2].startswith("waiting_line"):
            continue
        fam = "A" if x["family"] == "A" else "base"
        key = (fam, x["level"] if fam == "A" else 0)
        res, score = r.get("result") or {}, r.get("score") or {}
        ok = bool(score.get("success"))
        tab[key]["n"] += 1
        tab[key]["ok"] += ok
        if res.get("status") != "ok":
            tab[key]["call_failed"] += 1
            continue
        people = {q["number"]: q["xy"] for q in res.get("people") or []}
        listed = set()
        for c in res.get("constraints") or []:
            if c["constraint"]["kind"] == "come_after" and c["status"] == "used":
                listed |= set(c["constraint"]["people"])
        joined = set((res.get("solver") or {}).get("joined_waiting") or [])
        # which detected number is the added look-alike (nearest within 0.6 m)
        conf = None
        if fam == "A":
            a = x["added"][0]
            best = min(people.items(), key=lambda kv: math.hypot(kv[1][0] - a["x"], kv[1][1] - a["y"]), default=None)
            if best and math.hypot(best[1][0] - a["x"], best[1][1] - a["y"]) < 0.6:
                conf = best[0]
        if fam == "A":
            if conf is None:
                cls = "look-alike not detected"
            elif conf in listed:
                cls = "model listed look-alike as waiting"
            elif conf in joined:
                cls = "solver absorbed look-alike"
            elif not listed:
                cls = "no come_after used"
            else:
                cls = "look-alike kept out"
        else:
            cls = "absorbed someone" if joined else ("no come_after used" if not listed else "plain")
        tab[key][cls] += 1
        tab[key][cls + " & ok"] += ok
for key in sorted(tab):
    c = tab[key]
    print(key, dict(c))
tot = Counter()
for key, c in tab.items():
    if key[0] == "A":
        tot.update(c)
print("A total", dict(tot))
