"""Fresh crowd test of the joining gate, zero calls: on seed 3, good spots for pointing with worked
examples (fsc), our method as recorded (now) and our method's same answers re-solved with the gate (gate).
    PYTHONPATH=src python results/still-scenes/crowds/fresh-seed3/fresh_score.py"""
import collections
import json
import re
import sys
from math import comb
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir() and (p / "src").is_dir())
HARNESS = REPO / "results/still-scenes/analysis-scripts/harness"
sys.path.insert(0, str(HARNESS))
from replay_solver import _scenes_folder, solve_again  # noqa: E402
from sgi.evaluate import score  # noqa: E402
from sgi.gt.derive import prepare  # noqa: E402
from sgi.rescore import _recorded_config, _records  # noqa: E402
from sgi.scene import load_scene  # noqa: E402
from sgi.types import GoalResult  # noqa: E402

MODELS = {"claude_haiku55": "Claude Haiku 5.5", "claude_sonnet55_low": "Claude Sonnet 5.5",
          "gemini_fast": "Gemini 3.8 Flash", "soclaas_dsv41": "DeepSeek V4.1"}
GATE = {"join_facing_deg": 90.0, "join_spacing_ratio": 1.5}


def family(scene_id):
    m = re.match(r"^i06_s\d+_([A-E])(\d)_", scene_id)
    return (m.group(1), int(m.group(2))) if m else ("base", 0)


def latest(model, arm):
    runs = sorted((REPO / "runs").glob(f"*_crowd_vlm_seed3_{model}_{arm}"))
    return runs[-1] if runs and (runs[-1] / "results.jsonl").exists() else None


def records(path):
    out = []
    for line in open(path / "results.jsonl"):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def sign_p(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n)


rows = {}                                          # (model, scene, episode, trial) -> {arm: ok}
for model in MODELS:
    fsc, con = latest(model, "fsc"), latest(model, "con")
    if fsc is None or con is None:
        print(f"{model}: run missing (fsc {fsc is not None}, con {con is not None})")
        continue
    for r in records(fsc):
        rows.setdefault((model, r["scene_id"], r["episode_id"], r.get("trial")), {})["fsc"] = \
            bool((r.get("score") or {}).get("success"))
    doc, cfg = _recorded_config(con / "config.yaml")
    th = cfg.thresholds
    gated = type(th).model_validate({**th.model_dump(), **GATE})
    folder, loaded = _scenes_folder(con, doc, None), {}
    for r in records(con):
        key = (model, r["scene_id"], r["episode_id"], r.get("trial"))
        recorded = bool((r.get("score") or {}).get("success"))
        result = GoalResult.model_validate(r["result"])
        ok = {"now": recorded, "gate": recorded}
        if (result.source == "constraints" and result.pose_world is not None and result.people is not None
                and result.constraints is not None):
            if r["scene_id"] not in loaded:
                inputs, truth = load_scene(folder / r["scene_id"])
                loaded[r["scene_id"]] = (inputs.observation, truth, prepare(inputs, truth, cfg))
            observation, truth, truths = loaded[r["scene_id"]]
            gt = truths[r["episode_id"]].gt
            for arm, t in (("now", th), ("gate", gated)):
                _, again = solve_again(result, observation, t)
                ok[arm] = bool(score(again, observation, truth, gt, t).success)
            ok["recorded"] = recorded
        rows.setdefault(key, {}).update(ok)

done = sorted({k[0] for k, v in rows.items() if "now" in v and "fsc" in v})
print(f"models with both arms: {', '.join(MODELS[m] for m in done)}")
paired = {k: v for k, v in rows.items() if "now" in v and "fsc" in v}
redo = [v for v in paired.values() if "recorded" in v and v["recorded"] != v["now"]]
print(f"records paired: {len(paired)}; re-solving without the gate differs from the recorded success in {len(redo)}")
for name, keep in (("base (no one added)", lambda f: f[0] == "base"),
                   ("A, all levels", lambda f: f[0] == "A"),
                   ("A, 1.3 m and more (A1-A3)", lambda f: f[0] == "A" and f[1] <= 3),
                   ("A, 1.0 m and less (A4-A6)", lambda f: f[0] == "A" and f[1] >= 4)):
    sel = {k: v for k, v in paired.items() if keep(family(k[1]))}
    if not sel:
        continue
    n = len(sel)
    count = {arm: sum(v[arm] for v in sel.values()) for arm in ("fsc", "now", "gate")}
    gained = sum(v["gate"] and not v["now"] for v in sel.values())
    lost = sum(v["now"] and not v["gate"] for v in sel.values())
    print(f"\n## {name}: {n} tries")
    print("  " + ", ".join(f"{arm} {count[arm]} ({100 * count[arm] / n:.1f}%)" for arm in ("fsc", "now", "gate")))
    print(f"  gate against now, by try: gained {gained}, lost {lost}")
    for a, b in (("gate", "now"), ("gate", "fsc"), ("now", "fsc")):
        by_scene = collections.Counter()
        for k, v in sel.items():
            by_scene[k[1]] += v[a] - v[b]
        w, l = sum(d > 0 for d in by_scene.values()), sum(d < 0 for d in by_scene.values())
        print(f"  {a} minus {b}, by scene: better {w}, worse {l}, level {len(by_scene) - w - l}, sign p {sign_p(w, l):.4f}")
    per = collections.defaultdict(lambda: collections.Counter())
    for k, v in sel.items():
        for arm in ("fsc", "now", "gate"):
            per[k[0]][arm] += v[arm]
        per[k[0]]["n"] += 1
    print("  per model: " + "; ".join(f"{MODELS[m]} fsc {c['fsc']}, now {c['now']}, gate {c['gate']} of {c['n']}"
                                      for m, c in sorted(per.items())))
