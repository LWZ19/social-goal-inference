"""X3 post hoc (reported): where CTe's first-answer delay comes from, and quality measured only after each arm's own
delivery (strict share from the first service after delivery), DeepSeek and Qwen. Zero calls. Run from moving/."""
import json, glob, statistics, sys
sys.path.insert(0, "sweep")
from score_sweep import episode_scores
for root in ("runs_x3/dsv41", "runs_x3/soclaas_qwen38_low"):
    by = {}
    for f in glob.glob(f"{root}/**/episode.json", recursive=True):
        e = json.load(open(f)); p = f.split("/")
        by.setdefault((p[-4], p[-3]), {})[p[-2]] = e
    wall = {a: [] for a in ("CTd", "CTe")}; after = []
    for k, v in by.items():
        for a in ("CTd", "CTe"):
            ans = (v.get(a) or {}).get("answers") or []
            if ans and ans[0].get("wall_s") is not None:
                wall[a].append(ans[0]["wall_s"])
        if "CTd" in v and "CTe" in v:
            x, y = episode_scores(v["CTe"])["after_service_strict"], episode_scores(v["CTd"])["after_service_strict"]
            if x is not None and y is not None:
                after.append(x - y)
    print(root, "| first-answer wall time median s: CTd", round(statistics.median(wall["CTd"]), 2) if wall["CTd"] else None,
          "(a cache replay of CT2's stored answer)", "CTe", round(statistics.median(wall["CTe"]), 1) if wall["CTe"] else None, "(live)")
    w, l = sum(d >= 0.01 for d in after), sum(d <= -0.01 for d in after)
    print(f"  strict from the first service after delivery, CTe - CTd: n {len(after)}, W/L/T {w}/{l}/{len(after)-w-l}, "
          f"mean {statistics.mean(after) if after else float('nan'):+.3f}")
