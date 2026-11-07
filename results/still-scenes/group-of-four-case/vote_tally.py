"""Zero calls. Would a majority-of-3 vote over independent draws fix the conversation_group_4 target misses?
Each (model, variant, episode) has 3 recorded trials (independent plan draws). Majority-correct is the vote of the 3
target outcomes; also the vote of the 3 click pixels (their median pixel, lifted with the pipeline's own lift)."""
import sys, collections, statistics
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / '_lib' / 'paths.py').exists()) / '_lib'))
import paths  # noqa: E402  (results/_lib/paths.py)
sys.path.insert(0, str(paths.HARNESS))   # research_harness.py and its run list
import research_harness as H
from sgi.geometry import lift
from sgi.evaluate import match_target
wl = H.whitelist()
for stage, model in [('s21', 'qwen38_low'), ('s23', 'glm53'), ('s23', 'dsv41'), ('s23', 'sonnet55_low'),
                     ('s23', 'haiku55'), ('s23', 'gemini38_low')]:
    groups = collections.defaultdict(list)
    for run in H.runs(stage=stage, method='con', model=model, wl=wl):
        for rec in H.records(run, wl=wl):
            if 'conversation_group_4' in rec.scene_id and (rec.result or {}).get('target'):
                groups[(rec.scene_id, rec.episode_id)].append(rec)
    single = sum(bool(r.score.get('target_correct')) for g in groups.values() for r in g)
    n = sum(len(g) for g in groups.values())
    maj = med = 0
    for (scene, ep), g in groups.items():
        oks = [bool(r.score.get('target_correct')) for r in g]
        maj += sum(oks) * 2 > len(oks)
        sc, gt, th, _ = H._against(g[0])
        u = statistics.median(r.result['target']['pixel'][0] for r in g)
        v = statistics.median(r.result['target']['pixel'][1] for r in g)
        w = lift(sc.observation.depth, sc.observation.camera, u, v, th.lift_window)
        med += w is not None and match_target(w, sc.truth, gt, th) is not None
    print(f'{model:13} single draws {single}/{n} = {single / n:.2f}; majority of 3: {maj}/{len(groups)}; median click of 3: {med}/{len(groups)}')
