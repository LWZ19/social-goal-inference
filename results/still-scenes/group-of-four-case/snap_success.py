"""Zero calls. For the records whose target the detector snap changes (snap_sweep.py, 'det'), re-run the recorded
constraint solver with the snapped target (H.resolve: recorded people and constraints, new target world) and score
the full success. Also: does H.resolve with the recorded target reproduce the recorded success (sanity)?"""
import sys, json, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / '_lib' / 'paths.py').exists()) / '_lib'))
import paths  # noqa: E402  (results/_lib/paths.py)
sys.path.insert(0, str(paths.HARNESS))   # research_harness.py and its run list
import research_harness as H
from sgi.evaluate import match_target
import importlib.util
spec = importlib.util.spec_from_file_location('ss', Path(__file__).resolve().parent / 'snap_sweep.py')
src = open(spec.origin).read().split('tally = collections')[0]      # definitions only, not the sweep
ns = {'__file__': spec.origin}                       # snap_sweep.py finds results/_lib from its own path
exec(compile(src, spec.origin, 'exec'), ns)
DET, snap, wl = ns['DET'], ns['snap'], ns['wl']
tot = collections.Counter()
for stage, model in [('s21', 'qwen38_low'), ('s23', 'glm53'), ('s23', 'dsv41'), ('s23', 'sonnet55_low'),
                     ('s23', 'haiku55'), ('s23', 'gemini38_low')]:
    for run in H.runs(stage=stage, method='con', model=model, wl=wl):
        for rec in H.records(run, wl=wl):
            t = (rec.result or {}).get('target')
            if not t or t.get('pixel') is None or t.get('world') is None:
                continue
            folder = str(Path(H.check_scene(rec.ref, rec.scene_id, wl=wl)).resolve())
            if folder not in DET:
                continue
            w = snap(*t['pixel'], DET[folder]['people'])
            if w is None:
                continue
            sc, gt, th, _ = H._against(rec)
            if (match_target(w, sc.truth, gt, th) is not None) == bool(rec.score.get('target_correct')):
                continue
            new_t = dict(t, world=list(w))
            try:
                same = H.resolve(rec)
                again = H.resolve(rec, target=new_t)
            except (SystemExit, ValueError) as e:
                print(model, rec.scene_id, rec.episode_id, rec.trial, "source", rec.result.get("source"), "resolve refused:", e)
                tot[(model, 'refused')] += 1
                continue
            s = again.score
            print(f'{model:13} {rec.scene_id:38} {rec.episode_id} t{rec.trial} source={rec.result.get("source")} '
                  f'recorded={rec.score.get("success")} replay={same.score.success} '
                  f'snapped: success={s.success} target={s.target_correct} region={s.in_region} heading={s.heading_ok} '
                  f'reach={s.reachable} dist={s.distance_to_region_m}')
            tot[(model, 'changed')] += 1
            tot[(model, 'success_after')] += bool(s.success)
            tot[(model, 'replay_ok')] += bool(same.score.success) == bool(rec.score.get('success'))
for k in sorted(tot):
    print(k, tot[k])
