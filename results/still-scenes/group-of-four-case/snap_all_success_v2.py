"""POST HOC variant (chosen after snap_all_success.py showed 4 breaks): snap only when the depth-lifted point is more than 0.5 m from the clicked person.
Zero calls. Every recorded con record where the detector snap fires (click inside a detected person's box) and moves
the target world point: re-solve with the recorded people and constraints, recorded target vs snapped target, and
compare full success. Records without a solver step (source 'standard') are counted apart."""
import sys, math, collections
from pathlib import Path
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
src = open('results/2026-11-01-push/group4/snap_sweep.py').read().split('tally = collections')[0]
ns = {}
exec(compile(src, 'snap_sweep.py', 'exec'), ns)
DET, snap, wl = ns['DET'], ns['snap'], ns['wl']
tot = collections.defaultdict(collections.Counter)
moved = []
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
            c = tot[model]
            c['records'] += 1
            w = snap(*t["pixel"], DET[folder]["people"])
            if w is not None and math.dist(w[:2], t["world"][:2]) <= 0.5:
                w = None                     # post hoc: keep the depth lift when it agrees with the clicked person
            if w is None:
                continue
            c['snap_fires'] += 1
            d = math.dist(w[:2], t['world'][:2])
            moved.append(d)
            if d < 0.05:
                c['moved<5cm'] += 1
                continue
            try:
                a = H.resolve(rec).score
                b = H.resolve(rec, target=dict(t, world=list(w))).score
            except (SystemExit, ValueError):
                c['no_solver_step'] += 1
                continue
            c['resolved'] += 1
            c['succ_before'] += bool(a.success)
            c['succ_after'] += bool(b.success)
            if a.success and not b.success:
                c['BROKEN'] += 1
                print('BROKEN', model, rec.scene_id, rec.episode_id, rec.trial, f'moved {d:.2f} m', b)
            if b.success and not a.success:
                c['fixed'] += 1
for m, c in tot.items():
    print(m, dict(c))
moved.sort()
print('target moved by the snap (m): median %.2f, 90th %.2f, max %.2f' % (moved[len(moved)//2], moved[int(len(moved)*.9)], moved[-1]))
