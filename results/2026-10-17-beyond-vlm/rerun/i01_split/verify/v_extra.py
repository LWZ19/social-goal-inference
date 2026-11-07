"""Extra checks: no_one pose identity across methods; unreachable in-region gap values; Gemini wrong targets by arm."""
import sys, collections, io
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
OUT = 'results/2026-10-17-beyond-vlm/rerun/i01_split/verify/v_extra_out.txt'
buf = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); buf.write(s + '\n')
REAL = ('real', 'staffed', 'plain')
M = {'R': 'cot', 'E': 'fsc', 'C': 'con'}
NEW = [('qwen38_low', 's21'), ('glm53', 's23'), ('dsv41', 's23'), ('haiku55', 's23'), ('sonnet55_low', 's23'), ('gemini38_low', 's23')]
def load(stage, model, method):
    out = []
    for arm in REAL:
        (r,) = H.runs(stage=stage, arm=arm, model=model, method=method)
        out += list(H.records(r))
    return out
key = lambda r: (r.arm, r.scene_id, r.episode_id, r.trial)
for model, st in NEW:
    d = {m: {key(r): r for r in load(st, model, meth)} for m, meth in M.items()}
    src = collections.Counter((m, r.result.get('source')) for m in 'REC' for r in d[m].values() if r.situation == 'no_one')
    same_pose = sum(1 for k, r in d['R'].items() if r.situation == 'no_one' and
                    all(d[m][k].pose_world is not None and r.pose_world is not None and
                        max(abs(a - b) for a, b in zip(d[m][k].pose_world, r.pose_world)) < 1e-6 for m in 'EC'))
    same_succ = sum(1 for k, r in d['R'].items() if r.situation == 'no_one' and len({bool(d[m][k].score['success']) for m in 'REC'}) == 1)
    p(f'{model}: no_one sources {dict(src)}; R/E/C pose identical {same_pose}/108; success identical {same_succ}/108')
    for m in 'REC':
        g = sorted(round(r.score['min_body_gap_m'], 3) for r in d[m].values()
                   if r.score['target_correct'] and r.score['in_region'] and not r.score['reachable'] and r.score.get('min_body_gap_m') is not None)
        strip = sum(-0.05 <= x <= 0 for x in g)
        p(f'   {m} in-region unreachable gaps: {g}  strip(-0.05..0]={strip}')
    if model == 'gemini38_low':
        c = collections.Counter((r.arm, r.scene_id.split('_', 1)[1]) for r in d['R'].values() if not r.score['target_correct'])
        p('   gemini wrong by arm,scene', sorted(c.items()))
open(OUT, 'w').write(buf.getvalue())
