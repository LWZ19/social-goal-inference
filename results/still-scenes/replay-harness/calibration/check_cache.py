"""Do s8_conorc and s8_convlm share plan/context answers (and so the target) per (scene, episode, trial)?"""
import sys, collections, json
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
OUT = 'results/still-scenes/replay-harness/calibration/check_cache.txt'
MODELS = ['glm53', 'ornith15', 'qwen36_35', 'qwen38', 'qwen3vl']
lines = []
def key(rec): return (rec.scene_id, rec.episode_id, rec.trial)
def stage_text(rec, stage):
    return tuple(c.get('raw_text') for c in rec.result.get('calls') or [] if c['stage'] == stage)
for m in MODELS:
    a = {key(r): r for r in H.records(H.runs(stage='s8', model=m, method='convlm')[0])}
    b = {key(r): r for r in H.records(H.runs(stage='s8', model=m, method='conorc')[0])}
    n = H.runs(stage='s8', model=m, method='none')[0]
    c = {key(r): r for r in H.records(n)}
    assert set(a) == set(b) == set(c), m
    t = collections.Counter()
    for k in a:
        ra, rb, rc = a[k], b[k], c[k]
        t['keys'] += 1
        t['plan same'] += stage_text(ra, 'plan') == stage_text(rb, 'plan')
        t['context same'] += stage_text(ra, 'context') == stage_text(rb, 'context')
        t['locate same'] += stage_text(ra, 'locate') == stage_text(rb, 'locate')
        t['target same'] += json.dumps(ra.target, sort_keys=True) == json.dumps(rb.target, sort_keys=True)
        t['target_correct same'] += ra.score.get('target_correct') == rb.score.get('target_correct')
        t['source same'] += ra.result.get('source') == rb.result.get('source')
        t['both solver steps'] += ra.solver_step and rb.solver_step
        t['plan same as none'] += stage_text(ra, 'plan') == stage_text(rc, 'plan')
        t['context same as none'] += stage_text(ra, 'context') == stage_text(rc, 'context')
        for stage in ('plan', 'context', 'people', 'constraints'):
            for rr, nm in ((ra, 'convlm'), (rb, 'conorc')):
                calls = [x for x in rr.result.get('calls') or [] if x['stage'] == stage]
                t[f'{nm} {stage} calls'] += len(calls)
                t[f'{nm} {stage} cached'] += sum(bool(x.get('cached')) for x in calls)
    lines.append(f'{m}: ' + ', '.join(f'{k} {v}' for k, v in t.items()))
open(OUT, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
