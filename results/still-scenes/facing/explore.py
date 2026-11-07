"""A0 exploration: which C runs exist on the realistic arms, what their people calls hold."""
import collections, json, sys
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H

REAL = ('real', 'staffed', 'plain')
for st in ('s16b', 's19', 's20', 's21', 's22', 's23'):
    for ref in H.runs(stage=st):
        if ref.arm in REAL:
            print(st, ref.arm, ref.model, ref.method, ref.folder)
print()
for st, model in (('s19', None), ('s20', None), ('s21', None), ('s23', None)):
    for ref in H.runs(stage=st, method='con'):
        if ref.arm not in REAL:
            continue
        n = 0; npc = 0; cached = 0; sits = collections.Counter(); trials = collections.Counter(); raw_ok = 0
        scenes = set()
        for rec in H.records(ref):
            n += 1; sits[rec.situation] += 1; trials[rec.trial] += 1; scenes.add(rec.scene_id)
            for c in rec.result.get('calls') or []:
                if c['stage'] == 'people':
                    npc += 1; cached += bool(c.get('cached')); raw_ok += c.get('raw_text') is not None
        print(st, ref.arm, ref.model, 'recs', n, 'scenes', len(scenes), 'peoplecalls', npc, 'cached', cached, 'raw', raw_ok,
              dict(trials), dict(sits))
