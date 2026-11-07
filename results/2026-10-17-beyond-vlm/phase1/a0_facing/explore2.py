"""A0 exploration 2: raw people answers, facing labels, which records carry a people call."""
import collections, json, sys, hashlib
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H

REAL = ('real', 'staffed', 'plain')
shown = set()
for st, model in (('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41'), ('s21', 'qwen38_low'), ('s23', 'glm53'),
                  ('s23', 'dsv41'), ('s23', 'haiku55'), ('s23', 'sonnet55_low'), ('s23', 'gemini38_low')):
    labels = collections.Counter(); keys = collections.Counter(); per_key_texts = collections.defaultdict(set)
    sit_pc = collections.Counter(); sit_people = collections.Counter(); faced = collections.Counter()
    status = collections.Counter(); eff = collections.Counter()
    for ref in H.runs(stage=st, model=model, method='con'):
        if ref.arm not in REAL:
            continue
        for rec in H.records(ref):
            pcs = [c for c in rec.result.get('calls') or [] if c['stage'] == 'people']
            sit_pc[(rec.situation, len(pcs))] += 1
            sit_people[(rec.situation, rec.people is not None)] += 1
            for c in pcs:
                status[c['status']] += 1
                eff[(c['model'], c['reasoning_effort'], c['point_format'], tuple(c['image_size'] or ()))] += 1
                per_key_texts[(ref.arm, rec.scene_id, rec.trial)].add(c['raw_text'])
                if (st, model) not in shown:
                    shown.add((st, model)); print('=====', st, model, rec.scene_id, rec.episode_id, c['raw_text'][:1500])
    for k, texts in per_key_texts.items():
        keys[len(texts)] += 1
        for t in texts:
            try:
                d = json.loads(t)
            except Exception as e:
                labels['<unparsable>'] += 1; continue
            for p in d.get('people', []):
                labels[p.get('facing')] += 1
                faced[p.get('faced_point') is not None] += 1
    # identical across trials?
    by_scene = collections.defaultdict(list)
    for (arm, sc, tr), texts in per_key_texts.items():
        by_scene[(arm, sc)].append(hashlib.md5(''.join(sorted(texts)).encode()).hexdigest())
    same = sum(len(set(v)) == 1 and len(v) > 1 for v in by_scene.values())
    print(st, model, 'answers per (arm,scene,trial) distinct-text counts:', dict(keys), 'n keys', len(per_key_texts),
          '| scenes with identical text over trials:', same, 'of', len(by_scene))
    print('   labels', dict(labels), 'faced_point given', dict(faced))
    print('   status', dict(status), 'entry', dict(eff))
    print('   situation x n people calls', dict(sit_pc))
    print('   situation x people list present', dict(sit_people))
