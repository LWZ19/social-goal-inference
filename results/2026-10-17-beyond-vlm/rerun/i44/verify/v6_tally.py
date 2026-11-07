import sys, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
refs = [r for r in H.runs(stage='s8') if r.model in ('qwen36_35', 'qwen38')] + H.runs(stage='s10', model='qwen38_low') + H.runs(stage='s16b', arm='ph', model='qwen38_low') + H.runs(stage='s19', arm='ph', model='qwen38_low')
for r in refs:
    t = collections.Counter(); n = sum(1 for _ in H.records(r, tally=t))
    print(r.name, dict(t))
