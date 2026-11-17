"""Search for the pairing that gives 4 wins and 9 losses (qwen3.6:35b against a Qwen 3.8 run). Exploratory."""
import sys, collections, itertools
from pathlib import Path
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
OUT = Path(__file__).resolve().parent / 'search_q2.txt'
L = []
def out(*a):
    s = ' '.join(map(str, a)); print(s); L.append(s)
A_runs = [r for r in H.runs(stage='s8', model='qwen36_35')]
B_runs = [r for r in H.runs(stage='s8', model='qwen38')]
B_runs += [r for st in ('s10', 's16b', 's19') for r in H.runs(stage=st) if r.model.startswith('qwen38') and (st == 's10' or r.arm == 'ph')]
R = {r.folder: list(H.records(r)) for r in A_runs + B_runs}
FIELDS = ['success', 'target_correct', 'in_region', 'heading_ok', 'reachable']
def val(x, f): return bool(x.score.get(f))
hits = []
for a in A_runs:
    for b in B_runs:
        for f in FIELDS:
            for sit in [None] + sorted({x.situation for x in R[a.folder]}):
                for keying in ('trial', 'last', 'first', 'episode_majority', 'episode_count', 'scene_rate', 'scene_trial_rate'):
                    def mk(recs):
                        rs = [x for x in recs if sit is None or x.situation == sit]
                        if keying == 'trial': return {(x.scene_id, x.episode_id, x.trial): val(x, f) for x in rs}
                        if keying == 'last': return {(x.scene_id, x.episode_id): val(x, f) for x in rs}
                        if keying == 'first':
                            d = {}
                            for x in rs: d.setdefault((x.scene_id, x.episode_id), val(x, f))
                            return d
                        c = collections.defaultdict(list)
                        if keying == 'scene_rate':
                            for x in rs: c[x.scene_id].append(val(x, f))
                            return {k: sum(v) / len(v) for k, v in c.items()}
                        if keying == 'scene_trial_rate':
                            for x in rs: c[(x.scene_id, x.trial)].append(val(x, f))
                            return {k: sum(v) / len(v) for k, v in c.items()}
                        for x in rs: c[(x.scene_id, x.episode_id)].append(val(x, f))
                        if keying == 'episode_majority': return {k: sum(v) * 2 > len(v) for k, v in c.items()}
                        return {k: sum(v) / len(v) for k, v in c.items()}
                    A, B = mk(R[a.folder]), mk(R[b.folder])
                    ks = set(A) & set(B)
                    w = sum(1 for k in ks if A[k] > B[k]); l = sum(1 for k in ks if A[k] < B[k])
                    if (w, l) in ((4, 9), (9, 4)):
                        hits.append((a.name, b.name, f, sit, keying, len(ks), w, l))
                    if f == 'success' and sit is None:
                        out(f'{a.name:<22} {b.name:<26} {keying:<17} n {len(ks):>3} wins {w:>2} losses {l:>2}')
out('\nHITS (4/9 or 9/4):')
for h in hits: out(h)
OUT.write_text('\n'.join(L) + '\n')
