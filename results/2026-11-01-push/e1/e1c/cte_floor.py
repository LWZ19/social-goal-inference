"""Post hoc, zero calls: (1) the direct redraw floor from the Sonnet fresh repeats (CT, CT2 on seeds 10-12, same cached
first answers; only the sim's timing differs) against CTe vs CTd; (2) how often the VLM-people first answers (C arms,
seeds 10-18) contain keep_out_of_shared_space, against CTe's detector-people answers."""
import itertools, json, re, statistics, sys
from pathlib import Path
M = Path('results/2026-11-01-push/moving')
sys.path.insert(0, str(M))
import summarise_fresh as S
A, B = M / 'runs/sonnet55_low', M / 'runs/sonnet55_low_fresh_repeat'

def get(root):
    out = {}
    for e in S.load([root]):
        if e['scenario'] in S.BREAK: continue
        out[(e['scenario'], e['seed'], e['arm'])] = (S.measures(e), e)
    return out
a, b = get(A), get(B)

def flip_p(d):
    obs = abs(statistics.mean(d)); n = len(d)
    return sum(abs(sum(s * x for s, x in zip(sg, d)) / n) >= obs - 1e-12 for sg in itertools.product((1, -1), repeat=n)) / 2 ** n

def tot(ms):
    return (f"share {statistics.mean(m['share_strict'] for m in ms):.3f} end {sum(m['end_strict'] for m in ms)} "
            f"cut {sum(m['cut'] for m in ms)} back {sum(m['back10'] for m in ms)}/{sum(m['events'] for m in ms)}")
print('(1) Redraw floor: original vs repeat, same cached first answer, seeds 10-12')
for arm in ('CT', 'CT2'):
    ks = sorted(k for k in b if k[2] == arm and k in a)
    same = sum(a[k][1]['answers'][0].get('pose_world') == b[k][1]['answers'][0].get('pose_world') for k in ks)
    d = [b[k][0]['share_strict'] - a[k][0]['share_strict'] for k in ks]
    print(f"  {arm}: n {len(ks)} same first pose {same}/{len(ks)} | original {tot([a[k][0] for k in ks])} | repeat {tot([b[k][0] for k in ks])}")
    print(f"       repeat-original mean {statistics.mean(d):+.3f} mean|d| {statistics.mean(abs(x) for x in d):.3f} sd {statistics.pstdev(d):.3f} flip-p {flip_p(d):.2f}")
ks = sorted(k[:2] for k in a if k[2] == 'CTe')
d = [a[k + ('CTe',)][0]['share_strict'] - a[k + ('CTd',)][0]['share_strict'] for k in ks]
print(f"  CTe-CTd: n {len(d)} mean {statistics.mean(d):+.3f} mean|d| {statistics.mean(abs(x) for x in d):.3f} sd {statistics.pstdev(d):.3f} flip-p {flip_p(d):.2f}")
# CT2 four draws per cell where available: original, repeat (timing only)
print()
print('(2) keep_out_of_shared_space in the first answer (used constraints)')
def cons(p):
    t = (p / '001' / 'inference.txt').read_text()
    return [c for u, c in re.findall(r'\[(used|unused|dropped)\] ([a-z_]+) \(', t) if u == 'used']
rows = {}
for f in sorted(A.glob('*/seed*/CT/episode.json')):
    sc, seed = f.parts[-4], int(f.parts[-3][4:])
    if sc in S.BREAK or seed < 10: continue
    src = next((f.parent.parent / arm for arm in ('CT', 'CT2', 'CT2m', 'C1', 'CTd')
                if (f.parent.parent / arm / '001' / 'inference.txt').exists()), None)
    if src is None:
        print('  no first-answer text for', sc, seed); continue
    rows[(sc, seed)] = 'keep_out_of_shared_space' in cons(src)
by_sc = {}
for (sc, seed), v in rows.items():
    by_sc.setdefault(sc, []).append(v)
print(f"  VLM people (CT first answer), seeds 10-18: {sum(rows.values())}/{len(rows)}")
for sc, v in sorted(by_sc.items()):
    print(f"     {sc:26s} {sum(v)}/{len(v)}")
print(f"  of which seeds 10-12: {sum(v for (sc, s), v in rows.items() if s <= 12)}/{sum(1 for (sc, s) in rows if s <= 12)}")
cte = {k: 'keep_out_of_shared_space' in cons(A / k[0] / f'seed{k[1]}' / 'CTe') for k in ks}
print(f"  detector people (CTe first answer), seeds 10-12: {sum(cte.values())}/{len(cte)}")
