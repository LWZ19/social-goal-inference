import json, re, math
from pathlib import Path
R = Path('results/2026-11-01-push/moving/runs/sonnet55_low')
def parse(p):
    t = (p / '001' / 'inference.txt').read_text()
    ppl = [tuple(map(float, m)) for m in re.findall(r'#\d+ .*?at \((-?[\d.]+), (-?[\d.]+)\)', t, re.S)]
    ppl = [tuple(map(float, m)) for m in re.findall(r'^\s+#\d+[^\n]*?at \((-?[\d.]+), (-?[\d.]+)\)', t, re.M)] or ppl
    cons = re.findall(r'\[(used|unused|dropped|[a-z ]+)\] ([a-z_]+) \(', t)
    goal = re.search(r'GOAL \((-?[\d.]+), (-?[\d.]+), (-?\d+) deg\)', t)
    flags = re.search(r'flags: (.*)', t)
    return ppl, cons, goal.groups() if goal else None, flags.group(1) if flags else ''
def err(ppl, truth):
    # per true visible person nearest detection distance; plus count of answer people > 0.5 m from any truth
    tr = [(x, y) for _, x, y, vis in truth]
    near = [min(math.hypot(px - x, py - y) for x, y in tr) for px, py in ppl]
    return sum(d <= 0.5 for d in near), sum(d > 0.5 for d in near), (sorted(near)[len(near)//2] if near else None)
for sc in sorted(p.name for p in R.iterdir() if p.is_dir()):
    for s in (10, 11, 12):
        base = R / sc / f'seed{s}'
        if not (base / 'CTe').exists(): continue
        e = json.loads((base / 'CTe' / 'episode.json').read_text())
        truth = e['det'][0]['truth']
        nvis = sum(1 for t in truth if t[3])
        out = []
        for arm in ('CTd', 'CTe'):
            ppl, cons, goal, flags = parse(base / arm)
            ok, bad, med = err(ppl, truth)
            out.append(f"{arm}: ppl {len(ppl)} ok {ok} far {bad} med {med:.2f} cons {[c[1] for c in cons if c[0]=='used']} goal {goal} {flags}")
        print(f'{sc} s{s} (truth {len(truth)}, visible-flag {nvis})'); [print('   ', o) for o in out]
