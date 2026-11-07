"""POST HOC (after the judged cells): the frozen readings restricted to cells whose model answer is a constraint
answer (6 of 15 judged cells were standard-goal answers: no re-solving, every C arm identical). Zero calls."""
import json, sys
sys.path.insert(0, 'results/2026-11-01-push/moving')
from bystanders_score import load, cells, capture, paired, BYS
from summarise_fresh import sign_p
by = cells(load(['runs/dsv41'], set(BYS), set(range(10, 15))))
keep = {k: v for k, v in by.items() if (v['CT'].get('answers') or [{}])[0].get('source') == 'constraints'}
print(f'cells with a constraint answer: {len(keep)} of {len(by)}')
cap = {(k, a): capture(v[a]) for k, v in keep.items() for a in ('CT2', 'Gl', 'CT3', 'CT3-j')}
p3 = sum(cap[(k, 'CT2')][0] >= 1.0 for k in keep)
print(f'CT2 captured >= 1 s in {p3} of {len(keep)}')
either = [k for k in keep if cap[(k, 'Gl')][0] >= 1.0 or cap[(k, 'CT3')][0] >= 1.0]
print(f"Gl capture > CT3 in {sum(cap[(k, 'Gl')][0] > cap[(k, 'CT3')][0] for k in either)} of {len(either)} cells where either is captured")
for a, b in (('CT3', 'CT2'), ('CT3', 'Gl'), ('CT2', 'Gl'), ('CT3-j', 'CT2')):
    _, w, l, t, md, _ = paired(keep, a, b)
    print(f'{a} vs {b}: {w}/{l}/{t}, mean {md:+.3f}, sign p {sign_p(w, l):.3f}')
cap_cells = [k for k in keep if cap[(k, 'CT2')][0] >= 1.0]
_, w, l, t, md, _ = paired({k: keep[k] for k in cap_cells}, 'CT3', 'CT2')
print(f'CT3 vs CT2 where CT2 was captured ({len(cap_cells)} cells): {w}/{l}/{t}, mean {md:+.3f}')
