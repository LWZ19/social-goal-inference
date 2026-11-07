"""Zero calls, scenario files only (written before any bystander cell exists, for BYSTANDERS_FROZEN.md).
Which bystanders two model-free membership rules take into the line, for each line length, at the nominal places and
under the sim's fidget (each bystander uniformly within 0.35 m of their place; 4000 draws).

  G-chain : start from the person nearest the target point (taken as slot 0's face point), add anyone within 1.0 m of
            a member, repeat (IDEAS_interface F2 / A00).
  absorb  : the solver's place-in-turn rule (src/sgi/solver.py _turn) with the true members as the waiting people,
            the line direction front->back from the slots, standoff 1.0 m, behind = within 35 deg of the line, and
            nobody exempt (no keep_out_of_shared_space in the answer). Iterated as the solver iterates.
Wanderers / rejoiners are left out (their places vary); so is the robot.

    python3 bystanders_geometry.py
"""
import math
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
SCENES = ['cafe_line2_bystanders', 'room_coffee_cart_bystanders', 'sweet_spot_line2_bystanders']
STANDOFF, HALF, CHAIN, FIDGET, N = 1.0, math.radians(35.0), 1.0, 0.35, 4000


def load(name):
    d = yaml.safe_load((ROOT / 'sim/scenarios' / f'{name}.yaml').read_text())
    (q, line), = d['queues'].items()
    slots = [(s['x'], s['y']) for s in line['slots']]
    target = tuple(line['slots'][0]['face'])
    members = [h for h in d['humans'] if h.get('queue')]
    bys = [(h['avatar'], tuple(h['stand'])) for h in d['humans'] if h.get('bystander')]
    others = [(h['avatar'], tuple(h['stand'])) for h in d['humans'] if 'stand' in h and not h.get('bystander')]
    return slots, target, len(members), bys, others


def g_chain(people, target):
    start = min(people, key=lambda p: math.dist(p[1], target))
    chain, grew = {start[0]}, True
    while grew:
        grew = False
        for n, xy in people:
            if n not in chain and any(math.dist(xy, q) < CHAIN for m, q in people if m in chain):
                chain.add(n)
                grew = True
    return chain


def absorb(waiting, others, along):
    waiting = list(waiting)
    taken_all = []
    while True:
        last = waiting[-1]
        place = (last[1][0] + STANDOFF * along[0], last[1][1] + STANDOFF * along[1])
        taken = []
        for n, xy in others:
            if n in taken_all:
                continue
            back = (xy[0] - last[1][0], xy[1] - last[1][1])
            nb = math.hypot(*back)
            behind = back[0] * along[0] + back[1] * along[1] >= math.cos(HALF) * nb - 1e-9
            gap = math.dist(xy, place)
            if gap < STANDOFF and behind:
                taken.append((gap, n, xy))
        if not taken:
            return taken_all
        g, n, xy = min(taken)
        taken_all.append(n)
        waiting.append((n, xy))


def jitter(xy, rng):
    r, a = FIDGET * math.sqrt(rng.random()), rng.random() * 2 * math.pi
    return (xy[0] + r * math.cos(a), xy[1] + r * math.sin(a))


def main():
    rng = random.Random(0)
    for name in SCENES:
        slots, target, n_members, bys, others = load(name)
        along = (slots[1][0] - slots[0][0], slots[1][1] - slots[0][1])
        L = math.hypot(*along)
        along = (along[0] / L, along[1] / L)
        print(f'\n## {name}: {n_members} members at the start, target point {target}, line direction {along}')
        for n in range(1, n_members + 2):
            members = [(f'm{i}', slots[i]) for i in range(n)]
            nom_g = sorted(g_chain(members + bys + others, target) - {m for m, _ in members})
            nom_a = absorb(members, bys + others, along)
            cg, ca = {}, {}
            for _ in range(N):
                jb = [(b, jitter(xy, rng)) for b, xy in bys]
                for b in g_chain(members + jb + others, target) - {m for m, _ in members}:
                    cg[b] = cg.get(b, 0) + 1
                for b in absorb(members, jb + others, along):
                    ca[b] = ca.get(b, 0) + 1
            fmt = lambda c: ', '.join(f'{b} {c[b] / N:.0%}' for b in sorted(c)) or 'none'
            print(f'  line of {n}: G-chain takes {nom_g or "none"} (fidget: {fmt(cg)}); '
                  f'absorb takes {nom_a or "none"} (fidget: {fmt(ca)})')


if __name__ == '__main__':
    main()
