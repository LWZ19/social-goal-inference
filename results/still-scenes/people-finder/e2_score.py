"""E2 scorer (E2_FROZEN.md sections 5-6), zero calls.

    PYTHONPATH= .venv/bin/python e2_score.py --people e2_people.json \
        --scenes run_scenes runs/e2_qwen38_low runs/e2_glm53 runs/e2_dsv41 [runs/e2_sonnet55_low]

Each run folder holds one model's records_<i>of<n>.jsonl shards and run_meta.json. The last row of a unit counts. Units
still marked vlm_error are left out of every arm and counted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

FREE = ('soclaas_qwen38_low', 'soclaas_glm53', 'soclaas_dsv41')
GOAL_STAGES = ('people', 'constraints')
MATCH_M = 0.5


def signflip_p(nets):
    """Exact two-sided sign-flip p over integer nets: P(|sum of +-x| >= |observed|), by counting sums."""
    xs = [abs(int(x)) for x in nets if x]
    obs = abs(sum(int(x) for x in nets))
    if not xs:
        return 1.0
    counts = {0: 1}
    for x in xs:
        nxt = defaultdict(int)
        for s, c in counts.items():
            nxt[s + x] += c
            nxt[s - x] += c
        counts = nxt
    return sum(c for s, c in counts.items() if abs(s) >= obs) / 2 ** len(xs)


def sign_p(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(folders):
    rows, metas = {}, defaultdict(set)
    for f in folders:
        f = Path(f)
        meta = (f / 'run_meta.json').read_text()
        for p in sorted(f.glob('records_*of*.jsonl')):
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[(r['model'], r['scene'], r['episode'], r['t'])] = r
                    metas[r['model']].add(hashlib.sha256(meta.encode()).hexdigest())
    for model, m in metas.items():
        if len(m) > 1:
            raise SystemExit(f'{model}: shards with different run_meta.json: refusing')
    return list(rows.values())


def layout_row(scene):
    """(seed, row) from s<N>_real_eval_<nn>_<row>; anything else is its own layout and row."""
    if scene.startswith('s') and '_real_eval_' in scene:
        seed, rest = scene[1:].split('_real_eval_', 1)
        return seed, rest.split('_', 1)[1]
    return scene, scene


def ok(arm):
    return bool(arm['score']['success'])


def goal_seconds(arm, name, det_s):
    s = sum(c['latency_s'] or 0.0 for c in arm['calls'] if c['stage'] in GOAL_STAGES)
    return s + (det_s if name == 'D' else 0.0)


def unit_seconds(arm, name, det_s):
    return sum(c['latency_s'] or 0.0 for c in arm['calls']) + (det_s if name == 'D' else 0.0)


def all_ok(r):
    return all(c['status'] == 'ok' for a in r['arms'].values() for c in a['calls'])


def detector(people_path, scenes_dir):
    from sgi.config import load_config
    from sgi.gt.derive import is_visible
    from sgi.scene import load_scene
    doc = json.loads(Path(people_path).read_text())
    th = load_config().thresholds
    warm = [v['t_det_s'] + v['t_pose_s'] for v in doc['scenes'].values()
            if v.get('order', 1) > 0 and v.get('t_det_s') is not None and v.get('t_pose_s') is not None]
    det_s = st.median(warm) if warm else 0.0596          # E1b's A1 warm time when the pass logged none
    found = {'staff': [0, 0], 'other': [0, 0]}
    false, images = 0, 0
    for key, v in doc['scenes'].items():
        folder = Path(key)
        if scenes_dir is not None:
            folder = Path(scenes_dir) / folder.name
        inputs, truth = load_scene(folder)
        vis = [p for p in truth.people if is_visible(p, inputs.observation, th)]
        pairs = sorted((math.hypot(d['x'] - p.x, d['y'] - p.y), i, j) for i, d in enumerate(v['people'])
                       for j, p in enumerate(vis))
        used_d, used_p = set(), set()
        for dist, i, j in pairs:
            if dist <= MATCH_M and i not in used_d and j not in used_p:
                used_d.add(i)
                used_p.add(j)
        for j, p in enumerate(vis):
            k = 'staff' if p.role == 'staff' else 'other'
            found[k][1] += 1
            found[k][0] += j in used_p
        false += len(v['people']) - len(used_d)
        images += 1
    return det_s, found, false, images


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--people', required=True)
    ap.add_argument('--scenes', default=None)
    ap.add_argument('folders', nargs='+')
    a = ap.parse_args(argv)
    det_s, found, false, images = detector(a.people, a.scenes)
    out = []
    say = out.append
    say('# E2 score (E2_FROZEN.md)')
    say(f"- detector check (reported): recall other {found['other'][0]}/{found['other'][1]}, staff "
        f"{found['staff'][0]}/{found['staff'][1]}; false people {false} in {images} images "
        f"({false / max(images, 1):.3f} per image); warm detector time {det_s:.3f} s per image")
    rows = load(a.folders)
    claim, pooled_net = {}, defaultdict(int)
    for model in sorted({r['model'] for r in rows}):
        rs = [r for r in rows if r['model'] == model]
        err = [r for r in rs if r['vlm_error']]
        rs = [r for r in rs if not r['vlm_error']]
        pm = [r for r in rs if r['people_matter']]
        say(f'\n## {model}: {len(rs)} units ({len(err)} left out for vlm_error), {len(pm)} people-matter')
        for name in ('A0', 'A0b', 'D'):
            say(f'- success {name}: people-matter {sum(ok(r["arms"][name]) for r in pm)}/{len(pm)}, '
                f'all units {sum(ok(r["arms"][name]) for r in rs)}/{len(rs)}')
        rw = sum(1 for r in pm if ok(r['arms']['A0']) and not ok(r['arms']['A0b']))
        wr = sum(1 for r in pm if not ok(r['arms']['A0']) and ok(r['arms']['A0b']))
        gain = sum(1 for r in pm if not ok(r['arms']['A0']) and ok(r['arms']['D']))
        harm = sum(1 for r in pm if ok(r['arms']['A0']) and not ok(r['arms']['D']))
        lay, row = defaultdict(int), defaultdict(int)
        for r in pm:
            d = int(ok(r['arms']['D'])) - int(ok(r['arms']['A0']))
            lay[r['scene']] += d
            row[layout_row(r['scene'])[1]] += d
            if model in FREE:
                pooled_net[r['scene']] += d
        net = gain - harm
        p_lay, p_row = signflip_p(list(lay.values())), signflip_p(list(row.values()))
        say(f'- redraw floor A0 -> A0b: right->wrong {rw}, wrong->right {wr}')
        say(f'- D vs A0: gain {gain}, harm {harm}, net {net:+d}; layout sign-flip p {p_lay:.4f} over '
            f'{sum(1 for v in lay.values() if v)} layouts; row sign-flip p {p_row:.4f} over {sum(1 for v in row.values() if v)} rows')
        people_in_d = sum(1 for r in pm for c in r['arms']['D']['calls'] if c['stage'] == 'people')
        own = lambda r, n: sum(1 for c in r['arms'][n]['calls'] if c['stage'] in GOAL_STAGES)
        a0_people_ok = [r for r in pm if any(c['stage'] == 'people' and c['status'] == 'ok'
                                             for c in r['arms']['A0']['calls'])]
        one_fewer = all(own(r, 'A0') - own(r, 'D') == 1 for r in a0_people_ok)
        calls = {n: st.mean(sum(1 for _ in r['arms'][n]['calls']) for r in pm) if pm else float('nan')
                 for n in ('A0', 'A0b', 'D')}
        say(f'- calls per people-matter unit (with plan and context): A0 {calls["A0"]:.2f}, A0b {calls["A0b"]:.2f}, '
            f'D {calls["D"]:.2f}; people calls in D {people_in_d}; one fewer wherever A0 people ok: {one_fewer}')
        retries = {n: sum((c.get('validation_retries') or 0) + (c.get('transient_retries') or 0)
                          for r in pm for c in r['arms'][n]['calls'] if c['stage'] in GOAL_STAGES)
                   for n in ('A0', 'A0b', 'D')}
        say(f'- goal-step retries (validation + transient) on people-matter units: {retries}')
        tm = [r for r in pm if all_ok(r)]
        g = {n: [goal_seconds(r['arms'][n], n, det_s) for r in tm] for n in ('A0', 'A0b', 'D')}
        u = {n: [unit_seconds(r['arms'][n], n, det_s) for r in tm] for n in ('A0', 'A0b', 'D')}
        faster = sum(1 for x, y in zip(g['A0'], g['D']) if y < x)
        slower = sum(1 for x, y in zip(g['A0'], g['D']) if y > x)
        pf = sign_p(faster, slower)
        b_lo = sum(1 for x, y in zip(g['A0'], g['A0b']) if y < x)
        b_hi = sum(1 for x, y in zip(g['A0'], g['A0b']) if y > x)
        pp = sign_p(b_lo, b_hi)
        if tm:
            say(f'- goal-step seconds, {len(tm)} people-matter units with every call ok: median A0 '
                f'{st.median(g["A0"]):.1f}, A0b {st.median(g["A0b"]):.1f}, D {st.median(g["D"]):.1f} (constraints '
                f'{st.median([x - det_s for x in g["D"]]):.1f} + detector {det_s:.2f}); mean A0 {st.mean(g["A0"]):.1f}, '
                f'D {st.mean(g["D"]):.1f}')
            say(f'- whole unit seconds, median: A0 {st.median(u["A0"]):.1f}, A0b {st.median(u["A0b"]):.1f}, '
                f'D {st.median(u["D"]):.1f}')
        say(f'- D faster than A0 in {faster}, slower in {slower}: sign p {pf:.4f}; placebo A0b faster than A0 in {b_lo}, '
            f'slower in {b_hi}: sign p {pp:.4f}')
        bar_a = (harm - gain) <= rw
        bar_g = not (p_lay <= 0.05 and net < 0)
        bar_s1 = people_in_d == 0 and one_fewer
        judgeable = pp > 0.05
        bar_s2 = judgeable and faster > slower and pf <= 0.05
        h2 = net > 0 and p_lay <= 0.05
        say(f'- BARS: A (harm - gain <= {rw}) {bar_a}; G {bar_g}; S1 {bar_s1}; S2 '
            f'{bar_s2 if judgeable else "not judgeable (placebo p <= 0.05)"}; H2 (secondary) {h2}')
        if model in FREE:
            claim[model] = (bar_a and bar_g and bar_s1, bar_s2, judgeable, h2)
    say('\n## The claim (free models)')
    missing = [m for m in FREE if m not in claim]
    if missing:
        say(f'- not judged: no records for {missing}')
    else:
        acc = all(v[0] for v in claim.values())
        time_ok = all(v[1] or not v[2] for v in claim.values())
        holds = acc and time_ok and all(v[1] for v in claim.values() if v[2])
        not_j = [m for m, v in claim.items() if not v[2]]
        say(f'- "D beats C on fresh static scenes, faster at no loss of accuracy": {holds}'
            + (f' (time not judgeable for {not_j}: calls only there)' if holds and not_j else ''))
        if holds:
            pp_pool = signflip_p(list(pooled_net.values()))
            net_pool = sum(pooled_net.values())
            say(f'- H2 pooled over the free models: net {net_pool:+d}, layout sign-flip p {pp_pool:.4f}: '
                f'{net_pool > 0 and pp_pool <= 0.05}; per model: {[m for m, v in claim.items() if v[3]]}')
    text = '\n'.join(out)
    print(text)
    Path(a.folders[0], 'e2_score_out.txt').write_text(text + '\n')


if __name__ == '__main__':
    main()
