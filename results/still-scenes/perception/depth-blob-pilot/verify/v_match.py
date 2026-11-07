"""Matching, recall, misses, false blobs, facing and position error from verify/per_scene/*.json (v_blobs.py)."""
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
SC = [json.loads(p.read_text()) for p in sorted((OUT / 'per_scene').glob('*.json'))]
SETS = ('plain', 'real', 'staffed', 'ph', 'seed0', 'seed1', 'seed2')
lines = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    lines.append(s)


def match(people, blobs, R):
    """greedy one-to-one by distance within R; returns {person index: blob index}."""
    pairs = []
    for i, p in enumerate(people):
        for j, b in enumerate(blobs):
            d = math.hypot(p['x'] - b['pos'][0], p['y'] - b['pos'][1])
            if d <= R:
                pairs.append((d, i, j))
    pairs.sort()
    mp, used = {}, set()
    for d, i, j in pairs:
        if i in mp or j in used:
            continue
        mp[i] = j
        used.add(j)
    return mp


def axerr(a, b):
    d = abs((a - b) % math.pi)
    return min(d, math.pi - d)


def run(variant, foot, R, verbose=False):
    st = defaultdict(lambda: defaultdict(int))
    misses = []
    perr = defaultdict(list)
    fac = defaultdict(lambda: defaultdict(list))
    for s in SC:
        S = s['set']
        blobs = [b for b in s['blobs'][variant] if b.get(f'kept_{foot}')]
        mp = match(s['people'], blobs, R)
        cam = s['cam']
        for i, p in enumerate(s['people']):
            st[S]['people'] += 1
            st[S]['in_img'] += p['in_img']
            if i in mp:
                st[S]['hit_all'] += 1
                st[S]['hit_img'] += p['in_img']
                b = blobs[mp[i]]
                perr[S].append(math.hypot(p['x'] - b['pos'][0], p['y'] - b['pos'][1]))
                for key in ('ax_pts', 'ax_cells'):
                    if b.get(key) is not None:
                        fac[S][key].append(axerr(b[key], p['yaw']) <= math.pi / 4 + 1e-9)
                # camera-bearing axis null (axis from camera to blob)
                bear = math.atan2(b['pos'][1] - cam[1], b['pos'][0] - cam[0])
                fac[S]['cam_axis'].append(axerr(bear, p['yaw']) <= math.pi / 4)
                fac[S]['yaw'].append(p['yaw'])
                fac[S]['ax_pts_v'].append(b.get('ax_pts'))
                fac[S]['sit'].append(s['scene_id'])
            elif p['in_img']:
                misses.append((S, s['scene_id'], p['id'], p['role'], p['visible'], p['low_hidden'], p['low_seen']))
        st[S]['blobs'] += len(blobs)
        st[S]['false'] += len(blobs) - len(mp)
        st[S]['images'] += 1
        # false blobs excluding those within 0.5 m of a target anchor
        used = set(mp.values())
        fa = 0
        for j, b in enumerate(blobs):
            if j in used:
                continue
            near_t = any(math.hypot(b['pos'][0] - a[0], b['pos'][1] - a[1]) <= 0.5 for a in s['targets'].values())
            fa += not near_t
        st[S]['false_notgt'] += fa
    return st, misses, perr, fac


def main():
    say('=== recall (in-image people / all truth people), split variant x footprint def x match radius ===')
    say('variant     foot  R     ' + '  '.join(f'{S:>14s}' for S in SETS))
    for variant in ('ws_0.15', 'ws_0.25', 'ws_0.35', 'vor_0.15', 'vor_0.25', 'vor_0.35'):
        for foot in ('pca', 'bbox'):
            for R in (0.25, 0.4, 0.6):
                st, misses, _, _ = run(variant, foot, R)
                cells = []
                for S in SETS:
                    d = st[S]
                    cells.append(f"{d['hit_img']:>2d}/{d['in_img']:<2d}({d['hit_all']}/{d['people']})")
                ph42 = st['ph']['hit_img'] + st['seed0']['hit_img']
                seeds = sum(st[k]['hit_img'] for k in ('seed0', 'seed1', 'seed2'))
                nonstaff = [m for m in misses if m[3] is None]
                say(f'{variant:10s}  {foot:4s}  {R:<4}  ' + '  '.join(f'{c:>14s}' for c in cells) +
                    f'  ph+seed0={ph42}/42  ph+3seeds={st["ph"]["hit_img"] + seeds}/58  misses={len(misses)}'
                    f' nonstaff_miss={len(nonstaff)}')
    say('')
    for variant, foot in (('ws_0.25', 'pca'), ('vor_0.25', 'pca'), ('ws_0.15', 'pca'), ('ws_0.35', 'pca')):
        R = 0.4
        st, misses, perr, fac = run(variant, foot, R)
        say(f'=== {variant} {foot} R={R}: misses (in-image), false blobs, position error, facing ===')
        for m in misses:
            say('  miss', m)
        hid = sum(1 for m in misses if m[5] >= 6)
        say(f'  misses={len(misses)} staff={sum(1 for m in misses if m[3] == "staff")} '
            f'lower-body hidden (>=6 of 8 samples) = {hid}')
        for S in SETS:
            d = st[S]
            say(f"  {S:8s} blobs={d['blobs']} false={d['false']} per_image={d['false'] / d['images']:.2f} "
                f"false_not_near_target={d['false_notgt']} ({d['false_notgt'] / d['images']:.2f}/img) "
                f"pos_err_median={np.median(perr[S]) if perr[S] else float('nan'):.3f}")
        for grp, names in (('plain', ['plain']), ('real', ['real']), ('staffed', ['staffed']), ('ph', ['ph']),
                           ('ph+seed0', ['ph', 'seed0']), ('ph+3seeds', ['ph', 'seed0', 'seed1', 'seed2'])):
            pts = sum((fac[n]['ax_pts'] for n in names), [])
            cel = sum((fac[n]['ax_cells'] for n in names), [])
            cama = sum((fac[n]['cam_axis'] for n in names), [])
            yaws = sum((fac[n]['yaw'] for n in names), [])
            axv = sum((fac[n]['ax_pts_v'] for n in names), [])
            # shuffled-yaw null mod 180, 2000 permutations
            rng = random.Random(0)
            sh = []
            ok = [(a, y) for a, y in zip(axv, yaws) if a is not None]
            for _ in range(2000):
                ys = [y for _, y in ok]
                rng.shuffle(ys)
                sh.append(np.mean([axerr(a, y) <= math.pi / 4 for (a, _), y in zip(ok, ys)]))
            say(f'  facing {grp:10s} n={len(pts)} pts={np.mean(pts):.3f} ({sum(pts)}) cells={np.mean(cel):.3f} '
                f'({sum(cel)})  null_cam_axis_mod180={np.mean(cama):.3f}  null_shuffled_yaw_mod180={np.mean(sh):.3f}')
        say('')
    # analytic / simulated nulls
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 2 * math.pi, 200000)
    t = rng.uniform(0, 2 * math.pi, 200000)
    d180 = np.abs((a - t) % math.pi)
    d180 = np.minimum(d180, math.pi - d180)
    d360 = np.abs((a - t) % (2 * math.pi))
    d360 = np.minimum(d360, 2 * math.pi - d360)
    say(f'null uniform axis mod 180 within 45 deg: {np.mean(d180 <= math.pi / 4):.4f} (analytic 0.5)')
    say(f'null uniform direction mod 360 within 45 deg: {np.mean(d360 <= math.pi / 4):.4f} (analytic 0.25)')
    (OUT / 'v_match.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
