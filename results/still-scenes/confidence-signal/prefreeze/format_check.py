"""Pre-freeze format check for the phase-3 confidence protocol (FROZEN.md is written after this, from its output).

Computes NO signal and NO signal-outcome relation: it never reads a success, a score, a target_correct flag, a
detector distance or a re-solved place. It counts records, keys, trials, fields and joins only, and times one
H.resolve call. Runs and scenes only through the step-0 harness; zero calls.

    PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
        .venv/bin/python -B <this>
"""
from __future__ import annotations

import collections
import json
import re
import sys
import time
from pathlib import Path

import sys  # noqa: E402
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
RD = paths.STILL
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
sys.path.insert(0, str(RD / 'metrics-and-statistics' / 'metric-v0'))
import research_harness as H  # noqa: E402
import i14  # noqa: E402  (unchanged; plan() and key() only)

S23 = [('s23', m) for m in ('glm53', 'dsv41', 'haiku55', 'sonnet55_low', 'gemini38_low')]
CELL_NAME = {'qwen38_low': 'qwen', 'glm53': 'glm', 'dsv41': 'deepseek', 'haiku55': 'haiku', 'sonnet55_low': 'sonnet',
             'gemini38_low': 'gemini'}
LAYOUT = re.compile(r'eval_[0-9]{2}')


def plan():
    out = [(label, ref) for label, ref in i14.plan('oldwatch')]
    for stage, model in S23:
        for arm in i14.REAL_ARMS:
            refs = H.runs(stage=stage, arm=arm, model=model, method='con')
            assert len(refs) == 1, (stage, arm, model, refs)
            out.append(('new', refs[0]))
    return out


def main():
    t0 = time.monotonic()
    i14_keys = set()
    for name in ('rows_oldwatch.jsonl', 'rows_new.jsonl'):
        for line in open(RD / 'metrics-and-statistics' / 'metric-v0' / name):
            r = json.loads(line)
            if r['set'] == 'new' and r['stage'] == 's21':
                continue
            i14_keys.add(f"{r['run']}/{r['scene']}/{r['ep']}/{r['trial']}")
    ep = json.load(open(RD / 'people-finder' / 'early-detector' / 'phase1c' / 'a1' / 'eval_people.json'))
    det_listed = {(sd, sid) for v in ep['scenes'].values() for sd, sid in v['listed']}
    a2 = {}
    for w, models in (('old', ('qwen38_low', 'glm53', 'dsv41')), ('new', ('glm53', 'dsv41', 'haiku55', 'sonnet55_low', 'gemini38_low'))):
        for m in models:
            for line in open(RD / 'people-finder' / 'early-detector' / 'phase1c' / 'a2' / f'rows_{w}_{m}.jsonl'):
                r = json.loads(line)
                ok_d = 'D_d' in r and r['D_d'].get('moved_from_C_m') is not None and 'tnum' in r['D_d'] and 'tnum_C' in r
                a2[(r['run'], r['scene'], r['episode'], r['trial'])] = (r['solver_step'], ok_d)
    i04 = {}
    for p in sorted((RD / 'learned-models' / 'soft-density').glob('rows_*.jsonl')):
        for line in open(p):
            r = json.loads(line)
            i04[r['key']] = (r['cell'], r['layout'], r['flags'])

    cells = collections.Counter()
    keys_here = set()
    trial_groups = collections.defaultdict(list)       # (run, scene, ep) -> [(trial, has pose, solver step, has target world)]
    counts = collections.Counter()
    layouts = collections.defaultdict(set)
    radius = set()
    timed = None
    for label, ref in plan():
        th = H.recorded_config(ref)[1].thresholds
        radius.add((th.person_radius_m, th.match_tolerance_m))
        cell = f"{label}_{CELL_NAME[ref.model]}"
        goal_eps = set()
        recs = list(H.records(ref))
        for rec in recs:
            if label == 'watch' and rec.situation != 'audience':
                continue
            tw = (rec.target or {}).get('world') if rec.target else None
            trial_groups[(ref.folder, rec.scene_id, rec.episode_id)].append(
                (rec.trial, rec.pose_world is not None, rec.solver_step, tw is not None, rec.result.get('source'),
                 rec.result.get('status')))
            if not rec.solver_step:
                continue
            k = i14.key(rec)
            keys_here.add(k)
            cells[cell] += 1
            goal_eps.add((rec.scene_id, rec.episode_id))
            counts['trial values ' + str(rec.trial)] += 1
            # fields
            fl = rec.result.get('flags') or {}
            for f in ('exclusion_broken', 'far_from_target', 'constraint_unusable'):
                counts[f'flag field present {f}'] += f in fl
            if k in i04:
                counts['i04 join'] += 1
                c4, lay4, fl4 = i04[k]
                counts['i04 flags equal recorded flags'] += all(bool(fl.get(f)) == bool(v) for f, v in fl4.items())
                counts['i04 cell name equal'] += c4 == cell.replace('old_', 'old_').replace('new_', 'new_')
                layouts[cell].add(lay4)
                if label != 'watch':
                    m = LAYOUT.search(rec.scene_id)
                    counts['i04 layout equals eval index of scene id'] += m is not None and m.group(0) == lay4
            else:
                counts['i04 MISSING'] += 1
            # people and constraints structure (no detector, no outcome)
            nums = [p['number'] for p in rec.people]
            counts['people numbers unique'] += len(set(nums)) == len(nums)
            counts['people list empty'] += len(nums) == 0
            cons = [u['constraint'] for u in rec.constraints]
            named = {n for c in cons for n in (c.get('people') or [])}
            counts['records naming >=1 person'] += bool(named)
            counts['records naming a number absent from the list'] += bool(named - set(nums))
            counts['named numbers absent from the list (total)'] += len(named - set(nums))
            counts['named numbers (total distinct per record)'] += len(named)
            counts['target world present'] += bool(rec.target and rec.target.get('world'))
            # detector people mapping
            counts['detector mapping present'] += (str(ref.scenes_dir), rec.scene_id) in det_listed
            # A2 join (realistic only)
            if label in ('old', 'new'):
                a = a2.get((ref.folder, rec.scene_id, rec.episode_id, rec.trial))
                counts['A2 join (solver step with D row, D place and both target numbers present)'] += a == (True, True)
            if timed is None and label == 'old':
                s = time.monotonic()
                H.resolve(rec)
                timed = time.monotonic() - s
        print(f"{time.monotonic() - t0:7.1f} s {cell:16s} {ref.folder}: allowed {len(recs)}, goal steps so far {cells[cell]}",
              flush=True)
    print('\n== goal-step records per cell ==')
    for c, n in cells.items():
        print(f'  {c:16s} {n}')
    print(f'  total {sum(cells.values())}')
    print(f'  key set equals I14 rows (s21 excluded): {keys_here == i14_keys}; here {len(keys_here)}, I14 {len(i14_keys)}')
    print('\n== layouts (from I04 rows) per cell ==')
    for c, s in layouts.items():
        print(f'  {c:16s} {len(s)}: {sorted(s)}')
    print('\n== field and join counts (over goal-step records) ==')
    for k in sorted(counts):
        print(f'  {k}: {counts[k]}')
    print(f'\n== thresholds (person_radius_m, match_tolerance_m) seen: {sorted(radius)}')
    # trial structure over the (run, scene, episode) keys that hold at least one goal-step record
    print('\n== trial groups (run, scene, episode) holding >= 1 goal-step record ==')
    tg = collections.Counter()
    for key, items in trial_groups.items():
        if not any(i[2] for i in items):
            continue
        trials = sorted(i[0] for i in items)
        tg[f'n trials {len(items)} {trials}'] += 1
        tg[f'trials with pose {sum(i[1] for i in items)} of {len(items)}'] += 1
        tg[f'goal-step trials {sum(i[2] for i in items)} of {len(items)}'] += 1
        tg[f'trials with target world {sum(i[3] for i in items)} of {len(items)}'] += 1
        tg[f'trials with pose but no target world: {sum(i[1] and not i[3] for i in items)}'] += 1
        tg[f'trials with target world but no pose: {sum(i[3] and not i[1] for i in items)}'] += 1
        for i in items:
            if not i[2]:
                tg[f'non-goal-step trial: source {i[4]} status {i[5]} pose {i[1]}'] += 1
    for k in sorted(tg):
        print(f'  {k}: {tg[k]}')
    print(f'\none H.resolve: {timed:.2f} s; total {time.monotonic() - t0:.1f} s')


if __name__ == '__main__':
    main()
