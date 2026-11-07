"""Phase 2 (C5 / I52 flow pilot): placeholder TRAINING seeds 110-199, adapted from phase1b/i03/gen_fresh.py.

By the ruling of 18 Oct, seeds 110-199 are training data for this pilot. They are kept out of the I46
sealed set (seeds 3-99, refused here) and apart from the I03/I09 fit seeds 100-109 (also refused here).

For each seed k, exactly as gen_fresh.py: rng = default_rng(k) over every row of _SPLITS, then
more = default_rng([k, _MORE_STREAM]) over _MORE_SPLITS with indices from len(dict(_SPLITS)[split]); a row is
written with the package's own _write_scene only when its situation is in cfg.dev_situations, and a row of any
other situation (group_photo, lift_wait) is skipped before any random draw, so no held-out scene is ever rendered.
Output: phase2/seeds/placeholder/seed<k>/{eval,examples}/.

A seed is complete when seed<k>/DONE.json exists; a complete seed is skipped, an incomplete one is removed and
written again from its first row (one random stream per seed). Up to 4 seeds run at once, each in its own process.
No model, no network, no torch.

Usage (project venv, from the worktree):
  gen_placeholder.py                      generate every missing seed of 110-199, then write the manifest
  gen_placeholder.py --manifest-only      rebuild manifest_placeholder.json from the DONE files
  gen_placeholder.py --verify             check DONE files, counts, situations and fingerprints on disk
  gen_placeholder.py --repro K --tmp DIR  write seed K again under DIR (a new folder) and compare its sha256s
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402
from sgi.config import DEFAULT_CONFIG, load_config  # noqa: E402
from sgi.scene import read_scene_file  # noqa: E402
from sgi.synth.generate import _MORE_SPLITS, _MORE_STREAM, _REASONS, _SPLITS, _write_scene  # noqa: E402

WT = Path('.-scenes')
HERE = WT / 'results/2026-10-17-beyond-vlm/phase2/seeds'
OUT = HERE / 'placeholder'
MANIFEST = HERE / 'manifest_placeholder.json'
SEEDS = tuple(range(110, 200))
WORKERS = 4
DEV = {'no_one', 'waiting_line', 'conversation_group', 'audience'}
HELD = {'group_photo', 'lift_wait'}


def check_seed(k: int) -> None:
    assert 110 <= k <= 199, f"seed {k} refused: only 110-199 (training seeds of this pilot) are generated here"


def rows_of():
    """(pass, split, index, row) in the order the generator draws them."""
    out = []
    for split, rows in _SPLITS:
        for index, row in enumerate(rows):
            out.append(('first', split, index, row))
    for split, rows in _MORE_SPLITS:
        for index, row in enumerate(rows, start=len(dict(_SPLITS)[split])):
            out.append(('more', split, index, row))
    return out


def one_seed(k: int, out_root: str | None = None) -> dict:
    check_seed(k)
    cfg = load_config(DEFAULT_CONFIG)
    assert set(cfg.dev_situations) == DEV, cfg.dev_situations
    assert set(cfg.held_out_situations) == HELD, cfg.held_out_situations
    root = Path(out_root) if out_root else OUT
    H.guard(root)
    seed_dir = root / f'seed{k}'
    done = seed_dir / 'DONE.json'
    if done.exists():
        print(f'seed {k}: complete, skipped', flush=True)
        return json.loads(done.read_text())
    if seed_dir.exists():
        print(f'seed {k}: incomplete, removed and written again', flush=True)
        shutil.rmtree(seed_dir)
    for split, _ in _SPLITS:
        (seed_dir / split).mkdir(parents=True)
    counts = dict.fromkeys(_REASONS, 0)
    rng = np.random.default_rng(k)
    more = np.random.default_rng([k, _MORE_STREAM])
    scenes, skipped = [], []
    started = time.monotonic()
    for which, split, index, row in rows_of():
        if row[0] not in cfg.dev_situations:          # before any random draw
            assert row[0] in HELD, row
            skipped.append(f'{split}_{index:02d} ({row[0]})')
            continue
        split_dir = seed_dir / split
        before = {p.name for p in split_dir.iterdir()}
        t0 = time.monotonic()
        _write_scene(split_dir, index, row, rng if which == 'first' else more, cfg, counts)
        dt = time.monotonic() - t0
        new = sorted({p.name for p in split_dir.iterdir()} - before)
        assert len(new) == 1, (k, split, index, new)
        scene_id = new[0]
        H.guard(scene_id)
        sf = read_scene_file(split_dir / scene_id)
        sits = sorted({ep.ground_truth.situation for ep in sf.episodes})
        scenes.append({'split': split, 'index': index, 'scene_id': scene_id, 'row': list(row),
                       'situations': sits, 'seconds': round(dt, 1)})
        print(f'seed {k} {scene_id:44s} {"+".join(sits):22s} {dt:7.1f} s', flush=True)
    held = set(cfg.held_out_situations)
    bad = [s['scene_id'] for s in scenes if set(s['situations']) & held or not set(s['situations']) <= DEV]
    assert not bad, f'held-out or unknown episodes written: {bad}'
    for s in scenes:
        s['sha256'] = H.fingerprint(seed_dir / s['split'] / s['scene_id'])
    info = {'seed': k, 'scenes': scenes, 'skipped_rows': skipped, 'rejections': counts,
            'n_scenes': len(scenes), 'seconds': round(time.monotonic() - started, 1),
            'held_out_check': 'no written episode is of ' + ', '.join(sorted(held))}
    done.write_text(json.dumps(info, indent=1))
    print(f'seed {k}: done, {len(scenes)} scenes in {info["seconds"]} s', flush=True)
    return info


def git_state():
    commit = subprocess.run(['git', '-C', str(WT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(WT), 'status', '--porcelain', 'src', 'configs'], capture_output=True,
                           text=True).stdout.strip()
    return commit, dirty or None


def write_manifest():
    seeds = []
    for k in SEEDS:
        done = OUT / f'seed{k}' / 'DONE.json'
        if done.exists():
            seeds.append(json.loads(done.read_text()))
    commit, dirty = git_state()
    flat = [{'seed': d['seed'], **s} for d in seeds for s in d['scenes']]
    rejections = dict.fromkeys(_REASONS, 0)
    for d in seeds:
        for reason, n in d['rejections'].items():
            rejections[reason] += n
    manifest = {
        'note': ('Phase 2 placeholder TRAINING seeds 110-199 (decided 18 Oct), development situations only; '
                 'written by phase2/seeds/gen_placeholder.py, adapted from phase1b/i03/gen_fresh.py. Seeds 3-99 '
                 '(I46 sealed set) and 100-109 (I03/I09 fit data) are not generated here.'),
        'config': str(DEFAULT_CONFIG), 'worktree_commit': commit, 'src_or_configs_modified': dirty,
        'seeds_complete': [d['seed'] for d in seeds],
        'seeds_missing': [k for k in SEEDS if k not in {d['seed'] for d in seeds}],
        'per_seed': [{k: d[k] for k in ('seed', 'n_scenes', 'seconds', 'skipped_rows', 'rejections')}
                     for d in seeds],
        'scenes': flat,
        'totals': {'seeds': len(seeds), 'scenes': len(flat),
                   'scene_seconds': round(sum(s['seconds'] for s in flat), 1),
                   'rejections': rejections,
                   'by_situation': dict(collections.Counter('+'.join(s['situations']) for s in flat)),
                   'by_split': dict(collections.Counter(s['split'] for s in flat)),
                   'held_out_episodes_written': sum(bool(set(s['situations']) & HELD) for s in flat)},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {len(flat)} scenes over {len(seeds)} seeds -> {MANIFEST}", flush=True)
    return manifest


def generate():
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [k for k in SEEDS if not (OUT / f'seed{k}' / 'DONE.json').exists()]
    print(f'{len(SEEDS) - len(todo)} seeds complete, {len(todo)} to write, {WORKERS} at once', flush=True)
    started = time.monotonic()
    failed = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(one_seed, k): k for k in todo}
        for fut in as_completed(futures):
            k = futures[fut]
            try:
                fut.result()
            except BaseException as exc:  # noqa: BLE001 - one bad seed must not stop the others
                failed.append(k)
                print(f'seed {k}: FAILED: {type(exc).__name__}: {exc}', flush=True)
    print(f'wall time {time.monotonic() - started:.0f} s; failed seeds: {failed or "none"}', flush=True)
    write_manifest()
    return 1 if failed else 0


def verify():
    """Every check on disk: 90 DONE files, scene count per seed and situation, situations re-read from every
    scene.json, no held-out situation, no forbidden scene name, fingerprints as recorded."""
    problems = []
    dones = sorted(OUT.glob('seed*/DONE.json'))
    names = {p.parent.name for p in dones}
    expect = {f'seed{k}' for k in SEEDS}
    if names != expect:
        problems.append(f'DONE files: missing {sorted(expect - names)}, extra {sorted(names - expect)}')
    stray = sorted(p.name for p in OUT.iterdir() if p.name not in expect)
    if stray:
        problems.append(f'stray entries in placeholder/: {stray}')
    manifest = json.loads(MANIFEST.read_text())
    recorded = {(s['seed'], s['split'], s['scene_id']): s['sha256'] for s in manifest['scenes']}
    per_seed_sit = collections.Counter()
    per_seed_n = {}
    all_sits = collections.Counter()
    for k in SEEDS:
        seed_dir = OUT / f'seed{k}'
        d = json.loads((seed_dir / 'DONE.json').read_text())
        on_disk = sorted((split, p.name) for split, _ in _SPLITS for p in (seed_dir / split).iterdir())
        listed = sorted((s['split'], s['scene_id']) for s in d['scenes'])
        if on_disk != listed:
            problems.append(f'seed {k}: folders on disk differ from DONE.json')
        per_seed_n[k] = len(on_disk)
        for split, scene_id in on_disk:
            if H.forbidden(scene_id):
                problems.append(f'seed {k}: forbidden scene name {scene_id}')
                continue
            folder = seed_dir / split / scene_id
            sf = read_scene_file(folder)
            sits = {ep.ground_truth.situation for ep in sf.episodes}
            if sits & HELD or not sits <= DEV:
                problems.append(f'seed {k}: {scene_id} has situations {sorted(sits)}')
            for sit in sits:
                all_sits[sit] += 1
            per_seed_sit[(k, '+'.join(sorted(sits)))] += 1
            if H.fingerprint(folder) != recorded.get((k, split, scene_id)):
                problems.append(f'seed {k}: {scene_id} fingerprint differs from the manifest')
    counts = collections.Counter(per_seed_n.values())
    sit_profiles = collections.Counter(
        tuple(sorted((s, n) for (kk, s), n in per_seed_sit.items() if kk == k)) for k in SEEDS)
    print(json.dumps({'done_files': len(dones), 'scenes_per_seed': dict(counts),
                      'situation_profile_per_seed': {str(p): n for p, n in sit_profiles.items()},
                      'scenes_by_situation_total': dict(all_sits),
                      'problems': problems}, indent=1), flush=True)
    return 1 if problems else 0


def repro(k: int, tmp: Path):
    check_seed(k)
    tmp = Path(tmp)
    assert not tmp.exists(), f'{tmp} exists: the reproduction goes into a new folder'
    assert tmp.resolve().is_relative_to(HERE.resolve()), tmp
    tmp.mkdir(parents=True)
    info = one_seed(k, str(tmp))
    ref = json.loads((OUT / f'seed{k}' / 'DONE.json').read_text())
    a = {(s['split'], s['scene_id']): s['sha256'] for s in ref['scenes']}
    b = {(s['split'], s['scene_id']): s['sha256'] for s in info['scenes']}
    same = a == b
    diff = sorted(key for key in a.keys() | b.keys() if a.get(key) != b.get(key))
    print(json.dumps({'seed': k, 'scenes_ref': len(a), 'scenes_repro': len(b), 'identical': same,
                      'differing': [list(x) for x in diff], 'rejections_ref': ref['rejections'],
                      'rejections_repro': info['rejections']}, indent=1), flush=True)
    return 0 if same else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest-only', action='store_true')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--repro', type=int)
    ap.add_argument('--tmp', type=Path)
    args = ap.parse_args()
    for k in SEEDS:
        check_seed(k)
    if args.manifest_only:
        write_manifest()
        return 0
    if args.verify:
        return verify()
    if args.repro is not None:
        return repro(args.repro, args.tmp)
    return generate()


if __name__ == '__main__':
    raise SystemExit(main())
