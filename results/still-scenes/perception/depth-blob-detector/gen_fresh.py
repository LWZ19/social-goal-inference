"""I03 phase 1b: the fresh fit seeds 100-109, as FROZEN.md section 1 fixes them. A single process.

For each seed k: rng = default_rng(k) over every row of _SPLITS, then more = default_rng([k, _MORE_STREAM]) over
_MORE_SPLITS with indices from len(dict(_SPLITS)[split]); a row is written with the package's own _write_scene only
when its situation is in cfg.dev_situations, and a row of any other situation (group_photo, lift_wait) is skipped
before any random draw, so no held-out scene is ever rendered. Output: perception/fresh_seeds/seed<k>/{eval,examples}/.

A seed is complete when seed<k>/DONE.json exists; a complete seed is skipped, an incomplete one is removed and
written again from its first row (one random stream per seed). Seeds 3-99 are refused. No model, no network.
Writes fresh_seeds/manifest.json and logs the time per scene to stdout.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402
from sgi.config import DEFAULT_CONFIG, load_config  # noqa: E402
from sgi.scene import read_scene_file  # noqa: E402
from sgi.synth.generate import _MORE_SPLITS, _MORE_STREAM, _REASONS, _SPLITS, _write_scene  # noqa: E402

PHASE = paths.STILL / 'perception'
OUT = PHASE / 'fresh_seeds'
SEEDS = tuple(range(100, 110))
WT = paths.REPO


def rows_of(cfg):
    """(pass, split, index, row) in the order the generator draws them, held-out rows flagged."""
    out = []
    for split, rows in _SPLITS:
        for index, row in enumerate(rows):
            out.append(('first', split, index, row))
    for split, rows in _MORE_SPLITS:
        for index, row in enumerate(rows, start=len(dict(_SPLITS)[split])):
            out.append(('more', split, index, row))
    return out


def one_seed(k: int, cfg) -> dict:
    assert k >= 100, f"seed {k} refused: seeds 3-99 are reserved for I46"
    seed_dir = OUT / f'seed{k}'
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
    for which, split, index, row in rows_of(cfg):
        if row[0] not in cfg.dev_situations:          # before any random draw
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
    bad = [s['scene_id'] for s in scenes if set(s['situations']) & held]
    assert not bad, f'held-out episodes written: {bad}'
    for s in scenes:
        s['sha256'] = H.fingerprint(seed_dir / s['split'] / s['scene_id'])
    info = {'seed': k, 'scenes': scenes, 'skipped_rows': skipped, 'rejections': counts,
            'n_scenes': len(scenes), 'seconds': round(time.monotonic() - started, 1),
            'held_out_check': 'no written episode is of ' + ', '.join(sorted(held))}
    done.write_text(json.dumps(info, indent=1))
    return info


def main():
    cfg = load_config(DEFAULT_CONFIG)
    assert set(cfg.dev_situations) == {'no_one', 'waiting_line', 'conversation_group', 'audience'}, cfg.dev_situations
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [one_seed(k, cfg) for k in SEEDS]
    commit = subprocess.run(['git', '-C', str(WT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(WT), 'status', '--porcelain', 'src', 'configs'], capture_output=True,
                           text=True).stdout.strip()
    manifest = {'note': 'I03 phase 1b fresh fit seeds (FROZEN.md section 1); written by i03/gen_fresh.py',
                'config': str(DEFAULT_CONFIG), 'worktree_commit': commit, 'src_or_configs_modified': dirty or None,
                'seeds': seeds,
                'totals': {'scenes': sum(s['n_scenes'] for s in seeds),
                           'held_out_episodes_written': 0}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {manifest['totals']['scenes']} scenes over {len(seeds)} seeds", flush=True)


if __name__ == '__main__':
    main()
