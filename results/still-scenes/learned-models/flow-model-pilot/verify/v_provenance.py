"""Verifier step 3 (project venv): provenance and leakage checks. Reads only phase2/flow_pilot/, phase2/seeds/
placeholder (DONE.json and scene.json after H.guard_scene) and the harness-loaded development arrays I rebuilt.
Writes verify/out/provenance.json."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v_common as V  # noqa: E402
import research_harness as H  # noqa: E402

P = Path(V.PILOT)
SEEDS = Path(V.SEEDS)
res = {}


def mtime(p):
    return dt.datetime.fromtimestamp(os.stat(p).st_mtime).isoformat(timespec='seconds')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# 1. the note and the hash files
res['frozen_sha256'] = sha(P / 'FROZEN.md')
res['frozen_sha256_ok'] = res['frozen_sha256'] == 'a60f1d623168d5455a38e889d246fc3965d84d63a447641401bf2c5565da767f'
res['frozen_sha256_file'] = (P / 'FROZEN.sha256').read_text().split()[0]
res['deviations_sha256_ok'] = sha(P / 'DEVIATIONS.md') == (P / 'DEVIATIONS.sha256').read_text().split()[0]
ck = {}
for line in (P / 'CHECKPOINTS.sha256').read_text().splitlines():
    h, f = line.split()
    ck[f] = dict(listed=h, now=sha(P / f), ok=h == sha(P / f), mtime=mtime(P / f))
res['checkpoints'] = ck
times = {}
for f in ('FROZEN.md', 'FROZEN.sha256', 'prefreeze/fmt_check.log', 'prefreeze/fmt_check2.log', 'DEVIATIONS.md',
          'DEVIATIONS.sha256', 'CHECKPOINTS.sha256', 'data/train.npz', 'data/val.npz', 'data/dev.npz', 'data/manifest.json',
          'models/main/best.npz', 'models/loso_audience/best.npz', 'samples/val_main.npz', 'samples/dev_b_main.npz',
          'tables.txt', 'README.md'):
    times[f] = mtime(P / f)
seed_files = sorted((P / 'data' / 'seeds').glob('seed*.npz'))
times['data/seeds first'] = min(mtime(f) for f in seed_files)
times['data/seeds last'] = max(mtime(f) for f in seed_files)
times['data/png first'] = min(mtime(f) for f in (P / 'data' / 'png').glob('*.png'))
code_times = {f.name: mtime(f) for f in sorted((P / 'code').glob('*'))}
done_times = [mtime(SEEDS / f'seed{k}' / 'DONE.json') for k in range(110, 200)]
times['seed DONE.json last'] = max(done_times)
times['seed manifest'] = mtime(SEEDS.parent / 'manifest_placeholder.json')
log_starts = {}
for f in sorted((P / 'logs').glob('*.log')):
    first = f.read_text().splitlines()[:1]
    log_starts[f.name] = first[0][:60] if first else ''
res['times'] = times
res['code_times'] = code_times
res['log_first_lines'] = log_starts
# everything in the pilot folder older than the note?
older = []
for root, dirs, files in os.walk(P):
    if '/verify' in root or root.endswith('/verify'):
        continue
    for f in files:
        p = Path(root) / f
        if os.stat(p).st_mtime < os.stat(P / 'FROZEN.md').st_mtime:
            older.append(str(p.relative_to(P)))
res['files_older_than_FROZEN'] = sorted(older)

# 2. training data keys
tr = np.load(P / 'data' / 'train.npz')
va = np.load(P / 'data' / 'val.npz')
seed_re = re.compile(r'^seed(\d+)/(eval|examples)/[^/]+/e\d+$')


def seeds_of(keys):
    out = set()
    for k in keys:
        m = seed_re.match(str(k))
        assert m, k
        out.add(int(m.group(1)))
    return out


trs, vas = seeds_of(tr['keys']), seeds_of(va['keys'])
res['train_keys'] = dict(n=int(len(tr['keys'])), seeds_min=min(trs), seeds_max=max(trs), n_seeds=len(trs),
                         all_110_189=trs == set(range(110, 190)), labelled=int(tr['has_label'].sum()),
                         label_poses=int(tr['has_label'].sum()) * 32,
                         by_situation={s: int((tr['situation'] == s).sum()) for s in np.unique(tr['situation'])},
                         labelled_by_situation={s: int(((tr['situation'] == s) & tr['has_label']).sum())
                                                for s in np.unique(tr['situation'])})
res['val_keys'] = dict(n=int(len(va['keys'])), seeds=sorted(vas), all_190_199=vas == set(range(190, 200)),
                       labelled=int(va['has_label'].sum()))
res['train_val_key_overlap'] = len(set(map(str, tr['keys'])) & set(map(str, va['keys'])))
# the joined arrays are the per-seed arrays in order
cat_keys = []
for k in range(110, 190):
    cat_keys += [str(x) for x in np.load(P / 'data' / 'seeds' / f'seed{k}.npz')['keys']]
res['train_npz_is_seeds_110_189_in_order'] = cat_keys == [str(x) for x in tr['keys']]

# 3. dev records vs training: keys, raster identity
dv = np.load(P / 'data' / 'dev.npz')
dev_keys = [str(k) for k in dv['keys']]
res['dev_n'] = len(dev_keys)
res['dev_train_key_overlap'] = len(set(dev_keys) & set(map(str, tr['keys'])))
hb = lambda a: hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()  # noqa: E731
train_h = {hb(r) for r in tr['rasters']}
val_h = {hb(r) for r in va['rasters']}
dev_h = {hb(r) for r in dv['raster_b']} | {hb(r) for r in dv['raster_a']}
res['dev_rasters_identical_to_a_train_raster'] = len(dev_h & train_h)
res['dev_rasters_identical_to_a_val_raster'] = len(dev_h & val_h)
res['val_rasters_identical_to_a_train_raster'] = len(val_h & train_h)
# nearest train raster to each dev raster (b), by L1 over the floor+people channels, on a 4x-downsampled copy
trd = tr['rasters'][:, :2, ::4, ::4].reshape(len(tr['rasters']), -1).astype(np.float32)
dvd = dv['raster_b'][:, :2, ::4, ::4].reshape(len(dv['raster_b']), -1).astype(np.float32)
mins = []
for i in range(0, len(dvd), 4):
    d = np.abs(dvd[i:i + 4, None, :] - trd[None, :, :]).sum(-1)
    mins += list(d.min(1))
res['dev_to_train_min_L1_downsampled'] = dict(min=float(np.min(mins)), median=float(np.median(mins)),
                                               n_zero=int(np.sum(np.array(mins) == 0)))
# development scene names / run folders appearing anywhere in the training keys
dev_scenes = sorted({k.split('/')[1] for k in dev_keys})
res['dev_scene_names_in_train_keys'] = sum(any(f'/{s}/' in str(k) for s in dev_scenes) for k in tr['keys'])
res['dev_scene_examples'] = dev_scenes[:5]

# 4. seeds 110-199: held-out situations
seed_dirs = sorted(p.name for p in SEEDS.iterdir() if p.is_dir())
res['seed_dirs'] = dict(n=len(seed_dirs), first=seed_dirs[0], last=seed_dirs[-1],
                        all_110_199=seed_dirs == [f'seed{k}' for k in range(110, 200)])
bad, n_sc, n_ep, sit_count, listed_vs_disk = [], 0, 0, {}, []
for k in range(110, 200):
    sd = SEEDS / f'seed{k}'
    done = json.loads((sd / 'DONE.json').read_text())
    listed = {(s['split'], s['scene_id']) for s in done['scenes']}
    on_disk = {(split, p.name) for split in ('eval', 'examples') if (sd / split).exists() for p in (sd / split).iterdir()
               if p.is_dir()}
    if listed != on_disk:
        listed_vs_disk.append((k, sorted(listed ^ on_disk)))
    for split, sid in sorted(on_disk):
        folder = sd / split / sid
        H.guard_scene(folder)
        d = json.loads((folder / 'scene.json').read_text())
        n_sc += 1
        if re.search(r'eval_1[6-9](?![0-9])', sid) or 'group_photo' in sid or 'lift_wait' in sid:
            bad.append(('name', k, sid))
        for ep in d['episodes']:
            n_ep += 1
            s = ep['ground_truth']['situation']
            sit_count[s] = sit_count.get(s, 0) + 1
            if s not in V.DEV_SITS:
                bad.append(('situation', k, sid, s))
res['seeds_scan'] = dict(scenes=n_sc, episodes=n_ep, situations=sit_count, held_out_or_non_dev=bad,
                         listed_vs_disk_mismatch=listed_vs_disk)
(Path(V.VER) / 'out' / 'provenance.json').write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
