"""I03 Stage A: the fit data (FROZEN.md section 1), loaded. F-real: the 21 realistic worked examples; F-ph: the 8
placeholder worked examples plus the 260 fresh-seed scenes (seeds 100-109). Worked examples load through
research_harness.example under the default config; fresh seeds with sgi.scene.load_scene (they are not on the
harness's lists and are the only scenes read outside it). No evaluation scene is loaded here."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, 'docs/model_notes/tools')
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1b/i03')
import research_harness as H  # noqa: E402
import people_scorer as PS  # noqa: E402

FRESH = Path('results/2026-10-17-beyond-vlm/phase1b/fresh_seeds')


def fit_refs(require_fresh=True):
    """[(domain, set, scene id, loader)] for every fit scene, in a fixed order; require_fresh=False skips seeds
    that are not complete yet (for the self-check only)."""
    out = []
    for folder, scene_id in H.examples():
        s = str(folder)
        if 'realistic_final' in s:
            arm = s.split('realistic_final/')[1].split('/')[0]
            out.append(('F-real', arm, scene_id, ('example', folder, scene_id)))
        elif s.endswith('placeholder/examples'):
            out.append(('F-ph', 'ph_examples', scene_id, ('example', folder, scene_id)))
        else:
            raise SystemExit(f'unknown examples folder {folder}')
    for k in range(100, 110):
        done = FRESH / f'seed{k}' / 'DONE.json'
        if not require_fresh and not done.exists():
            continue
        info = json.loads(done.read_text())
        for s in info['scenes']:
            assert k >= 100
            out.append(('F-ph', f'seed{k}', s['scene_id'], ('fresh', FRESH / f'seed{k}' / s['split'], s['scene_id'])))
    return out


def load(loader):
    """A scene object with .observation and .truth, and its thresholds (default config)."""
    kind, folder, scene_id = loader
    if kind == 'example':
        sc = H.example(scene_id, folder)
        return sc, PS.default_thresholds()
    from sgi.scene import load_scene
    H.guard_scene(folder / scene_id)
    sc = PS.FreshScene(load_scene(folder / scene_id))
    assert sc.inputs.scene_id == scene_id
    return sc, PS.default_thresholds()
