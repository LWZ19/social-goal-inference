"""Verifier: fresh seeds hold only seeds 100-109, 26 scenes each, and no held-out situation in any episode."""
import collections, json, sys
from pathlib import Path
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
from sgi.config import load_config
FRESH = Path('results/still-scenes/perception/fresh_seeds')
cfg = load_config()
held = set(cfg.held_out_situations)
print("held_out_situations", sorted(held), "dev", sorted(cfg.dev_situations))
print("top-level", sorted(p.name for p in FRESH.iterdir()))
scene_sit = collections.Counter(); ep_sit = collections.Counter(); bad = []
man = json.loads((FRESH / 'manifest.json').read_text())
mf = {(s['seed'], sc['scene_id']): sc['sha256'] for s in man['seeds'] for sc in s['scenes']}
nfp = 0; fpbad = []
for seed in sorted(FRESH.glob('seed*')):
    n = 0
    for split in sorted(seed.iterdir()):
        if not split.is_dir():
            continue
        for sc in sorted(split.iterdir()):
            H.guard(sc.name)
            d = json.loads((sc / 'scene.json').read_text())
            sits = [e['ground_truth']['situation'] for e in d['episodes']]
            ep_sit.update(sits)
            if set(sits) & held or any(s in sc.name for s in held):
                bad.append(str(sc))
            scene_sit[sc.name.split('_', 2)[2].rsplit('_', 1)[0] if sc.name[-1].isdigit() else sc.name.split('_', 2)[2]] += 1
            n += 1
            k = int(seed.name[4:])
            fp = H.fingerprint(sc)
            nfp += 1
            if mf.get((k, sc.name)) != fp:
                fpbad.append(sc.name)
    print(seed.name, n, "scenes")
print("episode situations", dict(ep_sit))
print("held-out hits", bad)
print("fingerprints checked", nfp, "mismatches", fpbad)
print("manifest totals", man['totals'], "commit", man['worktree_commit'], "modified", man['src_or_configs_modified'])
print("rejections", [s['rejections'] for s in man['seeds']][:2], "skipped rows seed100", man['seeds'][0]['skipped_rows'])
