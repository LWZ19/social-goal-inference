"""C5 flow pilot 6(ii)-(iii): development records, scored by the package scorer (H.score_poses: the record's
recorded result and target, the run's thresholds) in the project venv.

For each sample file samples/dev_<setting>_<model>.npz (the sidecar's snapped samples, validity, top pose, hit@5
places and raw top pose), per record:
  top1        the top pose's success (the solver's place under the fallback);
  psucc       P(success): mean success of the valid snapped samples; under the fallback the success of the solver's
              place (choice); beside it psucc_all64 (all 64 snapped samples) and valid_share;
  hit         hit@1..5 over the places (carried forward when the list stops);
  raw_top1    the raw (unsnapped) top pose's success (density over all 64 raw samples; secondary).
Writes rows_<setting>_<cell>.jsonl per primary cell (main model) and rows_b_<cell>_<loso model>.jsonl.
No model call, no network, no torch.
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import research_harness as H  # noqa: E402
from build_dev import plan  # noqa: E402

OUT = Path(C.OUT)
FILES = [('b', 'main'), ('a', 'main'), ('b', 'loso_waiting_line'), ('b', 'loso_conversation_group'),
         ('b', 'loso_audience')]


def records_by_key():
    out = {}
    for label, cell, ref in plan():
        for rec in H.records(ref):
            if label == 'watch' and rec.situation != 'audience':
                continue
            if not rec.solver_step:
                continue
            out[f"{rec.ref.folder}/{rec.scene_id}/{rec.episode_id}/{rec.trial}"] = rec
    return out


def main():
    dev = json.loads((OUT / 'data' / 'dev.json').read_text())
    meta = {m['key']: m for m in dev['records']}
    recs = records_by_key()
    assert set(recs) == set(meta), (len(recs), len(meta))
    t0 = time.monotonic()
    for setting, model in FILES:
        s = np.load(OUT / 'samples' / f'dev_{setting}_{model}.npz')
        rows = collections.defaultdict(list)
        for i, key in enumerate(s['keys']):
            key = str(key)
            rec, m = recs[key], meta[key]
            top = tuple(float(v) for v in s['top'][i])
            npl = int(s['n_places'][i])
            places = [tuple(float(v) for v in s['places'][i][j]) for j in range(npl)]
            snapped = [(float(s['sp'][i][j, 0]), float(s['sp'][i][j, 1]), float(s['head'][i][j])) for j in range(64)]
            raw_top = tuple(float(v) for v in s['raw_top'][i])
            poses = [top] + places + snapped + [raw_top]
            uniq = list(dict.fromkeys(poses))
            sc = dict(zip(uniq, H.score_poses(rec, uniq)))
            ok = {p: bool(v.success) for p, v in sc.items()}
            valid = s['valid'][i].astype(bool)
            fb = bool(s['fallback'][i])
            succ64 = np.array([ok[p] for p in snapped])
            if fb:
                assert top == places[0]
                psucc = float(ok[top])
            else:
                psucc = float(succ64[valid].mean())
            hit = []
            for kk in range(5):
                hit.append(int(any(ok[p] for p in places[:kk + 1])))
            row = dict(key=key, setting=setting, model_flow=model, cell=m['cell'], model=m['model'], arm=m['arm'],
                       sit=m['sit'], layout=m['layout'], scene=m['scene'], target_ok=m['target_ok'],
                       top1=bool(ok[top]), psucc=psucc, psucc_all64=float(succ64.mean()),
                       valid_share=float(valid.mean()), fallback=fb, n_places=npl, hit=hit,
                       raw_top1=bool(ok[raw_top]), top_in_region=bool(sc[top].in_region),
                       top_heading_ok=bool(sc[top].heading_ok), top_reach=bool(sc[top].reachable),
                       rho_room=m[f'rho_{setting}'], C_top1=m['C_top1'], P_top1=m['P_top1'],
                       latency_s=None if np.isnan(s['latency_s'][i]) else float(s['latency_s'][i]),
                       top_forbidden={k: bool(v) for k, v in sc[top].forbidden.items()})
            rows[m['cell']].append(row)
        for cell, rr in rows.items():
            name = f'rows_{setting}_{cell}.jsonl' if model == 'main' else f'rows_{setting}_{cell}_{model}.jsonl'
            with open(OUT / name, 'w') as f:
                for r in rr:
                    f.write(json.dumps(r) + '\n')
        print(f"{time.monotonic() - t0:7.1f} s  {setting} {model}: {sum(len(v) for v in rows.values())} records",
              flush=True)


if __name__ == '__main__':
    main()
