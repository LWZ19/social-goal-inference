"""Extra checks: my P verdicts vs I01 rows_old_*; unlabelled training units by row; LOSO curves' best epochs; by-row
table (post hoc in the compute agent's write-up) from my proto rows."""
import collections, json
from pathlib import Path
import numpy as np
R = Path('results/2026-10-17-beyond-vlm')
P = R / 'phase2/flow_pilot'
out = {}
meta = {}
for p in sorted((P / 'verify/data').glob('vdev_*.json')):
    for m in json.loads(p.read_text()):
        meta[m['key']] = m
i01 = {}
for f in sorted((R / 'phase1/i01_repairs').glob('rows_old_*.jsonl')):
    for line in f.read_text().splitlines():
        r = json.loads(line)
        i01[f"{r['run']}/{r['scene']}/{r['episode']}/{r['trial']}"] = r
real = [k for k, m in meta.items() if m['cell'].startswith('old_')]
out['P_vs_I01'] = dict(n=len(real), in_i01=sum(k in i01 for k in real), mismatch=sum(bool(i01[k]['P']) != meta[k]['P_top1'] for k in real if k in i01))
out['C_vs_I01_C'] = dict(mismatch=sum(bool(i01[k]['C']) != meta[k]['C_rec'] for k in real if k in i01))
tr = np.load(P / 'data/train.npz')
unl = [str(k) for k, h in zip(tr['keys'], tr['has_label']) if not h]
out['train_unlabelled'] = dict(n=len(unl), rows=dict(collections.Counter(k.split('/')[2].split('_', 2)[2] for k in unl)))
va = np.load(P / 'data/val.npz')
unv = [str(k) for k, h in zip(va['keys'], va['has_label']) if not h]
out['val_unlabelled'] = dict(n=len(unv), rows=dict(collections.Counter(k.split('/')[2].split('_', 2)[2] for k in unv)))
for name in ('main', 'loso_waiting_line', 'loso_conversation_group', 'loso_audience'):
    c = json.loads((P / f'models/{name}/curve.json').read_text())
    e = json.loads((P / f'models/{name}/env.json').read_text())
    out[f'curve_{name}'] = dict(epochs=len(c['epoch']), argmin_epoch=int(np.argmin(c['val_loss'])) + 1, min_val=float(min(c['val_loss'])),
                                env_best_epoch=e['best_epoch'], env_best=e['best_val_loss'], wall_s=e['wall_s'], stopped=e['stopped'],
                                left_out=e['left_out'], train_units=e['train_units'], val_units=e['val_units'],
                                train_by_sit=e['train_units_by_situation'], val_by_sit=e['val_units_by_situation'])
rows = [json.loads(x) for x in (P / 'verify/out/dev_b_main_proto.jsonl').read_text().splitlines() if x.strip()]
br = collections.defaultdict(lambda: [0, 0, 0, 0])
for r in rows:
    row = r['scene'].split('eval_', 1)[1].split('_', 1)[1]
    b = br[row]; b[0] += 1; b[1] += r['C']; b[2] += r['top1']; b[3] += r['fallback']
out['by_row_b_proto (n, C, flow, fallbacks)'] = dict(sorted(br.items()))
(P / 'verify/out/extra.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
