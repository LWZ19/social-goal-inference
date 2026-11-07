"""C5 flow pilot: the tables and C5's criteria (FROZEN.md sections 6-7), computed once from the rows. Exploratory.
Writes tables.txt and summary.json. No model call, no network, no torch.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = Path(C.OUT)
I04 = Path(C.ROOT) / 'phase1d' / 'i04'
CELLS = ('old_qwen', 'old_glm', 'old_deepseek', 'watch_qwen', 'watch_glm', 'watch_deepseek')
MODELS = (('qwen', 'Qwen 3.8 (qwen38_low)'), ('glm', 'GLM-5.3-Flash (glm53)'), ('deepseek', 'DeepSeek V4.1 Flash (dsv41)'))
SITS = ('no_one', 'waiting_line', 'conversation_group', 'audience')
LOSO = ('loso_waiting_line', 'loso_conversation_group', 'loso_audience')
EXPL = ('EXPLORATORY. Gate 1 overridden by the project; train and test layouts share the generator templates (22 '
        'configurations); labels are the project\'s reference rules, so agreement measures fit to those rules, not '
        'social correctness.')

lines = []


def p(*a):
    lines.append(' '.join(str(x) for x in a))


def jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else f"{k}/0"


def model_of(cell):
    return cell.split('_', 1)[1]


def main():
    summ = {}
    p('C5 (I52) flow-matching pilot: tables. Protocol: FROZEN.md (sha256 a60f1d62...767f); deviations and '
      'clarifications: DEVIATIONS.md (hash in DEVIATIONS.sha256); checkpoints hashed in CHECKPOINTS.sha256 before '
      'any evaluation sample was drawn.')
    p(EXPL)
    p('')
    # ------------------------------------------------------------------------------------------- data
    man = json.loads((OUT / 'data' / 'manifest.json').read_text())
    p('== D1. Data (seeds 110-189 train, 190-199 validation; units = (scene, episode); e0 and e1 both kept) ==')
    for part in ('train', 'val'):
        m = man[part]
        p(f"{part}: {m['n_units']} units in {m['n_scenes']} scenes; left out: {dict(m['left_out']) or 'none'}")
        for s, v in m['by_situation'].items():
            p(f"  {s:20s} units {v['units']:5d} scenes {v['scenes']:5d} labelled {v['labelled']:5d} "
              f"(label poses {v['label_poses']}); unlabelled {v['units'] - v['labelled']}; lift failed "
              f"{v['lift_failed']}; seen face none {v['face_none']}; heading not shared {v['heading_not_shared']}; "
              f"robot disc in raster {v['robot_disc_in_raster']}")
        p(f"  unlabelled units by situation / row: {m['unlabelled_by_situation_row']}")
    p('  share of the extended region cells inside the raster, by row (train): '
      + ', '.join(f"{k}: {v}" for k, v in man['train']['region_share_inside_raster_by_row'].items()))
    sc = man['train']['label_selfcheck']
    p('  label self-check (train, I14 replica on every label pose): '
      + '; '.join(f"{s}: n {v['n']}, in region {v['share_inreg']:.4f}, heading ok {v['share_heading_ok']:.4f}, "
                  f"reachable {v['share_reach']:.4f}, all three {v['share_success']:.4f}" for s, v in sc.items()))
    p('')
    # -------------------------------------------------------------------------------------- training
    p('== D2. Training (rectified flow; AdamW 1e-3, wd 0.01, cosine over 30 epochs; batch 256; fp32; seed 0) ==')
    p(f"{'model':26s} {'left out':20s} {'tr units':>8s} {'tr poses':>8s} {'va poses':>8s} {'epochs':>6s} "
      f"{'best ep':>7s} {'best val':>9s} {'wall s':>7s} {'peak MiB':>8s} stopped")
    summ['training'] = {}
    for name in ('main',) + LOSO:
        e = json.loads((OUT / 'models' / name / 'env.json').read_text())
        cur = json.loads((OUT / 'models' / name / 'curve.json').read_text())
        summ['training'][name] = dict(best_epoch=e['best_epoch'], best_val=e['best_val_loss'], wall_s=e['wall_s'],
                                      epochs=len(cur['epoch']), stopped=e['stopped'],
                                      peak_alloc_mib=e['peak_mem_allocated_mib'])
        p(f"{name:26s} {e['left_out']:20s} {e['train_units']:8d} {e['train_poses']:8d} {e['val_poses']:8d} "
          f"{len(cur['epoch']):6d} {e['best_epoch']:7d} {e['best_val_loss']:9.5f} {e['wall_s']:7.1f} "
          f"{e['peak_mem_allocated_mib']:8.0f} {e['stopped']}")
    e = json.loads((OUT / 'models' / 'main' / 'env.json').read_text())
    p(f"env: {e['gpu']}, driver {e['driver']}, torch {e['torch']}, CUDA {e['cuda']}, cuDNN {e['cudnn']}, "
      f"{e['params']} parameters")
    p('')
    # ---------------------------------------------------------------------------------- (i) validation
    p('== T1. (i) Validation seeds 190-199 (in-distribution test; also picked the checkpoint, so mildly optimistic) ==')
    p('Success = in the target\'s acceptable region, reachable on robot_grid(truth), heading within 45 deg (I14 '
      'replica, checked against sgi.evaluate.score). Flow: 64 raw samples per unit; top pose = densest of the 64 raw '
      'samples. Coverage = share of region cells (0.10 m lattice extended beyond the raster, on the floor) with a raw '
      'sample within 0.2 m; n = units with at least one region cell. B0/B1 = solver top-1 with oracle people, oracle '
      'target and standard_goal (B1: reference constraints, circular, a reference only).')
    vrows = {m: jl(OUT / f'rows_val_{m}.jsonl') for m in ('main',) + LOSO}
    brows = {r['key']: r for r in jl(OUT / 'rows_val_baselines.jsonl')}
    p(f"{'situation':20s} {'n ep':>5s} {'n sc':>5s} | {'sample succ':>11s} | {'coverage (n)':>14s} | "
      f"{'flow top-1':>16s} | {'B0 top-1':>16s} | {'B1 top-1':>16s}")
    summ['val'] = {}
    for s in SITS + ('POOLED',):
        rr = [r for r in vrows['main'] if s == 'POOLED' or r['sit'] == s]
        n = len(rr)
        nsc = len({(r['seed'], r['split'], r['scene']) for r in rr})
        ss = np.mean([r['share_success'] for r in rr])
        cov = [r['coverage'] for r in rr if r['coverage'] is not None]
        top = sum(r['top_success'] for r in rr)
        b0 = sum(brows[r['key']]['B0'] for r in rr)
        b1 = sum(brows[r['key']]['B1'] for r in rr)
        summ['val'][s] = dict(n=n, n_scenes=nsc, share_success=float(ss), coverage=float(np.mean(cov)) if cov else None,
                              n_cov=len(cov), top=top, B0=b0, B1=b1)
        p(f"{s:20s} {n:5d} {nsc:5d} | {ss:11.3f} | {np.mean(cov) if cov else float('nan'):6.3f} ({len(cov):5d}) | "
          f"{pct(top, n):>16s} | {pct(b0, n):>16s} | {pct(b1, n):>16s}")
    lab = [r for r in vrows['main'] if not r['labelled']]
    p(f"units without labels (no free region cell in the raster): {len(lab)}; flow top-1 on them "
      f"{sum(r['top_success'] for r in lab)}; their rows: {sorted(collections.Counter(r['scene'].split('_', 2)[2] for r in lab).items())}")
    lift = sum(r['lift_failed'] for r in vrows['main'])
    p(f"oracle lift failed (3-D anchor used): {lift} units")
    p('')
    p('== T2. (i) Leave-one-situation-out: each LOSO model on its left-out situation, beside the main model on the same '
      'units ==')
    p(f"{'left out':20s} {'n ep':>5s} {'n sc':>5s} | {'sample succ LOSO/main':>22s} | {'coverage LOSO/main':>19s} | "
      f"{'top-1 LOSO':>16s} | {'top-1 main':>16s} | {'B0':>16s}")
    summ['val_loso'] = {}
    for m in LOSO:
        s = m[len('loso_'):]
        rl = {r['key']: r for r in vrows[m]}
        rm = [r for r in vrows['main'] if r['sit'] == s]
        assert set(rl) == {r['key'] for r in rm}
        n = len(rm)
        nsc = len({(r['seed'], r['split'], r['scene']) for r in rm})
        ssl, ssm = np.mean([rl[r['key']]['share_success'] for r in rm]), np.mean([r['share_success'] for r in rm])
        cl = [rl[r['key']]['coverage'] for r in rm if r['coverage'] is not None]
        cm = [r['coverage'] for r in rm if r['coverage'] is not None]
        tl, tm = sum(rl[r['key']]['top_success'] for r in rm), sum(r['top_success'] for r in rm)
        b0 = sum(brows[r['key']]['B0'] for r in rm)
        summ['val_loso'][s] = dict(n=n, share_loso=float(ssl), share_main=float(ssm), top_loso=tl, top_main=tm, B0=b0)
        p(f"{s:20s} {n:5d} {nsc:5d} | {ssl:10.3f} / {ssm:9.3f} | {np.mean(cl):8.3f} / {np.mean(cm):8.3f} | "
          f"{pct(tl, n):>16s} | {pct(tm, n):>16s} | {pct(b0, n):>16s}")
    p('')
    # ------------------------------------------------------------------------------------ dev records
    rows = {}
    for st in ('b', 'a'):
        for c in CELLS:
            rows[(st, c)] = jl(OUT / f'rows_{st}_{c}.jsonl')
    i04 = {}
    for f in sorted(I04.glob('rows_old_*.jsonl')) + sorted(I04.glob('rows_watch_*.jsonl')):
        for r in jl(f):
            i04[r['key']] = r
    allb = [r for c in CELLS for r in rows[('b', c)]]
    alla = [r for c in CELLS for r in rows[('a', c)]]
    assert len(allb) == 549 and len(alla) == 549
    assert all(r['key'] in i04 for r in allb)
    lat = json.loads((OUT / 'samples' / 'latency.json').read_text())
    dev = json.loads((OUT / 'data' / 'dev.json').read_text())

    def agg(rr, setting):
        n = len(rr)
        t1 = sum(r['top1'] for r in rr)
        ps = sum(r['psucc'] for r in rr)
        rho = sum(r['rho_room'] for r in rr)
        hit = [sum(r['hit'][k] for r in rr) for k in range(5)]
        return dict(n=n, top1=t1, top1_pct=100 * t1 / n, psucc_sum=ps, psucc_mean=ps / n,
                    psucc64=float(np.mean([r['psucc_all64'] for r in rr])),
                    valid_share=float(np.mean([r['valid_share'] for r in rr])),
                    fallback=sum(r['fallback'] for r in rr), rho_sum=rho,
                    lift_room=ps / rho if rho else float('nan'), lift_i14=t1 / rho if rho else float('nan'),
                    hit=hit, raw_top1=sum(r['raw_top1'] for r in rr),
                    C=sum(r['C_top1'] for r in rr), P=sum(r['P_top1'] for r in rr))

    def i04agg(rr, v):
        S = sum(i04[r['key']][v]['S'] for r in rr)
        rho = sum(i04[r['key']]['rho_room'] for r in rr)
        return dict(meanS=S / len(rr), lift=S / rho, hit5=sum(i04[r['key']][v]['hit_pkg'][4] for r in rr),
                    lex_hit5=sum(i04[r['key']]['lex_hit'][4] for r in rr))

    groups = [(c, lambda r, c=c: r['cell'] == c) for c in CELLS]
    groups += [(f'MODEL {m}', lambda r, m=m: model_of(r['cell']) == m) for m, _ in MODELS]
    groups += [('POOLED 549', lambda r: True)]
    summ['dev'] = {}
    for st, title, ref in (('b', 'T3. (ii) Setting (b): each model\'s recorded people (the fair comparison with C)', 'C'),
                           ('a', 'T4. (iii) Setting (a): true people, I01 P\'s people and constraints (upper bound)', 'P')):
        rr_all = allb if st == 'b' else alla
        p(f'== {title} ==')
        p(f"Package scorer (H.score_poses, recorded target). {ref} top-1 = "
          + ('recorded C (I14 S1)' if ref == 'C' else 'P\'s own solver top-1, recomputed with repairs.py\'s code')
          + '. P(success) = mean success of the valid snapped samples (fallback: the solver\'s place); Lift_room = '
          'sum P(success) / sum rho_room; I14 form = sum top-1 / sum rho_room; hit@k over the flow\'s places.')
        p(f"{'group':18s} {'n':>4s} | {ref + ' top-1':>15s} | {'flow top-1':>15s} | {'diff pp':>7s} | {'P(succ)':>7s} "
          f"{'all64':>6s} {'valid':>6s} {'fb':>3s} | {'Lift_room':>9s} {'I14 form':>8s} | hit@1..5 | raw top-1")
        for g, f in groups:
            rr = [r for r in rr_all if f(r)]
            a = agg(rr, st)
            refv = a['C'] if ref == 'C' else a['P']
            summ['dev'][f'{st}:{g}'] = dict(a, ref=refv, ref_pct=100 * refv / a['n'])
            p(f"{g:18s} {a['n']:4d} | {pct(refv, a['n']):>15s} | {pct(a['top1'], a['n']):>15s} | "
              f"{a['top1_pct'] - 100 * refv / a['n']:+7.1f} | {a['psucc_mean']:7.3f} {a['psucc64']:6.3f} "
              f"{a['valid_share']:6.3f} {a['fallback']:3d} | {a['lift_room']:9.2f} {a['lift_i14']:8.2f} | "
              f"{a['hit']} | {a['raw_top1']}")
        p('')
    # P reproduction against I01
    pm = [m for m in dev['records'] if m['set'] == 'old' and m['I01_P'] is not None and m['I01_P'] != m['P_top1']]
    p(f"P reproduction: realistic records compared with I01 rows_old_*: {dev['summary']['stats'].get('P_compared')}, "
      f"mismatches {len(pm)}" + (f": {[m['key'] for m in pm]}" if pm else ''))
    p('')
    # -------------------------------------------------------------------------------------- (iv) I04
    p('== T5. (iv) Comparison with I04 V1 (soft density) and V3 (Gaussian control), held-out settings, matched by key ==')
    p('I04: P(success) = mean S, Lift_room = sum S / sum rho_room, hit@5 = hit_pkg[4]. Flow: setting (b). C\'s '
      'lexicographic hit@5 beside. Note: I04 was fitted on the development records themselves (leave one layout out); '
      'the flow never saw a development record.')
    p(f"{'group':18s} {'n':>4s} | {'P(succ) flow/V1/V3':>22s} | {'Lift_room flow/V1/V3':>22s} | "
      f"{'hit@5 flow/V1/V3/C':>22s}")
    summ['i04'] = {}
    for g, f in groups:
        rr = [r for r in allb if f(r)]
        a = agg(rr, 'b')
        v1, v3 = i04agg(rr, 'V1'), i04agg(rr, 'V3')
        summ['i04'][g] = dict(flow=dict(psucc=a['psucc_mean'], lift=a['lift_room'], hit5=a['hit'][4]), V1=v1, V3=v3)
        p(f"{g:18s} {a['n']:4d} | {a['psucc_mean']:6.3f} {v1['meanS']:6.3f} {v3['meanS']:6.3f}   | "
          f"{a['lift_room']:6.2f} {v1['lift']:6.2f} {v3['lift']:6.2f}   | {a['hit'][4]:4d} {v1['hit5']:4d} "
          f"{v3['hit5']:4d} {v1['lex_hit5']:4d}")
    p('')
    # ------------------------------------------------------------------------------------ latency
    p('== T6. Latency on the 4060 (sidecar; arrays in host memory -> top pose and hit@5 places: encoder, 16 Euler '
      'steps over 64 samples, copy back, snap, validity, density, picks; 10 untimed warm-ups, then each record once) ==')
    for st in ('b', 'a'):
        L = lat[f'main_{st}']
        p(f"setting ({st}): n {L['n']}, median {1000 * L['median_s']:.2f} ms, p90 {1000 * L['p90_s']:.2f} ms, mean "
          f"{1000 * L['mean_s']:.2f} ms, max {1000 * L['max_s']:.2f} ms; torch peak alloc "
          f"{L['peak_mem_allocated_mib']:.0f} MiB; nvidia-smi {L['nvidia_smi']}")
    ds = dev['summary']
    p(f"reported apart (project venv, setting b, n {ds['n']}): painting the raster median "
      f"{1000 * ds['paint_b_s']['median']:.1f} ms (p90 {1000 * ds['paint_b_s']['p90']:.1f}); _over_floor (solver work C "
      f"already does) median {1000 * ds['over_floor_b_s']['median']:.1f} ms (p90 {1000 * ds['over_floor_b_s']['p90']:.1f}); "
      'the hand-off between the two environments (files on disk) was not timed per record.')
    p(f"env: {lat['env']['gpu']}, driver {lat['env']['driver']}, torch {lat['env']['torch']}, CUDA {lat['env']['cuda']}")
    p('')
    # ------------------------------------------------------------------------------------ criteria
    p('== T7. C5 criteria (FROZEN.md section 7; exploratory) ==')
    per = {}
    for m, label in MODELS:
        b = summ['dev'][f'b:MODEL {m}']
        a = summ['dev'][f'a:MODEL {m}']
        per[m] = dict(n=b['n'], flow_b=b['top1_pct'], C=b['ref_pct'], flow_a=a['top1_pct'], P=a['ref_pct'])
        p(f"{label:30s} n {b['n']}: flow(b) {b['top1_pct']:.1f}% vs C {b['ref_pct']:.1f}% ({b['top1_pct'] - b['ref_pct']:+.1f} pp); "
          f"flow(a) {a['top1_pct']:.1f}% vs P {a['ref_pct']:.1f}% ({a['top1_pct'] - a['ref_pct']:+.1f} pp)")
    s1_one = sum(per[m]['flow_b'] >= per[m]['C'] - 3 for m, _ in MODELS)
    s1_two = sum(abs(per[m]['flow_b'] - per[m]['C']) <= 3 for m, _ in MODELS)
    pool = summ['i04']['POOLED 549']
    S1 = s1_one >= 2
    S2 = pool['flow']['lift'] > pool['V1']['lift'] and pool['flow']['hit5'] > pool['V1']['hit5']
    med = lat['main_b']['median_s']
    S3 = med < 0.5
    K1 = all(per[m]['flow_a'] < per[m]['P'] - 10 for m, _ in MODELS)
    K2 = all(per[m]['flow_b'] < per[m]['C'] - 10 for m, _ in MODELS)
    K3 = pool['flow']['lift'] <= pool['V3']['lift'] and pool['flow']['psucc'] <= pool['V3']['meanS']
    p(f"S1 flow(b) top-1 >= C - 3 pp for >= 2 of 3 models (one-sided): {s1_one} of 3 -> {'met' if S1 else 'not met'} "
      f"(two-sided |diff| <= 3: {s1_two} of 3)")
    p(f"S2 pooled Lift_room flow {pool['flow']['lift']:.2f} > V1 {pool['V1']['lift']:.2f} and hit@5 flow "
      f"{pool['flow']['hit5']} > V1 {pool['V1']['hit5']} (strict): {'met' if S2 else 'not met'}")
    p(f"S3 median latency {1000 * med:.2f} ms < 500 ms: {'met' if S3 else 'not met'}")
    p(f"K1 flow(a) top-1 < P - 10 pp for every model: {'fires' if K1 else 'does not fire'}")
    p(f"K2 flow(b) top-1 < C - 10 pp for every model: {'fires' if K2 else 'does not fire'}")
    p(f"K3 pooled flow(b) Lift_room {pool['flow']['lift']:.2f} <= V3 {pool['V3']['lift']:.2f} and P(success) "
      f"{pool['flow']['psucc']:.3f} <= V3 mean S {pool['V3']['meanS']:.3f}: {'fires' if K3 else 'does not fire'}")
    kill = K1 or K2 or K3
    succ = S1 and S2 and S3
    verdict = 'kill' if kill else ('success' if succ else 'neither')
    p(f"VERDICT (kill overrides success; anything else is neither): {verdict.upper()} (exploratory)")
    p("C5's 'holds on fresh seeds' is reported from T1, not judged.")
    summ['criteria'] = dict(per_model=per, S1=S1, S1_one_sided_count=s1_one, S1_two_sided_count=s1_two, S2=S2, S3=S3,
                            K1=K1, K2=K2, K3=K3, verdict=verdict, median_latency_s=med)
    p('')
    # ------------------------------------------------------------------------------------ secondary
    p('== T8. Secondary (decides nothing): layout-cluster bootstrap of flow(b) - C top-1, percentage points ==')
    p('2,000 draws, default_rng(20261018); realistic layouts (13) drawn first (2000 x 13), then watching (2000 x 6); a '
      'model row uses its records in the drawn layouts of both strata; mean over drawn records; percentile 95%.')
    real_l = sorted({r['layout'] for r in allb if r['cell'].startswith('old_')})
    watch_l = sorted({r['layout'] for r in allb if r['cell'].startswith('watch_')})
    rng = np.random.default_rng(20261018)
    dr = rng.integers(0, len(real_l), size=(2000, len(real_l)))
    dw = rng.integers(0, len(watch_l), size=(2000, len(watch_l)))
    lid = {l: i for i, l in enumerate(real_l)}
    wid = {l: i for i, l in enumerate(watch_l)}
    summ['bootstrap'] = {}
    for g, f in [(f'MODEL {m}', lambda r, m=m: model_of(r['cell']) == m) for m, _ in MODELS] + [('POOLED 549', lambda r: True)]:
        rr = [r for r in allb if f(r)]
        sr = np.zeros(len(real_l))
        nr = np.zeros(len(real_l))
        sw = np.zeros(len(watch_l))
        nw = np.zeros(len(watch_l))
        for r in rr:
            d = int(r['top1']) - int(r['C_top1'])
            if r['cell'].startswith('old_'):
                sr[lid[r['layout']]] += d
                nr[lid[r['layout']]] += 1
            else:
                sw[wid[r['layout']]] += d
                nw[wid[r['layout']]] += 1
        num = sr[dr].sum(1) + sw[dw].sum(1)
        den = nr[dr].sum(1) + nw[dw].sum(1)
        est = 100 * (sr.sum() + sw.sum()) / (nr.sum() + nw.sum())
        b = 100 * num / den
        lo, hi = np.percentile(b, [2.5, 97.5])
        summ['bootstrap'][g] = dict(est=est, lo=float(lo), hi=float(hi))
        p(f"{g:18s} n {len(rr):4d}: {est:+.1f} pp [{lo:+.1f}, {hi:+.1f}]")
    p('')
    p('== T9. Secondary: by situation (setting b and a; flow top-1 vs C / P top-1; P(success)) ==')
    for st, rr_all, ref in (('b', allb, 'C_top1'), ('a', alla, 'P_top1')):
        for s in SITS:
            rr = [r for r in rr_all if r['sit'] == s]
            if not rr:
                continue
            n = len(rr)
            p(f"({st}) {s:20s} n {n:4d}: {ref[0]} {pct(sum(r[ref] for r in rr), n):>16s}  flow {pct(sum(r['top1'] for r in rr), n):>16s}  "
              f"P(succ) {np.mean([r['psucc'] for r in rr]):.3f}  hit@5 {sum(r['hit'][4] for r in rr)}  raw top-1 "
              f"{sum(r['raw_top1'] for r in rr)}  fallbacks {sum(r['fallback'] for r in rr)}")
    p('')
    p('== T10. Secondary: each LOSO model on its left-out situation\'s development records, setting (b) ==')
    for m in LOSO:
        s = m[len('loso_'):]
        rl = []
        for c in CELLS:
            fp = OUT / f'rows_b_{c}_{m}.jsonl'
            if fp.exists():
                rl += jl(fp)
        if not rl:
            p(f"{s}: no development records")
            continue
        keys = {r['key'] for r in rl}
        rm = [r for r in allb if r['key'] in keys]
        n = len(rl)
        p(f"{s:20s} n {n:4d}: C {pct(sum(r['C_top1'] for r in rm), n):>16s}  main {pct(sum(r['top1'] for r in rm), n):>16s}  "
          f"LOSO {pct(sum(r['top1'] for r in rl), n):>16s}  P(succ) main {np.mean([r['psucc'] for r in rm]):.3f} "
          f"LOSO {np.mean([r['psucc'] for r in rl]):.3f}  fallbacks LOSO {sum(r['fallback'] for r in rl)}")
    p('')
    p('== T11. Secondary: the raw (unsnapped) flow top pose on the development records (density over all 64 samples) ==')
    for g, f in groups:
        rb = [r for r in allb if f(r)]
        ra = [r for r in alla if f(r)]
        p(f"{g:18s} n {len(rb):4d}: (b) raw {pct(sum(r['raw_top1'] for r in rb), len(rb)):>16s} snapped "
          f"{pct(sum(r['top1'] for r in rb), len(rb)):>16s} | (a) raw {pct(sum(r['raw_top1'] for r in ra), len(ra)):>16s} "
          f"snapped {pct(sum(r['top1'] for r in ra), len(ra)):>16s}")
    p('')
    p(EXPL)
    (OUT / 'tables.txt').write_text('\n'.join(lines) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1, default=float))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
