"""I04 step 1-2 checks (FROZEN.md sections 2, 3 and 7): reproduction, replica validation, the I14 S1 reproduction.

Reads only the cache written by i04_cache.py and I14 v0's own per-record rows (phase1/i14_v0/rows_*.jsonl) for a
cross-check of the region shares. Exits non-zero if a stop condition of the note fires. Writes cache/check.txt.
"""
from __future__ import annotations

import collections
import json
import sys

HERE = 'results/2026-10-17-beyond-vlm/phase1d/i04'
I14 = 'results/2026-10-17-beyond-vlm/phase1/i14_v0'
PRIMARY = ['old_qwen', 'old_glm', 'old_deepseek', 'watch_qwen', 'watch_glm', 'watch_deepseek']
NEW = ['new_qwen', 'new_glm', 'new_deepseek', 'new_haiku', 'new_sonnet', 'new_gemini']
# I14 v0 tables.txt T2/T3: n and solver hit@1..5 (package)
I14_N = dict(old_qwen=143, old_glm=141, old_deepseek=157, watch_qwen=36, watch_glm=36, watch_deepseek=36,
             new_qwen=144, new_glm=141, new_deepseek=151, new_haiku=146, new_sonnet=143, new_gemini=135)
I14_HIT = dict(old_qwen=[134] * 5, old_glm=[96, 96, 98, 98, 99], old_deepseek=[109, 120, 121, 123, 123],
               watch_qwen=[32, 34, 35, 35, 35], watch_glm=[23, 25, 27, 29, 29], watch_deepseek=[23, 29, 30, 33, 33],
               new_qwen=[131] * 5, new_glm=[86, 87, 88, 89, 89], new_deepseek=[106, 112, 116, 116, 118],
               new_haiku=[134, 136, 136, 137, 141], new_sonnet=[134] * 5, new_gemini=[109] * 5)


def load_meta():
    rows = []
    for s in (0, 1):
        with open(f'{HERE}/cache/meta_shard{s}.jsonl') as f:
            rows += [json.loads(line) for line in f]
    return rows


def main():
    rows = load_meta()
    out = []
    p = out.append
    stop = []
    by = collections.defaultdict(list)
    for r in rows:
        by[r['cell']].append(r)
    skipped = collections.Counter()
    for s in (0, 1):
        with open(f'{HERE}/cache/skipped_shard{s}.json') as f:
            for k, v in json.load(f).items():
                skipped[k] += v
    p(f'records in cache: {len(rows)} (distinct keys {len({r["key"] for r in rows})})')
    p('\n== records per cell ==')
    for c in PRIMARY + NEW:
        rs = by[c]
        p(f'{c:15s} goal {len(rs):4d} (I14 {I14_N[c]})  standard-path left out {skipped[c + "|standard"]:4d}  '
          f'other situations left out {skipped[c + "|other_situation"]:4d}  layouts {len({r["layout"] for r in rs})}')
        if len(rs) != I14_N[c]:
            stop.append(f'{c}: n {len(rs)} != I14 {I14_N[c]}')
    p('\n== reproduction (section 2): picks[0] and picks[1:5] vs recorded places, headings ==')
    bad = [r for r in rows if not r['repro_ok']]
    worst_m = max(max(r['moved']) for r in rows)
    worst_t = max(max(r['dtheta']) for r in rows)
    p(f'records reproduced: {len(rows) - len(bad)} of {len(rows)}; worst place move {worst_m:.3e} m; worst heading '
      f'diff {worst_t:.3e} rad; picks per record {collections.Counter(r["n_picks"] for r in rows)}')
    for r in bad:
        p(f'  NOT REPRODUCED: {r["key"]} moved {r["moved"]} dtheta {r["dtheta"]}')
    p('\n== replica vs package (section 3) ==')
    val, mism = collections.Counter(), collections.Counter()
    for r in rows:
        val.update(r['val'])
        mism.update(r['mism'])
    for k in sorted(val):
        p(f'  {k:18s} checked {val[k]:8d}  disagreements {mism.get(k, 0)}')
    if sum(mism.values()):
        stop.append(f'replica disagreements: {dict(mism)}')
    hok_bad = [r['key'] for r in rows if r['hok_agree'] != r['n_cells']]
    p(f'  kappa-inf arc indicator equals the replica heading verdict on all F cells: {len(rows) - len(hok_bad)} of '
      f'{len(rows)} records')
    top_bad = [r['key'] for r in rows if r['top_rep_success'] != r['top1']]
    p(f'  replica success at picks[0] (section-4 heading) == official top-1: {len(rows) - len(top_bad)} of {len(rows)}')
    if top_bad:
        stop.append(f'top-1 replica mismatch on {len(top_bad)} records')
    off = sum(r['rec_success'] != r['top1'] for r in rows)
    p(f'  official top-1 (package now) == recorded success: {len(rows) - off} of {len(rows)}')
    dens_bad = sum(1 for r in rows for v in r['dens'].values() if v['succ_rep'] != v['succ_pkg'])
    p(f'  density place lists whose replica and package verdicts differ: {dens_bad}')
    p(f'  sampled cells per record: min {min(r["n_val"] for r in rows)}, mean '
      f'{sum(r["n_val"] for r in rows) / len(rows):.1f}; F cells held by more than one region (right target, '
      f'reachable): {sum(r["n_multi"] for r in rows)} over {sum(1 for r in rows if r["n_multi"])} records; '
      f'degenerate-heading cells: {sum(r["n_deg"] for r in rows)}')
    p('\n== S1 reproduction (section 7): lexicographic hit@1..5, package verdicts on the recorded places ==')
    for c in PRIMARY + NEW:
        rs = [r for r in by[c] if r['repro_ok']]
        hit = [sum(r['lex_hit'][k] for r in rs) for k in range(5)]
        ok = hit == I14_HIT[c]
        p(f'{c:15s} n {len(rs):4d}  hit@1..5 {hit}  I14 {I14_HIT[c]}  {"match" if ok else "MISMATCH"}')
        if not ok and c in PRIMARY:
            stop.append(f'S1 mismatch at {c}')
    p('\n== cross-check against I14 v0 rows (same floor F, same disc) ==')
    i14 = {}
    for fn in ('rows_oldwatch.jsonl', 'rows_new.jsonl'):
        with open(f'{I14}/{fn}') as f:
            for line in f:
                q = json.loads(line)
                i14[(q['run'], q['scene'], q['ep'], q['trial'])] = q
    dr = dd = 0.0
    nn = ndisc = nsel = 0
    for r in rows:
        q = i14.get((r['run'], r['scene'], r['ep'], r['trial']))
        if q is None:
            continue
        nn += 1
        dr = max(dr, abs(q['rho_room'] - r['rho_room']))
        dd = max(dd, abs(q['rho_disc'] - r['rho_disc']))
        ndisc += q['n_disc'] != r['n_disc']
        nsel += q['n_sel'] != r['n_cells']
    p(f'matched rows {nn} of {len(rows)}; F size differs {nsel}; disc size differs {ndisc}; max |rho_room diff| '
      f'{dr:.2e}; max |rho_disc diff| {dd:.2e}')
    prim = [r for r in rows if r['cell'] in PRIMARY and r['repro_ok']]
    p(f'primary: n {len(prim)}, top-1 {sum(r["top1"] for r in prim)}, any-of-5 {sum(r["lex_hit"][4] for r in prim)}, '
      f'right target {sum(r["target_ok"] for r in prim)}, sum rho_room {sum(r["rho_room"] for r in prim):.3f}')
    p('\n== layouts of the primary records (folds) ==')
    lay = collections.Counter(r['layout'] for r in prim)
    p(f'{len(lay)} layouts: ' + ', '.join(f'{k} ({v})' for k, v in sorted(lay.items())))
    newl = collections.Counter(r['layout'] for r in rows if r['cell'] in NEW)
    p(f'new wording: {len(newl)} layouts, all among the primary ones: {set(newl) <= set(lay)}')
    p('\nSTOP CONDITIONS: ' + ('none' if not stop else '; '.join(stop)))
    text = '\n'.join(out)
    print(text)
    with open(f'{HERE}/cache/check.txt', 'w') as f:
        f.write(text + '\n')
    sys.exit(1 if stop else 0)


if __name__ == '__main__':
    main()
