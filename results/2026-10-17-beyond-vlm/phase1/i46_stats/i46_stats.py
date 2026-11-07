"""I46, the statistics half (phase 1). Offline only: research_harness reads, no model or network call, nothing tuned.

Reuses rerun/stats/stats.py (load, pairs, analyse, signflip_int, power_exact, icc_oneway, t_quant, layout_of).

Sets (realistic arms real, staffed, plain; 252 records per model and method; 14 layouts x 18 records each,
18 = 3 arms x 2 episodes x 3 trials; layout = scene id without the arm prefix):
  old wording: Qwen  R/E s16b {cot,fsc}, C s19 con;  GLM R/E s16b, C s19;  DeepSeek R/E/C s20.
  new wording: Qwen s21 cot/fsc/con; glm53, dsv41, haiku55, sonnet55_low, gemini38_low s23 cot/fsc/con.
  Qwen 9-draw pool (R, E): s16b trials 0-2 + s22 trials 3-5 + s21 trials 0-2 (s22 trials 0-2 dropped: copies of s16b).

Parts (each writes one .txt table beside this file; everything also goes to i46_stats_output.txt):
  1 icc_table.txt     ICC(1) of success by layout (one-way ANOVA, the judge's estimator, stats.icc_oneway) for R, E, C
                      per model at both wordings; design effect DEFF = 1 + (m - 1) ICC with m = 18; n_eff = 252 / DEFF;
                      a cluster-bootstrap 95% interval (layouts resampled, 2,000 draws, seed 2); the ICC of the
                      paired differences C-E, C-R, E-R (the contrast's own DEFF); the ICC by item (84 items x 3 trials).
  2 tci_table.txt     t(13) 95% intervals on the 14 layout mean differences, C-E and C-R (and E-R, supplementary) at the
                      new wording, scaled to records of 252; checked against RERUN.md's "New rows" table.
  3 power_table.txt   power of a two-sided exact sign-flip test by layout at alpha 0.05 for a sealed set of n fresh
                      layouts (n = 8, 14, 20, 30): n layouts drawn with replacement from the 14 observed integer layout
                      differences, exact test, 10,000 draws per cell (stats.power_exact). Post-hoc and optimistic.
                      Also: per-situation power (lines, groups; n up to 40) for I46's kill line; a replay of
                      stats.py's old-wording power (same seed and order) as a check of the reused code.
  4 noise_table.txt   the batch SD of one 3-trial batch (252 records) from Qwen's 9-draw pool, R and E, by three
                      estimators; paired E-R noise; within-batch estimates for the six models at the new wording.
  5 mdd_table.txt     minimum detectable differences (80% power, two-sided alpha 0.05) for live A/B runs.
"""
import collections
import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402,F401

RD = Path('results/2026-10-17-beyond-vlm')
sys.path.insert(0, str(RD / 'rerun' / 'stats'))
import stats as S  # noqa: E402

OUT = Path(__file__).resolve().parent
ARMS = S.ARMS
M_PER_LAYOUT = 18
N_LAYOUTS = 14
N_REC = 252
Z975, Z80 = 1.959963985, 0.841621234
ALL = []
_section = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    ALL.append(s)
    _section.append(s)


def flush(name):
    (OUT / name).write_text('\n'.join(_section) + '\n', encoding='utf-8')
    _section.clear()


# ---------------------------------------------------------------- loading
def cell(stage, model, method, trials=None):
    """Records of the three arms of one run triple as dicts: arm, scene, episode, trial, layout, situation, s."""
    out, folders = [], []
    for arm in ARMS:
        folder, recs = S.load(stage, arm, model, method)
        folders.append(folder)
        for r in recs:
            if trials is not None and r.trial not in trials:
                continue
            out.append(dict(arm=arm, scene=r.scene_id, ep=r.episode_id, trial=r.trial, layout=S.layout_of(r.scene_id),
                            sit=r.situation, s=S.success(r)))
    return out, folders


def by_layout(recs, key='s'):
    g = collections.defaultdict(list)
    for r in recs:
        g[r['layout']].append(r[key])
    return dict(g)


def icc_np(groups):
    """Independent numpy ICC(1) (one-way ANOVA) for equal group sizes; cross-check of stats.icc_oneway."""
    X = np.array([groups[k] for k in sorted(groups)], dtype=float)
    g, k = X.shape
    gm = X.mean()
    msb = k * ((X.mean(axis=1) - gm) ** 2).sum() / (g - 1)
    msw = ((X - X.mean(axis=1, keepdims=True)) ** 2).sum() / (g * (k - 1))
    den = msb + (k - 1) * msw
    return (msb - msw) / den if den > 0 else float('nan')


def icc_boot(groups, rng, B=2000):
    X = np.array([groups[k] for k in sorted(groups)], dtype=float)
    g, k = X.shape
    out = np.empty(B)
    for b in range(B):
        Y = X[rng.integers(0, g, size=g)]
        gm = Y.mean()
        msb = k * ((Y.mean(axis=1) - gm) ** 2).sum() / (g - 1)
        msw = ((Y - Y.mean(axis=1, keepdims=True)) ** 2).sum() / (g * (k - 1))
        den = msb + (k - 1) * msw
        out[b] = (msb - msw) / den if den > 0 else np.nan
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5)), int(np.isnan(out).sum())


def deff(icc, m=M_PER_LAYOUT):
    return 1 + (m - 1) * icc


def paired(recs_a, recs_b):
    """Pair two cells on (arm, scene, episode, trial); returns dicts with s = sA - sB."""
    kb = {(r['arm'], r['scene'], r['ep'], r['trial']): r for r in recs_b}
    assert len(kb) == len(recs_b)
    out = []
    for r in recs_a:
        o = kb[(r['arm'], r['scene'], r['ep'], r['trial'])]
        assert o['sit'] == r['sit']
        out.append(dict(r, s=r['s'] - o['s']))
    assert len(out) == len(recs_a) == len(recs_b)
    return out


def item_icc(recs):
    """ICC(1) with the item (arm, scene, episode) as the group: 84 items x 3 trials."""
    g = collections.defaultdict(list)
    for r in recs:
        g[(r['arm'], r['scene'], r['ep'])].append(r['s'])
    sizes = set(len(v) for v in g.values())
    assert len(sizes) == 1, sizes
    return icc_np(dict(g)), len(g), sizes.pop()


OLD = {  # model label -> {method letter: (stage, model, method)}
    'Qwen': {'R': ('s16b', 'qwen38_low', 'cot'), 'E': ('s16b', 'qwen38_low', 'fsc'), 'C': ('s19', 'qwen38_low', 'con')},
    'GLM': {'R': ('s16b', 'glm53', 'cot'), 'E': ('s16b', 'glm53', 'fsc'), 'C': ('s19', 'glm53', 'con')},
    'DeepSeek': {'R': ('s20', 'dsv41', 'cot'), 'E': ('s20', 'dsv41', 'fsc'), 'C': ('s20', 'dsv41', 'con')},
}
NEWM = [('Qwen', 'qwen38_low', 's21'), ('GLM', 'glm53', 's23'), ('DeepSeek', 'dsv41', 's23'), ('Haiku', 'haiku55', 's23'),
        ('Sonnet', 'sonnet55_low', 's23'), ('Gemini', 'gemini38_low', 's23')]
NEW = {lab: {'R': (st, m, 'cot'), 'E': (st, m, 'fsc'), 'C': (st, m, 'con')} for lab, m, st in NEWM}


# ---------------------------------------------------------------- part 1: ICC and design effect
def part1():
    say('PART 1. Intra-layout correlation (ICC) of success, design effect for 14 layouts x 18 records')
    say('ICC(1) = (MSB - MSW) / (MSB + (m - 1) MSW), one-way ANOVA by layout, m = 18 records per layout, 14 layouts, '
        'n = 252 records per cell (stats.icc_oneway; cross-checked by an independent numpy version).')
    say('DEFF = 1 + 17 ICC; n_eff = 252 / DEFF (records worth of independent information). Boot = cluster bootstrap '
        'over layouts, 2,000 draws, numpy default_rng(2), percentile 95%; it is rough with 14 clusters.')
    say('ICC_item = ICC(1) by item (arm, scene, episode; 84 items x 3 trials): the share of variance between items; '
        '1 - ICC_item is the share that is trial-to-trial (sampling) noise.')
    say('')
    rng = np.random.default_rng(2)
    res = {}
    hdr = (f"{'wording':8s} {'model':9s} {'M':1s} {'succ':>4s} {'ICC':>6s} {'boot 95%':>16s} {'DEFF':>5s} {'n_eff':>6s} "
           f"{'ICC_item':>8s}  runs")
    for wording, table in (('old', OLD), ('new', NEW)):
        say(f'-- {wording} wording')
        say(hdr)
        for lab, specs in table.items():
            cells = {}
            for M in 'REC':
                recs, folders = cell(*specs[M])
                assert len(recs) == N_REC, (lab, M, len(recs))
                g = by_layout(recs)
                assert len(g) == N_LAYOUTS and all(len(v) == M_PER_LAYOUT for v in g.values())
                icc = S.icc_oneway(g)
                assert abs(icc - icc_np(g)) < 1e-12
                lo, hi, nnan = icc_boot(g, rng)
                ii, nitems, ktr = item_icc(recs)
                d = deff(icc)
                res[(wording, lab, M)] = dict(icc=icc, deff=d, neff=N_REC / d, lo=lo, hi=hi, succ=sum(r['s'] for r in recs),
                                              icc_item=ii)
                cells[M] = recs
                say(f"{wording:8s} {lab:9s} {M:1s} {sum(r['s'] for r in recs):4d} {icc:6.3f} [{lo:+6.3f}, {hi:+6.3f}] "
                    f"{d:5.2f} {N_REC / d:6.1f} {ii:8.3f}  {specs[M][0]} {specs[M][2]}" + (f' (boot NaN {nnan})' if nnan else ''))
            for a, b in (('C', 'E'), ('C', 'R'), ('E', 'R')):
                P = paired(cells[a], cells[b])
                g = by_layout(P)
                icc = S.icc_oneway(g)
                assert abs(icc - icc_np(g)) < 1e-12
                lo, hi, nnan = icc_boot(g, rng)
                d = deff(icc)
                same_stage = specs[a][0] == specs[b][0]
                res[(wording, lab, a + '-' + b)] = dict(icc=icc, deff=d, neff=N_REC / d, lo=lo, hi=hi, same_stage=same_stage)
                say(f"{wording:8s} {lab:9s} {a}-{b} diff {icc:6.3f} [{lo:+6.3f}, {hi:+6.3f}] {d:5.2f} {N_REC / d:6.1f} "
                    f"{'':8s}  paired on (arm, scene, episode, trial); "
                    f"{'same stage (shared target draw)' if same_stage else 'different stages (target draws not shared)'}"
                    + (f' (boot NaN {nnan})' if nnan else ''))
        say('')
    say('Summary, ICC of success by method (range over models):')
    for wording, table in (('old', OLD), ('new', NEW)):
        for M in 'REC':
            v = [res[(wording, lab, M)]['icc'] for lab in table]
            dv = [res[(wording, lab, M)]['deff'] for lab in table]
            say(f'  {wording} {M}: ICC {min(v):.3f} to {max(v):.3f} (median {float(np.median(v)):.3f}); '
                f'DEFF {min(dv):.2f} to {max(dv):.2f}; n_eff {N_REC / max(dv):.0f} to {N_REC / min(dv):.0f} of 252')
        for c in ('C-E', 'C-R', 'E-R'):
            v = [res[(wording, lab, c)]['icc'] for lab in table]
            dv = [res[(wording, lab, c)]['deff'] for lab in table]
            say(f'  {wording} {c} (paired difference): ICC {min(v):.3f} to {max(v):.3f}; DEFF {min(dv):.2f} to {max(dv):.2f}; '
                f'n_eff {N_REC / max(dv):.0f} to {N_REC / min(dv):.0f} of 252')
    say('')
    say('Cross-check against RERUN stats_output.txt (ICC(1) printed there for C and the comparator):')
    ref = {('new', 'Qwen'): (0.122, 0.119, 0.264), ('new', 'GLM'): (0.261, 0.413, 0.350),
           ('new', 'DeepSeek'): (0.125, 0.321, 0.390), ('new', 'Haiku'): (0.265, 0.678, 0.747),
           ('new', 'Sonnet'): (0.244, 0.544, 0.718), ('old', 'Qwen'): (0.080, 0.114, None),
           ('old', 'GLM'): (0.317, 0.486, None), ('old', 'DeepSeek'): (0.168, 0.331, None)}
    bad = 0
    for (w, lab), (c, e, r) in ref.items():
        for M, v in (('C', c), ('E', e), ('R', r)):
            if v is None:
                continue
            ok = abs(round(res[(w, lab, M)]['icc'], 3) - v) < 1e-9
            bad += not ok
    say(f'  {len(ref)} model rows, mismatches at 3 decimals: {bad}')
    say('Notes: the C answers of s21 Qwen and s23 GLM/DeepSeek reuse the s19/s20 people answers (cache copies), so '
        'new-wording C ICCs for those three are not independent of the old-wording ones. Old-wording R/E (s16b) and C '
        '(s19) of Qwen and GLM come from different stages, so their C-E and C-R pairs do not share a target draw.')
    flush('icc_table.txt')
    return res


# ---------------------------------------------------------------- part 2: t-intervals
RERUN_NEW = {  # RERUN.md "New rows" table: C-X, p, layouts +/-/0, t CI low, high (records of 252)
    ('Qwen', 'E'): (32, 0.0078, (8, 0, 6), 8.4, 55.6), ('Qwen', 'R'): (49, 0.0156, (7, 0, 7), 11.9, 86.1),
    ('GLM', 'E'): (47, 0.0078, (8, 0, 6), 17.3, 76.7), ('GLM', 'R'): (38, 0.0156, (7, 0, 7), 8.2, 67.8),
    ('DeepSeek', 'E'): (59, 0.0078, (8, 0, 6), 20.4, 97.6), ('DeepSeek', 'R'): (65, 0.0078, (9, 2, 3), 22.4, 107.6),
    ('Haiku', 'E'): (92, 0.0039, (9, 0, 5), 38.1, 145.9), ('Haiku', 'R'): (89, 0.0039, (9, 0, 5), 32.4, 145.6),
    ('Sonnet', 'E'): (64, 0.0156, (7, 0, 7), 14.4, 113.6), ('Sonnet', 'R'): (64, 0.0156, (7, 0, 7), 8.0, 120.0),
    ('Gemini', 'E'): (41, 0.0078, (8, 0, 6), 11.0, 71.0), ('Gemini', 'R'): (50, 0.0078, (8, 0, 6), 16.5, 83.5),
}


def part2():
    say('PART 2. t-intervals on the 14 layout differences, new wording (Qwen s21, others s23)')
    say('Per layout l: d_l = mean over its 18 paired records of (s_A - s_B); interval = mean(d) +/- t(0.975, 13) sd(d)/sqrt(14), '
        'scaled by 252 to records. p = two-sided exact sign-flip by layout (2^14 patterns; statistic sum of d_l). '
        'Layouts = favouring A / B / tied. n = 252 pairs, 14 layouts per row.')
    say('C-E and C-R were already in RERUN.md ("New rows", script rerun/stats/stats.py); they are recomputed here and '
        'checked. E-R is supplementary (needed for the MDD part).')
    say('')
    say(f"{'model':9s} {'A-B':4s} {'A':>4s} {'B':>4s} {'A-B':>5s} {'layouts':>8s} {'p2 exact':>9s} {'mean d':>7s} {'sd d':>6s} "
        f"{'t 95% CI (records of 252)':>27s} {'t p':>8s}  RERUN")
    res = {}
    mism = 0
    for lab, specs in NEW.items():
        for a, b in (('C', 'E'), ('C', 'R'), ('E', 'R')):
            P, _ = S.pairs(specs[a], specs[b])
            an = S.analyse(f'{lab} {a}-{b}', P)
            assert an['n'] == N_REC and an['L'] == N_LAYOUTS and set(an['npairs'].values()) == {M_PER_LAYOUT}
            lo, hi = an['tci']
            res[(lab, a + '-' + b)] = an
            chk = ''
            if a == 'C':
                r = RERUN_NEW[(lab, b)]
                ok = (an['C'] - an['X'] == r[0] and abs(an['p2'] - r[1]) < 6e-5 and (an['npos'], an['nneg'], an['L'] - an['nz']) == r[2]
                      and abs(round(lo * N_REC, 1) - r[3]) < 1e-9 and abs(round(hi * N_REC, 1) - r[4]) < 1e-9)
                mism += not ok
                chk = 'cited, matches' if ok else f'MISMATCH {r}'
            say(f"{lab:9s} {a}-{b:2s} {an['C']:4d} {an['X']:4d} {an['C'] - an['X']:+5d} {an['npos']:>2d}/{an['nneg']}/{an['L'] - an['nz']:<3d} "
                f"{an['p2']:9.5f} {an['mean']:+7.4f} {an['sd']:6.4f}   [{lo * N_REC:+7.1f}, {hi * N_REC:+7.1f}] {an['tp2']:8.5f}  {chk}")
    say('')
    say(f'Rows checked against RERUN.md: 12; mismatches: {mism}. t(0.975, 13) = {S.t_quant(0.975, 13):.4f}.')
    flush('tci_table.txt')
    return res


# ---------------------------------------------------------------- part 3: power
NS = (8, 14, 20, 30)


def part3(tres):
    say('PART 3. Power of a sealed set of n fresh layouts, new wording, all six models')
    say('Design (stats.py Design A): draw n layouts with replacement from the 14 observed integer layout differences '
        '(sum over 18 paired records), apply the exact sign-flip test by layout (two-sided, statistic = sum), reject at '
        'p <= 0.05; 10,000 draws per cell; numpy default_rng((0, model index, contrast index)). MC SE <= 0.005.')
    say('POST-HOC AND OPTIMISTIC: it resamples the observed effects of the same 14 layouts that showed the effect (no '
        'shrinkage for selection), assumes fresh layouts mix situations as these do (6 nobody-near, 5 lines, 3 groups), '
        'that new situations behave like the old ones, and that the run noise of the fresh run equals that of these runs.')
    say('')
    labs = list(NEW)
    pw = {}
    say(f"{'model':9s} {'A-B':4s} " + ' '.join(f'{"n=" + str(n):>8s}' for n in NS) + '   (two-sided power; one-sided in brackets)')
    for mi, lab in enumerate(labs):
        for ci, c in enumerate(('C-E', 'C-R')):
            an = tres[(lab, c)]
            v = [an['dsum'][l] for l in an['lays']]
            rng = np.random.default_rng((0, mi, ci))
            row = []
            for n in NS:
                p2, p1 = S.power_exact(v, n, 10000, rng)
                pw[(lab, c, n)] = (p2, p1)
                row.append(f'{p2:5.3f} [{p1:5.3f}]')
            say(f"{lab:9s} {c:4s} " + ' '.join(f'{x:>8s}' for x in row))
    say('')
    say('Smallest n on the grid with two-sided power >= 0.80:')
    for lab in labs:
        parts = []
        for c in ('C-E', 'C-R'):
            ok = [n for n in NS if pw[(lab, c, n)][0] >= 0.8]
            parts.append(f'{c}: {ok[0] if ok else ">30"}')
        say(f'  {lab:9s} ' + ', '.join(parts))
    say('')
    say('Why n = 8 is weak: two-sided exact sign-flip p >= 2 / 2^k with k non-tied layouts, so p <= 0.05 needs k >= 6. '
        'With 6 of 14 layouts usually tied (nobody-near), 8 fresh layouts give about 4.6 non-tied on average.')
    say('')
    # per-situation power for the kill line ("more than 40 layouts needed per situation")
    say('Per-situation power (for I46 kill: "more than 40 layouts needed per situation"): fresh layouts of one situation '
        'only, drawn from that situation\'s observed layout differences (lines: 5 layouts, groups: 3). Even more '
        'optimistic: it resamples 5 or 3 values. Nobody-near is not powered for C vs E or C vs R at any n (its C-E '
        'differences are almost all 0: C does not act there).')
    NS2 = (6, 8, 14, 20, 30, 40)
    say(f"{'model':9s} {'A-B':4s} {'situation':18s} {'obs diffs':22s} " + ' '.join(f'{"n=" + str(n):>6s}' for n in NS2) + '  min n (>=0.8)')
    need = {}
    for mi, lab in enumerate(labs):
        for ci, c in enumerate(('C-E', 'C-R')):
            an = tres[(lab, c)]
            for si, sit in enumerate(('waiting_line', 'conversation_group')):
                v = [an['dsum'][l] for l in an['lays'] if an['sit'][l] == sit]
                rng = np.random.default_rng((1, mi, ci, si))
                row = []
                minn = None
                for n in NS2:
                    p2, _ = S.power_exact(v, n, 4000, rng)
                    row.append(p2)
                    if minn is None and p2 >= 0.8:
                        minn = n
                need[(lab, c, sit)] = minn
                say(f"{lab:9s} {c:4s} {sit:18s} {str(v):22s} " + ' '.join(f'{x:6.3f}' for x in row) + f'  {minn if minn else ">40"}')
    over40 = [k for k, v in need.items() if v is None]
    say(f'Cells needing more than 40 layouts (lines and groups; 4,000 draws per cell, rng (1, model, contrast, situation)): '
        f'{len(over40)} of {len(need)}: {over40}')
    say('Degenerate by construction: where every observed layout difference of a situation is positive, any resample of '
        'n >= 6 layouts is all positive and p = 2 / 2^n <= 0.05, so power is 1. Where one of 3 group layouts is 0, power '
        'follows the chance of drawing at least 6 non-zero layouts. This shows only that the kill line does not fire under '
        'this design; it says nothing about how fresh layouts of a situation will vary.')
    say('')
    # replay of stats.py old-wording Design A as a check of the reused code
    say('Check: replay of stats.py PART 1 Design A (old wording, C-E, rng default_rng(0) shared in the order Qwen, GLM, '
        'DeepSeek; 10,000 draws), against stats_output.txt:')
    rng = np.random.default_rng(0)
    old_ref = {'qwen': (0.072, 0.603, 0.924, 0.998), 'glm': (0.373, 0.919, 0.994, 1.000), 'ds': (0.767, 0.999, 1.000, 1.000)}
    old = {'qwen': (('s19', 'qwen38_low', 'con'), ('s16b', 'qwen38_low', 'fsc')),
           'glm': (('s19', 'glm53', 'con'), ('s16b', 'glm53', 'fsc')),
           'ds': (('s20', 'dsv41', 'con'), ('s20', 'dsv41', 'fsc'))}
    allok = True
    for m, (sc, sx) in old.items():
        P, _ = S.pairs(sc, sx)
        an = S.analyse(m, P)
        v = [an['dsum'][l] for l in an['lays']]
        got = []
        for n in NS:
            p2, _ = S.power_exact(v, n, 10000, rng)
            got.append(f'{float(p2):.3f}')   # stats.py printed with :.3f (numpy round() would round 0.6025 down)
        ok = tuple(got) == tuple(f'{x:.3f}' for x in old_ref[m])
        allok &= ok
        say(f'  {m}: {got} reference {list(old_ref[m])} -> {"identical" if ok else "DIFFERENT"}')
    say(f'  replay identical: {allok}')
    flush('power_table.txt')
    return pw, need


# ---------------------------------------------------------------- part 4: noise
def part4():
    say('PART 4. Run-to-run noise: SD of one 3-trial batch of 252 records')
    say("Controller's figures (REPORT section 8): one 3-trial batch has an SD of about 6-8 records (Qwen 7.9).")
    say('Qwen 9-draw pool (R = cot, E = fsc): batch 1 = s16b trials 0-2, batch 2 = s22 trials 3-5, batch 3 = s21 trials '
        '0-2; s22 trials 0-2 dropped (cache copies of s16b). Each batch: 3 trials x 84 items (arm, scene, episode) = 252.')
    say('Estimators of the SD of one batch total:')
    say('  (a) SD of the 3 batch totals (df 2 per method; R and E pooled: df 4).')
    say('  (b) SD of the 9 single-trial totals (84 records each) x sqrt(3) (df 8 per method; assumes independent trials).')
    say('  (c) item binomial: sqrt(3 * sum_i s_i^2), s_i^2 = unbiased variance of item i\'s 9 outcomes (holds the items '
        'fixed; assumes independence across items and trials).')
    say('')
    pool = {}
    for M, meth in (('R', 'cot'), ('E', 'fsc')):
        a, fa = cell('s16b', 'qwen38_low', meth)
        b, fb = cell('s22', 'qwen38_low', meth, trials={3, 4, 5})
        c, fc = cell('s21', 'qwen38_low', meth)
        assert len(a) == len(b) == len(c) == N_REC
        for r in a:
            r['batch'] = 's16b'
        for r in b:
            r['batch'] = 's22t3-5'
        for r in c:
            r['batch'] = 's21'
        pool[M] = a + b + c
        say(f'  {M}: runs ' + ', '.join(fa + fb + fc))
    say('')
    out = {}
    for M in 'RE':
        recs = pool[M]
        bt = collections.Counter()
        tt = collections.Counter()
        items = collections.defaultdict(list)
        for r in recs:
            bt[r['batch']] += r['s']
            tt[(r['batch'], r['trial'])] += r['s']
            items[(r['arm'], r['scene'], r['ep'])].append(r['s'])
        assert len(tt) == 9 and len(items) == 84 and all(len(v) == 9 for v in items.values())
        btv = np.array([bt[k] for k in ('s16b', 's22t3-5', 's21')], dtype=float)
        ttv = np.array([tt[k] for k in sorted(tt)], dtype=float)
        sa = float(btv.std(ddof=1))
        sb = float(ttv.std(ddof=1) * math.sqrt(3))
        var_c = 3 * sum(np.var(v, ddof=1) for v in items.values())
        sc = math.sqrt(var_c)
        out[M] = dict(bt=btv, tt=ttv, sa=sa, sb=sb, sc=sc, va=float(btv.var(ddof=1)), vb=float(ttv.var(ddof=1) * 3), vc=var_c)
        say(f"  Qwen {M}: batch totals s16b/s22t3-5/s21 = {int(btv[0])} / {int(btv[1])} / {int(btv[2])} (sum {int(btv.sum())} of 756, "
            f"mean {btv.mean():.1f} of 252)")
        say(f"    trial totals (of 84): {[int(x) for x in ttv]} (keys {sorted(tt)})")
        say(f"    (a) {sa:.2f}   (b) {sb:.2f}   (c) {sc:.2f}")
    pa = math.sqrt((out['R']['va'] + out['E']['va']) / 2)
    pb = math.sqrt((out['R']['vb'] + out['E']['vb']) / 2)
    pc = math.sqrt((out['R']['vc'] + out['E']['vc']) / 2)
    say(f'  Pooled R and E (root mean variance): (a) {pa:.2f} (df 4)   (b) {pb:.2f} (df 16)   (c) {pc:.2f}')
    # approximate 95% CI for (a) and (b) by chi-square: use Wilson-Hilferty for chi2 quantiles
    def chi2q(p, df):
        z = {0.025: -1.959963985, 0.975: 1.959963985}[p]
        return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3
    for nm, s_, df in (('(a)', pa, 4), ('(b)', pb, 16)):
        lo = s_ * math.sqrt(df / chi2q(0.975, df))
        hi = s_ * math.sqrt(df / chi2q(0.025, df))
        say(f'    approx 95% CI for pooled {nm}: {lo:.1f} to {hi:.1f} (chi-square, df {df}, Wilson-Hilferty quantiles)')
    say(f'  -> the run\'s Qwen 7.9 is reproduced by (a) pooled over R and E: {pa:.2f}.')
    say('')
    # paired E-R
    say('Paired E-R (same stage and trial share the target draw; no_one records are identical in R and E):')
    pr = {(r['batch'], r['arm'], r['scene'], r['ep'], r['trial']): r['s'] for r in pool['R']}
    dbt = collections.Counter()
    ditems = collections.defaultdict(list)
    for r in pool['E']:
        d = r['s'] - pr[(r['batch'], r['arm'], r['scene'], r['ep'], r['trial'])]
        dbt[r['batch']] += d
        ditems[(r['arm'], r['scene'], r['ep'])].append(d)
    dv = np.array([dbt[k] for k in ('s16b', 's22t3-5', 's21')], dtype=float)
    sd_pa = float(dv.std(ddof=1))
    sd_pc = math.sqrt(3 * sum(np.var(v, ddof=1) for v in ditems.values()))
    say(f'  E-R per batch: {[int(x) for x in dv]} (mean {dv.mean():+.1f}); SD of batch differences (df 2) {sd_pa:.2f}; '
        f'item-level paired SD {sd_pc:.2f}; independent-batch reference sqrt(2) x pooled (a) = {math.sqrt(2) * pa:.2f}, '
        f'sqrt(2) x pooled (c) = {math.sqrt(2) * pc:.2f}')
    out['pair'] = dict(dv=dv, sd_pa=sd_pa, sd_pc=sd_pc)
    out['pooled'] = dict(a=pa, b=pb, c=pc)
    say('  Item-level variance of the mean over 9 draws per item ignores between-batch shifts (wording, gateway day); '
        '(a) includes them but has only 2-4 df.')
    say('')
    # supplementary: within-batch estimates for the six models at the new wording
    say('Supplementary: within-batch noise at the new wording (one batch of 3 trials per model and method).')
    say('  (b\') SD of the 3 trial totals x sqrt(3), df 2 per method, pooled over R, E, C (df 6);')
    say('  (c\') item binomial over the 3 trials: sqrt(3 * sum_i s_i^2) per method, pooled (root mean variance) over R, E, C;')
    say('  paired item-level SD of C-E, C-R, E-R within the batch (same trial = same target draw).')
    say('  These see only within-batch (trial-to-trial) variation, not batch-to-batch shifts, so they can be low.')
    say(f"  {'model':9s} {'(b) R/E/C':>20s} {'pooled':>7s} {'(c) R/E/C':>20s} {'pooled':>7s} {'pair C-E':>8s} {'C-R':>6s} {'E-R':>6s}")
    sup = {}
    for lab, specs in NEW.items():
        cells = {M: cell(*specs[M])[0] for M in 'REC'}
        bvals, cvals = [], []
        for M in 'REC':
            tt = collections.Counter()
            items = collections.defaultdict(list)
            for r in cells[M]:
                tt[r['trial']] += r['s']
                items[(r['arm'], r['scene'], r['ep'])].append(r['s'])
            assert len(tt) == 3 and len(items) == 84
            bvals.append(float(np.var([tt[k] for k in sorted(tt)], ddof=1) * 3))
            cvals.append(3 * sum(np.var(v, ddof=1) for v in items.values()))
        prs = {}
        for a, b in (('C', 'E'), ('C', 'R'), ('E', 'R')):
            P = paired(cells[a], cells[b])
            items = collections.defaultdict(list)
            for r in P:
                items[(r['arm'], r['scene'], r['ep'])].append(r['s'])
            prs[a + '-' + b] = math.sqrt(3 * sum(np.var(v, ddof=1) for v in items.values()))
        sup[lab] = dict(b=[math.sqrt(x) for x in bvals], bp=math.sqrt(np.mean(bvals)), c=[math.sqrt(x) for x in cvals],
                        cp=math.sqrt(np.mean(cvals)), pair=prs)
        say(f"  {lab:9s} {'/'.join(f'{math.sqrt(x):.1f}' for x in bvals):>20s} {math.sqrt(np.mean(bvals)):7.2f} "
            f"{'/'.join(f'{math.sqrt(x):.1f}' for x in cvals):>20s} {math.sqrt(np.mean(cvals)):7.2f} "
            f"{prs['C-E']:8.2f} {prs['C-R']:6.2f} {prs['E-R']:6.2f}")
    allc = [math.sqrt(v ** 2) for lab in sup for v in sup[lab]['c'][:2]]
    say(f"  Item-binomial (c') for R and E across the six models: {min(allc):.1f} to {max(allc):.1f} records.")
    say('  Caveat: C people answers of Qwen (s21), GLM and DeepSeek (s23) are cache copies of s19/s20, so their C rows '
        'are a re-draw of everything except the people answers.')
    out['sup'] = sup
    flush('noise_table.txt')
    return out


# ---------------------------------------------------------------- part 5: MDD
def part5(noise, tres):
    say('PART 5. Minimum detectable differences (MDD) for live A/B runs: 80% power, two-sided alpha 0.05')
    zz = Z975 + Z80
    say(f'Normal approximation, MDD = (z0.975 + z0.80) x SD of the difference = {zz:.3f} x SD. Records of 252 (pp = /2.52).')
    say('')
    say('M1. Two independent live batches (A and B each one 3-trial batch, nothing shared): SD diff = sqrt(2) x batch SD.')
    for lab, s in (('5 (old first figure)', 5.0), ('6', 6.0), ('7', 7.0), ('Qwen (a) pooled', noise['pooled']['a']),
                   ('8', 8.0), ('Qwen (c) item binomial', noise['pooled']['c'])):
        m = zz * math.sqrt(2) * s
        say(f'  batch SD {lab:24s} = {s:5.2f}: MDD {m:5.1f} records ({m / 2.52:4.1f} pp)')
    say('  With b batches per arm (3b trials), divide by sqrt(b): Qwen (a) 7.93 -> MDD '
        + ', '.join(f'b={b}: {zz * math.sqrt(2) * noise["pooled"]["a"] / math.sqrt(b):.1f}' for b in (1, 2, 3, 4)) + ' records of 252.')
    say('')
    say('M2. Paired live A/B in one stage (same trials share the plan/context target draw; ideally cached perception too):')
    m = zz * noise['pair']['sd_pc']
    say(f'  Qwen pool E-R item-level paired SD {noise["pair"]["sd_pc"]:.2f} -> MDD {m:.1f} records ({m / 2.52:.1f} pp) per 3-trial batch.')
    say(f'  Qwen pool E-R SD of the 3 batch differences {noise["pair"]["sd_pa"]:.2f} (df 2) -> MDD {zz * noise["pair"]["sd_pa"]:.1f} '
        'records (includes between-batch shifts; very imprecise).')
    for lab, d in noise['sup'].items():
        say(f'  {lab:9s} within-batch paired SD C-E {d["pair"]["C-E"]:.1f}, C-R {d["pair"]["C-R"]:.1f}, E-R {d["pair"]["E-R"]:.1f} '
            f'-> MDD {zz * d["pair"]["C-E"]:.0f} / {zz * d["pair"]["C-R"]:.0f} / {zz * d["pair"]["E-R"]:.0f} records')
    say('')
    say('M3. Generalising over layouts (the reported unit): t over the 14 layout differences. MDD(n) = (t0.975,n-1 + t0.80,n-1) '
        'x sd_layout / sqrt(n) x 252, with sd_layout the observed SD of the 14 layout mean differences.')
    t975, t80 = S.t_quant(0.975, 13), S.t_quant(0.80, 13)
    say(f'  t0.975,13 = {t975:.3f}, t0.80,13 = {t80:.3f}')
    rows = []
    for lab in NEW:
        for c in ('C-E', 'C-R', 'E-R'):
            an = tres[(lab, c)]
            mdd = (t975 + t80) * an['sd'] / math.sqrt(14) * N_REC
            rows.append(mdd)
            say(f'  {lab:9s} {c}: sd_layout {an["sd"]:.4f} -> MDD at 14 layouts {mdd:5.1f} records ({mdd / 2.52:4.1f} pp); '
                f'observed {an["C"] - an["X"]:+d}')
    say(f'  Range over the 18 rows: {min(rows):.0f} to {max(rows):.0f} records.')
    say('  Floor of the exact test: two-sided p >= 2 / 2^k with k non-tied layouts, so at least 6 layouts must move. On these sets '
        'a change confined to lines (5 layouts, floor 0.0625) or groups (3 layouts, floor 0.25) cannot reach p <= 0.05 two-sided.')
    flush('mdd_table.txt')


def main():
    say('I46 statistics, phase 1 (research_harness only; no model call; nothing tuned). Definitions in the module docstring.')
    say('=' * 110)
    icc = part1()
    say('=' * 110)
    tres = part2()
    say('=' * 110)
    part3(tres)
    say('=' * 110)
    noise = part4()
    say('=' * 110)
    part5(noise, tres)
    (OUT / 'i46_stats_output.txt').write_text('\n'.join(ALL) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
