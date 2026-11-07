"""Layout-level exact sign-flip tests for the still-scene method comparison (items 1 and 2 of the audit fill).

No model or provider call: this reads results.jsonl and config.yaml of finished runs and nothing else, and imports
no project code. Definitions are in README.md beside this file.

Item 1, development still-scene headline, runs from the research whitelist (research_runs.txt) only:
  new wording: Qwen 3.8 low s21; GLM-5.3, DeepSeek V4.1, Haiku 5.5, Sonnet 5.5 low, Gemini 3.8 low s23.
  old wording: Qwen and GLM R and E from s16b, C from s19; DeepSeek R, E and C from s20 (never s20_held).
Item 2, the seven older models of E3 (descriptive): runs s24_{variant}_{entry}_{method}, the folders run_static.sh
  names; never e3_models/held_configs/ or any held run.

Unit: the layout (scene id without its variant prefix; 14 layouts). Records are paired across methods by
(variant, scene, episode, trial). net(A-B) of a layout = sum over its paired records of success(A) - success(B).
Test: exact two-sided sign-flip over layouts, p = #{sign patterns s of the non-zero nets: |sum s_l |net_l|| >=
|observed sum|} / 2^k, k = number of non-zero nets, counted exactly by dynamic programming over integer sums and
checked by full enumeration of the 2^k patterns. Smallest attainable p with k non-zero layouts: 2 / 2^k (1 if k <= 1).
"""
import collections
import itertools
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import yaml

ROOT = Path('.')
RUNS = ROOT / 'runs'
WHITELIST = Path('docs/model_notes/tools/research_runs.txt')
OLD_STATS = Path('results/2026-10-17-beyond-vlm/rerun/'
                 'stats/stats_output.txt')
E3 = ROOT / 'results/2026-11-01-push/e3_models'
E3_TOTALS = E3 / 'results_all_out.txt'
E3_SCENES = E3 / 'scenes_dirs.yaml'
OUT = Path(__file__).resolve().parent / 'layout_tests_out.txt'

VARIANTS = ('real', 'staffed', 'plain')
METHOD_RUN = {'R': 'cot', 'E': 'fsc', 'C': 'con'}
COMPARISONS = (('C', 'R'), ('C', 'E'), ('E', 'R'))
HELD_DEFAULT = frozenset({'group_photo', 'lift_wait', 'machine_use'})   # src/sgi/config.py default
FORBIDDEN = re.compile(r'(?:^|_)s(?:tage)?18|(?:^|_)s(?:tage)?20[a-z]*_held|held|(?:^|/)\.env|'
                       r'(?:^|_)eval_1[6-9](?![0-9])', re.I)
HELD_SCENE = re.compile(r'(?:^|_)eval_1[6-9](?![0-9])', re.I)

# Item 1 models: (display name, short name in run names).
NEW_WORDING = (('Qwen 3.8 low', 'qwen38_low', 's21'), ('GLM-5.3', 'glm53', 's23'), ('DeepSeek V4.1', 'dsv41', 's23'),
               ('Haiku 5.5', 'haiku55', 's23'), ('Sonnet 5.5 low', 'sonnet55_low', 's23'),
               ('Gemini 3.8 low', 'gemini38_low', 's23'))
OLD_WORDING = (('Qwen 3.8 low', 'qwen38_low', {'R': 's16b', 'E': 's16b', 'C': 's19'}),
               ('GLM-5.3', 'glm53', {'R': 's16b', 'E': 's16b', 'C': 's19'}),
               ('DeepSeek V4.1', 'dsv41', {'R': 's20', 'E': 's20', 'C': 's20'}))
# Totals (R, E, C of 252) the earlier report quotes: EVIDENCE_AUDIT.md A1 and A2, docs/results/README.md line 99.
REPORT_TOTALS = {('new', 'Qwen 3.8 low'): (179, 196, 228), ('new', 'GLM-5.3'): (132, 123, 170),
                 ('new', 'DeepSeek V4.1'): (120, 126, 185), ('new', 'Haiku 5.5'): (151, 148, 240),
                 ('new', 'Sonnet 5.5 low'): (178, 178, 242), ('new', 'Gemini 3.8 low'): (164, 173, 214),
                 ('old', 'Qwen 3.8 low'): (175, 214, 234), ('old', 'GLM-5.3'): (129, 145, 186),
                 ('old', 'DeepSeek V4.1'): (113, 120, 182)}
# Item 2 models: (display name as results_all_out.txt prints it, E3 entry name).
E3_MODELS = (('Gemini 3 Flash', 'gemini3_flash'), ('Gemini 3.1 Flash Lite', 'gemini31_flash_lite'),
             ('Gemini 3.5 Flash Lite', 'gemini35_flash_lite'), ('Gemini 3.6 Flash', 'gemini36_flash'),
             ('Claude Haiku 4.5', 'claude_haiku45'), ('Claude Sonnet 4.6', 'claude_sonnet46_low'),
             ('Claude Sonnet 5', 'claude_sonnet5_low'))
# The three current models results_all_out.txt also lists, checked against the s23 runs of item 1.
E3_CURRENT = (('Claude Sonnet 5.5', 'Sonnet 5.5 low'), ('Claude Haiku 5.5', 'Haiku 5.5'),
              ('Gemini 3.8 Flash', 'Gemini 3.8 low'))

LINES = []


def say(s=''):
    print(s)
    LINES.append(s)


def guard(text):
    if FORBIDDEN.search(str(text)):
        sys.exit(f'refused: held out or excluded: {text}')
    return text


# ---------------------------------------------------------------------------------------------------- run lookup
def read_whitelist():
    names = {}
    for line in WHITELIST.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        run, scenes = line.split('=', 1)
        folder = run.removeprefix('runs/')
        name = folder[16:]                       # drop 'YYYYMMDD-HHMMSS_'
        assert re.fullmatch(r'\d{8}-\d{6}_', folder[:16]), line
        assert name not in names, ('two whitelist lines for one name', name)
        names[name] = (folder, (ROOT / scenes).resolve())
    return names


def whitelisted_run(wl, name):
    if name not in wl:
        sys.exit(f'not on the whitelist: {name}')
    folder, scenes = wl[name]
    return RUNS / guard(folder), scenes


def e3_run(name, scenes_by_variant, variant):
    """The one folder runs/<timestamp>_<name> (run_static.sh's --name), never a held run."""
    hits = [p for p in RUNS.iterdir() if p.is_dir() and p.name[16:] == name and not FORBIDDEN.search(p.name)]
    assert len(hits) == 1, (name, [p.name for p in hits])
    return hits[0], Path(scenes_by_variant[variant]).resolve()


# ---------------------------------------------------------------------------------------------------- records
def load_run(folder, scenes, variant, name, tally):
    """Records of one run as {(variant, scene, episode, trial): (success, situation)}; held-out records skipped."""
    cfg = yaml.safe_load((folder / 'config.yaml').read_text(encoding='utf-8'))
    assert cfg['name'] == name, (folder, cfg['name'])
    assert Path(guard(cfg['scenes_dir'])).resolve() == scenes, (folder, cfg['scenes_dir'], scenes)
    held = set(HELD_DEFAULT)

    def recorded_held(node):                     # every held_out_situations list the run's config records
        if isinstance(node, dict):
            for k, v in node.items():
                if k == 'held_out_situations':
                    held.update(v)
                else:
                    recorded_held(v)
        elif isinstance(node, list):
            for v in node:
                recorded_held(v)
    recorded_held(cfg)
    out = {}
    for n, line in enumerate((folder / 'results.jsonl').read_text(encoding='utf-8').splitlines(), 1):
        rec = json.loads(line)
        situation = (rec.get('labels') or {}).get('situation')
        if situation is None or situation in held:
            tally['skipped: held-out or unknown situation'] += 1
            continue
        sid = rec['scene_id']
        if HELD_SCENE.search(sid):
            tally['skipped: held-out scene'] += 1
            continue
        assert sid.startswith(variant + '_'), (folder, n, sid)
        key = (variant, sid, rec['episode_id'], rec['trial'])
        assert key not in out, ('duplicate record', folder, key)
        out[key] = (int(bool(rec['score']['success'])), situation)
        tally['allowed'] += 1
    return out


def layout_of(scene_id):
    return re.sub(r'^(?:real|staffed|plain)_', '', scene_id)


def load_model(locate):
    """locate(method, variant) -> (folder, scenes, run name). Returns ({method: {key: success}}, sit, folders, tally)."""
    data, folders, tally, situation = {}, {}, collections.Counter(), {}
    for m in METHOD_RUN:
        data[m] = {}
        folders[m] = []
        for variant in VARIANTS:
            folder, scenes, name = locate(m, variant)
            recs = load_run(folder, scenes, variant, name, tally)
            assert len(recs) == 84, (folder, len(recs))
            for key, (s, sit) in recs.items():
                data[m][key] = s
                lay = layout_of(key[1])
                assert situation.setdefault(lay, sit) == sit, (folder, key, sit, situation[lay])
            folders[m].append(folder.name)
    keys = set(data['R'])
    assert len(keys) == 252, len(keys)
    assert all(set(data[m]) == keys for m in data), 'methods do not pair record for record'
    return data, situation, folders, tally


# ---------------------------------------------------------------------------------------------------- the test
def signflip(nets):
    """Exact two-sided sign-flip p over layouts: (p as Fraction, hits, k). Zero nets are dropped (same p)."""
    nz = [abs(x) for x in nets if x != 0]
    k = len(nz)
    t = abs(sum(nets))
    dist = {0: 1}
    for x in nz:
        new = collections.Counter()
        for s, c in dist.items():
            new[s + x] += c
            new[s - x] += c
        dist = new
    hits = sum(c for s, c in dist.items() if abs(s) >= t)
    # Cross-check: enumerate every sign pattern.
    hits_enum = sum(1 for signs in itertools.product((1, -1), repeat=k)
                    if abs(sum(a * b for a, b in zip(signs, nz))) >= t)
    assert hits == hits_enum, (nets, hits, hits_enum)
    return Fraction(hits, 2 ** k), hits, k


def min_p(k):
    return Fraction(1) if k <= 1 else Fraction(2, 2 ** k)


def compare(data, a, b, layouts):
    net = {l: 0 for l in layouts}
    for key in data[a]:
        net[layout_of(key[1])] += data[a][key] - data[b][key]
    nets = [net[l] for l in layouts]
    p, hits, k = signflip(nets)
    return dict(A=sum(data[a].values()), B=sum(data[b].values()), W=sum(x > 0 for x in nets),
                L=sum(x < 0 for x in nets), T=sum(x == 0 for x in nets), net=sum(nets), nets=nets, p=p, hits=hits,
                k=k, minp=min_p(k))


def fp(p):
    return f'{float(p):.4g}'


# ---------------------------------------------------------------------------------------------------- reports
HEAD = (f"  {'model':22s} {'R/E/C':>11s}  {'cmp':4s} {'A':>4s} {'B':>4s}  {'W/L/T':>8s} {'net':>5s}  "
        f"{'p two-sided':>11s} {'= hits/2^k':>12s}  {'min p':>9s}")


def table_rows(model, totals, res):
    for i, (a, b) in enumerate(COMPARISONS):
        r = res[(a, b)]
        wlt = f"{r['W']}/{r['L']}/{r['T']}"
        tot = '/'.join(str(x) for x in totals) if i == 0 else ''
        say(f"  {model if i == 0 else '':22s} {tot:>11s}  {a}-{b:2s} {r['A']:4d} {r['B']:4d}  {wlt:>8s} {r['net']:+5d}  "
            f"{fp(r['p']):>11s} {str(r['hits']) + '/' + str(2 ** r['k']):>12s}  {fp(r['minp']):>9s}")


def analyse_block(title, models):
    """models: list of (display, locate). Returns {display: (totals R/E/C, res, layouts, situation, data)}."""
    say('=' * 118)
    say(title)
    say('=' * 118)
    results = {}
    layouts_ref = None
    for display, locate in models:
        data, situation, folders, tally = load_model(locate)
        assert set(tally) == {'allowed'} and tally['allowed'] == 756, (display, tally)
        layouts = sorted(situation)
        assert len(layouts) == 14, layouts
        per_layout = collections.Counter(layout_of(k[1]) for k in data['R'])
        assert set(per_layout.values()) == {18}, per_layout
        if layouts_ref is None:
            layouts_ref = (layouts, situation)
        assert (layouts, situation) == layouts_ref, display
        res = {(a, b): compare(data, a, b, layouts) for a, b in COMPARISONS}
        totals = tuple(sum(data[m].values()) for m in ('R', 'E', 'C'))
        results[display] = (totals, res, layouts, situation, data, folders)
    say('Successes of 252 per method (R / E / C) and layout-level tests; A-B rows give A, B totals, layouts A>B / '
        'A<B / tied (W/L/T), net = A-B summed, exact two-sided sign-flip p, smallest p attainable with k non-zero '
        'layouts.')
    say(HEAD)
    for display, (totals, res, *_rest) in results.items():
        table_rows(display, totals, res)
    say()
    layouts, situation = layouts_ref
    say('Per-layout nets (layout order: ' + ', '.join(f'{i}={l}[{situation[l]}]' for i, l in enumerate(layouts)) + ')')
    for display, (totals, res, layouts, situation, data, folders) in results.items():
        for a, b in COMPARISONS:
            say(f"  {display:22s} {a}-{b}: " + ' '.join(f'{x:+3d}' for x in res[(a, b)]['nets']))
    say('Per-layout successes of 18 (R / E / C)')
    for display, (totals, res, layouts, situation, data, folders) in results.items():
        cells = []
        for l in layouts:
            c = [sum(v for k, v in data[m].items() if layout_of(k[1]) == l) for m in ('R', 'E', 'C')]
            cells.append('/'.join(str(x) for x in c))
        say(f'  {display:22s} ' + ' '.join(f'{c:>8s}' for c in cells))
    say('Runs read (results.jsonl, 84 records each, none skipped)')
    for display, (totals, res, layouts, situation, data, folders) in results.items():
        for m in ('R', 'E', 'C'):
            say(f'  {display:22s} {m}: ' + ', '.join(folders[m]))
    say()
    return results


def parse_old_stats():
    """The earlier stats.py output: {block name: (C total, X name, X total, two-sided p string)}."""
    blocks, name = {}, None
    for line in OLD_STATS.read_text(encoding='utf-8').splitlines():
        m = re.match(r'^-- (.+?): runs ', line)
        if m:
            name = m.group(1)
            continue
        m = re.match(r'^  pairs n=252 .*?; C=(\d+) ([ER])=(\d+) ', line)
        if m and name:
            blocks[name] = [int(m.group(1)), m.group(2), int(m.group(3)), None]
            continue
        m = re.search(r'exact sign-flip.*?two-sided p = ([0-9.e+-]+) ', line)
        if m and name:
            blocks[name][3] = m.group(1)
            continue
        m = re.match(r'^  per layout \(situation: C-[ER] summed over its pairs\): (.*)$', line)
        if m and name:
            blocks[name].append({lay: int(v) for lay, v in re.findall(r'(eval_\w+?)\[\w+\]:([+-]\d+)', m.group(1))})
            name = None
    return blocks


def parse_e3_totals():
    out, on = {}, False
    for line in E3_TOTALS.read_text(encoding='utf-8').splitlines():
        if line.startswith('STILL SCENES'):
            on = True
            continue
        if on and not line.startswith('  '):
            break
        m = re.match(r'^  (.+?)\s{2,}(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s', line)
        if on and m:
            out[m.group(1)] = tuple(int(m.group(i)) for i in (2, 3, 4))
    return out


def main():
    wl = read_whitelist()
    say('Layout-level exact sign-flip tests, still scenes (layout_tests.py; no model call; definitions in README.md).')
    say('Methods: R = marked point (cot runs), E = R with worked examples (fsc runs), C = constraint method (con runs).')
    say('Each method: 3 variants (real, staffed, plain) x 14 layouts x 2 episodes x 3 trials = 252 records.')
    say()

    def wl_locator(stage_of, short):
        def locate(m, variant):
            name = f'{stage_of[m]}_{variant}_{short}_{METHOD_RUN[m]}'
            folder, scenes = whitelisted_run(wl, name)
            return folder, scenes, name
        return locate

    new = analyse_block('ITEM 1a. Development still scenes, NEW wording (Qwen 3.8 low s21; the other five s23)',
                        [(d, wl_locator({'R': st, 'E': st, 'C': st}, s)) for d, s, st in NEW_WORDING])
    old = analyse_block('ITEM 1b. Development still scenes, OLD wording (Qwen, GLM: R and E s16b, C s19; '
                        'DeepSeek: R, E, C s20)', [(d, wl_locator(stages, s)) for d, s, stages in OLD_WORDING])

    scenes_by_variant = yaml.safe_load(E3_SCENES.read_text(encoding='utf-8'))

    def e3_locator(entry):
        def locate(m, variant):
            name = f's24_{variant}_{entry}_{METHOD_RUN[m]}'
            folder, scenes = e3_run(name, scenes_by_variant, variant)
            return folder, scenes, name
        return locate

    e3 = analyse_block('ITEM 2. Seven older models, E3 (s24 runs, descriptive; same scenes and protocol as s23)',
                       [(d, e3_locator(e)) for d, e in E3_MODELS])

    # ------------------------------------------------------------------------------------------- checks
    say('=' * 118)
    say('CHECKS')
    say('=' * 118)
    bad = []
    say('Totals R/E/C against the earlier report (EVIDENCE_AUDIT.md A1, A2; docs/results/README.md):')
    for (wording, display), want in REPORT_TOTALS.items():
        got = (new if wording == 'new' else old)[display][0]
        ok = got == want
        bad += [] if ok else [f'{wording} {display} totals {got} != report {want}']
        say(f"  {wording:3s} {display:16s} got {got} report {want}  {'OK' if ok else 'MISMATCH'}")
    say('Totals R/E/C against e3_models/results_all_out.txt:')
    e3_want = parse_e3_totals()
    for display, _entry in E3_MODELS:
        got, want = e3[display][0], e3_want[display]
        ok = got == want
        bad += [] if ok else [f'E3 {display} totals {got} != results_all_out {want}']
        say(f"  {display:22s} got {got} results_all_out {want}  {'OK' if ok else 'MISMATCH'}")
    for e3_name, display in E3_CURRENT:
        got, want = new[display][0], e3_want[e3_name]
        ok = got == want
        bad += [] if ok else [f'{e3_name} totals {got} != results_all_out {want}']
        say(f"  {e3_name:22s} (s23 {display}) got {got} results_all_out {want}  {'OK' if ok else 'MISMATCH'}")
    say('Earlier layout-level p-values (rerun/stats/stats_output.txt, two-sided, printed to 5 significant digits):')
    blocks = parse_old_stats()
    want_blocks = [('qwen', old, 'Qwen 3.8 low', 'E'), ('glm', old, 'GLM-5.3', 'E'), ('ds', old, 'DeepSeek V4.1', 'E')]
    want_blocks += [(f'{s} C vs {x}', new, d, x) for d, s, _st in NEW_WORDING for x in ('E', 'R')]
    for block, res, display, x in want_blocks:
        c_tot, xname, x_tot, p_str, lay_nets = blocks[block]
        r = res[display][1][('C', x)]
        mine = dict(zip(res[display][2], r['nets']))
        ok = (xname == x and c_tot == r['A'] and x_tot == r['B'] and f"{float(r['p']):.5g}" == p_str
              and lay_nets == mine)
        bad += [] if ok else [f'stats_output {block}: C={c_tot} {xname}={x_tot} p={p_str} vs mine C={r["A"]} '
                              f'{x}={r["B"]} p={float(r["p"]):.5g}']
        say(f"  {block:22s} earlier C={c_tot} {xname}={x_tot} p={p_str:>9s} | here C={r['A']} {x}={r['B']} "
            f"p={float(r['p']):.5g}, 14 layout nets {'equal' if lay_nets == mine else 'DIFFER'}  {'OK' if ok else 'MISMATCH'}")
    say(f'Earlier blocks checked: {len(want_blocks)} (3 old-wording C-E, 12 new-wording C-E and C-R); '
        'the old-wording C-R and E-R and the new-wording E-R tests are new here.')
    say()
    say('ALL CHECKS PASS' if not bad else 'DISCREPANCIES:\n  ' + '\n  '.join(bad))
    OUT.write_text('\n'.join(LINES) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
