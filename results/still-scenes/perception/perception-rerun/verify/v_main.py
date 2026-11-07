"""Independent verifier for the perception claims (P1, P2, P3, I03-why, N1-N6). Reads through research_harness only."""
import sys, math, re, json, collections, statistics
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
from sgi.gt.derive import is_visible

OUT = 'results/still-scenes/perception/perception-rerun/verify/v_main_out.txt'
lines_out = []
def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s); lines_out.append(s)

REAL = ('real', 'staffed', 'plain')


def my_match(dets, vis, radius):
    """Greedy one-to-one, nearest first, within 2*radius. Returns {number: person_id}."""
    pairs = []
    for d in dets:
        for i, p in enumerate(vis):
            pairs.append((math.dist(d['xy'], (p.x, p.y)), d['number'], i))
    pairs.sort()
    m, taken = {}, set()
    for g, n, i in pairs:
        if g <= 2 * radius and n not in m and i not in taken:
            m[n] = vis[i].id; taken.add(i)
    return m


def analyse(stage, model, label):
    runs = [r for r in H.runs(stage=stage, model=model, method='con') if r.arm in REAL]
    assert len(runs) == 3, runs
    rows = []
    for r in runs:
        th = H.recorded_config(r)[1].thresholds
        for rec in H.records(r):
            if not rec.solver_step:
                continue
            sc = H.scene(r, rec.scene_id)
            truth, obs = sc.truth, sc.observation
            vis = [p for p in truth.people if is_visible(p, obs, th)]
            m = my_match(rec.people, vis, th.person_radius_m)
            ep = truth.episodes[rec.episode_id]
            chosen = rec.score.get('matched_target_id') or ep.target_ids[0]
            grp = truth.group_for(rec.episode_id, chosen)
            members = [p.id for p in vis if grp is not None and p.group == grp]
            ca = [c for c in rec.constraints if c['constraint']['kind'] == 'come_after']
            listed = {n for c in ca for n in c['constraint']['people']}
            anylisted = {n for c in rec.constraints for n in c['constraint']['people']}
            named = {m.get(n) for n in listed} - {None}
            named_any = {m.get(n) for n in anylisted} - {None}
            found_ids = set(m.values())
            missed = [pid for pid in members if pid not in named]
            missed_any = [pid for pid in members if pid not in named_any]
            nf = [pid for pid in missed if pid not in found_ids]
            nf_any = [pid for pid in missed_any if pid not in found_ids]
            share = len(m) / len(vis) if vis else None
            recall = (len([x for x in listed if m.get(x) in members]) / len(members)) if (members and rec.situation == 'waiting_line') else None
            ms = rec.measures
            s = rec.score
            rows.append(dict(arm=r.arm, scene=rec.scene_id, ep=rec.episode_id, trial=rec.trial, sit=rec.situation,
                             labels=rec.labels, members=members, missed=missed, nf=nf, missed_any=missed_any, nf_any=nf_any,
                             share=share, rec_share=ms.get('people_found_share'), recall=recall, rec_recall=ms.get('members_recall'),
                             unmatched=ms.get('people_unmatched'), n_ca=len(ca), success=s.get('success'),
                             tc=s.get('target_correct'), in_region=s.get('in_region'), reachable=s.get('reachable'),
                             people=rec.people, truth_people=list(truth.people), vis=vis, rec=rec, run=r, m=m,
                             listed=listed))
    # agreement with recorded
    agree_share = sum(1 for x in rows if x['rec_share'] is not None and x['share'] is not None and abs(x['share'] - x['rec_share']) < 1e-9)
    n_share = sum(1 for x in rows if x['rec_share'] is not None)
    agree_rec = sum(1 for x in rows if x['rec_recall'] is not None and x['recall'] is not None and abs(x['recall'] - x['rec_recall']) < 1e-9)
    n_rec = sum(1 for x in rows if x['rec_recall'] is not None)
    P(f'== {label}: solver steps {len(rows)}; share agree {agree_share}/{n_share}; recall agree {agree_rec}/{n_rec}')
    line = [x for x in rows if x['sit'] == 'waiting_line']
    def ratio(sub, key_m='missed', key_n='nf'):
        a = sum(len(x[key_n]) for x in sub); b = sum(len(x[key_m]) for x in sub)
        return f'{a}/{b} = {a/b:.3f}' if b else 'n/a'
    fails = [x for x in line if x['tc'] and not x['success'] and (not x['in_region'] or not x['reachable'])]
    failed_any = [x for x in line if not x['success']]
    P(f'  P1 lines (all {len(line)} line steps): never-found/missed = {ratio(line)}')
    P(f'  P1 line failures ({len(fails)}): {ratio(fails)}')
    P(f'  P1 all failed line steps ({len(failed_any)}): {ratio(failed_any)}')
    grp = [x for x in rows if x['sit'] != 'waiting_line' and x['members']]
    P(f'  P1 groups (any constraint naming): {ratio(grp, "missed_any", "nf_any")} over {len(grp)} steps with members')
    P(f'  P1 lines+groups pooled: lines come_after + groups any = '
      f'{sum(len(x["nf"]) for x in line)+sum(len(x["nf_any"]) for x in grp)}/{sum(len(x["missed"]) for x in line)+sum(len(x["missed_any"]) for x in grp)}')
    # P3
    r1 = sum(1 for x in fails if x['rec_recall'] is not None and x['rec_recall'] < 1)
    r0 = sum(1 for x in fails if x['rec_recall'] == 0)
    f1 = sum(1 for x in fails if x['rec_share'] is not None and x['rec_share'] < 1)
    f0 = sum(1 for x in fails if x['rec_share'] == 0)
    nca = sum(1 for x in fails if x['n_ca'] == 0)
    um = sum(1 for x in fails if (x['unmatched'] or 0) > 0)
    unr = sum(1 for x in fails if not x['reachable'])
    outr = sum(1 for x in fails if not x['in_region'])
    rnone = sum(1 for x in fails if x['rec_recall'] is None)
    P(f'  P3 line failures {len(fails)}: recall<1 {r1} (=0 {r0}, None {rnone}); found<1 {f1} (=0 {f0}); unmatched>0 {um}; no come_after {nca}; unreachable {unr}; outside region {outr}')
    # alt failure def: all line steps not success
    P(f'  P3-alt all non-success line steps: {len(failed_any)}; with target correct {sum(1 for x in failed_any if x["tc"])}')
    # I03-why
    sh = [x['rec_share'] for x in rows if x['rec_share'] is not None]
    shl = [x['rec_share'] for x in line if x['rec_share'] is not None]
    single = [x for x in line if (x['labels'].get('factors') or {}).get('group_size') == 1]
    shs = [x['rec_share'] for x in single if x['rec_share'] is not None]
    rl = [x['rec_recall'] for x in line if x['rec_recall'] is not None]
    P(f'  I03 found mean {statistics.mean(sh):.3f} (n {len(sh)}); lines {statistics.mean(shl):.3f} (n {len(shl)}); single-person line {statistics.mean(shs):.3f} (n {len(shs)}); members_recall mean {statistics.mean(rl):.3f} (n {len(rl)})')
    return rows


def single_line_success(stage, model):
    runs = [r for r in H.runs(stage=stage, model=model, method='con') if r.arm in REAL]
    n = s = 0
    for r in runs:
        for rec in H.records(r):
            if rec.situation == 'waiting_line' and (rec.labels.get('factors') or {}).get('group_size') == 1:
                n += 1; s += bool(rec.score.get('success'))
    return s, n


def phantoms(rows, label):
    seen = {}
    per_rec = [0, 0]
    for x in rows:
        if x['sit'] != 'waiting_line':
            continue
        key = (x['arm'], x['scene'], tuple(tuple(round(v, 6) for v in d['xy']) for d in x['people']))
        far = 0
        for d in x['people']:
            dmin = min(math.dist(d['xy'], (p.x, p.y)) for p in x['truth_people'])
            if dmin > 1.0:
                far += 1
        per_rec[0] += far; per_rec[1] += len(x['people'])
        seen[key] = (far, len(x['people']))
    a = sum(v[0] for v in seen.values()); b = sum(v[1] for v in seen.values())
    P(f'  N5 {label}: distinct line people answers {len(seen)}: far(>1m from every true person) {a} of {b}; per record {per_rec[0]} of {per_rec[1]}')


def wall_check(rows, label):
    """Signed distance along the camera ray from the far detection to the bounds polygon exit; listed in come_after."""
    dists, listed, farthest, n = [], 0, 0, 0
    recs_with_phantom = 0; recs_fail = 0
    for x in rows:
        if x['sit'] != 'waiting_line':
            continue
        r = x['run']
        sc = H.scene(r, x['scene'])
        cam = sc.observation.camera.T_world_camera
        cx, cy = float(cam[0, 3]), float(cam[1, 3])
        poly = sc.truth.bounds
        has = False
        tgt = x['rec'].target.get('world') if x['rec'].target else None
        for d in x['people']:
            dmin = min(math.dist(d['xy'], (p.x, p.y)) for p in x['truth_people'])
            if dmin <= 1.0:
                continue
            has = True; n += 1
            dx, dy = d['xy'][0] - cx, d['xy'][1] - cy
            L = math.hypot(dx, dy); ux, uy = dx / L, dy / L
            # ray from camera: find largest t where ray exits polygon
            ts = []
            for i in range(len(poly)):
                (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
                ex, ey = x2 - x1, y2 - y1
                den = ux * ey - uy * ex
                if abs(den) < 1e-12:
                    continue
                t = ((x1 - cx) * ey - (y1 - cy) * ex) / den
                u = ((x1 - cx) * uy - (y1 - cy) * ux) / den
                if t > 0 and -1e-9 <= u <= 1 + 1e-9:
                    ts.append(t)
            texit = max(ts) if ts else float('nan')
            dists.append(texit - L)
            if d['number'] in x['listed']:
                listed += 1
        if has:
            recs_with_phantom += 1
            recs_fail += not x['success']
    if dists:
        P(f'  N6 {label}: far detections {n}; wall gap (exit - det, m) median {statistics.median(dists):.3f} min {min(dists):.3f} max {max(dists):.3f}; within 0.3 m of wall {sum(abs(v) <= 0.3 for v in dists)}; listed under come_after {listed}; records with phantom {recs_with_phantom}, failing {recs_fail}')


def p2(stage, model, arms, label):
    runs = [r for r in H.runs(stage=stage, model=model, method='con') if r.arm in arms]
    texts = {}
    reason = 0
    for r in runs:
        for rec in H.records(r):
            if rec.people is None:
                continue
            for c in rec.result.get('calls') or []:
                if c['stage'] == 'constraints' and c.get('raw_text'):
                    texts[c['raw_text']] = 1
                    reason += (c.get('usage') or {}).get('reasoning_tokens') or 0
    tot = vals = withkey = 0
    for t in texts:
        tot += len(t)
        for mm in re.finditer(r'"people"\s*:\s*\[([^\]]*)\]', t):
            vals += len(mm.group(1)); withkey += len(mm.group(0))
    P(f'  P2 {label}: distinct constraints answers {len(texts)}; chars {tot}; list contents {vals} ({100*vals/tot:.2f}%); with key {withkey} ({100*withkey/tot:.2f}%); reasoning tokens total {reason}')
    # token proxy: words/punct split
    tok = lambda s: re.findall(r"\w+|[^\w\s]", s)
    T = sum(len(tok(t)) for t in texts)
    V = sum(len(tok(mm.group(1))) for t in texts for mm in re.finditer(r'"people"\s*:\s*\[([^\]]*)\]', t))
    K = sum(len(tok(mm.group(0))) for t in texts for mm in re.finditer(r'"people"\s*:\s*\[([^\]]*)\]', t))
    P(f'     token proxy (word/punct): contents {100*V/T:.2f}%  with key {100*K/T:.2f}%')


if __name__ == '__main__':
    g19 = analyse('s19', 'glm53', 'GLM s19 con')
    s, n = single_line_success('s19', 'glm53'); P(f'  I03 single-person line C success {s} of {n}')
    phantoms(g19, 'GLM s19')
    d20 = analyse('s20', 'dsv41', 'DeepSeek s20 con')
    phantoms(d20, 'DeepSeek s20')
    g23 = analyse('s23', 'glm53', 'GLM s23 con')
    phantoms(g23, 'GLM s23')
    wall_check(g23, 'GLM s23')
    d23 = analyse('s23', 'dsv41', 'DeepSeek s23 con')
    phantoms(d23, 'DeepSeek s23')
    wall_check(d23, 'DeepSeek s23')
    q19 = analyse('s19', 'qwen38_low', 'Qwen s19 con')
    phantoms(q19, 'Qwen s19')
    p2('s19', 'glm53', REAL, 'GLM s19 realistic')
    p2('s20', 'dsv41', REAL, 'DeepSeek s20 realistic')
    p2('s19', 'glm53', ('dev_seed0', 'dev_seed1', 'dev_seed2'), 'GLM s19 dev seeds')
    p2('s20', 'dsv41', ('dev_seed0', 'dev_seed1', 'dev_seed2'), 'DeepSeek s20 dev seeds')
    p2('s19', 'glm53', ('ph',), 'GLM s19 ph')
    p2('s20', 'dsv41', ('ph',), 'DeepSeek s20 ph')
    p2('s23', 'glm53', REAL, 'GLM s23 realistic')
    p2('s23', 'dsv41', REAL, 'DeepSeek s23 realistic')
    # s23 vs s19 people identity and constraints freshness
    for (sa, ma, sb) in (('s19', 'glm53', 's23'), ('s20', 'dsv41', 's23')):
        A = {}
        for r in [r for r in H.runs(stage=sa, model=ma, method='con') if r.arm in REAL]:
            for rec in H.records(r):
                if rec.people is not None:
                    A[(r.arm, rec.scene_id, rec.episode_id, rec.trial)] = rec
        same = tot = fresh = ctot = 0
        for r in [r for r in H.runs(stage=sb, model=ma, method='con') if r.arm in REAL]:
            for rec in H.records(r):
                if rec.people is None:
                    continue
                k = (r.arm, rec.scene_id, rec.episode_id, rec.trial)
                for c in rec.result.get('calls') or []:
                    if c['stage'] == 'constraints':
                        ctot += 1; fresh += not c.get('cached')
                if k in A:
                    tot += 1; same += json.dumps(A[k].people) == json.dumps(rec.people)
        P(f'  {sb} {ma}: people identical to {sa} in {same}/{tot}; constraints calls fresh {fresh}/{ctot}')
    open(OUT, 'w').write('\n'.join(lines_out) + '\n')
