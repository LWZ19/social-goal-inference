"""M3 variants: which reading could give the report's 3.9%? Reads only through research_harness."""
import sys, collections, math
from pathlib import Path
sys.path.insert(0, 'docs/model_notes/tools')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_harness as H
from flags_misc import seg_poly
from sgi.polygons import point_in_polygon, disc
OUT = Path(__file__).resolve().parent
lines = []
def out(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); lines.append(s)
C_RUNS = {'qwen': ('s19', 'qwen38_low'), 'glm': ('s19', 'glm53'), 'ds': ('s20', 'dsv41')}
acc = collections.Counter()
for arm in ('real', 'staffed', 'plain', 'ph', 'dev_seed0', 'dev_seed1', 'dev_seed2'):
    grp = 'realistic' if arm in ('real', 'staffed', 'plain') else ('ph' if arm == 'ph' else 'dev')
    for key, (stage, model) in C_RUNS.items():
        run = H.runs(stage=stage, arm=arm, model=model, method='con')[0]
        th = H.recorded_config(run)[1].thresholds
        for r in H.records(run):
            if r.pose_world is None or r.result.get('status') != 'ok':
                continue
            sc = H.scene(run, r.scene_id)
            robot = sc.observation.robot
            a = (float(robot.x), float(robot.y)); b = (float(r.pose_world[0]), float(r.pose_world[1]))
            regs = sc.truths[r.episode_id].gt.forbidden_regions or []
            succ = bool(r.score['success'])
            matched = r.score.get('matched_target_id')
            hits = [seg_poly(a, b, reg.polygon) for reg in regs]
            A = any(h[0] for h in hits)
            Am = any(h[0] for h, reg in zip(hits, regs) if reg.target_id in (None, matched))
            # people bodies as forbidden discs (person_radius_m), the target person excluded
            tgt_pid = None
            if matched is not None:
                t = sc.truth.targets[matched]
                tgt_pid = t.person_id if t.kind == 'person' else None
            P = False
            for p in sc.truth.people:
                if p.id == tgt_pid:
                    continue
                if seg_poly(a, b, disc((p.x, p.y), th.person_radius_m, 24))[0]:
                    P = True
            for scope in ((grp,), ('all arms',)):
                for den, keep in (('all', True), ('succ', succ), ('solver', r.result.get('source') == 'constraints'),
                                  ('has_forbidden', bool(regs))):
                    if not keep:
                        continue
                    k = (scope[0], key, den)
                    acc[k + ('n',)] += 1
                    acc[k + ('A',)] += A
                    acc[k + ('A_matched',)] += Am
                    acc[k + ('people',)] += P
                    acc[k + ('A_or_people',)] += A or P
                    acc[k + ('pairs_n',)] += len(regs)
                    acc[k + ('pairs_hit',)] += sum(h[0] for h in hits)
for scope in ('realistic', 'ph', 'dev', 'all arms'):
    for den in ('all', 'succ', 'solver', 'has_forbidden'):
        tot = collections.Counter()
        for key in C_RUNS:
            k = (scope, key, den)
            for m in ('n', 'A', 'A_matched', 'people', 'A_or_people', 'pairs_n', 'pairs_hit'):
                tot[m] += acc[k + (m,)]
            n = acc[k + ('n',)]
            if n:
                out(f'{scope:9s} {den:13s} {key:5s} n={n:4d} A {acc[k+("A",)]} ({acc[k+("A",)]/n:.1%}) A_matched {acc[k+("A_matched",)]}'
                    f' people {acc[k+("people",)]} ({acc[k+("people",)]/n:.1%}) A|people {acc[k+("A_or_people",)]} ({acc[k+("A_or_people",)]/n:.1%})'
                    f' pairs {acc[k+("pairs_hit",)]}/{acc[k+("pairs_n",)]}')
        n = tot['n']
        if n:
            out(f'{scope:9s} {den:13s} POOL  n={n:4d} A {tot["A"]} ({tot["A"]/n:.1%}) A_matched {tot["A_matched"]} ({tot["A_matched"]/n:.1%})'
                f' people {tot["people"]} ({tot["people"]/n:.1%}) A|people {tot["A_or_people"]} ({tot["A_or_people"]/n:.1%})'
                f' pairs {tot["pairs_hit"]}/{tot["pairs_n"]} ({(tot["pairs_hit"]/tot["pairs_n"] if tot["pairs_n"] else 0):.1%})')
(OUT / 'm3_variants_output.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
