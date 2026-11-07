"""Extras (independent verifier): P5 and ALL on the old wording (all three models, all solver steps), the facing
error of matched detections (old three + new Haiku/Sonnet/Gemini), the stay-categories of GLM's and DeepSeek's
line failures under P, and R4 vs R4s per record. Prints to stdout."""
import collections
import json
import math
import statistics
import sys

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1/i01_repairs/verify')
from vrun import reference_constraints  # noqa: E402

V = 'results/2026-10-17-beyond-vlm/phase1/i01_repairs/verify'
ARMS = ('real', 'staffed', 'plain')


def angd(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi) * 180 / math.pi


def main():
    from sgi.gt.derive import is_visible, oracle_target
    from sgi.geometry import project
    from sgi.run import _matched_people
    from sgi.types import DetectedPerson
    do_p5 = {('old', 's19', 'qwen38_low'), ('old', 's19', 'glm53'), ('old', 's20', 'dsv41')}
    cells = [('old', 's19', 'qwen38_low'), ('old', 's19', 'glm53'), ('old', 's20', 'dsv41'),
             ('new', 's23', 'haiku55'), ('new', 's23', 'sonnet55_low'), ('new', 's23', 'gemini38_low')]
    for wording, stage, model in cells:
        vrows = {(r['arm'], r['scene'], r['episode'], r['trial']): r
                 for r in map(json.loads, open(f'{V}/vrows_{wording}_{model}.jsonl'))}
        errs, none = [], 0
        p5 = all_ = 0
        lf_p5 = lf_all = n_lf = 0
        br_p5 = br_all = 0
        cats = collections.Counter()
        for arm in ARMS:
            ref = H.runs(stage=stage, arm=arm, model=model, method='con')[0]
            th = H.recorded_config(ref)[1].thresholds
            for rec in H.records(ref):
                if not rec.solver_step:
                    continue
                row = vrows[(arm, rec.scene_id, rec.episode_id, rec.trial)]
                sc = H.scene(ref, rec.scene_id)
                obs, truth = sc.observation, sc.truth
                vis = [p for p in truth.people if is_visible(p, obs, th)]
                byid = {p.id: p for p in vis}
                det = [DetectedPerson.model_validate(p) for p in rec.people]
                m = _matched_people(det, vis, th)
                for d in det:
                    if d.number in m:
                        if d.yaw is None:
                            none += 1
                        else:
                            errs.append(angd(d.yaw, byid[m[d.number]].yaw))
                lf = row['situation'] == 'waiting_line' and row['target_correct'] and not row['C'] and (
                    not row['in_region'] or not row['reachable'])
                cons = [u['constraint'] for u in rec.constraints]
                if lf and not row['P']:
                    ca = [c for c in cons if c['kind'] == 'come_after']
                    if not ca:
                        cats['no come_after'] += 1
                    elif not any(c['people'] for c in ca):
                        cats['come_after with no numbers'] += 1
                    elif all(n not in m for c in ca for n in c['people']):
                        cats['come_after numbers all unmatched'] += 1
                    else:
                        cats['come_after keeps a matched number'] += 1
                    cats[f'recorded people: missed {len(vis) - len(m)} spurious {len(det) - len(m)}'] += 1
                if (wording, stage, model) not in do_p5:
                    continue
                # P people
                spur = {d.number for d in det if d.number not in m}
                nxt = max([d.number for d in det], default=0) + 1
                people, num = [], {}
                for d in det:
                    if d.number in m:
                        t = byid[m[d.number]]
                        people.append(d.model_copy(update={'xy': (t.x, t.y), 'yaw': t.yaw}))
                        num[t.id] = d.number
                found = set(m.values())
                for p in vis:
                    if p.id in found:
                        continue
                    mk = project(obs.camera, (p.x, p.y, th.person_mark_height_m))
                    if mk is None:
                        continue
                    people.append(DetectedPerson(number=nxt, xy=(p.x, p.y), yaw=p.yaw, pixel=tuple(mk[:2])))
                    num[p.id] = nxt
                    nxt += 1
                ep = truth.episodes[rec.episode_id]
                chosen = (rec.score.get('matched_target_id') if row['target_correct'] else None) or ep.target_ids[0]
                group = truth.group_for(rec.episode_id, chosen)
                members = [num[p.id] for p in truth.group(group) if p.id in num] if group is not None else []
                pid = truth.targets[chosen].person_id
                needed = num.get(pid) if pid else None
                rc = reference_constraints(rec.situation, members, needed)
                s5 = bool(H.resolve(rec, people=people, constraints=rc).score.success)
                ot = oracle_target(obs, truth.targets[chosen], th)
                sa = bool(H.resolve(rec, people=people, constraints=rc, target=ot).score.success) if ot else s5
                p5 += s5 - row['C']
                all_ += sa - row['C']
                br_p5 += row['C'] and not s5
                br_all += row['C'] and not sa
                if lf:
                    n_lf += 1
                    lf_p5 += s5
                    lf_all += sa
        C = sum(r['C'] for r in vrows.values())
        n45 = sum(e < 45 for e in errs)
        print(f'{wording} {model}: facing matched with yaw {len(errs)}, within 45 deg {n45}, median '
              f'{statistics.median(errs):.1f}, yaw None {none}')
        if cats:
            print('   line failures P leaves:', dict(cats))
        if (wording, stage, model) in do_p5:
            print(f'   P5 total {C + p5} (C {C}), ALL total {C + all_}; line failures {n_lf}: P5 flips {lf_p5}, '
                  f'ALL flips {lf_all}; breaks P5 {br_p5}, ALL {br_all}')
        r4diff = sum(r['solver_step'] and (r['R4'] != r['R4s'] or r['P'] != r['Ps']) for r in vrows.values())
        print(f'   steps where R4 != R4s or P != Ps: {r4diff}')


if __name__ == '__main__':
    main()
