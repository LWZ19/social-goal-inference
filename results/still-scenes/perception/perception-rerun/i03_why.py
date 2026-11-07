"""I03 'Why' figures: GLM people_found_share overall / lines / groups (C goal steps, realistic s19), and the
single-person line (group_size 1): found share, members recall, C successes. Also s23 GLM for the new wording."""
import sys, collections
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
for stage, model in [('s19', 'glm53'), ('s23', 'glm53'), ('s20', 'dsv41'), ('s23', 'dsv41')]:
    recs = [r for arm in ('real', 'staffed', 'plain') for r in H.records(H.runs(stage=stage, arm=arm, model=model, method='con')[0])]
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs) / len(xs), len(xs)) if xs else (float('nan'), 0)
    gs = [r for r in recs if r.solver_step]
    out = [f'{stage} {model}: pfs all {mean([r.measures.get("people_found_share") for r in gs])}',
           f'lines {mean([r.measures.get("people_found_share") for r in gs if r.situation == "waiting_line"])}',
           f'groups {mean([r.measures.get("people_found_share") for r in gs if r.situation == "conversation_group"])}',
           f'members_recall {mean([r.measures.get("members_recall") for r in gs])}',
           f'members_precision {mean([r.measures.get("members_precision") for r in gs])}']
    print('; '.join(out))
    one = [r for r in recs if r.situation == 'waiting_line' and (r.labels.get('factors') or {}).get('group_size') == 1]
    print(f'   single-person line: records {len(one)}, C success {sum(bool(r.score.get("success")) for r in one)}, '
          f'goal steps {sum(r.solver_step for r in one)}, pfs {mean([r.measures.get("people_found_share") for r in one if r.solver_step])}, '
          f'members_recall {mean([r.measures.get("members_recall") for r in one])}, scenes {sorted(set(r.scene_id for r in one))}')
