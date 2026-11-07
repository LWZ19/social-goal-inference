"""E2 driver (E2_FROZEN.md): the current C (A0), a second draw of it (A0b) and C with I09 A1's detector people (D), on
the fresh static E2 scenes, paired in each unit. A unit is one (scene, episode, trial) on one model:
  - the plan call, the lifted target and the context call run once, through sgi's own Pipeline, and are shared;
  - where the model's context check says people matter, each arm runs its goal step on the same inputs the pipeline
    would give it (the model's own check included), in a rotating order; otherwise C's standard goal is every arm's.
Run by the run from the main checkout's venv:
  python -B e2_driver.py --models soclaas_qwen38_low --scenes run_scenes --people e2_people.json \
      --people-sha256 <hex> --out runs/e2_qwen38_low --rate 6 [--shard 0/4] [--resume]
  python -B e2_driver.py --models soclaas_qwen38_low --scenes <dir> --people <file> --people-sha256 <hex> \
      --out /tmp --dry --limit 3                                                     (mock back end, zero calls)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

TRIAL_BASE = 50            # A0, D, plan and context at TRIAL_BASE + t
FLOOR_OFFSET = 100         # A0b at TRIAL_BASE + FLOOR_OFFSET + t: fresh draws of the same two calls
ORDERS = (("A0", "D", "A0b"), ("D", "A0b", "A0"), ("A0b", "A0", "D"))
MAX_UNIT_CALLS = 7


class DetectorPeople:
    """A1's frozen detector people of one scene as a people source (E1b's class, unchanged): each a FoundPerson at
    the detector's (x, y) and yaw, its pixel the projection of the point 1.1 m above it. No model is asked."""

    def __init__(self, people, th):
        self.people = sorted(people, key=lambda p: -p['score'])
        self.th = th

    def __call__(self, observation):
        from sgi.geometry import project
        from sgi.stages.people import FoundPerson, PeopleFound
        started = time.perf_counter()
        found = []
        for p in self.people:
            m = project(observation.camera, (p['x'], p['y'], self.th.person_mark_height_m))
            if m is not None:
                found.append(FoundPerson(xy=(float(p['x']), float(p['y'])),
                                         yaw=None if p['yaw'] is None else float(p['yaw']), pixel=m[:2], label=None))
        return PeopleFound(found=found, calls=[], call_time_s=0.0, geometry_time_s=time.perf_counter() - started)


class Capture:
    """Stands in for the goal step once, so that Pipeline.infer_goal runs its own target and context steps and hands
    over exactly what it would give a method: (observation, request, subtask, target, the model's own check)."""

    def __init__(self):
        self.args = None

    def infer(self, observation, request, subtask, target, context_check):
        from sgi.types import GoalResult
        self.args = (observation, request, subtask, target, context_check)
        return GoalResult(status='ok', source='capture')


def merged(shared, res):
    """The arm's result as Pipeline.infer_goal would have returned it with that arm's method (its last lines)."""
    timings = dict(shared.timings_s)
    for stage, seconds in res.timings_s.items():
        timings[stage] = timings.get(stage, 0.0) + seconds
    return res.model_copy(update={'calls': shared.calls + res.calls, 'timings_s': timings, 'target': shared.target,
                                  'location': shared.location, 'context_check': shared.context_check})


def with_plan(result, outcome):
    """As sgi.run.run_episode: the plan call belongs to every result of a model-made plan."""
    return result.model_copy(update={'calls': [outcome.call, *result.calls],
                                     'timings_s': {**result.timings_s, 'plan': outcome.time_s}})


def arm_entry(result, sc, wall):
    return dict(status=result.status, source=result.source,
                score=dict(success=bool(sc.success), target_correct=bool(sc.target_correct),
                           in_region=bool(sc.in_region), heading_ok=bool(sc.heading_ok), reachable=bool(sc.reachable)),
                calls=[dict(stage=c.stage, status=c.status, latency_s=c.latency_s, latency_total_s=c.latency_total_s,
                            validation_retries=c.validation_retries, transient_retries=c.transient_retries,
                            cached=c.cached, usage=c.usage.model_dump(mode='json')) for c in result.calls],
                timings_s=result.timings_s, n_people=None if result.people is None else len(result.people),
                pose_world=result.pose_world,
                constraints=None if result.constraints is None else [
                    dict(kind=getattr(c.constraint, 'kind', None) or type(c.constraint).__name__, status=c.status)
                    for c in result.constraints],
                wall_s=wall)


def run_unit(pipeline, cfg, inputs, truth, et, episode, t, people, order):
    from sgi.evaluate import score
    from sgi.gt.derive import reference_people_matter
    from sgi.run import _UNMATCHED, _matched
    from sgi.types import GoalResult
    obs, ep, th = inputs.observation, et.gt, cfg.thresholds
    trial = TRIAL_BASE + t
    started = time.time()
    outcome = pipeline.plan(episode.request, episode.context, observation=obs, trial=trial)
    plan = outcome.plan
    subtasks = [] if plan is None else plan.subtasks
    index = next((i for i, s in enumerate(subtasks) if s.type == 'navigate' and s.target_point is not None), None)
    chosen = None
    capture = Capture()
    if plan is None:
        shared = GoalResult(status=outcome.status)
    elif index is None:
        shared = GoalResult(status='no_target')
    else:
        matched = _matched(subtasks[index], obs, truth, ep, th)
        chosen = matched if matched != _UNMATCHED else None
        shared = pipeline.infer_goal(obs, episode.request, episode.context, plan, index, inferrer=capture,
                                     trial=trial)
    people_matter = capture.args is not None
    arms = {}
    for arm in order:
        t0 = time.perf_counter()
        if not people_matter:
            res = shared
        else:
            if arm == 'A0':
                inferrer = pipeline.constraint_inferrer(None, trial=trial)
            elif arm == 'A0b':
                inferrer = pipeline.constraint_inferrer(None, trial=trial + FLOOR_OFFSET)
            else:
                inferrer = pipeline.constraint_inferrer(DetectorPeople(people, th), trial=trial)
            res = merged(shared, inferrer.infer(*capture.args))
        res = with_plan(res, outcome)
        arms[arm] = arm_entry(res, score(res, obs, truth, ep, th), time.perf_counter() - t0)
    calls = [c for a in arms.values() for c in a['calls']]
    check = shared.context_check
    return dict(scene=inputs.scene_id, episode=episode.episode_id, t=t, trial=trial, order=list(order),
                plan_status=outcome.status, evaluated_subtask=index, chosen=chosen, people_matter=people_matter,
                said_people_matter=None if check is None else bool(check.people_matter),
                reference_people_matter=bool(reference_people_matter(et, chosen)),
                vlm_error=any(c['status'] == 'vlm_error' for c in calls), arms=arms, started=started)


def unit_calls(row):
    """Distinct model calls of a unit: the shared ones once, each arm's own goal-step calls."""
    shared = [c for c in row['arms'][row['order'][0]]['calls'] if c['stage'] in ('plan', 'context')]
    own = [c for a in row['arms'].values() for c in a['calls'] if c['stage'] not in ('plan', 'context')]
    return len(shared) + (len(own) if row['people_matter'] else 0)


def run_meta(cfg, args, people_sha):
    from sgi.config import REPO_ROOT
    git = lambda *a: subprocess.run(['git', *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout
    diff = git('diff', 'HEAD', '--', 'src', str(cfg.run.prompts_dir))
    return dict(git_head=git('rev-parse', 'HEAD').strip(), diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),
                config_sha256=hashlib.sha256(json.dumps(cfg.model_dump(mode='json'), sort_keys=True).encode()).hexdigest(),
                people_sha256=people_sha, model=args.models, trials=args.trials, scenes=str(Path(args.scenes).resolve()),
                trial_base=TRIAL_BASE, floor_offset=FLOOR_OFFSET, dry=args.dry)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True, help='one model entry per process')
    ap.add_argument('--scenes', required=True)
    ap.add_argument('--people', required=True)
    ap.add_argument('--people-sha256', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--trials', type=int, default=2)
    ap.add_argument('--rate', type=int, default=6, help="this process's requests a minute (the client's own limit)")
    ap.add_argument('--max-calls', type=int, default=1150)
    ap.add_argument('--shard', default='0/1', help='i/n; keep n fixed for a model across resumes')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='first N units of this shard (smoke)')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args(argv)
    from sgi.gt.derive import prepare
    from sgi.pipeline import Pipeline, build_pipeline
    from sgi.prompts import stages_of
    from sgi.run import _load_scenes
    from sgi.serve import make_config
    from sgi.vlm.factory import make_client
    data = Path(a.people).read_bytes()
    people_sha = hashlib.sha256(data).hexdigest()
    if people_sha != a.people_sha256:
        raise SystemExit(f'people file sha256 {people_sha} is not the one given: refusing')
    table = {str(Path(k).resolve()): v['people'] for k, v in json.loads(data)['scenes'].items()}
    if ',' in a.models:
        raise SystemExit('one model per process')
    cfg = make_config(a.models, max_rpm=a.rate)
    run = cfg.run
    if (run.goal, run.people, run.constraints, run.context_check, run.grounding, run.grounding_source) != (
            'constraints', 'vlm', 'vlm', 'vlm', 'vlm', 'view'):
        raise SystemExit(f'not the current C: {run}')
    if a.dry:
        from sgi.vlm.mock import MockBackend, centre_responder
        clients = {s: make_client(cfg.stages[s].model, cfg, backend=MockBackend(centre_responder))
                   for s in stages_of(run)}
        for c in clients.values():
            c._cache = None
        pipeline = Pipeline(cfg, clients)
    else:
        pipeline = build_pipeline(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = run_meta(make_config(a.models), a, people_sha)     # the method's config: the rate is pacing, not method
    mp = out / 'run_meta.json'
    if mp.exists() and json.loads(mp.read_text()) != meta:
        raise SystemExit(f'{mp} differs from this run (code, config, people or settings changed): refusing')
    mp.write_text(json.dumps(meta, indent=1))
    units = []
    for folder, inputs, truth in _load_scenes(Path(a.scenes), cfg):
        people = table[str(folder.resolve())]
        ets = prepare(inputs, truth, cfg)
        for episode in inputs.episodes:
            for t in range(a.trials):
                units.append((folder, inputs, truth, ets[episode.episode_id], episode, t, people))
    i, n = (int(x) for x in a.shard.split('/'))
    rp = out / f'records_{i}of{n}.jsonl'
    done = set()
    if a.resume and rp.exists():
        for line in rp.read_text().splitlines():
            r = json.loads(line)
            if not r['vlm_error']:
                done.add((r['scene'], r['episode'], r['t']))
    log = rp.open('a')
    n_calls, ran = 0, 0
    for k, (folder, inputs, truth, et, episode, t, people) in enumerate(units):
        if k % n != i or (inputs.scene_id, episode.episode_id, t) in done:
            continue
        if a.limit and ran >= a.limit:
            break
        if n_calls + MAX_UNIT_CALLS > a.max_calls:
            print(f'stop: --max-calls {a.max_calls}', flush=True)
            return 3
        row = run_unit(pipeline, cfg, inputs, truth, et, episode, t, people, ORDERS[k % 3])
        row.update(model=a.models, unit=k)
        row['n_calls'] = unit_calls(row)
        n_calls += row['n_calls']
        ran += 1
        log.write(json.dumps(row, default=str) + '\n')
        log.flush()
        ok = {arm: v['score']['success'] for arm, v in row['arms'].items()}
        print(f"{n_calls:>5} {a.models} {inputs.scene_id} {episode.episode_id} t{t} pm={row['people_matter']} "
              f"{ok}{' VLM_ERROR' if row['vlm_error'] else ''}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
