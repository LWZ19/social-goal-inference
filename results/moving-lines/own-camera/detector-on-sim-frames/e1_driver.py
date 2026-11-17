"""E1b driver (E1_FROZEN.md): detector people in place of the VLM people call, on the new-wording C records whose
goal step called people (the people-matter records), realistic sets. Per record, the recorded plan, evaluated subtask,
target point and context check are given to `pipeline.infer_goal`, and only the constraint method runs:
  A0: the VLM people call + the constraints call (the current C), at a fresh trial;
  D : I09 A1's frozen detector people (no people call) + the constraints call, at the same fresh trial.
Run by the run from the main checkout's venv:
  python -B e1_driver.py --models soclaas_glm53 --out runs/e1_glm53 --rate 13           (live)
  python -B e1_driver.py --models soclaas_glm53 --out /tmp --limit 4 --dry          (zero calls)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

sys.path.insert(0, str(paths.REPO / 'docs/model_notes/tools'))
import research_harness as H  # noqa: E402

EVAL_PEOPLE = paths.STILL / 'people-finder/early-detector/phase1c/a1/eval_people.json'
EVAL_PEOPLE_SHA256 = 'ffa595453c94b444c8dd030d053d37ac103441a7e74f07e3f172836b3d0c24b3'
TRIAL_OFFSET = 30
MODEL_RUNS = {'soclaas_qwen38_low': ('s21', 'qwen38_low'), 'soclaas_glm53': ('s23', 'glm53'),
              'soclaas_dsv41': ('s23', 'dsv41'), 'claude_sonnet55_low': ('s23', 'sonnet55_low'),
              'claude_haiku55': ('s23', 'haiku55'), 'gemini_fast': ('s23', 'gemini38_low')}


class DetectorPeople:
    """A1's frozen detector people of one scene as a people source (the D-live driver's class): each a FoundPerson at
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


def detector_table():
    data = EVAL_PEOPLE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == EVAL_PEOPLE_SHA256, 'eval_people.json changed'
    doc = json.loads(data)
    return {str(Path(k).resolve()): v for k, v in doc['scenes'].items() if v['domain'] == 'E-real'}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--arms', default='A0,D')
    ap.add_argument('--out', required=True)
    ap.add_argument('--rate', type=float, default=13.0)
    ap.add_argument('--max-calls', type=int, default=1000)
    ap.add_argument('--limit', type=int, default=0, help='first N records per model (smoke)')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args(argv)
    from sgi.pipeline import Pipeline, build_pipeline
    from sgi.serve import make_config
    from sgi.types import ContextCheck, Plan, Point
    from sgi.vlm.factory import make_client
    from sgi.prompts import stages_of
    table = detector_table()
    wl = H.whitelist()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / 'records.jsonl').open('a')
    stamps, n_calls = [], 0
    for model in a.models.split(','):
        stage, short = MODEL_RUNS[model]
        cfg = make_config(model)
        if a.dry:
            from sgi.vlm.mock import MockBackend, centre_responder
            clients = {s: make_client(cfg.stages[s].model, cfg, backend=MockBackend(centre_responder))
                       for s in stages_of(cfg.run)}
            for c in clients.values():
                c._cache = None
            pipeline = Pipeline(cfg, clients)
        else:
            pipeline = build_pipeline(cfg)
        th = cfg.thresholds
        done = 0
        for run in H.runs(stage=stage, method='con', model=short, wl=wl):
            for rec in H.records(run, wl=wl):
                if not rec.solver_step:
                    continue                       # no people call recorded: both arms would be identical
                if a.limit and done >= a.limit:
                    break
                done += 1
                sc = H.scene(rec.ref, rec.scene_id, wl=wl)
                folder = str(Path(H.check_scene(rec.ref, rec.scene_id, wl=wl)).resolve())
                people = table[folder]['people']
                obs = sc.observation
                plan = Plan(**rec.raw['plan'])
                idx = rec.raw['evaluated_subtask']
                pt = rec.result['target']['point']
                ctx = ContextCheck(**rec.result['context_check'])
                ep = next(e for e in sc.inputs.episodes if e.episode_id == rec.episode_id)
                trial = TRIAL_OFFSET + rec.trial
                for arm in a.arms.split(','):
                    need = 2 if arm == 'A0' else 1
                    if n_calls + need > a.max_calls:
                        print(f'stop: --max-calls {a.max_calls}', flush=True)
                        return 3
                    for _ in range(need):
                        now = time.time()
                        stamps = [x for x in stamps if now - x < 60.0]
                        if len(stamps) >= a.rate:
                            time.sleep(60.0 - (now - stamps[0]) + 0.05)
                        stamps.append(time.time())
                    n_calls += need
                    source = None if arm == 'A0' else DetectorPeople(people, th)
                    inferrer = pipeline.constraint_inferrer(source, trial=trial)
                    started = time.perf_counter()
                    res = pipeline.infer_goal(obs, ep.request, ep.context, plan, idx, target_point=Point(**pt),
                                              context_check=ctx, inferrer=inferrer, trial=trial)
                    wall = time.perf_counter() - started
                    sc_ok = None
                    if res.status == 'ok' and res.pose_world is not None:
                        s = H.score_pose(rec, res.pose_world, target=res.target)
                        sc_ok = dict(success=bool(s.success), target_correct=bool(s.target_correct),
                                     in_region=bool(s.in_region), heading_ok=bool(s.heading_ok),
                                     reachable=bool(s.reachable))
                    row = dict(model=model, arm=arm, run=str(rec.run), scene=rec.scene_id, episode=rec.episode_id,
                               rec_trial=rec.trial, trial=trial, status=res.status, score=sc_ok,
                               recorded_success=bool(rec.score.get('success')), timings_s=res.timings_s,
                               calls=[dict(stage=c.stage, status=c.status, latency_s=c.latency_s) for c in res.calls],
                               n_people=None if res.people is None else len(res.people), wall_s=wall)
                    log.write(json.dumps(row, default=str) + '\n')
                    log.flush()
                    print(f'{n_calls:>5} {model} {arm} {rec.scene_id} {rec.episode_id} t{trial} {res.status} '
                          f'{None if sc_ok is None else sc_ok["success"]}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
