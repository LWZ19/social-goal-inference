"""Independent recomputation for i01_baselines: per C record, recorded C, SA (solver, recorded people, no
constraints), CA (solver, only come_after kept), SG (people-blind standard goal). Writes one jsonl row per record."""
import sys, json, time
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
from sgi.stages.goal import standard_goal
from sgi.types import GroundedTarget

OUT = 'results/still-scenes/replay-harness/baselines/verify/'
which = sys.argv[1]
ARMS = ("real", "staffed", "plain")
if which == "new":
    sel = [("s21", "qwen38_low")] + [("s23", m) for m in ("glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low")]
else:
    sel = [("s19", "qwen38_low"), ("s19", "glm53"), ("s20", "dsv41")]
rows = []
t0 = time.time()
for stage, model in sel:
    for arm in ARMS:
        runs = H.runs(stage=stage, arm=arm, model=model, method="con")
        assert len(runs) == 1, (stage, arm, model, runs)
        run = runs[0]
        th = H.recorded_config(run)[1].thresholds
        for rec in H.records(run):
            r = rec.result
            row = dict(stage=stage, model=model, arm=arm, scene=rec.scene_id, ep=rec.episode_id, trial=rec.trial,
                       sit=rec.situation, status=r.get("status"), source=r.get("source"),
                       has_target=rec.target is not None, people=None if rec.people is None else len(rec.people),
                       ncons=None if rec.constraints is None else len(rec.constraints),
                       kinds=None if rec.constraints is None else [u["constraint"]["kind"] for u in rec.constraints],
                       solver_step=rec.solver_step, C=bool(rec.score["success"]),
                       C_now=bool(H.rescore(rec).success))
            # SG: people-blind standard goal
            if rec.target is not None:
                sc = H.scene(run, rec.scene_id)
                g = standard_goal(sc.observation, GroundedTarget.model_validate(rec.target), th)
                row["SG"] = bool(H.score_pose(rec, tuple(g.pose_world)).success)
                if r.get("source") == "standard" and rec.pose_world is not None:
                    import math
                    row["SG_moved"] = math.dist(g.pose_world[:2], rec.pose_world[:2])
            else:
                row["SG"] = None
            # SA and CA
            if rec.target is not None and rec.people is not None:
                row["SA"] = bool(H.resolve(rec, constraints=[]).score.success)
                ca = [u["constraint"] for u in (rec.constraints or []) if u["constraint"]["kind"] == "come_after"]
                row["n_ca"] = len(ca)
                row["CA"] = bool(H.resolve(rec, constraints=ca).score.success)
                if rec.constraints is not None:
                    full = H.resolve(rec)
                    row["C_resolve"] = bool(full.score.success)
                    row["C_moved"] = full.moved_m
            else:
                row["SA"] = row["CA"] = None
            rows.append(row)
    print(stage, model, len(rows), round(time.time() - t0, 1), flush=True)
with open(OUT + f"v1_rows_{which}.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
print("done", len(rows), round(time.time() - t0, 1))
