"""n12 check: recorded score fields vs the current package scorer (H.rescore), on a stated subset."""
import sys, time, io
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
OUT = 'results/2026-10-17-beyond-vlm/rerun/i01_split/verify/v_rescore_out.txt'
buf = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); buf.write(s + '\n')
FIELDS = ('success', 'target_correct', 'in_region', 'reachable', 'heading_ok')
SETS = [(st, model) for st, model in (('s23', 'gemini38_low'), ('s21', 'qwen38_low'), ('s23', 'sonnet55_low'), ('s23', 'haiku55'), ('s23', 'glm53'), ('s23', 'dsv41'))]
if len(sys.argv) > 1:
    SETS = SETS[:int(sys.argv[1])]
total = mism = 0
t0 = time.time()
for st, model in SETS:
    for meth in ('cot', 'fsc', 'con'):
        for arm in ('real', 'staffed', 'plain'):
            (run,) = H.runs(stage=st, arm=arm, model=model, method=meth)
            for r in H.records(run):
                s = H.rescore(r)
                sd = s.model_dump() if hasattr(s, 'model_dump') else dict(s)
                bad = [f for f in FIELDS if bool(sd.get(f)) != bool(r.score.get(f))]
                total += 1
                if bad:
                    mism += 1
                    p('MISMATCH', run.name, r.scene_id, r.episode_id, r.trial, bad)
    p(f'{st} {model}: cumulative records {total}, mismatches {mism}, {time.time()-t0:.0f}s')
open(OUT, 'w').write(buf.getvalue())
