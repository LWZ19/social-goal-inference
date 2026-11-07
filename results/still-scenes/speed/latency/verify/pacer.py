"""Split record latency_s - sum(call latency_s) into pacer/retry time (latency_total_s - latency_s) and time outside calls."""
import sys, statistics as st
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
for stage, model in (("s19", "qwen38_low"), ("s19", "glm53"), ("s21", "qwen38_low")):
    pac, out, n = [], [], 0
    for ref in H.runs(stage=stage, model=model, method="con"):
        if ref.arm not in ("real", "staffed", "plain"): continue
        for rec in H.records(ref):
            calls = rec.result.get("calls") or []
            if not any(c["stage"] == "constraints" for c in calls): continue
            n += 1
            lt = sum(c.get("latency_total_s") or 0 for c in calls); ls = sum(c.get("latency_s") or 0 for c in calls)
            pac.append(lt - ls); out.append(rec.latency_s - lt)
    print(f"{stage} {model} goal-step n={n}: pacer+retry (total-latency) mean {st.mean(pac):.2f} median {st.median(pac):.3f}; "
          f"outside calls mean {st.mean(out):.2f} median {st.median(out):.2f}")
