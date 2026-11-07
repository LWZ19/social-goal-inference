import json,sys,statistics as st,collections
for d in sys.argv[1:]:
    recs=[json.loads(l) for l in open(f"runs/{d}/results.jsonl")]
    A=collections.defaultdict(list)
    for r in recs:
        calls=(r.get("result") or {}).get("calls") or []
        L=collections.defaultdict(float)
        for c in calls: L[c["stage"]]+=c.get("latency_s") or 0
        if "constraints" not in L: continue
        p,x,pe,k=L["plan"],L["context"],L["people"],L["constraints"]
        A["serial"].append(p+x+pe+k); A["A1"].append(max(p+x,pe)+k)
        A["A2"].append(max(p,pe)+max(x,k)); A["A2_fastpeople"].append(p+max(x,k))
        A["provider_floor_k"].append(k)
    q=lambda v:sorted(v)[int(0.9*len(v))]
    print(d[16:], " ".join(f"{n}:{st.mean(v):.1f}/{st.median(v):.1f}/p90 {q(v):.1f}" for n,v in A.items()))
