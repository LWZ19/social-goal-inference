import json,sys,statistics as st,collections
for d in sys.argv[1:]:
    recs=[json.loads(l) for l in open(f"runs/{d}/results.jsonl")]
    ser=[];cp=[];cp_share=[];ser_share=[];n=0;cachedpeople=0
    seen=set()
    # order by file order
    for r in recs:
        res=r.get("result") or {}
        calls=res.get("calls") or []
        lat=collections.defaultdict(float)
        for c in calls: lat[c["stage"]]+=c.get("latency_s") or 0
        if "constraints" not in lat or "people" not in lat: continue
        n+=1
        planlat=0
        # plan call stored where?
        pc=r.get("plan_call") or None
        plan=lat.get("plan",0)
        pre=plan+lat.get("locate",0)+lat.get("context",0)
        s=pre+lat["people"]+lat["constraints"]
        c=max(pre,lat["people"])+lat["constraints"]
        ser.append(s);cp.append(c)
        key=(r["scene_id"],r["trial"])
        if key in seen:  # people shared: free in a live run
            ser_share.append(pre+lat["constraints"]);cp_share.append(pre+lat["constraints"])
        else:
            seen.add(key);ser_share.append(s);cp_share.append(c)
    if n==0: print(d,"none");continue
    m=lambda x:st.mean(x)
    print(f"{d[16:]:40s} n={n} serial {m(ser):.1f} cp {m(cp):.1f} ({100*(1-m(cp)/m(ser)):.0f}%) | per-scene-shared serial {m(ser_share):.1f} cp {m(cp_share):.1f} ({100*(1-m(cp_share)/m(ser_share)):.0f}%) stages={sorted(set(c['stage'] for r in recs for c in (r.get('result') or {}).get('calls') or []))}")
