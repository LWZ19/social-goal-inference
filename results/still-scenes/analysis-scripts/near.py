import json,glob,collections,statistics as st
def latest(pat):
    c=[d for d in sorted(glob.glob(pat)) if glob.glob(d+'/results.jsonl')]
    return c[-1] if c else None
first=None
for model,pfx in [('qwen38_low','s16b'),('glm53','s16b'),('dsv41','s20')]:
  for arm_m in ['cot','fsc']:
    rows=[]
    for arm in ['real','staffed','plain']:
        d=latest(f'runs/*_{pfx}_{arm}_{model}_{arm_m}')
        if not d: continue
        for l in open(d+'/results.jsonl'):
            r=json.loads(l)
            if first is None: first=r['labels']; print('labels',first)
            rows.append(r)
    by=collections.defaultdict(list)
    for r in rows:
        s=r['score']; sit=r['labels'].get('situation') or r['labels'].get('kind')
        if s.get('success') or not s.get('target_correct'): continue
        if r['result'].get('status')!='ok' and s.get('distance_to_region_m') is None: continue
        dd=s.get('distance_to_region_m')
        if dd is None: continue
        by[sit].append((dd,s.get('min_body_gap_m'),s.get('in_region'),s.get('reachable'),s.get('heading_ok')))
    for sit,v in by.items():
        ds=[x[0] for x in v]
        print(model,arm_m,sit,'n',len(v),'med %.2f'%st.median(ds),'<=0.15:',sum(x<=0.15 for x in ds),'<=0.3:',sum(x<=0.3 for x in ds),'<=0.5:',sum(x<=0.5 for x in ds),'inreg_fail',sum(1 for x in v if x[2]))
print('---- split near misses (<=0.3) by reachable/body gap')
for model,pfx in [('qwen38_low','s16b'),('glm53','s16b'),('dsv41','s20')]:
  for arm_m in ['fsc']:
    c=collections.Counter()
    for arm in ['real','staffed','plain']:
        d=latest(f'runs/*_{pfx}_{arm}_{model}_{arm_m}')
        for l in open(d+'/results.jsonl'):
            r=json.loads(l); s=r['score']; sit=r['labels']['situation']
            if sit=='no_one' or s.get('success') or not s.get('target_correct'): continue
            dd=s.get('distance_to_region_m')
            if dd is None: continue
            g=s.get('min_body_gap_m')
            c[(sit, 'near' if dd<=0.3 else 'far', 'unreach' if not s.get('reachable') else 'reach', 'gap<0.3' if (g is not None and g<0.3) else 'gap>=0.3')]+=1
    for k,v in sorted(c.items()): print(model,k,v)
