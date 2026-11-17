import json,sys,glob,statistics as st,collections
pats=sys.argv[1:]
for p in pats:
  for d in sorted(glob.glob('runs/*'+p+'*')):
    if any(x in d for x in ['_s18_','_s20_held','_s21_']): continue
    recs=[json.loads(l) for l in open(d+'/results.jsonl')]
    succ=sum(1 for r in recs if r.get('score',{}).get('success') or r.get('metrics',{}).get('success'))
    by=collections.defaultdict(list); tps=[]
    for r in recs:
      for c in (r.get('result') or {}).get('calls',[]):
        u=c.get('usage') or {}
        by[c['stage']].append((c['latency_s'],u.get('output_tokens') or 0))
        if c['latency_s'] and u.get('output_tokens'): tps.append(u['output_tokens']/c['latency_s'])
    s=' '.join(f"{k}:{st.median([a for a,b in v]):.1f}s/{st.median([b for a,b in v]):.0f}t" for k,v in by.items())
    print(d.split('/')[-1], len(recs), 'succ',succ, 'tok/s med %.0f'%(st.median(tps) if tps else 0), s)
