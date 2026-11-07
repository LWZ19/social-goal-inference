import json,glob,collections,sys
def latest(pat):
    c=[d for d in sorted(glob.glob('runs/'+pat)) if glob.glob(d+'/results.jsonl')]
    return c[-1]
for model,pats in [('glm',['*_s19_%s_glm53_con']),('ds',['*_s20_%s_dsv41_con']),('qwen',['*_s19_%s_qwen38_low_con'])]:
    eps=collections.defaultdict(list)
    for arm in ['real','staffed','plain']:
        d=latest(pats[0]%arm)
        for l in open(d+'/results.jsonl'):
            r=json.loads(l)
            s=r['score'];res=r['result']
            if s['success']: why='ok'
            elif not s['target_correct']: why='target'
            elif res.get('source')!='constraints': why='nonC:'+str(res.get('source'))
            else: why='goalC'
            eps[(arm,r['scene_id'],r['episode_id'])].append((why,r['labels']['situation'],res.get('people') is not None))
    cnt=collections.Counter(); mixed=collections.Counter(); sit=collections.Counter()
    per_trial=sum(w=='ok' for v in eps.values() for w,_,_ in v)
    for k,v in eps.items():
        n=sum(w=='ok' for w,_,_ in v)
        cnt[n]+=1
        if 0<n<3:
            for w,s,_ in v:
                if w!='ok': mixed[w]+=1
            sit[v[0][1]]+=1
        if n==0:
            for w,s,_ in v: mixed['ALLFAIL_'+w]+=1
    allfail=collections.Counter(w for v in eps.values() for w,_,_ in v if w!='ok')
    print(model,len(eps),'eps; succ-count hist',dict(cnt),'per-trial succ',per_trial,'/',3*len(eps))
    print('  fail causes all',dict(allfail))
    print('  fail causes in mixed/allfail eps',dict(mixed))
    print('  mixed eps by situation',dict(sit))
