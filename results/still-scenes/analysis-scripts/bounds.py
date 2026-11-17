import json,glob,collections,math
def latest(pat):
    return [d for d in sorted(glob.glob('runs/'+pat)) if glob.glob(d+'/results.jsonl')][-1]
for model,pat in [('glm','*_s19_%s_glm53_con'),('ds','*_s20_%s_dsv41_con')]:
    eps=collections.defaultdict(list)
    for arm in ['real','staffed','plain']:
        for l in open(latest(pat%arm)+'/results.jsonl'):
            r=json.loads(l); eps[(arm,r['scene_id'],r['episode_id'])].append(r)
    maj=sum(sum(x['score']['success'] for x in v)>=2 for v in eps.values())
    tgtmaj_any=0; goalC_fixable=0; npeople=[]; spread=[]
    for v in eps.values():
        tc=sum(x['score']['target_correct'] for x in v)
        anys=any(x['score']['success'] for x in v)
        if tc>=2 and anys: tgtmaj_any+=1
        # episodes with >=2 C goal-step trials, correct target
        cs=[x for x in v if x['result'].get('source')=='constraints' and x['score']['target_correct']]
        if len(cs)>=2 and any(x['score']['success'] for x in cs): goalC_fixable+=1
        ps=[x['result'].get('people') for x in cs]
        if len(cs)==3 and all(p is not None for p in ps):
            ns=[len(p) if isinstance(p,list) else len(p.get('people',[])) for p in ps]
            npeople.append(ns)
        # spread of final positions among trials
        pw=[x['result'].get('pose_world') for x in v if x['result'].get('pose_world')]
        if len(pw)==3:
            d=max(math.dist(a[:2],b[:2]) for a in pw for b in pw); spread.append(d)
    print(model,'majority-of-3 eps',maj,'| tgt-majority & any-success',tgtmaj_any,'| >=2 C trials w/ correct tgt & one success',goalC_fixable)
    disagree=sum(len(set(n))>1 for n in npeople)
    print('  episodes with 3 C trials:',len(npeople),'people-count differs across trials:',disagree)
    spread.sort(); print('  max pairwise final-position spread median %.2f m, share>0.5m %.2f'%(spread[len(spread)//2],sum(s>0.5 for s in spread)/len(spread)))
