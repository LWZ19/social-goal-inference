import json,glob,collections
def latest(pat):
    return [d for d in sorted(glob.glob('runs/'+pat)) if glob.glob(d+'/results.jsonl')][-1]
for model,pat in [('glm','*_s19_%s_glm53_con'),('ds','*_s20_%s_dsv41_con')]:
    eps=collections.defaultdict(list)
    for arm in ['real','staffed','plain']:
        for l in open(latest(pat%arm)+'/results.jsonl'):
            r=json.loads(l); eps[(arm,r['scene_id'],r['episode_id'])].append(r)
    mixed=0; allfailC=0; pc_diff_mixed=0; kinds_diff_mixed=0; sits=collections.Counter()
    for v in eps.values():
        cs=[x for x in v if x['result'].get('source')=='constraints' and x['score']['target_correct']]
        if len(cs)<2: continue
        s=[x['score']['success'] for x in cs]
        if any(s) and not all(s):
            mixed+=1; sits[v[0]['labels']['situation']]+=1
            def plen(p):
                if p is None: return None
                return len(p) if isinstance(p,list) else len(p.get('people',p))
            pcs=[plen(x['result'].get('people')) for x in cs]
            if len(set(pcs))>1: pc_diff_mixed+=1
            def ck(c):
                if c is None: return None
                L=c if isinstance(c,list) else c.get('constraints',[])
                return tuple(sorted(str(y.get('kind')) for y in L if isinstance(y,dict)))
            if len(set(ck(x['result'].get('constraints')) for x in cs))>1: kinds_diff_mixed+=1
        elif not any(s): allfailC+=1
    print(model,'C-eligible eps with mixed success',mixed,dict(sits),'| all C trials fail',allfailC,'| mixed eps where people count differs',pc_diff_mixed,'| constraint-kind multiset differs',kinds_diff_mixed)
r=json.loads(open(latest('*_s19_real_glm53_con')+'/results.jsonl').readline())
for x in [json.loads(l) for l in open(latest('*_s19_real_glm53_con')+'/results.jsonl')]:
    if x['result'].get('people'):
        print(type(x['result']['people']).__name__, str(x['result']['people'])[:300]); print(str(x['result']['constraints'])[:400]); print(str(x['result']['candidates'])[:300]); print(str(x['result']['solver'])[:400]); break
