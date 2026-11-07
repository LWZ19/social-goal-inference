import json,glob,collections
def latest(pat):
    return [d for d in sorted(glob.glob('runs/'+pat)) if glob.glob(d+'/results.jsonl')][-1]
S={}
for model,pat in [('glm','*_s19_%s_glm53_con'),('ds','*_s20_%s_dsv41_con')]:
    for arm in ['real','staffed','plain']:
        for l in open(latest(pat%arm)+'/results.jsonl'):
            r=json.loads(l); S.setdefault((arm,r['scene_id'],r['episode_id']),[]).append(r['score']['success'])
print('eps',len(S),'any-of-6',sum(any(v) for v in S.values()),'>=4 of 6',sum(sum(v)>=4 for v in S.values()),'>=3 of 6',sum(sum(v)>=3 for v in S.values()),'mean per-call',sum(map(sum,S.values()))/6)
for seedpat,name in [('*_s19_dev_seed%d_glm53_con','glm'),('*_s20_dev_seed%d_dsv41_con','ds')]:
    E=collections.defaultdict(list)
    for s in range(3):
        for l in open(latest(seedpat%s)+'/results.jsonl'):
            r=json.loads(l); E[(s,r['scene_id'],r['episode_id'])].append(r['score']['success'])
    print(name,'dev watching eps',len(E),'trials',sum(map(len,E.values())),'succ',sum(map(sum,E.values())),'any',sum(any(v) for v in E.values()),'all',sum(all(v) for v in E.values()),'maj',sum(sum(v)*2>len(v) for v in E.values()))
