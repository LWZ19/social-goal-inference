import json,glob,itertools,random,re,statistics as st
R='runs/'
M={'qwen':('s16b_{s}_qwen38_low_fsc','s19_{s}_qwen38_low_con'),
   'glm':('s16b_{s}_glm53_fsc','s19_{s}_glm53_con'),
   'ds':('s20_{s}_dsv41_fsc','s20_{s}_dsv41_con')}
def load(pat):
    fs=sorted(glob.glob(R+'*_'+pat))
    if not fs: return None
    f=fs[-1]  # latest
    return [json.loads(l) for l in open(f+'/results.jsonl')]
random.seed(0)
for m,(pe,pc) in M.items():
    pairs=[]
    for s in ('real','staffed','plain'):
        E=load(pe.format(s=s)); C=load(pc.format(s=s))
        ke={(r['scene_id'],r['episode_id'],r['trial']):r for r in E}
        for r in C:
            k=(r['scene_id'],r['episode_id'],r['trial'])
            if k in ke:
                lay=re.sub(r'^[a-z]+_','',r['scene_id'])
                pairs.append((lay,r['labels']['situation'],int(r['score']['success']),int(ke[k]['score']['success'])))
    lays=sorted(set(p[0] for p in pairs))
    d={l:[p[2]-p[3] for p in pairs if p[0]==l] for l in lays}
    dm={l:sum(v)/len(v) for l,v in d.items()}
    obs=sum(dm.values())
    nz=[v for v in dm.values() if v!=0]
    cnt=0;tot=0
    for signs in itertools.product([1,-1],repeat=len(nz)):
        tot+=1
        if abs(sum(s*v for s,v in zip(signs,nz)))>=abs(obs)-1e-12: cnt+=1
    # ICC of C success by layout (one-way ANOVA)
    def icc(idx):
        g={l:[p[idx] for p in pairs if p[0]==l] for l in lays}
        k=st.mean(len(v) for v in g.values()); gm=st.mean(x for v in g.values() for x in v)
        msb=sum(len(v)*(st.mean(v)-gm)**2 for v in g.values())/(len(g)-1)
        msw=sum((x-st.mean(v))**2 for v in g.values() for x in v)/(sum(len(v) for v in g.values())-len(g))
        return (msb-msw)/(msb+(k-1)*msw) if msb+(k-1)*msw>0 else float('nan')
    sit={l:[p[1] for p in pairs if p[0]==l][0] for l in lays}
    print(m,'n',len(pairs),'layouts',len(lays),'C',sum(p[2] for p in pairs),'E',sum(p[3] for p in pairs),
          'nonzero layouts',len(nz),'p_clustered',round(cnt/tot,5),'ICC_C',round(icc(2),3),'ICC_E',round(icc(3),3),
          'ICC_diff', round(icc_d if (icc_d:=0) else 0,3))
    print('  layout mean diffs by situation:',{sit[l]:[] for l in lays} and sorted((sit[l],round(dm[l],2)) for l in lays))
    # power by resampling layouts (bootstrap) for n fresh layouts, sign-flip test (MC)
    for n in (8,14,20,30):
        pw=0;B=300
        for _ in range(B):
            smp=[dm[random.choice(lays)] for _ in range(n)]
            o=abs(sum(smp)); c=0
            for _ in range(400):
                if abs(sum(v*random.choice((1,-1)) for v in smp))>=o-1e-12: c+=1
            pw+= (c/400)<0.05
        print('   n_layouts',n,'power(bootstrap of observed effect)',round(pw/B,2))
