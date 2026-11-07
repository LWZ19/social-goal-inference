import json, collections
D='results/2026-10-17-beyond-vlm/phase1/i01_calibration/verify/'
rows=[json.loads(l) for l in open(D+'v1_rows.jsonl')]
by=collections.defaultdict(list)
for r in rows:
    if r['step_o'] and r['target_xy_o'] is not None:
        by[(r['model'],r['scene'],r['episode'])].append(r)
pa=ag=0
for k,l in by.items():
    for i in range(len(l)):
        for j in range(i+1,len(l)):
            if l[i]['target_xy_o']==l[j]['target_xy_o']:
                pa+=1; ag+= l[i]['res_o']==l[j]['res_o']
out=f'identical target world point pairs {pa}, agree {ag}'
open(D+'v5_ceiling_exact.txt','w').write(out+'\n'); print(out)
