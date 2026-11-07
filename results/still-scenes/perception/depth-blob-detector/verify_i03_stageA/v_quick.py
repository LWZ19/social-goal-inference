import sys, time
sys.path.insert(0, 'results/still-scenes/perception/depth-blob-detector/verify_i03_stageA')
import v_fit as F
from sgi.config import load_config
th = load_config().thresholds
sc = F.fit_scenes()
for i in (0, 10, 25, 100):
    dom, st, ld = sc[i]
    t = time.time()
    obs, truth = F.load(ld)
    res, nv, npl = F.run_scene(obs, truth, th)
    print(i, dom, st, ld[2], nv, npl, round(time.time()-t, 2), res['orig'][142], res['fix'][142])
