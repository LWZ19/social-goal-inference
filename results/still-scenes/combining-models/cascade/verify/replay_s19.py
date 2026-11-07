import sys
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/rerun/cascade/verify')
import verify as V
G = V.load('s19','glm53'); Q = V.load('s19','qwen38_low')
V.analyse(G, Q, 'K4 replay: s19 glm53 -> s19 qwen38_low', do_replay=True)
open('results/2026-10-17-beyond-vlm/rerun/cascade/verify/replay_s19_out.txt','w').write('\n'.join(V.OUT)+'\n')
