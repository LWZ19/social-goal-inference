"""E2, POST HOC (after e2_score_all_out.txt and e2_excluded_out.txt): pooled H2 with GLM's left-out units at their
worst case for D (each one -1 on its layout's net). Zero calls; reported, not a bar.
    PYTHONPATH= .venv/bin/python e2_pooled_worst.py runs/e2_soclaas_qwen38_low \
        runs/e2_soclaas_dsv41 runs/e2_soclaas_glm53
"""
import sys
from collections import defaultdict

from e2_score import FREE, load, ok, signflip_p

rows = load(sys.argv[1:])
net = defaultdict(int)
for r in rows:
    if r['model'] not in FREE:
        continue
    if r['vlm_error']:
        net[r['scene']] -= 1
    elif r['people_matter']:
        net[r['scene']] += int(ok(r['arms']['D'])) - int(ok(r['arms']['A0']))
print(f"pooled H2, left-out units at worst case for D: net {sum(net.values()):+d}, layout sign-flip p "
      f"{signflip_p(list(net.values())):.4f} over {sum(1 for v in net.values() if v)} layouts")
