"""POST HOC (written after tables.txt was read; decides nothing): for waiting_line units, read off each raster the
angle between the seen-face direction (the channel-6 cell farthest from the origin) and the direction from the target
point to the nearest people-disc cell, and that cell's distance; validation seeds (generator, n 140) against
development records in setting (b) (n 261). Rasters only; labels used only to select the waiting_line rows."""
import json
import math
from pathlib import Path
import numpy as np
B = Path(__file__).resolve().parents[1]
OFFS = (np.arange(64) - 31.5) * 0.1
XC, YC = np.meshgrid(OFFS, OFFS)


def geom(r):
    f = r[5] > 0.5
    p = r[1] > 0.5
    if not f.any() or not p.any():
        return None
    fd = np.hypot(XC[f], YC[f])
    j = int(np.argmax(fd))
    fa = math.atan2(YC[f][j], XC[f][j])
    pd = np.hypot(XC[p], YC[p])
    k = int(np.argmin(pd))
    pa = math.atan2(YC[p][k], XC[p][k])
    return abs((pa - fa + math.pi) % (2 * math.pi) - math.pi) * 180 / math.pi, float(pd[k])


out = []
va = np.load(B / 'data' / 'val.npz')
dv = np.load(B / 'data' / 'dev.npz')
for name, R, S in (('validation seeds (generator)', va['rasters'], va['situation']),
                   ('development records, setting b', dv['raster_b'], dv['situation'])):
    g = [geom(R[i]) for i in np.flatnonzero(S == 'waiting_line')]
    miss = sum(x is None for x in g)
    g = np.array([x for x in g if x is not None])
    out.append(f"{name}: n {len(g)} (+{miss} without seen face or people); angle face-to-nearest-person deg: median "
               f"{np.median(g[:, 0]):.1f}, IQR {np.percentile(g[:, 0], 25):.1f}-{np.percentile(g[:, 0], 75):.1f}, "
               f"share <= 30 deg {np.mean(g[:, 0] <= 30):.2f}; nearest person-disc cell distance m: median "
               f"{np.median(g[:, 1]):.2f}, IQR {np.percentile(g[:, 1], 25):.2f}-{np.percentile(g[:, 1], 75):.2f}")
txt = 'POST HOC, decides nothing.\n' + '\n'.join(out) + '\n'
(B / 'posthoc' / 'line_geometry.txt').write_text(txt)
print(txt)
