"""Item 10 amendment 3, reconstructed from the caches (zero calls): the rolling median latency_total_s of a lane's own
calls, window 30, at every call from the 30th on. While a lane ran it was the model's only caller (amendment 2), so the
model's calls in that time are the lane's. Usage: gateway_check_series.py MODEL_ID START END THRESHOLD"""
import json, os, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
CACHE = paths.REPO / ".cache"                # the model-call cache (not in this repository)
mid, a, b, thr = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
t = lambda s: time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))
lo, hi = t(a), t(b); calls = []
for root in [d for d in os.listdir(CACHE) if d.startswith("vlm")]:
    for e in os.scandir(CACHE / root):
        m = e.stat().st_mtime
        if not lo <= m <= hi: continue
        try: d = json.load(open(e.path))
        except Exception: continue
        if d.get("model") == mid and d.get("cached") is False and d.get("status") == "ok":
            calls.append((m, float(d.get("latency_total_s") or d.get("latency_s") or 0)))
calls.sort()
print(f"# {mid} calls {a} to {b}: {len(calls)}; threshold {thr} s; rolling median of the last 30 at each call from the 30th")
worst = (0, None)
for i in range(29, len(calls)):
    med = statistics.median(c[1] for c in calls[i - 29:i + 1])
    if med > worst[0]: worst = (med, calls[i][0])
    if (i - 29) % 10 == 0 or med > thr:
        print(f"{time.strftime('%H:%M:%S', time.localtime(calls[i][0]))} call {i + 1}: rolling median {med:.1f} s{'  OVER' if med > thr else ''}")
print(f"# maximum rolling median {worst[0]:.1f} s at {time.strftime('%H:%M:%S', time.localtime(worst[1])) if worst[1] else '-'}; "
      f"{'never over the threshold' if worst[0] <= thr else 'OVER the threshold'}")
