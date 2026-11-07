"""Item 9 amendment 2: list the arms whose first answer met an HTTP 429 (zero calls). Each 429 line in a service log
moving/runs/<model>/serve_w<w>_<C|R>.log goes to the latest episode on worker <w> using that service (C: C1/CT/CT2,
R: R1) that started at or before it; a service answers one request at a time, so that episode's first request was
the one in flight. Episode start = end - duration, from the run logs. Cross-check: cache entries of the model with
transient_retries > 0. Usage: map_retries.py MODEL_DIR ENTRY_MODEL_ID LOG_GLOB"""
import glob, json, os, re, sys, datetime as dt
mdir, model_id, log_glob = sys.argv[1:4]
base = f"moving/runs/{mdir}"
day = "2026-11-06"
t = lambda s: dt.datetime.strptime(f"{day} {s}", "%Y-%m-%d %H:%M:%S")
eps = []                                                    # (worker, kind, start, end, scenario, seed, arm, logfile)
for f in sorted(glob.glob(log_glob)):
    for line in open(f):
        m = re.match(r"(\d\d:\d\d:\d\d) w(\d) (\S+) seed(\d+) (\S+): exit (\S+) in (\d+) s", line)
        if m:
            end = t(m[1]); start = end - dt.timedelta(seconds=int(m[7]))
            eps.append((int(m[2]), "R" if m[5].startswith("R") else "C", start, end, m[3], int(m[4]), m[5], os.path.basename(f), m[6]))
hits = []
for f in glob.glob(f"{base}/serve_w*_*.log"):
    w, kind = re.search(r"serve_w(\d)_([CR])\.log", f).groups()
    for line in open(f):
        if " 429 " in line:
            m = re.match(r"(\d\d:\d\d:\d\d)", line)
            if not m: continue
            T = t(m[1]); cands = [e for e in eps if e[0] == int(w) and e[1] == kind and e[2] <= T and e[8] != "timeout"]   # a timed-out episode was blocked before sending
            e = max(cands, key=lambda e: e[2]) if cands else None
            hits.append((m[1], w, kind, e))
final = {}
for e in eps:                                               # the episode kept for each cell is the one with episode.json
    final.setdefault((e[4], e[5], e[6]), []).append(e)
out = ["# arms whose first answer met an HTTP 429 (map_retries.py); <run dir> <scenario> <seed> <arm>"]
for T, w, kind, e in sorted(hits):
    desc = "no episode" if e is None else f"{e[4]} seed{e[5]} {e[6]} ({e[7]}, exit {e[8]}, {e[2]:%H:%M:%S}-{e[3]:%H:%M:%S})"
    kept = e is not None and os.path.exists(f"{base}/{e[4]}/seed{e[5]}/{e[6]}/episode.json") and e[8] == "0"
    print(f"429 at {T} w{w} {kind}: {desc}{'' if kept else '  [not a kept episode]'}")
    if kept and e[6] in ("CT2", "R1"):
        line = f"{mdir} {e[4]} {e[5]} {e[6]}"
        if line not in out: out.append(line)
print(f"retried CT2/R1 arms: {len(out) - 1}")
open(f"fill/i9_qwen_glm/retried_{mdir}.txt", "w").write("\n".join(out) + "\n")
