import sys, statistics
from collections import Counter
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "round3"))
from i_followups import load, first_service_t, in_line, ARMS
sel, eps = load("runs_pilot3/dsv41")
print("| Arm | joins | leaves | leaves per join | samples joined (after 1st service) | strict in line when joined | when not joined | median time joined before leaving (s) |")
print("|---|---|---|---|---|---|---|---|")
for arm in ARMS:
    j = l = 0; nj = n = inj = innj = 0; stays = []
    for lay in sel:
        e = eps.get((lay["name"], arm))
        if not e: continue
        t_join = None
        for ev in e.get("joins", []):
            if ev["event"] == "joined": j += 1; t_join = ev["t"]
            elif ev["event"] == "left":
                l += 1
                if t_join is not None: stays.append(ev["t"] - t_join); t_join = None
        t0 = first_service_t(e)
        if t0 is None: continue
        for s in e["samples"]:
            if s["t"] < t0 - 1e-9: continue
            n += 1
            if s["joined"]: nj += 1; inj += in_line(s, "strict")
            else: innj += in_line(s, "strict")
    print(f"| {arm} | {j} | {l} | {l/max(j,1):.2f} | {nj/max(n,1):.3f} | {inj/max(nj,1):.3f} | {innj/max(n-nj,1):.3f} | "
          f"{statistics.median(stays) if stays else float('nan'):.1f} |")
