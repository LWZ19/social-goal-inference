"""The moving headline (C1, R1, CT, CT2) on Qwen 3.8 and GLM-5.3, seeds 10-14, beside every other
model's same 25 cells. Zero calls; read-only.

    .venv/bin/python results/moving-lines/track-vs-decide-once/track_score.py          (92 s, the pass mark)
    .venv/bin/python results/moving-lines/track-vs-decide-once/track_score.py --long   (240 s, secondary)
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import mcompare as C  # noqa: E402

JUDGED = {"Qwen 3.8": "soclaas_qwen38_low", "GLM-5.3": "soclaas_glm53"}
OTHERS = {"Sonnet 5.5": "sonnet55_low", "DeepSeek V4.1": "dsv41", "Haiku 5.5": "claude_haiku55",
          "Gemini 3.8": "gemini_fast", "Sonnet 4.6": "claude_sonnet46_low", "Sonnet 5": "claude_sonnet5_low",
          "Haiku 4.5": "claude_haiku45", "Gemini 3 Flash": "gemini3_flash", "Gemini 3.1 Flash Lite": "gemini31_flash_lite",
          "Gemini 3.5 Flash Lite": "gemini35_flash_lite", "Gemini 3.6 Flash": "gemini36_flash"}
SEEDS = range(10, 15)
# Arms that timed out before t=0 (blocked behind the previous episode's still-running request on the single-threaded
# service) are left out of the pass mark's pairs and of the latency label, and counted.
TIMEOUTS = Path(__file__).resolve().parent / "timeouts.txt"


def timeouts(d, arm):
    if not TIMEOUTS.exists():
        return 0
    rows = [l.split() for l in TIMEOUTS.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return sum(1 for r in rows if len(r) == 4 and r[0] == d and r[3] == arm and int(r[2]) in SEEDS)


# Arms whose first answer had transient_retries > 0 (HTTP 429 responses) are listed in retried.txt; the pass mark and
# the latency label are also reported without the cells where CT2 or R1 is listed (a sensitivity check).
RETRIED = Path(__file__).resolve().parent / "retried.txt"


def retried(d):
    if not RETRIED.exists():
        return set()
    rows = [l.split() for l in RETRIED.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return {(r[1], int(r[2])) for r in rows if len(r) == 4 and r[0] == d and r[3] in ("CT2", "R1")}


def latency(eps, arm, seconds):
    rows = [e for (_, _, a), e in eps.items() if a == arm and e is not None]
    lat = [e["answers"][0].get("latency_s") for e in rows if e.get("answers") and e["answers"][0].get("status") == "ok"]
    late = sum(1 for e in rows if not e.get("answers") or e["answers"][0].get("status") != "ok"
               or (e["answers"][0].get("latency_s") or 0) + 2.0 > seconds)
    err = sum(1 for e in rows if C.errors(e))
    med = statistics.median(lat) if lat else float("nan")
    return len(rows), med, late, err


def main():
    global RETRIED
    if "--retried" in sys.argv:                      # which retry list the sensitivity reads
        RETRIED = Path(sys.argv[sys.argv.index("--retried") + 1])
    long = "--long" in sys.argv
    base, seconds = ("runs_i9long", 240.0) if long else ("runs", 92.0)
    print(f"# The moving headline, seeds 10-14, {seconds:.0f} s episodes{' (SECONDARY, cache-only)' if long else ''}\n")
    print("| model | R1 / CT / CT2 strict | R1 / CT / CT2 primary | first answer median s, C / R | C episodes with no answer "
          "in time (or an error) | CT2 vs R1 strict W/L/T, p | CT vs R1 strict W/L/T, p | CT2 vs R1 primary W/L/T, p |")
    print("|---|---|---|---|---|---|---|---|")
    verdicts = []
    for name, d in {**JUDGED, **({} if long else OTHERS)}.items():
        eps = C.load(f"{base}/{d}", C.EVAL5, SEEDS, ("C1", "R1", "CT", "CT2"))
        if not any(e is not None for (_, _, a), e in eps.items() if a == "CT2"):
            print(f"| {name} | not run | | | | | | |")
            continue
        m = {a: C.means(eps, a) for a in ("R1", "CT", "CT2")}
        nc, mc, late, errc = latency(eps, "CT2", seconds)
        tout = 0 if long else timeouts(d, "CT2")         # counted, not added to the label
        _, mr, _, _ = latency(eps, "R1", seconds)
        r2, r1 = C.row("CT2 vs R1", C.pairs(eps, "CT2", "R1")), C.row("CT vs R1", C.pairs(eps, "CT", "R1"))
        print(f"| {name} | {m['R1'][1]:.2f} / {m['CT'][1]:.2f} / {m['CT2'][1]:.2f} | {m['R1'][2]:.2f} / {m['CT'][2]:.2f} / "
              f"{m['CT2'][2]:.2f} | {mc:.1f} / {mr:.1f} | {late} of {nc} (errors {errc}; timed out, excluded: {tout}) | "
              f"{'/'.join(map(str, r2['strict']))}, {r2['p_strict']:.4f} | {'/'.join(map(str, r1['strict']))}, "
              f"{r1['p_strict']:.4f} | {'/'.join(map(str, r2['frozen']))}, {r2['p_frozen']:.4f} |")
        if name in JUDGED and not long and retried(d):
            skip = retried(d)
            kept = {k: e for k, e in eps.items() if (k[0], k[1]) not in skip}
            rs = C.row("CT2 vs R1", C.pairs(kept, "CT2", "R1"))
            nk, _, latek, _ = latency(kept, "CT2", seconds)
            verdicts.append(f"- {name}: sensitivity, without the {len(skip)} cells with a retried first answer: "
                            f"CT2 vs R1 strict {'/'.join(map(str, rs['strict']))}, p {rs['p_strict']:.4f}, pass mark {C.better(rs, 'strict')}; "
                            f"no first answer in time {latek} of {nk}")
        if name in JUDGED:
            bound = late > nc / 2
            verdicts.append(f"- {name}: pass mark{' (secondary)' if long else ''} (CT2 vs R1, strict, wins > losses and "
                            f"p <= 0.05): {C.better(r2, 'strict')}; primary scoring (reported): {C.better(r2, 'frozen')}"
                            f"{'; LATENCY-BOUND: most C answers arrive after the episode ends' if bound else ''}"
                            f"{'' if r2['n'] == 25 else f'  [pairs {r2[chr(110)]}, expected 25: check mechanics]'}")
    print("\n" + "\n".join(verdicts) if verdicts else "\n(no model has run)")


if __name__ == "__main__":
    main()
