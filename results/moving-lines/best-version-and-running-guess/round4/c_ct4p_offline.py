"""Round 4, step C (written after CT4P.md was frozen and while CT4P runs; not a pre-registered prediction): what the
offline proxy says about CT4 against CT4P on the 24 pilot layouts, robot idle. Zero calls."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b_pl import VARIANTS, classify, line_of, pilot_cells, proxy_ok, run  # noqa: E402

HIGH = ["cafe_1002", "cafe_1000", "cafe_1001", "cafe_1160", "cafe_1031", "sweet_1112", "sweet_1130", "cafe_1081",
        "sweet_1011", "cafe_1183", "cafe_1090", "sweet_1129"]


def main():
    cs, _ = pilot_cells("runs_pilot3/dsv41")
    for name in ("CT4 (threshold, chord)", "CT4P (threshold, path)"):
        cls, params, dec, dr = VARIANTS[name]
        out = {}
        for grp in ("high", "low"):
            ok = n = okr = nr = 0
            for lay, c in cs:
                if (lay["name"] in HIGH) != (grp == "high"):
                    continue
                line = line_of(lay["name"])
                for r in run(c, None, cls, params, dec, dr):
                    if not r["q"]:
                        continue
                    g = proxy_ok(r, line)
                    n += 1
                    ok += g
                    if classify(r) == "right":
                        nr += 1
                        okr += g
            out[grp] = (ok / max(n, 1), okr / max(nr, 1), nr)
        print(f"{name}: goal in line (proxy), all ticks: high {out['high'][0]:.3f}, low {out['low'][0]:.3f}; "
              f"right-last ticks only: high {out['high'][1]:.3f} (n {out['high'][2]}), low {out['low'][1]:.3f} "
              f"(n {out['low'][2]})")


if __name__ == "__main__":
    main()
