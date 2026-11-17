"""Shared scorer for the fill-in moving designs (EVIDENCE_AUDIT.md Part 1). Zero calls; read-only.

Paired comparisons of whole-episode share in line by (model, scenario, seed), exactly as moving-lines/design-and-scoring/summarise_fresh.py's
`paired`:
- measures() gives share_strict (SCORING_STRICT.md) and share_frozen (SCORING.md, the primary scoring);
- a win, loss or tie compares the two shares, with a tie only when they are equal within 1e-9;
- the sign test is summarise_fresh.sign_p (exact, two-sided).
Pooled results sum wins and losses over models (each cell is one pair).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

MOVING = paths.MOVING_RUNS           # run folders are named relative to it, e.g. "runs_i10/dsv41"
sys.path.insert(0, str(paths.DESIGN))
from summarise_fresh import measures, sign_p  # noqa: E402

EVAL5 = ("cafe_slots", "room_coffee_cart_slots", "cafe_line2_moving", "cafe_line6_moving", "sweet_spot_line2_moving")


def load(root: str, scenarios, seeds, arms) -> dict:
    """{(scenario, seed, arm): episode or None} for one model's run folder (relative to moving/ or absolute)."""
    base = Path(root) if Path(root).is_absolute() else MOVING / root
    out = {}
    for sc in scenarios:
        for s in seeds:
            for a in arms:
                f = base / sc / f"seed{s}" / a / "episode.json"
                out[(sc, s, a)] = json.loads(f.read_text()) if f.exists() else None
    return out


def errors(e) -> int:
    """Answers or re-solves that ended in a vlm_error (a cache miss in a cache-only run shows as one)."""
    if e is None:
        return 0
    return sum(1 for a in e.get("answers", []) if a.get("status") == "vlm_error")


def pairs(eps: dict, a: str, b: str) -> list:
    keys = sorted({(sc, s) for (sc, s, _) in eps})
    return [(k, measures(eps[(*k, a)]), measures(eps[(*k, b)])) for k in keys
            if eps.get((*k, a)) is not None and eps.get((*k, b)) is not None]


def wlt(ps, key: str) -> tuple:
    w = sum(x[key] > y[key] + 1e-9 for _, x, y in ps)
    l = sum(x[key] < y[key] - 1e-9 for _, x, y in ps)
    return w, l, len(ps) - w - l


def row(name: str, ps) -> dict:
    sw, sl, st = wlt(ps, "share_strict")
    fw, fl, ft = wlt(ps, "share_frozen")
    d = statistics.mean(x["share_strict"] - y["share_strict"] for _, x, y in ps) if ps else float("nan")
    df = statistics.mean(x["share_frozen"] - y["share_frozen"] for _, x, y in ps) if ps else float("nan")
    return dict(name=name, n=len(ps), strict=(sw, sl, st), p_strict=sign_p(sw, sl), d_strict=d,
                frozen=(fw, fl, ft), p_frozen=sign_p(fw, fl), d_frozen=df)


def table(rows) -> str:
    out = ["| | pairs | strict W/L/T | strict mean d | strict p | primary W/L/T | primary mean d | primary p |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['name']} | {r['n']} | {'/'.join(map(str, r['strict']))} | {r['d_strict']:+.3f} | "
                   f"{r['p_strict']:.4f} | {'/'.join(map(str, r['frozen']))} | {r['d_frozen']:+.3f} | {r['p_frozen']:.4f} |")
    return "\n".join(out)


def means(eps: dict, arm: str) -> tuple:
    m = [measures(e) for (_, _, a), e in eps.items() if a == arm and e is not None]
    if not m:
        return 0, float("nan"), float("nan")
    return len(m), statistics.mean(x["share_strict"] for x in m), statistics.mean(x["share_frozen"] for x in m)


def better(r: dict, key: str = "strict", alpha: float = 0.05) -> bool:
    """The bar used by the fill-in designs: more wins than losses and an exact two-sided sign p <= alpha."""
    w, l, _ = r[key]
    return w > l and r[f"p_{key}"] <= alpha
