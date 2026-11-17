"""Round 3, step G: offline predictions for the closed-loop pilot (24 layouts, candidates 1000-1199), written and
hashed before any closed-loop arm other than CT2 is opened. Zero calls.

Reads only: test-layouts/pilot_selection.json, test-layouts/pilot_dump/<name>/seed<N>.json (robot idle), and each layout's CT2
first answer (001/answer.json) and CT2 episode (for the photo and delivery times and the robot's path). It does
not open CT3F, CT4, W0 or CT5 outputs.

Offline arms: CT4 (V0), W0 (CT4 + window), CT5 (W4). Per arm, pooled over layouts, from delivery: right / cut / far /
empty shares against the dump's truth; last switches per minute; roster changes per tick. Per layout: CT5 minus CT4
switches per minute, as the sign the closed loop's goal jumps per minute should show.

    .venv/bin/python round3/g_pilot_predict.py runs_pilot3/dsv41
    (from best-version-and-running-guess/; RUN_ROOT relative to results/data/moving/)
"""
import json
import sys
from collections import Counter
from pathlib import Path

from r3lib import CT4, CT5, classify, run  # first: it sets the import path
from replay_lib import MOVING, Cell  # noqa: E402
from sgi.types import Constraint, DetectedPerson  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

WIN = dict(STATIC_WIN_S=5.0, WALKERS_STATIC=True, LR_EXIT=1.0)
ARMS = {"CT4": (CT4, None, "threshold"), "W0": (CT5, WIN, "threshold"), "CT5": (CT5, WIN, "cost")}


def pilot_cells(root, arm="CT2"):
    sel = json.loads((paths.MOVING / "test-layouts" / "pilot_selection.json").read_text())["layouts"]
    out, missing = [], []
    for lay in sel:
        hits = sorted((MOVING / root).glob(f"**/{lay['name']}/seed{lay['seed']}/{arm}/episode.json"))
        dump_f = paths.MOVING / "test-layouts" / "pilot_dump" / lay["name"] / f"seed{lay['seed']}.json"
        if not hits or not dump_f.exists():
            missing.append((lay["name"], f"no {arm} episode" if not hits else "no dump"))
            continue
        f = hits[0]
        e = json.loads(f.read_text())
        a = f.parent / "001" / "answer.json"
        answers = [x for x in e.get("answers", []) if x.get("delivered_t") is not None]
        if not a.exists() or not answers or not e.get("samples"):
            missing.append((lay["name"], "no answer"))
            continue
        ans = json.loads(a.read_text())
        inf = ans.get("inference") or {}
        if ans.get("source") != "constraints" or not inf.get("people") or ans.get("target_world") is None:
            missing.append((lay["name"], "answer without people or target"))
            continue
        people = [DetectedPerson(number=p["number"], xy=tuple(p["xy"]), yaw=p.get("yaw"),
                                 pixel=tuple(p.get("pixel") or (0.0, 0.0)), label=p.get("label"))
                  for p in inf["people"]]
        cons = [Constraint(**c["constraint"]) for c in inf.get("constraints", [])]
        out.append((lay, Cell(model="dsv41", scenario=lay["name"], seed=lay["seed"], arm=arm, path=f.parent,
                              episode=e, people=people, constraints=cons, target=tuple(ans["target_world"][:2]),
                              asked_t=answers[0]["asked_t"], delivered_t=answers[0]["delivered_t"],
                              bystanders=set(e.get("bystanders") or []),
                              dump=json.loads(dump_f.read_text()))))
    return out, missing


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "runs_pilot3/dsv41"
    cs, missing = pilot_cells(root)
    print(f"# Offline predictions for the round 3 pilot: {len(cs)} layouts with a CT2 answer\n")
    for name, why in missing:
        print(f"- left out: {name} ({why})")
    per = {}
    print("\n| Arm | right | cut | far | empty | last switches / min | roster changes / tick |")
    print("|---|---|---|---|---|---|---|")
    for arm, (cls, params, decision) in ARMS.items():
        k, sw, rf, nt, mins = Counter(), 0, 0, 0, 0.0
        for lay, c in cs:
            rows = run(c, None, est_cls=cls, params=params, decision=decision)
            kk = Counter(classify(r) for r in rows)
            s = sum(1 for a, b in zip(rows, rows[1:]) if a["last"] != b["last"])
            m = max((rows[-1]["t"] - rows[0]["t"]) / 60, 1e-9) if rows else 1e-9
            k.update(kk)
            sw += s
            rf += sum(1 for a, b in zip(rows, rows[1:]) if a["roster"] != b["roster"])
            nt += len(rows)
            mins += m
            per[(lay["name"], arm)] = s / m
        scored = sum(v for key, v in k.items() if key is not None)
        print(f"| {arm} | " + " | ".join(f"{k[x] / max(scored, 1):.3f}" for x in ("right", "cut", "far", "empty")) +
              f" | {sw / max(mins, 1e-9):.1f} | {rf / max(nt, 1):.3f} |")
    fewer = sum(1 for lay, _ in cs if per[(lay["name"], "CT5")] < per[(lay["name"], "CT4")])
    print(f"\nPer layout, CT5 switches fewer than CT4's: {fewer} of {len(cs)}")


if __name__ == "__main__":
    main()
