"""Port checks for CONFIRM_CT4PJ.md, totals only (no per-layout lines). Zero calls; reads only the direction, join
and "behind" fields, the tracks and the cached first answers; no outcome measure.

    .venv/bin/python round4/p_port_checks.py RUN_ROOT SELECTION_JSON DUMP_DIR
    (from best-version-and-running-guess/; RUN_ROOT relative to results/data/moving/, e.g. runs_confirm4/dsv41;
    SELECTION_JSON and DUMP_DIR relative to results/moving-lines/, e.g. test-layouts/confirm_selection.json
    test-layouts/confirm_dump)

- CT4P (as e_ct4p_port.py): for every re-solve with direction_source "path", rebuild the remembered path's back
  tangent at the farthest roster member along it and compare with solve_direction (match within 1e-3). Every
  "belief" fallback should be a re-solve with no path in the rebuild too.
- CT4PJ (as g_ct4pj_port.py): on re-solves where the arm logs joined, rebuild "behind" (arc length on the path
  beyond the robot's, or distance to the target without a path; robot position interpolated from the 1-s samples)
  and compare with the logged ids. Count the solver's last being a "behind" id, which should never happen.
Bars: CT4P direction match ≥ 0.99 of path re-solves, with no fallback that has a path in the rebuild; CT4PJ behind
match ≥ 0.99 of joined re-solves, with no last that is "behind". Below a bar: stop, fix the port, scores unread.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "round3"))
from b_pl import Path_  # noqa: E402
from replay_lib import MOVING, Cell, robot_at  # noqa: E402
from sgi.types import Constraint, DetectedPerson  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402


def cells(root, sel_path, dump_dir, arm):
    """As g_pilot_predict.pilot_cells, with the selection, dumps and run folder as arguments."""
    sel = json.loads((paths.MOVING / sel_path).read_text())["layouts"]
    out, missing = [], Counter()
    for lay in sel:
        hits = sorted((MOVING / root).glob(f"**/{lay['name']}/seed{lay['seed']}/{arm}/episode.json"))
        dump_f = paths.MOVING / dump_dir / lay["name"] / f"seed{lay['seed']}.json"
        if not hits or not dump_f.exists():
            missing["no episode" if not hits else "no dump"] += 1
            continue
        f = hits[0]
        e = json.loads(f.read_text())
        a = f.parent / "001" / "answer.json"
        answers = [x for x in e.get("answers", []) if x.get("delivered_t") is not None]
        if not a.exists() or not answers:
            missing["no answer"] += 1
            continue
        ans = json.loads(a.read_text())
        inf = ans.get("inference") or {}
        if ans.get("source") != "constraints" or not inf.get("people") or ans.get("target_world") is None:
            missing["answer without people or target"] += 1
            continue
        people = [DetectedPerson(number=p["number"], xy=tuple(p["xy"]), yaw=p.get("yaw"),
                                 pixel=tuple(p.get("pixel") or (0.0, 0.0)), label=p.get("label"))
                  for p in inf["people"]]
        cons = [Constraint(**c["constraint"]) for c in inf.get("constraints", [])]
        out.append(Cell(model=root, scenario=lay["name"], seed=lay["seed"], arm=arm, path=f.parent, episode=e,
                        people=people, constraints=cons, target=tuple(ans["target_world"][:2]),
                        asked_t=answers[0]["asked_t"], delivered_t=answers[0]["delivered_t"],
                        bystanders=set(e.get("bystanders") or []), dump=json.loads(dump_f.read_text())))
    return out, missing


def check_ct4p(cs):
    k = Counter()
    for c in cs:
        path = Path_(c)
        for r in c.episode["resolves"]:
            if r.get("status") != "ok":
                continue
            pos = {int(i): (x, y) for i, x, y in r["tracks"]}
            pts = [pos[i] for i in (r.get("roster") or []) if i in pos]
            mine = path.r._on_path(max(pts, key=path.r._along))[2] if path.r._path is not None and pts else None
            if r.get("direction_source") == "path":
                k["path"] += 1
                d = r.get("solve_direction")
                if mine is not None and d and math.dist(mine, d) < 1e-3:
                    k["same"] += 1
            else:
                k["fallback"] += 1
                k["fallback with a path in the rebuild"] += mine is not None
    share = k["same"] / max(k["path"], 1)
    ok = k["path"] > 0 and share >= 0.99 and k["fallback with a path in the rebuild"] == 0
    print(f"- CT4P direction: {k['same']}/{k['path']} path re-solves match ({share:.4f}); fallbacks {k['fallback']}, "
          f"of which with a path in the rebuild {k['fallback with a path in the rebuild']} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_ct4pj(cs):
    k = Counter()
    for c in cs:
        path = Path_(c)
        for r in c.episode["resolves"]:
            if r.get("status") != "ok":
                continue
            pos = {int(i): (x, y) for i, x, y in r["tracks"]}
            beh = set(r.get("behind") or [])
            k["last is behind"] += r.get("last") is not None and r["last"] in beh
            if not r.get("joined"):
                continue
            k["joined"] += 1
            rob = robot_at(c.episode, r["sent_t"])
            if path.r._path is not None:
                mine = path.r._along(rob)
                rebuilt = {i for i, xy in pos.items() if path.r._along(xy) > mine + 1e-9}
            else:
                mine = math.dist(rob, c.target)
                rebuilt = {i for i, xy in pos.items() if math.dist(xy, c.target) > mine + 1e-9}
            if rebuilt == beh:
                k["same"] += 1
            else:
                k["boundary (differs by one person)"] += len(rebuilt ^ beh) == 1
                k["differs by more"] += len(rebuilt ^ beh) > 1
    share = k["same"] / max(k["joined"], 1)
    ok = share >= 0.99 and k["last is behind"] == 0
    print(f"- CT4PJ behind: {k['same']}/{k['joined']} joined re-solves match ({share:.4f}); boundary "
          f"{k['boundary (differs by one person)']}, differs by more {k['differs by more']}; last is a behind id "
          f"{k['last is behind']} -> {'PASS' if ok else 'FAIL'}"
          + ("" if k["joined"] else " (no joined re-solves: nothing to rebuild)"))
    return ok


def main():
    root, sel_path, dump_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"# Port checks (totals only): {root}\n")
    c4p, m4p = cells(root, sel_path, dump_dir, "CT4P")
    c4pj, m4pj = cells(root, sel_path, dump_dir, "CT4PJ")
    print(f"- episodes: CT4P {len(c4p)} (left out {dict(m4p)}), CT4PJ {len(c4pj)} (left out {dict(m4pj)})")
    a = check_ct4p(c4p)
    b = check_ct4pj(c4pj)
    print(f"\nVerdict: {'both PASS: scoring may go ahead' if a and b else 'FAIL: stop and fix the port; scores unread'}")


if __name__ == "__main__":
    main()
