"""Round 6, check on part B's scoring (EXPLORATORY): over joined re-solves where the sim agrees the robot is in line,
right last split by whether the sim's last is None (nobody ahead: the robot is first) or a person. Zero calls.

    .venv/bin/python round6/d_none.py SELECTION_JSON RUN_ROOT [RUN_ROOT ...]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "round3"))
from b_pred import find, ok_resolves, stretches  # noqa: E402
from i_followups import MOVING  # noqa: E402


def main():
    sel = json.loads((MOVING / sys.argv[1]).read_text())["layouts"]
    print(f"# Right last while in line, split by the sim's last ({sys.argv[1]}; exploratory)")
    for root in sys.argv[2:]:
        print(f"\n## {root}\n")
        print("| arm | sim's last is a person: n, right | sim's last is none: n, right | arm says none when a person is ahead |")
        print("|---|---|---|---|")
        for arm in ("CT3F", "CT4PJ"):
            k = {"pn": 0, "pr": 0, "nn": 0, "nr": 0, "fn": 0}
            for lay in sel:
                f = find(root, lay, arm)
                if f is None:
                    continue
                for s in stretches(ok_resolves(json.loads(f.read_text()))):
                    for r in s:
                        tr = r["truth"]
                        if not tr.get("joined"):
                            continue
                        if tr.get("last") is None:
                            k["nn"] += 1
                            k["nr"] += r.get("last") is None
                        else:
                            k["pn"] += 1
                            k["pr"] += r.get("last") == tr["last"]
                            k["fn"] += r.get("last") is None
            print(f"| {arm} | {k['pn']}, {k['pr'] / max(k['pn'], 1):.3f} | {k['nn']}, {k['nr'] / max(k['nn'], 1):.3f} | "
                  f"{k['fn'] / max(k['pn'], 1):.3f} |")


if __name__ == "__main__":
    main()
