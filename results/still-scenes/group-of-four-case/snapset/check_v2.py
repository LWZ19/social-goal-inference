"""Zero calls. Two checks of scan_v2.py's replay:
  1. its P0 rows reproduce layouts.csv (the plain default-bearing scan), every person target's share and twin id;
  2. for one seed and a patched configuration, a plain sgi.synth.generate.accepted_layouts run (the call the realistic
     generator makes) with the same in-process patch gives the same twin ids and the same people as scan_v2's replay
     with the saved and restored generator state.
    PYTHONPATH= nice -n 10 .venv/bin/python results/2026-11-01-push/group4/snapset/check_v2.py [SEED CONFIG]
"""
import csv
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scan_v2 as V                                # noqa: E402


def check_p0() -> bool:
    old = {(r["seed"], r["twin_id"]): r for r in csv.DictReader(open(HERE / "layouts.csv"))}
    new = [r for r in csv.DictReader(open(HERE / "layouts_v2.csv")) if r["config"] == "P0" and not r["error"]]
    bad = [r["twin_id"] for r in new if (r["seed"], r["twin_id"]) not in old or
           (r["target_kind"] == "person" and r["hidden_real"] != old[(r["seed"], r["twin_id"])]["hidden_real"])]
    seeds = {r["seed"] for r in new}
    missing = [k for k in old if k[0] in seeds and k not in {(r["seed"], r["twin_id"]) for r in new}]
    print(f"P0 vs layouts.csv: {len(new)} rows over {len(seeds)} seeds, mismatches {len(bad)}, missing {len(missing)}")
    return not bad and not missing


def check_patched(seed: int, config: str) -> bool:
    """A plain patched accepted_layouts run against scan_v2's replay: rows from the patched size on must equal the
    replay's rows of `config`, rows before it P0's; a failing replay must fail in both."""
    import sgi.synth.recipes as R
    from sgi.config import load_config
    from sgi.synth.generate import accepted_layouts
    from sgi.synth.realistic.library import load_library
    keys = ("twin_id", "bearing", "hidden_real", "occluders_real", "b_hidden_by_target", "b_person")
    default, library = dict(R._GROUP_BEARING_DEG), load_library()
    R._GROUP_BEARING_DEG.clear()
    R._GROUP_BEARING_DEG.update(V.CONFIGS[config])
    plain, plain_error = [], None
    try:
        with tempfile.TemporaryDirectory() as work:
            for split, index, layout, twin in accepted_layouts(Path(work), load_config(), seed):
                got = V._measure(seed, config, split, index, layout, twin, library, False)
                if got is not None:
                    plain.append(tuple(got.get(k) for k in keys))
    except RuntimeError as error:
        plain_error = str(error)
    finally:
        R._GROUP_BEARING_DEG.clear()
        R._GROUP_BEARING_DEG.update(default)
    only = dict(V.CONFIGS)
    V.CONFIGS.clear()
    V.CONFIGS.update({"P0": only["P0"], config: only[config]})
    try:
        rows = V._rows_for_seed(seed)
    finally:
        V.CONFIGS.clear()
        V.CONFIGS.update(only)
    replay_error = next((r["error"] for r in rows if r["config"] == config and r.get("error")), None)
    if plain_error or replay_error:
        same = bool(plain_error) and bool(replay_error)
        print(f"seed {seed} {config}: plain fails: {plain_error!r}; replay fails: {replay_error!r} -> agree: {same}")
        return same
    mine = [tuple(r.get(k) for k in keys) for r in rows if r["config"] == config]
    p0 = {tuple(r.get(k) for k in keys) for r in rows if r["config"] == "P0" and r["target_kind"] == "person"}
    ids = {row[0] for row in mine}
    after = [row for row in plain if row[0] in ids]
    before = [row for row in plain if row[0] not in ids]
    same = after == mine and all(row in p0 for row in before)
    print(f"seed {seed} {config}: rows from the patch on equal: {after == mine}; rows before it are P0's: "
          f"{all(row in p0 for row in before)}")
    for row in plain:
        print("   ", row)
    return same


if __name__ == "__main__":
    ok = True
    if (HERE / "layouts_v2.csv").is_file():
        ok &= check_p0()
    if len(sys.argv) == 3:
        ok &= check_patched(int(sys.argv[1]), sys.argv[2])
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)
