"""SNAP_FROZEN.md with amendments 1-3 scored (zero calls). For every plan-only record: target_correct with the
pipeline's depth lift (as recorded and re-matched) and with the frozen person snap (snap_sweep_v2.py's rule), per
scene and set; H1, H2 (exact sign-flip test over the 8 primary layouts), H3; per-scene table; every break listed.

    python3 snap_score.py --runs RUN_DIR [...] --scenes snapset/run_scenes_v2 --people snapset/snap_people.json
    python3 snap_score.py --dev ...    # mechanics check on development runs only (no sets, no bars)

Run with the main checkout's venv (sgi importable). Scenes are matched by folder name = scene_id; detector people by
the basename of their key."""
import argparse
import csv
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from sgi.evaluate import match_target
from sgi.gt.derive import prepare
from sgi.rescore import _recorded_config, _records
from sgi.scene import load_scene

HERE = Path(__file__).resolve().parent
PRIMARY = ["s403c3b50", "s404c4b10", "s405c2b20", "s406c3b50", "s410c4b75", "s411c4b70", "s412c2b15", "s415c4b10"]
FAR = ["s401c2b15", "s407c2b20", "s408c3b15", "s409c3b15"]
HELD = re.compile(r"eval_1[6-9]|group_photo|lift_wait")
MODELS = ("qwen38_low", "glm53", "dsv41")


def snap(u, v, people, lifted):
    """The frozen rule: the detected person whose box holds the click, nearest by |u - box centre u| / box width; used
    only when the depth-lifted point is more than 0.5 m from that person. Returns (world or None, chosen person)."""
    inside = [p for p in people if p["box"][0] <= u <= p["box"][2] and p["box"][1] <= v <= p["box"][3]]
    if not inside:
        return None, None
    best = min(inside, key=lambda p: abs(u - (p["box"][0] + p["box"][2]) / 2) / max(p["box"][2] - p["box"][0], 1.0))
    if math.dist((best["x"], best["y"]), lifted[:2]) <= 0.5:
        return None, best
    return (best["x"], best["y"], 1.1), best


def set_of(scene_id, cells):
    prefix = scene_id.split("_", 1)[0]
    if prefix in PRIMARY:
        return "primary"
    if prefix in FAR:
        return "far"
    cell = cells.get(scene_id, "")
    return {"A": "A", "B": "B"}.get(cell.split("/")[0], "other")


def model_of(run_dir):
    name = Path(run_dir).name
    return next((m for m in MODELS if m in name), name)


def sign_flip_p(values):
    """Exact two-sided sign-flip test of the sum of per-layout values (zeros carry no sign)."""
    obs = abs(sum(values))
    vals = [v for v in values if v != 0]
    n = len(vals)
    if n == 0:
        return 1.0
    hits = sum(1 for signs in itertools.product((1, -1), repeat=n) if abs(sum(s * v for s, v in zip(signs, vals))) >= obs - 1e-9)
    return hits / 2 ** n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--people", required=True)
    ap.add_argument("--sets", default=str(HERE / "snapset/assembled_v2.csv"))
    ap.add_argument("--gaps", default=str(HERE / "anchor_gap_out.txt"))
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args()

    people = {Path(k).name: v["people"] for k, v in json.loads(Path(a.people).read_text())["scenes"].items()}
    cells = {} if a.dev else {r["scene_name"]: r["cell"] for r in csv.DictReader(open(a.sets))}
    gaps = {}
    for line in Path(a.gaps).read_text().splitlines() if Path(a.gaps).exists() else []:
        m = re.match(r"(\S+)\s.*\(([\d.]+|inf) body half-widths\)", line)
        if m:
            gaps[m.group(1)] = float(m.group(2))
    scene_dirs = {d.name: d for d in Path(a.scenes).iterdir() if (d / "scene.json").exists()}

    rows = []
    loaded = {}
    for run in a.runs:
        _, cfg = _recorded_config(Path(run) / "config.yaml")
        th = cfg.thresholds
        for rec in _records(Path(run) / "results.jsonl"):
            sid, eid = rec["scene_id"], rec["episode_id"]
            if HELD.search(sid):
                continue                                  # never read a held-out row
            if sid not in scene_dirs:
                continue
            if sid not in loaded:
                inputs, truth = load_scene(scene_dirs[sid])
                loaded[sid] = (inputs, truth, prepare(inputs, truth, cfg))
            inputs, truth, truths = loaded[sid]
            gt = truths[eid].gt
            kinds = {truth.targets[i].kind for i in gt.target_ids}
            t = (rec.get("result") or {}).get("target")
            base = after = False
            chosen = None
            if t and t.get("world") is not None and t.get("pixel") is not None:
                base = match_target(t["world"], truth, gt, th) is not None
                w, chosen = snap(t["pixel"][0], t["pixel"][1], people.get(sid, []), t["world"])
                after = base if w is None else match_target(w, truth, gt, th) is not None
                fired = w is not None
            else:
                fired = False
            tp = next((truth.targets[i] for i in gt.target_ids if truth.targets[i].kind == "person"), None)
            recall = None
            if tp is not None:
                recall = any(math.dist((p["x"], p["y"]), tp.anchor[:2]) <= 0.5 for p in people.get(sid, []))
            rows.append(dict(model=model_of(run), scene=sid, ep=eid, trial=rec.get("trial"), kind="/".join(sorted(kinds)),
                             set="dev" if a.dev else set_of(sid, cells), base=base, after=after, fired=fired,
                             pixel=None if not t else t.get("pixel"), recall=recall,
                             boxes=[p["box"] for p in people.get(sid, []) if t and t.get("pixel") is not None
                                    and p["box"][0] <= t["pixel"][0] <= p["box"][2] and p["box"][1] <= t["pixel"][1] <= p["box"][3]]))

    print(f"# Snap test: {len(rows)} records, models {sorted({r['model'] for r in rows})}"
          f"{' (DEV MECHANICS CHECK, not a judgement)' if a.dev else ''}\n")
    by_set = defaultdict(list)
    for r in rows:
        by_set[r["set"]].append(r)
    print("| Set | Records | Depth lift right | Snap right | Fixed | Broken | Snap fired |")
    print("|---|---|---|---|---|---|---|")
    for s in ("primary", "far", "A", "B", "other", "dev"):
        rs = by_set.get(s, [])
        if rs:
            print(f"| {s} | {len(rs)} | {sum(r['base'] for r in rs)} | {sum(r['after'] for r in rs)} | "
                  f"{sum(r['after'] and not r['base'] for r in rs)} | {sum(r['base'] and not r['after'] for r in rs)} | "
                  f"{sum(r['fired'] for r in rs)} |")
    print("\nPer model:")
    for m in sorted({r["model"] for r in rows}):
        for s in ("primary", "far", "A", "B", "dev"):
            rs = [r for r in by_set.get(s, []) if r["model"] == m]
            if rs:
                print(f"  {m:12} {s:8} lift {sum(r['base'] for r in rs)}/{len(rs)}, snap {sum(r['after'] for r in rs)}/{len(rs)}, "
                      f"fixed {sum(r['after'] and not r['base'] for r in rs)}, broken {sum(r['base'] and not r['after'] for r in rs)}")
    print("\nPer scene (gap in body half-widths; detector recall of the target):")
    for sid in sorted({r["scene"] for r in rows}):
        rs = [r for r in rows if r["scene"] == sid]
        rec = {r["recall"] for r in rs}
        print(f"  {sid:52} {rs[0]['set']:7} gap {gaps.get(sid, '-')} recall {rec} lift {sum(r['base'] for r in rs)}/{len(rs)} "
              f"snap {sum(r['after'] for r in rs)}/{len(rs)} fixed {sum(r['after'] and not r['base'] for r in rs)} "
              f"broken {sum(r['base'] and not r['after'] for r in rs)}")
    breaks = [r for r in rows if r["base"] and not r["after"]]
    if breaks:
        print("\nEvery break:")
        for r in breaks:
            print(f"  {r['model']} {r['scene']} {r['ep']} t{r['trial']} ({r['set']}): click {r['pixel']}, boxes holding it {r['boxes']}")
    if a.dev:
        return

    print("\n## Bars (SNAP_FROZEN.md, amendments 1-3)\n")
    prim, setA = by_set.get("primary", []), by_set.get("A", [])
    rate_p = sum(r["base"] for r in prim) / max(len(prim), 1)
    rate_a = sum(r["base"] for r in setA) / max(len(setA), 1)
    h1 = (rate_a - rate_p) >= 0.10
    print(f"- H1: depth-lift target right on the primary 8: {rate_p:.3f} ({len(prim)} records) against set A {rate_a:.3f} "
          f"({len(setA)}); gap {100 * (rate_a - rate_p):.1f} points (bar >= 10): {'PASS' if h1 else 'FAIL -> \"not needed\"; H2 not judged'}")
    nets = []
    for sid in sorted({r["scene"] for r in prim}):
        rs = [r for r in prim if r["scene"] == sid]
        nets.append(sum(r["after"] and not r["base"] for r in rs) - sum(r["base"] and not r["after"] for r in rs))
    p = sign_flip_p(nets)
    net = sum(nets)
    h2 = h1 and net > 0 and p <= 0.05
    print(f"- H2: per-layout net (fixed - broken) {nets}, total {net:+d}, sign-flip p {p:.4f} (bar > 0 and p <= 0.05): "
          f"{'PASS' if h2 else ('FAIL' if h1 else 'not judged')}")
    brA = sum(r["base"] and not r["after"] for r in setA)
    brB = sum(r["base"] and not r["after"] for r in by_set.get("B", []))
    h3 = brA <= 1
    print(f"- H3: set A broken {brA} of {len(setA)} (bar <= 1): {'PASS' if h3 else 'FAIL'}; set B (weak fallbacks, no "
          f"weight) broken {brB} of {len(by_set.get('B', []))}")
    print(f"- Claim 'the snap beats C's lift on person targets' (a first replication on 8 layouts): "
          f"{'MADE' if h1 and h2 and h3 else 'NOT MADE'}")


if __name__ == "__main__":
    main()
