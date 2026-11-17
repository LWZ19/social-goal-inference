"""Independent verifier, I06 build: the expensive checks on a stated random sample of kept variants, on all 30 bases,
and the visibility claims on every kept E variant.

    v_sample.py

Sample: kept variants of the manifest grouped by (family, level), each group sorted by scene id, then
numpy default_rng(20261018).choice(n, min(5, n), replace=False) per group in sorted (family, level) order: 60
variants over all 12 family-levels. (20261018 is a sampling seed, not a generator seed.)

Per sampled variant and per base (the build's folder, which v_compare_bases.py shows equal to my reproduction):
- sgi.check_scene on the stored folder: no error and no warning but check 9;
- full oracle (grounding oracle, context_check always_yes, goal oracle, every stage on the mock entry, no cache):
  run.run_episode on both episodes; success, and the number of model calls;
- regions re-derived by sgi.gt.derive.prepare from the folder equal the stored polygons and representative points;
- L1 cells of e0 and L2 cells (my implementation of FROZEN section 5), L2-empty;
- per person: samples seen of 9 and relevant (e0, e1) against the manifest's person table; pixel share
  (generate._share_shown on that person's figure, figures from my replay) where the manifest records one.
For every kept E variant: each relevant base person's samples seen and pixel share in the variant vs the base.
Writes out/sample.jsonl and out/sample.txt.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import math
import time

import numpy as np

from sgi.check_scene import check_scene
from sgi.config import load_config, with_model
from sgi.freespace import connected, robot_grid
from sgi.gt.derive import _VISIBILITY_HEIGHTS_M, _sample_seen, prepare
from sgi.pipeline import build_pipeline
from sgi.polygons import grid_points_in
from sgi.run import run_episode
from sgi.scene import load_scene
from sgi.synth import generate as G

import vlib as V

LINES = []


def say(t=""):
    print(t, flush=True)
    LINES.append(t)


def l1_l2(inputs, truth, ets, th):
    obs = inputs.observation
    grid = robot_grid(truth, obs.robot, th)
    joined = connected(grid, (obs.robot.x, obs.robot.y))
    cells = set()
    for r in ets["e0"].gt.acceptable_regions or []:
        for x, y in grid_points_in(r.polygon, th.grid_resolution_m):
            idx = grid.index(float(x), float(y))
            if idx is not None and joined[idx]:
                cells.add((float(x), float(y)))
    gid = truth.episodes["e0"].group_at_target
    non = [p for p in truth.people if p.group != gid]
    groups = collections.defaultdict(list)
    for p in non:
        if p.group is not None:
            groups[p.group].append(p)
    circles = []
    for g, mem in groups.items():
        if len(mem) >= 2:
            cx, cy = sum(p.x for p in mem) / len(mem), sum(p.y for p in mem) / len(mem)
            circles.append((cx, cy, max(math.hypot(p.x - cx, p.y - cy) for p in mem)))
    l2 = 0
    for x, y in cells:
        if all(math.hypot(x - p.x, y - p.y) - th.person_radius_m - th.robot_radius_m >= th.personal_gap_m for p in non) \
                and all(math.hypot(x - cx, y - cy) >= r for cx, cy, r in circles):
            l2 += 1
    return len(cells), l2


def figures_of(b, fam, lv, seed, row):
    rng = np.random.default_rng([seed, 2, V.FAM_CODE[fam], row, lv])
    pairs, _, _ = V.place(rng, b, fam, lv)
    return [*b.layout.figures, *(f for _, f in pairs)], [p for p, _ in pairs]


def battery(folder, cfg, ocfg, pipeline):
    th = cfg.thresholds
    res = {}
    results = check_scene(folder, cfg, write_png=False)
    res["check_findings"] = [(r.number, r.level) for r in results if r.level in ("error", "warning")]
    res["check_pass"] = not [1 for r in results if r.level == "error" or (r.level == "warning" and r.number != 9)]
    inputs, truth = load_scene(folder)
    ets = prepare(inputs, truth, ocfg)
    oracle = {}
    for ep in inputs.episodes:
        rec = run_episode(pipeline, inputs, truth, ets[ep.episode_id], ep, ocfg, 0)
        oracle[ep.episode_id] = (bool(rec["score"]["success"]), len(rec["result"].get("calls") or []))
    res["oracle"] = oracle
    res["oracle_success"] = all(s for s, _ in oracle.values())
    res["calls"] = sum(c for _, c in oracle.values())
    same = True
    for eid, ep in truth.episodes.items():
        a = [(list(map(tuple, r.polygon)), tuple(r.representative_point)) for r in ep.acceptable_regions or []]
        d = [(list(map(tuple, r.polygon)), tuple(r.representative_point)) for r in ets[eid].gt.acceptable_regions or []]
        fa = [(f.kind, list(map(tuple, f.polygon))) for f in ep.forbidden_regions or []]
        fd = [(f.kind, list(map(tuple, f.polygon))) for f in ets[eid].gt.forbidden_regions or []]
        same &= a == d and fa == fd and len(a) > 0
    res["regions_package_own"] = same
    res["l1_cells"], res["l2_cells"] = l1_l2(inputs, truth, ets, th)
    res["l2_empty"] = res["l1_cells"] > 0 and res["l2_cells"] == 0
    rel = {eid: set(ets[eid].relevant_people) for eid in ets}
    persons = {}
    for p in truth.people:
        seen = int(sum(_sample_seen(inputs.observation, (p.x, p.y, z), th) for z in _VISIBILITY_HEIGHTS_M))
        persons[p.id] = {"seen": seen, "rel_e0": p.id in rel["e0"], "rel_e1": p.id in rel["e1"], "group": p.group}
    res["persons"] = persons
    return res, inputs, truth


def main():
    cfg = load_config()
    ocfg = with_model(cfg, "mock")
    ocfg = ocfg.model_copy(update={"run": ocfg.run.model_copy(update={
        "grounding": "oracle", "context_check": "always_yes", "goal": "oracle", "use_cache": False, "trials": 1})})
    pipeline = build_pipeline(ocfg)
    es = [json.loads(l) for l in (V.I06 / "build" / "manifest.jsonl").read_text().splitlines() if l]
    kept = [e for e in es if e["kind"] == "variant" and e["accepted"]]
    groups = collections.defaultdict(list)
    for e in kept:
        groups[(e["family"], e["level"])].append(e)
    rng = np.random.default_rng(20261018)
    sample = []
    for k in sorted(groups):
        g = sorted(groups[k], key=lambda e: e["scene_id"])
        idx = rng.choice(len(g), size=min(5, len(g)), replace=False)
        sample += [g[i] for i in sorted(idx)]
    say(f"sample: {len(sample)} kept variants: " + ", ".join(f"{k[0]}{k[1]} {min(5, len(v))}/{len(v)}" for k, v in sorted(groups.items())))
    bases = {s: V.load_bases(s) for s in (0, 1, 2)}
    out = open(V.HERE / "out" / "sample.jsonl", "w")
    t0 = time.perf_counter()
    agg = collections.Counter()
    # bases (the build's level-0 folders)
    for e in [x for x in es if x["kind"] == "base"]:
        res, inputs, truth = battery(V.I06 / e["folder"], cfg, ocfg, pipeline)
        row = {"scene_id": e["scene_id"], "seed": e["seed"], "kind": "base", **{k: v for k, v in res.items() if k != "persons"}}
        b = bases[e["seed"]][e["row"]]
        # person table vs the manifest's base table
        mp = e.get("persons") or {}
        row["persons_seen_match"] = all(mp.get(pid, {}).get("seen_of_9") == v["seen"] for pid, v in res["persons"].items())
        row["persons_rel_match"] = all(mp.get(pid, {}).get("relevant_e0") == v["rel_e0"] and
                                       mp.get(pid, {}).get("relevant_e1") == v["rel_e1"] for pid, v in res["persons"].items())
        share_diff = 0.0
        for pid, v in mp.items():
            if "pixel_share" in v:
                i = [p.id for p in truth.people].index(pid)
                s = G._share_shown(inputs.observation.camera, inputs.observation.depth, [b.layout.figures[i]])
                share_diff = max(share_diff, abs(s - v["pixel_share"]))
        row["share_max_diff"] = share_diff
        out.write(json.dumps(row) + "\n"); out.flush()
        for k in ("check_pass", "oracle_success", "regions_package_own", "persons_seen_match", "persons_rel_match"):
            agg[("base", k)] += bool(row[k])
        agg[("base", "l2_empty")] += row["l2_empty"]
        agg[("base", "calls")] += row["calls"]
        agg[("base", "n")] += 1
        print(f"base s{e['seed']} {e['scene_id']}: check {row['check_pass']} oracle {row['oracle_success']} "
              f"regions {row['regions_package_own']} seen {row['persons_seen_match']} rel {row['persons_rel_match']} "
              f"share {share_diff:.2e} [{time.perf_counter() - t0:.0f} s]", flush=True)
    # sampled variants
    for e in sample:
        res, inputs, truth = battery(V.I06 / e["folder"], cfg, ocfg, pipeline)
        b = bases[e["seed"]][e["row"]]
        figs, added = figures_of(b, e["family"], e["level"], e["seed"], e["row"])
        row = {"scene_id": e["scene_id"], "seed": e["seed"], "kind": "variant", "family": e["family"], "level": e["level"],
               **{k: v for k, v in res.items() if k != "persons"}}
        mp = e["persons"]
        row["persons_seen_match"] = all(mp[pid]["seen_of_9"] == v["seen"] for pid, v in res["persons"].items()) and \
            set(mp) == set(res["persons"])
        row["persons_rel_match"] = all(mp[pid]["relevant_e0"] == v["rel_e0"] and mp[pid]["relevant_e1"] == v["rel_e1"]
                                       for pid, v in res["persons"].items())
        share_diff = 0.0
        ids = [p.id for p in truth.people]
        for pid, v in mp.items():
            if "pixel_share" in v:
                s = G._share_shown(inputs.observation.camera, inputs.observation.depth, [figs[ids.index(pid)]])
                share_diff = max(share_diff, abs(s - v["pixel_share"]))
        row["share_max_diff"] = share_diff
        row["added_non_member"] = all(truth.people[ids.index(p.id)].group != truth.episodes["e0"].group_at_target
                                      and truth.people[ids.index(p.id)].role is None for p in added)
        out.write(json.dumps(row) + "\n"); out.flush()
        for k in ("check_pass", "oracle_success", "regions_package_own", "persons_seen_match", "persons_rel_match",
                  "added_non_member"):
            agg[("variant", k)] += bool(row[k])
        agg[("variant", "l2_empty")] += row["l2_empty"]
        agg[("variant", "calls")] += row["calls"]
        agg[("variant", "n")] += 1
        print(f"{e['scene_id']}: check {row['check_pass']} {row['check_findings']} oracle {row['oracle_success']} "
              f"regions {row['regions_package_own']} L1 {row['l1_cells']} L2 {row['l2_cells']} seen {row['persons_seen_match']} "
              f"rel {row['persons_rel_match']} share {share_diff:.2e} [{time.perf_counter() - t0:.0f} s]", flush=True)
    # every kept E variant: relevant base people's visibility, variant vs base
    say("\nE variants (all kept): relevant base members, samples seen of 9 and pixel share, variant vs base")
    e_rows = []
    th = cfg.thresholds
    for e in [x for x in kept if x["family"] == "E"]:
        b = bases[e["seed"]][e["row"]]
        vin, vtr = load_scene(V.I06 / e["folder"])
        bin_, btr = b.inputs, b.truth
        figs, added = figures_of(b, "E", e["level"], e["seed"], e["row"])
        bets = prepare(bin_, btr, cfg)
        vets = prepare(vin, vtr, cfg)
        relb = {pid for eid in bets for pid in bets[eid].relevant_people}
        changes = []
        for i, p in enumerate(btr.people):
            if p.id not in relb:
                continue
            sb = int(sum(_sample_seen(bin_.observation, (p.x, p.y, z), th) for z in _VISIBILITY_HEIGHTS_M))
            sv = int(sum(_sample_seen(vin.observation, (p.x, p.y, z), th) for z in _VISIBILITY_HEIGHTS_M))
            shb = G._share_shown(bin_.observation.camera, bin_.observation.depth, [b.layout.figures[i]])
            shv = G._share_shown(vin.observation.camera, vin.observation.depth, [figs[i]])
            changes.append((p.id, sb, sv, round(shb, 3), round(shv, 3)))
        ap = added[0]
        relv = {pid for eid in vets for pid in vets[eid].relevant_people}
        e_rows.append({"scene_id": e["scene_id"], "members": changes, "added_relevant": ap.id in relv})
        seen_changed = [c for c in changes if c[1] != c[2]]
        share_fell = [c for c in changes if c[4] < c[3] - 1e-9]
        say(f"  {e['scene_id']}: relevant {len(changes)}; seen changed {seen_changed or 'none'}; share fell {share_fell or 'none'}")
    n_unchanged = sum(all(c[1] == c[2] for c in r["members"]) for r in e_rows)
    n_fell = sum(1 for r in e_rows for c in r["members"] if c[4] < c[3] - 1e-9)
    say(f"  E kept: {len(e_rows)}; samples unchanged for every relevant base person: {n_unchanged}/{len(e_rows)}; "
        f"person-cases with a lower pixel share: {n_fell}")
    out.close()
    say("\nAggregates:")
    for kind in ("base", "variant"):
        n = agg[(kind, "n")]
        say(f"  {kind}s n={n}: " + ", ".join(f"{k} {agg[(kind, k)]}/{n}" for k in
                                              ("check_pass", "oracle_success", "regions_package_own", "persons_seen_match",
                                               "persons_rel_match", "added_non_member", "l2_empty")) +
            f", model calls {agg[(kind, 'calls')]}")
    (V.HERE / "out" / "sample.txt").write_text("\n".join(LINES) + "\n")
    (V.HERE / "out" / "e_visibility.json").write_text(json.dumps(e_rows, indent=1))
    print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
