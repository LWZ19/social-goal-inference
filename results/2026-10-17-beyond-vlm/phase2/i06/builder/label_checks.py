"""I06 step 4: label checks on every kept variant and every base (FROZEN.md sections 5, 9(d) and 11), and the manifest.

    label_checks.py rows SEED   # per kept scene of the seed: build/labels_rows_seed<SEED>.jsonl
    label_checks.py summary     # build/labels.txt, build/manifest.json, build/manifest.jsonl, build/counts.txt

Per kept scene (the 30 bases and every accepted variant):
- scene check: sgi.check_scene on the final folder, regions written: no error and no warning but check 9 (the
  generator's own acceptance rule, generate._rejection);
- full oracle (arm 1: grounding oracle, context_check always_yes, goal oracle, the mock entries, no cache, no call):
  run.run_episode on both episodes, success under L1 (the package scorer) for both;
- every episode has a non-empty acceptable region, and its L1 cells (0.05 m lattice points in the polygons on a free
  cell of the scorer's robot grid joined to the robot) are non-empty;
- every added person is a non-member (not in group_at_target, no role); no held-out situation anywhere;
- L1 invariance: every variant's acceptable polygons and headings equal its base's (representative points may move);
- L2 (the probe's pose-level rule, section 5): of e0's L1 cells, those with a body gap >= personal_gap_m to every
  non-member (centre distance - person radius - robot radius) and outside every other group's circle (members' mean,
  largest member distance); L2-empty when none is left;
- for E: every relevant person's visible fraction (samples seen of 9) and pixel share, in the variant and its base.
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys
import time

import numpy as np

from sgi.check_scene import check_scene
from sgi.config import load_config
from sgi.freespace import connected, robot_grid
from sgi.gt.derive import prepare
from sgi.pipeline import build_pipeline
from sgi.polygons import grid_points_in
from sgi.run import run_episode
from sgi.scene import load_scene

import crowd as C

FROZEN_SHA = "a83c0ffe0cf5fafd9f007fee478d5899c9f858d2eb77629f01885cbeccef693e"
HELD_OUT_ROWS = re.compile(r"eval_(1[6-9]|2[01])(?![0-9])")


def oracle_config(cfg):
    run = cfg.run.model_copy(update={"grounding": "oracle", "context_check": "always_yes", "goal": "oracle",
                                     "use_cache": False, "trials": 1})
    return cfg.model_copy(update={"run": run})


def l1_cells(truth, observation, regions, th) -> np.ndarray:
    grid = robot_grid(truth, observation.robot, th)
    joined = connected(grid, (observation.robot.x, observation.robot.y))
    cells = set()
    for region in regions:
        for x, y in grid_points_in(region.polygon, th.grid_resolution_m):
            index = grid.index(float(x), float(y))
            if index is not None and joined[index]:
                cells.add((round(float(x), 9), round(float(y), 9)))
    return np.array(sorted(cells), dtype=float).reshape(-1, 2)


def l2_mask(cells: np.ndarray, truth, group_id: str, th) -> np.ndarray:
    keep = np.ones(len(cells), dtype=bool)
    if not len(cells):
        return keep
    reach = th.person_radius_m + th.robot_radius_m + th.personal_gap_m          # 0.85 m centre distance
    for person in truth.people:
        if person.group != group_id:
            keep &= np.hypot(cells[:, 0] - person.x, cells[:, 1] - person.y) - th.person_radius_m - th.robot_radius_m \
                >= th.personal_gap_m
    groups = collections.defaultdict(list)
    for person in truth.people:
        if person.group is not None and person.group != group_id:
            groups[person.group].append(person)
    circles = []
    for gid, members in groups.items():
        if len(members) >= 2:
            cx, cy = np.mean([p.x for p in members]), np.mean([p.y for p in members])
            radius = max(math.hypot(p.x - cx, p.y - cy) for p in members)
            circles.append((gid, float(cx), float(cy), float(radius)))
            keep &= np.hypot(cells[:, 0] - cx, cells[:, 1] - cy) >= radius
    return keep


def polygons_of(truth, eid):
    return [(list(map(tuple, r.polygon)), r.heading.model_dump()) for r in truth.episodes[eid].acceptable_regions or []]


def rows(seed: int) -> None:
    cfg = load_config()
    th = cfg.thresholds
    ocfg = oracle_config(cfg)
    pipeline = build_pipeline(ocfg)
    entries = [json.loads(l) for l in (C.BUILD / f"bases_seed{seed}.jsonl").read_text().splitlines() if l]
    entries += [json.loads(l) for l in (C.BUILD / f"variants_seed{seed}.jsonl").read_text().splitlines() if l]
    base_truths = {}
    out = C.BUILD / f"labels_rows_seed{seed}.jsonl"
    started = time.perf_counter()
    with open(out, "w", encoding="utf-8") as f:
        for entry in entries:
            if not entry["accepted"]:
                continue
            folder = C.I06 / entry["folder"]
            row = {"scene_id": entry["scene_id"], "kind": entry["kind"], "seed": seed, "row": entry["row"],
                   "family": entry["family"], "level": entry["level"], "base_scene_id": entry["base_scene_id"]}
            # files on disk are those recorded
            row["sha256_match"] = C.folder_hashes(folder) == entry["sha256"]
            results = check_scene(folder, cfg, write_png=False)
            bad = [f"check_{r.number}:{r.level}" for r in results
                   if r.level == "error" or (r.level == "warning" and r.number != G_EXPECTED)]
            row["scene_check_pass"] = not bad
            row["scene_check_findings"] = bad
            row["scene_check_warnings_9"] = [r.number for r in results if r.level == "warning" and r.number == G_EXPECTED]
            inputs, truth = load_scene(folder)
            ets = prepare(inputs, truth, ocfg)
            gid = truth.episodes["e0"].group_at_target
            # full oracle, both episodes
            oracle = {}
            for episode in inputs.episodes:
                record = run_episode(pipeline, inputs, truth, ets[episode.episode_id], episode, ocfg, 0)
                calls = len(record["result"].get("calls") or [])
                oracle[episode.episode_id] = {"success": bool(record["score"]["success"]), "calls": calls,
                                              "pose": record["result"].get("pose_world"),
                                              "status": record["result"].get("status")}
            row["full_oracle"] = oracle
            row["full_oracle_success"] = all(o["success"] for o in oracle.values())
            row["model_calls"] = sum(o["calls"] for o in oracle.values())
            # regions non-empty and reachable
            region_rows = {}
            for eid in truth.episodes:
                regions = ets[eid].gt.acceptable_regions or []
                cells = l1_cells(truth, inputs.observation, regions, th)
                region_rows[eid] = {"regions": len(regions), "l1_cells": int(len(cells)),
                                    "representative_points": [list(r.representative_point) for r in regions]}
            row["regions"] = region_rows
            row["regions_nonempty_reachable"] = all(r["regions"] > 0 and r["l1_cells"] > 0 for r in region_rows.values())
            # situations
            situations = sorted({ep.situation for ep in truth.episodes.values()})
            row["situations"] = situations
            row["no_held_out"] = (all(s in cfg.dev_situations and s not in cfg.held_out_situations for s in situations)
                                  and not HELD_OUT_ROWS.search(entry["scene_id"])
                                  and entry["row_spec"][0] not in C.HELD_OUT_RECIPES)
            # added people are non-members
            added_ids = [a["id"] for a in entry.get("added", [])]
            on_disk = {p.id: p for p in truth.people}
            row["added_non_members"] = all(on_disk[i].group != gid and on_disk[i].role is None for i in added_ids)
            row["n_added"] = len(added_ids)
            row["n_people"] = len(truth.people)
            # L1 invariance against the base
            if entry["kind"] == "base":
                base_truths[entry["scene_id"]] = truth
                row["l1_invariant"] = True
                row["rep_point_moved_m"] = 0.0
            else:
                btruth = base_truths[entry["base_scene_id"]]
                row["l1_invariant"] = all(polygons_of(truth, eid) == polygons_of(btruth, eid) for eid in truth.episodes)
                mine = truth.episodes["e0"].acceptable_regions[0].representative_point
                theirs = btruth.episodes["e0"].acceptable_regions[0].representative_point
                row["rep_point_moved_m"] = math.dist(mine, theirs)
            # L2 cells on e0
            cells = l1_cells(truth, inputs.observation, ets["e0"].gt.acceptable_regions, th)
            keep = l2_mask(cells, truth, gid, th)
            row["l1_cells_e0"] = int(len(cells))
            row["l2_cells_e0"] = int(keep.sum())
            row["l2_empty"] = bool(len(cells) > 0 and keep.sum() == 0)
            # episodes agree on regions (one target, one region set)
            row["episodes_same_regions"] = polygons_of(truth, "e0") == polygons_of(truth, "e1")
            # E: visible fractions recorded
            if entry["family"] == "E":
                persons = entry.get("persons") or {}
                relevant = [pid for pid, p in persons.items() if p["relevant_e0"] or p["relevant_e1"]]
                row["e_visible_recorded"] = bool(relevant) and all(
                    "seen_of_9" in persons[pid] and "pixel_share" in persons[pid] for pid in relevant)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"seed {seed} {entry['scene_id']}: scene check {row['scene_check_pass']}, full oracle "
                  f"{row['full_oracle_success']}, regions {row['regions_nonempty_reachable']}, L1 invariant "
                  f"{row['l1_invariant']}, L1 cells {row['l1_cells_e0']}, L2 cells {row['l2_cells_e0']} "
                  f"[{time.perf_counter() - started:.0f} s]", flush=True)


G_EXPECTED = 9


def pct(a: int, b: int) -> str:
    return f"{a}/{b} = {100.0 * a / b:.1f}%" if b else f"{a}/0"


def summary() -> int:
    entries, label_rows = [], {}
    for seed in (0, 1, 2):
        entries += [json.loads(l) for l in (C.BUILD / f"bases_seed{seed}.jsonl").read_text().splitlines() if l]
        entries += [json.loads(l) for l in (C.BUILD / f"variants_seed{seed}.jsonl").read_text().splitlines() if l]
        for l in (C.BUILD / f"labels_rows_seed{seed}.jsonl").read_text().splitlines():
            if l:
                r = json.loads(l)
                label_rows[r["scene_id"]] = r
    base_tables = {}
    for seed in (0, 1, 2):
        p = C.BUILD / f"base_tables_seed{seed}.json"
        if p.exists():
            base_tables.update({(seed, k): v for k, v in json.loads(p.read_text()).items()})
    variants = [e for e in entries if e["kind"] == "variant"]
    bases = [e for e in entries if e["kind"] == "base"]
    kept = [e for e in variants if e["accepted"]]
    lines = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    say(f"I06 build: label checks (FROZEN.md sha256 {FROZEN_SHA[:12]}...; sections 5, 9(d), 11 step 4)")
    say(f"planned variants {len(variants)} (270 expected); kept {len(kept)}; rejected {len(variants) - len(kept)}; "
        f"bases {len(bases)}")
    say("")
    # counts by family and level
    say("Counts by family and level (n planned = base layouts the family applies to, over seeds 0-2):")
    say(f"  {'fam':4s}{'lvl':>4s}{'planned':>9s}{'kept':>6s}{'1st-att':>9s}{'no_spot':>9s}{'scene-chk':>11s}   rejection reasons (last attempt)")
    by = collections.defaultdict(list)
    for e in variants:
        by[(e["family"], e["level"])].append(e)
    family_totals = collections.defaultdict(lambda: [0, 0])
    for key in sorted(by):
        es = by[key]
        k = sum(e["accepted"] for e in es)
        first = sum(e["accepted"] and e["attempts"] == 1 for e in es)
        nospot = sum(e["rejected_reason"] == "no_spot" for e in es)
        sc = sum((not e["accepted"]) and e["rejected_reason"] != "no_spot" for e in es)
        reasons = collections.Counter(e["reasons"][-1] for e in es if not e["accepted"])
        family_totals[key[0]][0] += len(es)
        family_totals[key[0]][1] += k
        say(f"  {key[0]:4s}{key[1]:>4d}{len(es):>9d}{k:>6d}{first:>9d}{nospot:>9d}{sc:>11d}   {dict(reasons) or '-'}")
    say("  by family: " + "; ".join(f"{f} {v[1]}/{v[0]}" for f, v in sorted(family_totals.items())))
    attempts_all = collections.Counter(r for e in variants for r in e["reasons"])
    say(f"  every attempt's outcome, all variants: {dict(attempts_all)}")
    say("")
    # criterion (d)
    accepted, planned = len(kept), 270
    first_att = sum(e["accepted"] and e["attempts"] == 1 for e in variants)
    say("Criterion (d): 'At least 70% of variants pass check_scene, and full oracle scores 100%.'")
    say(f"  pass rate = accepted / planned = {pct(accepted, planned)} (no_spot counted as a rejection); "
        f"first-attempt pass rate {pct(first_att, planned)}")
    scenes = [label_rows.get(e["scene_id"]) for e in [*bases, *kept]]
    missing = [e["scene_id"] for e, r in zip([*bases, *kept], scenes) if r is None]
    rows_ok = [r for r in scenes if r is not None]
    fo_v = sum(r["full_oracle_success"] for r in rows_ok if r["kind"] == "variant")
    fo_b = sum(r["full_oracle_success"] for r in rows_ok if r["kind"] == "base")
    say(f"  full oracle (arm 1) success under L1, both episodes: variants {pct(fo_v, len(kept))}, bases "
        f"{pct(fo_b, len(bases))}; model calls made {sum(r['model_calls'] for r in rows_ok)}")
    d_met = accepted / planned >= 0.70 and fo_v == len(kept) and fo_b == len(bases) and not missing
    say(f"  criterion (d): {'MET' if d_met else 'NOT MET'}")
    say(f"  K3 reading ('more than 50% of layouts rejected', read as variants): rejected "
        f"{pct(planned - accepted, planned)} -> {'fires' if (planned - accepted) / planned > 0.5 else 'does not fire'}")
    say("")
    # label checks
    say(f"Label checks on every kept scene ({len(kept)} variants + {len(bases)} bases; rows missing: {missing or 'none'}):")
    checks = [("scene check passes (no error; no warning but 9)", "scene_check_pass"),
              ("full oracle succeeds on both episodes (L1)", "full_oracle_success"),
              ("every episode's acceptable region non-empty and reachable (L1 cells > 0)", "regions_nonempty_reachable"),
              ("every added person is a non-member (no role)", "added_non_members"),
              ("no held-out situation, row or scene", "no_held_out"),
              ("L1 invariance: polygons and headings equal the base's", "l1_invariant"),
              ("e0 and e1 have the same regions", "episodes_same_regions"),
              ("files on disk match the manifest sha256", "sha256_match")]
    results = {}
    for text, key in checks:
        good = sum(bool(r[key]) for r in rows_ok)
        results[key] = good == len(rows_ok)
        say(f"  {text}: {pct(good, len(rows_ok))}")
        if good != len(rows_ok):
            for r in rows_ok:
                if not r[key]:
                    say(f"      fails: {r['scene_id']} {r.get('scene_check_findings', '')}")
    e_rows = [r for r in rows_ok if r["family"] == "E"]
    e_ok = sum(bool(r.get("e_visible_recorded")) for r in e_rows)
    results["e_visible_recorded"] = e_ok == len(e_rows)
    say(f"  E: every relevant person's visible fraction and pixel share recorded: {pct(e_ok, len(e_rows))}")
    l2_empty_v = [r for r in rows_ok if r["kind"] == "variant" and r["l2_empty"]]
    l2_empty_b = [r for r in rows_ok if r["kind"] == "base" and r["l2_empty"]]
    say(f"  L2-empty (e0's L1 cells, none meeting L2): variants {pct(len(l2_empty_v), len(kept))}, bases "
        f"{pct(len(l2_empty_b), len(bases))} (descriptive here; criterion (c) and K2 are judged by the scoring step)")
    for r in l2_empty_v + l2_empty_b:
        say(f"      L2-empty: {r['scene_id']} (L1 cells {r['l1_cells_e0']})")
    moved = [r["rep_point_moved_m"] for r in rows_ok if r["kind"] == "variant"]
    say(f"  representative point moved from the base's (allowed): {sum(m > 1e-9 for m in moved)} of {len(moved)} variants; "
        f"max {max(moved, default=0):.3f} m")
    say("")
    # E visibility summary
    say("E: relevant people's visible fraction (samples seen of 9) and pixel share, variant vs base:")
    for e in [x for x in kept if x["family"] == "E"]:
        bt = base_tables.get((e["seed"], e["base_scene_id"]), {})
        parts = []
        for pid, p in sorted(e["persons"].items()):
            if not (p["relevant_e0"] or p["relevant_e1"]):
                continue
            b = bt.get(pid)
            added = pid in {a["id"] for a in e["added"]}
            if b is None:
                parts.append(f"{pid}(added) {p['seen_of_9']}/9 share {p.get('pixel_share', float('nan')):.2f}")
            else:
                parts.append(f"{pid} {p['seen_of_9']}/9 (base {b['seen_of_9']}/9) share {p.get('pixel_share', float('nan')):.2f} "
                             f"(base {b.get('pixel_share', float('nan')):.2f})")
            del added
        say(f"  {e['scene_id']}: " + "; ".join(parts))
    say("")
    # people under 9 of 9 in other families
    partial = collections.Counter()
    for e in kept:
        if e["family"] == "E":
            continue
        for pid, p in e["persons"].items():
            if p["seen_of_9"] < 9 and (p["relevant_e0"] or p["relevant_e1"]):
                partial[e["family"]] += 1
    say(f"Relevant people seen at under 9 of 9 samples in kept A-D variants (pixel share recorded): {dict(partial) or 'none'}")
    all_ok = all(results.values()) and not missing
    say("")
    say(f"RESULT: label checks {'ALL PASS' if all_ok else 'SOME FAIL'}; criterion (d) {'MET' if d_met else 'NOT MET'}")
    (C.BUILD / "labels.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # the manifest
    for e in entries:
        r = label_rows.get(e["scene_id"])
        if r is not None:
            e["labels"] = {k: r[k] for k in ("scene_check_pass", "full_oracle_success", "regions_nonempty_reachable",
                                             "added_non_members", "no_held_out", "l1_invariant", "l1_cells_e0",
                                             "l2_cells_e0", "l2_empty", "rep_point_moved_m")}
        if e["kind"] == "base":
            e["persons"] = base_tables.get((e["seed"], e["scene_id"]))
    counts = {f"{f}{l}": {"planned": len(es), "kept": sum(x["accepted"] for x in es),
                          "first_attempt": sum(x["accepted"] and x["attempts"] == 1 for x in es),
                          "no_spot": sum(x["rejected_reason"] == "no_spot" for x in es),
                          "scene_check_rejected": sum((not x["accepted"]) and x["rejected_reason"] != "no_spot" for x in es)}
              for (f, l), es in sorted(by.items())}
    doc = {"what": "I06 crowd variants (families A-E) and their 30 base layouts, seeds 0-2",
           "frozen_note": "phase2/i06/FROZEN.md", "frozen_sha256": FROZEN_SHA,
           "streams": "variant: numpy default_rng([seed, 2, family code (A1 B2 C3 D4 E5), row index, level]); "
                      "base: default_rng(seed) (eval_06..13) or default_rng([seed, 1]) (eval_14..15)",
           "planned_variants": len(variants), "kept_variants": len(kept), "bases": len(bases),
           "criterion_d": {"pass_rate": accepted / planned, "first_attempt_rate": first_att / planned,
                           "full_oracle_variants": f"{fo_v}/{len(kept)}", "full_oracle_bases": f"{fo_b}/{len(bases)}",
                           "met": d_met},
           "counts_by_family_level": counts, "entries": entries}
    (C.BUILD / "manifest.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    with open(C.BUILD / "manifest.jsonl", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "rows":
        rows(int(sys.argv[2]))
    else:
        raise SystemExit(summary())
