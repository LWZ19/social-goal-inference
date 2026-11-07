"""Independent verifier, I06 build: recount the manifest, check folders and hashes, held-out and seed compliance,
the timeline, and the descriptive numbers the build reports. Light (reads JSON and hashes files). Writes
out/counts.txt and out/counts.json."""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path

from sgi.config import load_config

import vlib as V

I06 = V.I06
BUILD = I06 / "build"
LINES = []
OUT = {}


def say(t=""):
    print(t, flush=True)
    LINES.append(t)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mtime(p):
    return dt.datetime.fromtimestamp(os.stat(p).st_mtime).isoformat(timespec="seconds")


def main():
    cfg = load_config()
    es = [json.loads(l) for l in (BUILD / "manifest.jsonl").read_text().splitlines() if l]
    variants = [e for e in es if e["kind"] == "variant"]
    bases = [e for e in es if e["kind"] == "base"]
    say(f"manifest.jsonl: {len(es)} entries: {len(variants)} variants, {len(bases)} bases")
    # manifest.json agrees with manifest.jsonl?
    doc = json.loads((BUILD / "manifest.json").read_text())
    say(f"manifest.json entries equal manifest.jsonl: {doc['entries'] == es}; its counts: kept {doc['kept_variants']}, "
        f"planned {doc['planned_variants']}")
    # planned set
    plan = set(V.planned())
    got = {(e["seed"], e["row"], e["family"], e["level"]) for e in variants}
    say(f"planned set from FROZEN (A on lines 6,7,8,9,13 x 6 levels; B, C 1; D 2, 5; E 1, 2; rows 6-15; seeds 0-2): "
        f"{len(plan)}; manifest set equal: {plan == got}; duplicates: {len(variants) - len(got)}")
    # streams
    st_ok = all(e["stream"] == [e["seed"], 2, V.FAM_CODE[e["family"]], e["row"], e["level"]] for e in variants)
    say(f"every variant stream == [seed, 2, family code, row, level]: {st_ok}; distinct: "
        f"{len({tuple(e['stream']) for e in variants}) == len(variants)}")
    say(f"base streams: {collections.Counter((e['row'] <= 13, e['stream']) for e in bases)}")
    say(f"seeds in manifest: {sorted({e['seed'] for e in es})}")
    # recount
    by = collections.defaultdict(list)
    for e in variants:
        by[(e["family"], e["level"])].append(e)
    say("\nRecount by family/level: planned, kept, kept on attempt 1, no_spot, scene-check rejected (reasons):")
    tot = collections.Counter()
    table = {}
    for k in sorted(by):
        g = by[k]
        kept = sum(e["accepted"] for e in g)
        first = sum(e["accepted"] and e["attempts"] == 1 and e["reasons"] == ["accepted"] for e in g)
        ns = sum((not e["accepted"]) and e["reasons"] == ["no_spot"] for e in g)
        sc = [e for e in g if not e["accepted"] and e["reasons"] != ["no_spot"]]
        scr = collections.Counter(r for e in sc for r in e["reasons"])
        table[f"{k[0]}{k[1]}"] = {"planned": len(g), "kept": kept, "first": first, "no_spot": ns, "scene_check": len(sc)}
        say(f"  {k[0]}{k[1]}: {len(g)} planned, {kept} kept, {first} first-attempt, {ns} no_spot, {len(sc)} scene-check "
            f"rejected {dict(scr) or ''} (attempts {[e['attempts'] for e in sc] or ''})")
        tot.update({"planned": len(g), "kept": kept, "first": first, "no_spot": ns, "scene_check": len(sc)})
    say(f"  total: {dict(tot)}; pass rate {tot['kept']}/{tot['planned']} = {100 * tot['kept'] / tot['planned']:.2f}%; "
        f"rejected {tot['planned'] - tot['kept']} = {100 * (tot['planned'] - tot['kept']) / tot['planned']:.2f}%")
    say(f"  attempts outcome counter: {dict(collections.Counter(r for e in variants for r in e['reasons']))}")
    weird = [e["scene_id"] for e in variants if e["accepted"] != (e["reasons"][-1] == "accepted")
             or (e["rejected_reason"] == "no_spot") != (e["reasons"][-1] == "no_spot")]
    say(f"  entries whose accepted flag / reason disagree: {weird or 'none'}")
    OUT["table"] = table
    fam = collections.defaultdict(lambda: [0, 0])
    for (f, l), g in by.items():
        fam[f][0] += sum(e["accepted"] for e in g)
        fam[f][1] += len(g)
    say("  by family: " + "; ".join(f"{f} {a}/{b}" for f, (a, b) in sorted(fam.items())))
    # by row
    say("\nKept/planned by base row and family-level (seeds pooled):")
    rows = sorted({e["row"] for e in variants})
    keys = sorted(by)
    for r in rows:
        cells = []
        for k in keys:
            g = [e for e in by[k] if e["row"] == r]
            cells.append(f"{k[0]}{k[1]} {sum(e['accepted'] for e in g)}/{len(g)}" if g else f"{k[0]}{k[1]} -")
        say(f"  eval_{r:02d} {variants[[e['row'] for e in variants].index(r)]['row_spec'][0][:12]:12s} " + "  ".join(cells))
    # folders on disk
    say("\nScene folders on disk vs manifest:")
    allok = True
    for seed in (0, 1, 2):
        disk = sorted(p.name for p in (I06 / "scenes" / f"seed{seed}").iterdir())
        want = sorted([e["scene_id"] for e in es if e["seed"] == seed and e["accepted"]])
        ok = disk == want
        allok &= ok
        say(f"  seed {seed}: {len(disk)} folders; equal to the kept variants + bases of the manifest: {ok}")
    nbad = 0
    nfiles = 0
    for e in es:
        if not e["accepted"]:
            continue
        folder = I06 / e["folder"]
        files = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
        nfiles += len(files)
        if files != e["sha256"]:
            nbad += 1
            say(f"    sha mismatch: {e['seed']} {e['scene_id']}")
    say(f"  sha256 of every file of the 229 kept folders vs manifest: {nfiles} files, {nbad} folders differ")
    # held-out and situations
    say("\nHeld-out rows, scenes and situations:")
    held = re.compile(r"eval_(1[6-9]|2[01])(?![0-9])")
    say(f"  scene ids matching eval_16..21: {[e['scene_id'] for e in es if held.search(e['scene_id'])] or 'none'}")
    say(f"  row recipes: {sorted({e['row_spec'][0] for e in es})}")
    sits = collections.Counter()
    for e in es:
        if not e["accepted"]:
            continue
        js = json.loads((I06 / e["folder"] / "scene.json").read_text())
        for ep in js["episodes"]:
            sits[ep["ground_truth"]["situation"]] += 1
    say(f"  situations in the 229 kept scene.json (episodes): {dict(sits)}; dev situations {cfg.dev_situations}; "
        f"held-out {cfg.held_out_situations}; any held-out: {any(s in cfg.held_out_situations for s in sits)}")
    # grep everything the build wrote for references to reserved seeds / held-out material
    say("\nText search of the build's own files (builder/, build/*.txt|*.jsonl|*.json|logs, precheck/, DEVIATIONS.md):")
    pats = {"held-out names": re.compile(r"group_photo|lift_wait|keep/heldout|s20_held|s21_replay|s22_replay|s21_sonnet|pf_"),
            "eval_16..21": held,
            "seed 3-219 in a seed position": re.compile(r"(?:seed\s*|default_rng\(\s*\[?\s*|seed=)(?:[3-9]|[1-9][0-9]|1[0-9][0-9]|2[01][0-9])\b"),
            "s18 run": re.compile(r"(?<![0-9a-z])s18_")}
    files = [*sorted((I06 / "builder").glob("*.py")), *sorted((I06 / "builder").glob("*.sh")), *sorted((I06 / "builder").glob("*.txt")),
             *sorted(BUILD.glob("*.txt")), *sorted((BUILD / "logs").glob("*.log")), *sorted((I06 / "precheck").glob("*")),
             I06 / "DEVIATIONS.md", *sorted(BUILD.glob("*.jsonl")), BUILD / "manifest.json", *sorted(BUILD.glob("base_tables_*.json"))]
    hits = collections.defaultdict(list)
    for f in files:
        if f.is_dir():
            continue
        text = f.read_text(errors="replace")
        for name, rx in pats.items():
            for m in rx.finditer(text):
                s = max(0, m.start() - 60)
                hits[name].append(f"{f.relative_to(I06)}: ...{text[s:m.end() + 40].replace(chr(10), ' ')}...")
    for name in pats:
        say(f"  {name}: {len(hits[name])} hit(s)")
        for h in hits[name][:12]:
            say(f"      {h}")
    # timeline
    say("\nTimeline (file modification times, local):")
    t_frozen = os.stat(I06 / "FROZEN.md").st_mtime
    say(f"  FROZEN.md {mtime(I06 / 'FROZEN.md')}; sha256 {sha(I06 / 'FROZEN.md')}; FROZEN.sha256 says "
        f"{(I06 / 'FROZEN.sha256').read_text().split()[0]}")
    say(f"  DEVIATIONS.md {mtime(I06 / 'DEVIATIONS.md')}; sha256 {sha(I06 / 'DEVIATIONS.md')}; DEVIATIONS.sha256 says "
        f"{(I06 / 'DEVIATIONS.sha256').read_text().split()[0]}")
    scene_files = [p for p in (I06 / "scenes").rglob("*") if p.is_file()]
    first_scene = min(scene_files, key=lambda p: os.stat(p).st_mtime)
    var_files = [p for p in scene_files if p.parent.name.startswith("i06_")]
    first_var = min(var_files, key=lambda p: os.stat(p).st_mtime)
    bi_files = [p for p in (BUILD / "byte_identity").rglob("*") if p.is_file()]
    first_bi = min(bi_files, key=lambda p: os.stat(p).st_mtime)
    say(f"  first file under scenes/: {mtime(first_scene)} ({first_scene.relative_to(I06)})")
    say(f"  first variant file under scenes/: {mtime(first_var)} ({first_var.relative_to(I06)})")
    say(f"  first file under build/byte_identity: {mtime(first_bi)} ({first_bi.relative_to(I06)})")
    for name in ("base_check.txt", "self_test.txt", "byte_identity.txt", "readings.txt", "labels.txt"):
        say(f"  build/{name}: {mtime(BUILD / name)}")
    say(f"  builder/crowd.py last modified: {mtime(I06 / 'builder' / 'crowd.py')}")
    lab = sorted((BUILD / "logs").glob("labels_seed*.log"))
    say(f"  first label-check (full-oracle scoring) log line: {lab[0].read_text().splitlines()[0][:40]}")
    pre = [p for p in (I06 / "precheck").iterdir()]
    say(f"  precheck files (before FROZEN, disclosed in its section 0): {[(p.name, mtime(p)) for p in pre]}")
    before = [p for p in [*scene_files, *bi_files] if os.stat(p).st_mtime < t_frozen]
    say(f"  scene or byte-identity files older than FROZEN.md: {len(before)}")
    # descriptive numbers
    say("\nDescriptive numbers recomputed from the stored scene.json files (all kept variants):")
    nm = collections.defaultdict(list)
    moved, npeople = [], []
    for e in variants:
        if not e["accepted"]:
            continue
        js = json.loads((I06 / e["folder"] / "scene.json").read_text())
        bj = json.loads((I06 / "scenes" / f"seed{e['seed']}" / e["base_scene_id"] / "scene.json").read_text())
        ppl = js["ground_truth"]["people"]
        npeople.append(len(ppl))
        gid = js["episodes"][0]["ground_truth"]["group_at_target"]
        mem = [p for p in ppl if p["group"] == gid]
        for p in ppl[len(bj["ground_truth"]["people"]):]:
            nm[f"{e['family']}{e['level']}"].append(min(math.hypot(p["x"] - m["x"], p["y"] - m["y"]) for m in mem))
        r0 = js["episodes"][0]["ground_truth"]["acceptable_regions"][0]["representative_point"]
        b0 = bj["episodes"][0]["ground_truth"]["acceptable_regions"][0]["representative_point"]
        moved.append(math.dist(r0, b0))
    for k in sorted(nm):
        v = nm[k]
        say(f"  {k}: n {len(v)} people; nearest member min {min(v):.2f} median {statistics.median(v):.2f} max {max(v):.2f} m")
    say(f"  representative point moved (e0 region 0, variant vs build's base): {sum(m > 1e-9 for m in moved)} of "
        f"{len(moved)}; max {max(moved):.3f} m")
    say(f"  people per kept variant: min {min(npeople)} max {max(npeople)}")
    (V.HERE / "out" / "counts.txt").write_text("\n".join(LINES) + "\n")
    (V.HERE / "out" / "counts.json").write_text(json.dumps(OUT, indent=1))


if __name__ == "__main__":
    main()
