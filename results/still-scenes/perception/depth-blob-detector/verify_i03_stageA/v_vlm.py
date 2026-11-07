"""Verifier: VLM people lists scored like for like (FROZEN.md section 6) with the verifier's own scorer, plus a check
of the verifier's matching against the recorded measures (people_found_share x visible, people_unmatched) of every
recorded record with a people list. Blob rows come from the verifier's own eval_rows_off0.000.jsonl."""
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_harness as H  # noqa: E402


def here(stored) -> str:
    """A scene folder as a stored file names it (the recorded absolute path without its local root), as it lies
    here: through the harness's relocated() and from the repository root."""
    return str((H.main_checkout() / H.relocated(stored)).resolve())
import v_core as C  # noqa: E402

HERE = Path(__file__).resolve().parent
I03 = HERE.parent
MODELS = [("Qwen 3.8", "s19", "qwen38_low", ("real", "staffed", "plain", "ph", "dev_seed0", "dev_seed1", "dev_seed2")),
          ("GLM-5.3", "s19", "glm53", ("real", "staffed", "plain", "ph", "dev_seed0", "dev_seed1", "dev_seed2")),
          ("DeepSeek V4.1", "s20", "dsv41", ("real", "staffed", "plain", "ph", "dev_seed0", "dev_seed1", "dev_seed2")),
          ("Haiku 5.5", "s23", "haiku55", ("real", "staffed", "plain")),
          ("Sonnet 5.5 low", "s23", "sonnet55_low", ("real", "staffed", "plain")),
          ("Gemini 3.8 low", "s23", "gemini38_low", ("real", "staffed", "plain"))]
blob = {}
for r in map(json.loads, open(HERE / "eval_rows_off0.000.jsonl")):
    blob[here(r["folder"])] = r


def main():
    rows = []
    mcheck = collections.Counter()
    mism = []
    for name, stage, model, arms in MODELS:
        for arm in arms:
            runs = H.runs(stage=stage, arm=arm, model=model, method="con")
            assert len(runs) == 1, (name, arm, runs)
            run = runs[0]
            th = H.recorded_config(run)[1].thresholds
            keys = collections.OrderedDict()
            for rec in H.records(run):
                keys.setdefault((rec.scene_id, rec.trial), []).append(rec)
            for (sid, trial), recs in keys.items():
                sc = H.scene(run, sid)
                folder = str((run.scenes_dir / sid).resolve())
                T = C.Truth(sc.observation, sc.truth, th)
                lists = [r.people for r in recs if r.people is not None]
                # matching check per record against the recorded measures
                for r in recs:
                    if r.people is None:
                        continue
                    ms = r.measures
                    dets = [(p["xy"][0], p["xy"][1], p.get("yaw")) for p in sorted(r.people, key=lambda p: p["number"])]
                    # use the record's own numbers for tie-breaks
                    m = C.match(dets, T.vis)
                    share, unm = ms.get("people_found_share"), ms.get("people_unmatched")
                    if share is None and unm is None:
                        mcheck["no measure"] += 1
                        continue
                    ok1 = share is None or (len(T.vis) and abs(share * len(T.vis) - len(m)) < 1e-6) or (not T.vis and len(m) == 0)
                    ok2 = unm is None or unm == len(dets) - len(m)
                    mcheck["agree" if ok1 and ok2 else "differ"] += 1
                    if not (ok1 and ok2):
                        mism.append((run.folder, sid, r.episode_id, trial, share, unm, len(m), len(T.vis), len(dets)))
                if not lists:
                    rows.append({"model": name, "arm": arm, "folder": folder, "scene": sid, "trial": trial, "has": False,
                                 "vis": len(T.vis)})
                    continue
                seen, union = [], []
                for lst in lists:
                    for p in lst:
                        k = (p["xy"][0], p["xy"][1], p["pixel"][0], p["pixel"][1])
                        if any(abs(k[0] - q[0]) <= 1e-9 and abs(k[1] - q[1]) <= 1e-9 and abs(k[2] - q[2]) <= 1e-6 and
                               abs(k[3] - q[3]) <= 1e-6 for q in seen):
                            continue
                        seen.append(k)
                        union.append((k, p.get("yaw")))
                union.sort(key=lambda t: t[0])
                dets = [(k[0], k[1], yaw) for k, yaw in union]
                r = C.score(dets, T)
                row = {k: r[k] for k in C.SUMK}
                row.update(model=name, arm=arm, folder=folder, scene=sid, trial=trial, has=True, errs=r["errs"],
                           n_lists=len(lists), misses=r["misses"], people=r["people"])
                rows.append(row)
    with open(HERE / "vlm_rows_verify.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("matching check vs recorded measures:", dict(mcheck))
    for x in mism[:10]:
        print("  MISMATCH", x)
    # tables
    for name, *_ in MODELS:
        for dom, arms in (("E-real", ("real", "staffed", "plain")), ("E-ph", ("ph", "dev_seed0", "dev_seed1", "dev_seed2"))):
            rs = [r for r in rows if r["model"] == name and r["arm"] in arms]
            if not rs:
                continue
            has = [r for r in rs if r["has"]]
            P = C.pool(has)
            nolist = len(rs) - len(has)
            lo, hi = C.wilson(P["matched"], P["vis"])
            Bp = C.pool([blob[r["folder"]] for r in has])
            vis_all = P["vis"] + sum(r["vis"] for r in rs if not r["has"])
            print(f"{name:15s} {dom:6s} keys {len(rs)} (no list {nolist}) vis {P['vis']} recall {P['matched']}/{P['vis']} = "
                  f"{P['recall']:.3f} [{lo:.3f}, {hi:.3f}] false/answer {P['false'] / len(has):.3f} (false {P['false']}; raw "
                  f"{P['false'] + P['exempt']}) mp {P['mpairs']}/{P['npairs']} med {P['med_err']:.3f} full {P['w45_full']}/"
                  f"{P['known']} unk {P['unknown']} axis {P['w45_axis']}/{P['known']} || blobs {Bp['matched']}/{Bp['vis']} = "
                  f"{Bp['recall']:.3f} false/key {Bp['false'] / len(has):.3f} mp {Bp['mpairs']}/{Bp['npairs']} axis "
                  f"{Bp['w45_axis']}/{Bp['known']} || sens recall {P['matched']}/{vis_all} = {P['matched'] / vis_all:.3f} "
                  f"false/answer {P['false'] / len(rs):.3f}")
    nl = [r for r in rows if not r["has"]]
    print("keys", len(rows), "without list", len(nl), "situations of no-list keys",
          collections.Counter(r["scene"].split("_", 3)[-1] if False else ("no_one" in r["scene"]) for r in nl))
    # compare with the compute agent's vlm_rows.jsonl (as data)
    theirs = {}
    for t in map(json.loads, open(I03 / "stageA" / "vlm_rows.jsonl")):
        theirs[(t["model"], t["arm"], here(t["folder"]), t["trial"])] = t
    nd = 0
    for r in rows:
        t = theirs.get((r["model"], r["arm"], r["folder"], r["trial"]))
        if t is None:
            nd += 1
            print("MISSING in theirs", r["model"], r["arm"], r["scene"], r["trial"])
            continue
        if not r["has"]:
            if t["has_list"]:
                nd += 1
                print("DIFF has_list", r["model"], r["scene"], r["trial"])
            continue
        a = (r["vis"], r["matched"], r["false"], r["exempt"], r["merges"], r["npairs"], r["mpairs"], r["n_det"],
             r["known"], r["unknown"], r["w45_full"], r["w45_axis"])
        b = (t["n_visible"], t["matched"], t["false"], t["exempt"], t["merges"], t["neighbour_pairs"], t["merged_pairs"],
             t["n_det"], t["facing"]["known"], t["facing"]["unknown"], t["facing"]["within45_full"], t["facing"]["within45_axis"])
        if a != b:
            nd += 1
            if nd <= 15:
                print("DIFF", r["model"], r["arm"], r["scene"], r["trial"], a, b)
    print(f"vlm_rows comparison: {nd} of {len(rows)} keys differ; theirs has {len(theirs)} keys")


if __name__ == "__main__":
    main()
