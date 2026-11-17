"""I03 Stage B step 1: the frozen Stage A kept blobs of every scene the replayed runs use, cached once per distinct
scene (FROZEN.md section 7, "Blobs are the frozen Stage A kept blobs of the record's scene").

The detector and setting are Stage A's, unchanged: blobs.detect with stageA/chosen.json's setting and drop_clip
(False: the one general fix was not adopted). Scenes are read only through research_harness (H.distinct_scenes,
H.scene). Each scene's kept blobs are checked against Stage A's rows.jsonl: the kept count, and the matched, false
and exempt counts when the blobs are scored again with the shared scorer under the same thresholds and exemption.

Writes blob_cache.json: {"meta": {...hashes...}, "scenes": {folder: {"scene_id", "listed": [[scenes_dir, id]...],
"blobs": [{"id", "x", "y", "top", "footprint", "cells", "axis_band"}...]}}}. No model, no network.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
I03 = HERE.parent
sys.path.insert(0, str(I03))
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402


def here(stored) -> str:
    """A scene folder as a stored file names it (the recorded absolute path without its local root), as it lies
    here: through the harness's relocated() and from the repository root."""
    return str((H.main_checkout() / H.relocated(stored)).resolve())
import blobs as B  # noqa: E402
import people_scorer as PS  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_setting():
    chosen_path = I03 / "stageA" / "chosen.json"
    chosen = json.loads(chosen_path.read_text())
    assert sha(I03 / "fit_selection.json") == (I03 / "FIT.sha256").read_text().split()[0]
    assert chosen["fit_selection_sha256"] == sha(I03 / "fit_selection.json")
    sel = json.loads((I03 / "fit_selection.json").read_text())
    assert chosen["setting"] == sel["chosen_setting"]
    fix = json.loads((I03 / "fit_selection_fix.json").read_text())
    assert chosen["detector"] == fix["union_choice"]["detector"] == "original"
    assert chosen["drop_clip"] is False
    for line in (I03 / "stageA" / "CHOSEN.sha256").read_text().splitlines():
        digest, name = line.split()
        if name == "chosen.json":
            assert digest == sha(chosen_path), "chosen.json does not match CHOSEN.sha256"
    return chosen


def main():
    t0 = time.monotonic()
    chosen = frozen_setting()
    s, drop_clip = chosen["setting"], bool(chosen["drop_clip"])
    stage_a = {}
    for line in (I03 / "stageA" / "rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        stage_a[here(r["folder"])] = r
    assert sha(I03 / "stageA" / "rows.jsonl") == (I03 / "stageA" / "ROWS.sha256").read_text().split()[0]
    distinct = H.distinct_scenes()
    assert len(distinct) == 68, len(distinct)
    out, mism = {}, []
    for sref in distinct:
        sc = H.scene(sref.runs[0], sref.scene_id)
        kb, _ = B.detect(sc.observation, s, drop_clip=drop_clip)
        blobs = [{"id": b["id"], "x": b["x"], "y": b["y"], "top": b["top"], "footprint": b["footprint"],
                  "cells": b["cells"], "axis_band": b.get("axis_band")} for b in kb]
        # consistency with Stage A's scored row (same thresholds rule: each listed run's recorded config)
        th = H.recorded_config(sref.runs[0])[1].thresholds
        r = PS.score_people([(b["x"], b["y"], None) for b in blobs], sc, th, exempt_xy=PS.exempt_points(sc.truth))
        a = stage_a[str(sref.folder)]
        got = (len(blobs), r["matched"], r["false"], r["exempt"], r["n_visible"])
        want = (a["kept"], a["matched"], a["false"], a["exempt"], a["n_visible"])
        if got != want:
            mism.append((str(sref.folder), got, want))
        out[str(sref.folder)] = {"scene_id": sref.scene_id,
                                 "listed": [[str(d), i] for d, i in sref.listed],
                                 "blobs": blobs}
        print(f"{sref.scene_id:36s} kept {len(blobs):2d} matched {r['matched']}/{r['n_visible']} false {r['false']}"
              f"{'' if got == want else '  MISMATCH ' + str(want)}", flush=True)
    meta = {"setting": s, "drop_clip": drop_clip, "top_hi": s["z_hi"] - 0.05,
            "chosen_json_sha256": sha(I03 / "stageA" / "chosen.json"),
            "fit_selection_sha256": sha(I03 / "fit_selection.json"),
            "blobs_py_sha256": sha(I03 / "blobs.py"), "people_scorer_py_sha256": sha(I03 / "people_scorer.py"),
            "stageA_rows_sha256": sha(I03 / "stageA" / "rows.jsonl"),
            "scenes": len(out), "blobs": sum(len(v["blobs"]) for v in out.values()),
            "stageA_mismatches": mism, "seconds": round(time.monotonic() - t0, 1)}
    (HERE / "blob_cache.json").write_text(json.dumps({"meta": meta, "scenes": out}, indent=1))
    print(json.dumps(meta, indent=1))
    if mism:
        raise SystemExit(f"{len(mism)} scenes disagree with Stage A rows.jsonl")


if __name__ == "__main__":
    main()
