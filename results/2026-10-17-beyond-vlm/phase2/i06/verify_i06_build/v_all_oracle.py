"""Independent verifier, I06 build: beyond the sample, the scene check, the full oracle (arm 1) and region re-derivation
on EVERY kept variant (199) of the manifest, with v_sample.battery. Writes out/all_oracle.jsonl and out/all_oracle.txt."""
from __future__ import annotations

import collections
import json
import time

from sgi.config import load_config, with_model
from sgi.pipeline import build_pipeline

import vlib as V
from v_sample import battery


def main():
    cfg = load_config()
    ocfg = with_model(cfg, "mock")
    ocfg = ocfg.model_copy(update={"run": ocfg.run.model_copy(update={
        "grounding": "oracle", "context_check": "always_yes", "goal": "oracle", "use_cache": False, "trials": 1})})
    pipeline = build_pipeline(ocfg)
    es = [json.loads(l) for l in (V.I06 / "build" / "manifest.jsonl").read_text().splitlines() if l]
    kept = [e for e in es if e["kind"] == "variant" and e["accepted"]]
    agg = collections.Counter()
    t0 = time.perf_counter()
    with open(V.HERE / "out" / "all_oracle.jsonl", "w") as f:
        for e in kept:
            res, _, _ = battery(V.I06 / e["folder"], cfg, ocfg, pipeline)
            row = {"scene_id": e["scene_id"], "seed": e["seed"], **{k: v for k, v in res.items() if k != "persons"}}
            f.write(json.dumps(row) + "\n")
            for k in ("check_pass", "oracle_success", "regions_package_own", "l2_empty"):
                agg[k] += bool(row[k])
            agg["calls"] += row["calls"]
            agg["l1_zero"] += row["l1_cells"] == 0
            print(f"{e['scene_id']}: check {row['check_pass']} oracle {row['oracle_success']} regions "
                  f"{row['regions_package_own']} L1 {row['l1_cells']} L2 {row['l2_cells']} [{time.perf_counter() - t0:.0f} s]",
                  flush=True)
    text = (f"all kept variants n={len(kept)}: check_pass {agg['check_pass']}, oracle_success (both episodes) "
            f"{agg['oracle_success']}, regions_package_own {agg['regions_package_own']}, e0 L1 cells zero {agg['l1_zero']}, "
            f"L2-empty {agg['l2_empty']}, model calls {agg['calls']}")
    print(text)
    (V.HERE / "out" / "all_oracle.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
