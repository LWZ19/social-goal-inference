"""Pre-freeze: does the reproduced seed-0 eval_06 render the same depth and rgb as the whitelisted scene? (no variant)"""
import sys, shutil, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
from sgi.config import load_config
from sgi.synth import generate as G
HERE = Path(__file__).resolve().parent
cfg = load_config()
work = Path(tempfile.mkdtemp(prefix="work_", dir=HERE))
try:
    rng = np.random.default_rng(0)
    counts = dict.fromkeys(G._REASONS, 0)
    for index, row in enumerate(G._EVAL[:7]):
        layout, scene, rgb, depth = G._accept(work / "eval", index, row, rng, cfg, counts)
    sc = H.scene(H.runs(stage="s10", model="qwen38_low", method="con")[0], "eval_06_waiting_line_1")
    d0 = np.asarray(sc.observation.depth); r0 = np.asarray(sc.observation.rgb)
    print("depth equal:", np.array_equal(np.nan_to_num(d0, nan=-1), np.nan_to_num(depth, nan=-1)),
          "max abs diff", float(np.nanmax(np.abs(d0 - depth))), "rgb equal:", np.array_equal(r0, rgb))
finally:
    shutil.rmtree(work, ignore_errors=True)
