"""I09 fix, diagnosis: shared project-venv code. Torch-free; writes nothing.

- A1's code is reused by import, never copied (pipeline.py = P, common.py = C, I03's people_scorer.py = PS), each
  pinned by sha256 as in phase2/i09/confirm/confirm_lib.py. Bytecode writing is off, so nothing lands in the earlier
  phase folders.
- Fit scenes (seeds 110-119) are path-guarded (pathguard_fit.check) and harness-guarded (H.guard_scene) before their
  scene.json is read, then loaded with sgi.scene.load_scene into PS.FreshScene.
- A1's development scenes are loaded only through the step-0 harness (H.distinct_scenes, H.scene), as c2_repro.py.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

WORKTREE = paths.REPO
assert HERE == paths.STILL / "people-finder/early-detector/fix", HERE
A1 = paths.STILL / "people-finder/early-detector/phase1c/a1"
I03 = paths.STILL / "perception/depth-blob-detector"
SEEDS = paths.STILL / "scene-seeds/phase3/realistic_train"

PINNED = {
    A1 / "pipeline.py": "4826be88d6939f3f5e4d14375a2fc9698abd94973ca46243d6697e263e5d063a",
    A1 / "common.py": "cdc33d742f6db5d09c3ad9a4557c1210ef10f8909484cee101ce983f9f8b2f53",
    I03 / "people_scorer.py": "e12776eaddac1de0af75b58f8a272cd0a13332596da626bcc81b085f5900d685",
    A1 / "sidecar/infer.py": "bee1322fd692bf0e85e820871413d486fda04d5770f72b5d3d75e18266709484",
    A1 / "chosen.json": "19b1e16340d810e75f8f772eeeabeaf4b788e09614e74d78d3afc206531119ff",
    A1 / "chosen_fix.json": "fe8177165773f7208d417227c358adae7d04326bd29cb6c75d73ed87b3f83f52",
    WORKTREE / "configs/default.yaml": "c651f4c484fff15d33290e13f28bc53c9221c8edd79c0b7a24f24a128098b4fc",
}
DEV_RAW = {A1 / "raw_eval/rtdetr_v2_r50vd.jsonl": "8a03b82ca89bb82bdd0b10a8e56944b8562f3a580967087646e408dd4862a97d",
           A1 / "eval_people.json": "ffa595453c94b444c8dd030d053d37ac103441a7e74f07e3f172836b3d0c24b3",
           A1 / "manifest_eval.json": "6c321c3800d0b76632da0711a2385dce4915a28e9cd1f563ff3f8a5782be0b22"}
FROZEN_SETTING = {"detector": "rtdetr_v2_r50vd", "tau": 0.25, "band": "wide", "flat": False, "d_dup": 0.5,
                  "d_push": 0.2}
VARIANT = "F5"
ROW_IDS = ("real_eval_00_no_one_place", "real_eval_01_no_one_place", "real_eval_02_no_one_place",
           "real_eval_03_no_one_deep_place", "real_eval_04_no_one_object", "real_eval_05_no_one_person",
           "real_eval_06_waiting_line_1", "real_eval_07_waiting_line_2", "real_eval_08_waiting_line_4",
           "real_eval_09_waiting_line_6", "real_eval_10_conversation_group_2", "real_eval_11_conversation_group_3",
           "real_eval_12_conversation_group_4", "real_eval_13_waiting_line_3")


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


for _p, _want in PINNED.items():
    assert sha(_p) == _want, f"code identity: {_p}"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(A1))
import common as C  # noqa: E402  (inserts the harness and phase1b/i03 paths)
import pathguard_fit as PG  # noqa: E402
import people_scorer as PS  # noqa: E402
import pipeline as P  # noqa: E402
import research_harness as H  # noqa: E402

assert Path(C.__file__).resolve() == A1 / "common.py" and Path(P.__file__).resolve() == A1 / "pipeline.py"
assert Path(PS.__file__).resolve() == I03 / "people_scorer.py"
assert Path(H.__file__).resolve() == WORKTREE / "results/still-scenes/analysis-scripts/harness/research_harness.py"


def frozen_setting():
    chosen = json.loads((A1 / "chosen.json").read_text())
    fix = json.loads((A1 / "chosen_fix.json").read_text())
    assert fix["setting"] == chosen["setting"] == FROZEN_SETTING
    assert fix["fix_adopted"] is True and fix["facing_variant"] == VARIANT
    return dict(FROZEN_SETTING), VARIANT


def load_fit_scene(folder):
    """A fit scene (seeds 110-119), guarded before scene.json is read; returns PS.FreshScene."""
    from sgi.scene import load_scene
    PG.check(folder)
    H.guard_scene(Path(folder))
    return PS.FreshScene(load_scene(Path(folder)))


def fit_manifest():
    m = json.loads((HERE / "manifest_fit.json").read_text())
    return m


def raw_lines(path):
    return {x["rgb"]: x for x in map(json.loads, Path(path).read_text().splitlines())}
