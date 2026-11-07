"""I09 fix, diagnosis: the fit-data path guard. Standard library only, so the torch sidecar (which never imports sgi)
and the project venv both load it.

check(path) -> path, or AssertionError. The path, as written and resolved, must lie under
<research>/phase3/seeds/realistic_train/seed<k>/eval/<scene>/ with 110 <= k <= 119, hold exactly one seed<k>
component, and contain none of keep/heldout, s18, s20_held, eval_16..eval_19, group_photo, lift_wait, heldout,
held_out (case ignored). The spent test seeds (phase2/seeds/realistic, seeds 200-219) and the sealed seeds 3-99 can
never pass: their folders are not under SEEDS_ROOT and their numbers are outside 110-119.
"""
import os
import re

RESEARCH = "results/2026-10-17-beyond-vlm"
SEEDS_ROOT = RESEARCH + "/phase3/seeds/realistic_train"
FORBIDDEN_SUBSTR = ("keep/heldout", "s18", "s20_held", "eval_16", "eval_17", "eval_18", "eval_19", "group_photo",
                    "lift_wait", "heldout", "held_out", "phase2/seeds")
SEED_RE = re.compile(r"seed(\d+)")
SEED_MIN, SEED_MAX = 110, 119
FIT_TRAIN = tuple(range(110, 116))          # design data
FIT_VALIDATE = tuple(range(116, 120))       # final confirmation only; never scored or viewed in the diagnosis


def _one(path: str) -> None:
    low = path.lower()
    for s in FORBIDDEN_SUBSTR:
        assert s not in low, f"refused: {s} in {path}"
    parts = path.split("/")
    seeds = [(i, int(m.group(1))) for i, part in enumerate(parts) if (m := SEED_RE.fullmatch(part.lower()))]
    assert len(seeds) == 1, f"refused: {len(seeds)} seed<k> components in {path}"
    i, k = seeds[0]
    assert SEED_MIN <= k <= SEED_MAX, f"refused: seed{k} outside {SEED_MIN}-{SEED_MAX}: {path}"
    assert "/".join(parts[:i]) == SEEDS_ROOT, f"refused: seed folder not directly under {SEEDS_ROOT}: {path}"
    assert len(parts) > i + 2 and parts[i + 1] == "eval" and parts[i + 2], f"refused: not under seed{k}/eval/: {path}"


def check(path) -> str:
    p = str(path)
    assert os.path.isabs(p), f"refused: relative path {p}"
    _one(os.path.normpath(p))
    _one(os.path.realpath(p))
    return str(path)


def seed_of(path) -> int:
    check(path)
    return int(next(m.group(1) for part in str(path).split("/") if (m := SEED_RE.fullmatch(part.lower()))))
