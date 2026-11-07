"""Verifier's own test-set path guard (standard library only; loaded by path in both environments).

check(path): the path, as written (normalised) and resolved, must be absolute, lie under
<research>/phase2/seeds/realistic/seed<k>/eval/<scene>/... with 200 <= k <= 219 and exactly one seed<k> component,
and contain none of the banned substrings (case ignored).
"""
import os
import re

RESEARCH = "results/2026-10-17-beyond-vlm"
SEEDS_ROOT = RESEARCH + "/phase2/seeds/realistic"
BANNED = ("keep/heldout", "s18", "s20_held", "eval_16", "eval_17", "eval_18", "eval_19", "group_photo", "lift_wait",
          "heldout", "held_out", "/.env", ".cache")
_SEED = re.compile(r"seed(\d+)")


def _check(p):
    low = p.lower()
    for b in BANNED:
        if b in low:
            raise AssertionError(f"v_guard refused ({b}): {p}")
    parts = p.split("/")
    hits = [(i, int(m.group(1))) for i, s in enumerate(parts) if (m := _SEED.fullmatch(s.lower()))]
    if len(hits) != 1:
        raise AssertionError(f"v_guard refused ({len(hits)} seed components): {p}")
    i, k = hits[0]
    if not (200 <= k <= 219):
        raise AssertionError(f"v_guard refused (seed{k}): {p}")
    if "/".join(parts[:i]) != SEEDS_ROOT:
        raise AssertionError(f"v_guard refused (not under {SEEDS_ROOT}): {p}")
    if not (len(parts) > i + 2 and parts[i + 1] == "eval" and parts[i + 2]):
        raise AssertionError(f"v_guard refused (not under seed{k}/eval/<scene>): {p}")
    return k


def check(path):
    p = str(path)
    if not os.path.isabs(p):
        raise AssertionError(f"v_guard refused (relative): {p}")
    k1 = _check(os.path.normpath(p))
    k2 = _check(os.path.realpath(p))
    if k1 != k2:
        raise AssertionError(f"v_guard refused (link changes seed): {p}")
    return k1
