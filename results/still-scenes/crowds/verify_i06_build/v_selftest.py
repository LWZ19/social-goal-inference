"""Independent verifier, I06 build: FROZEN section 3's self-test with MY pre-checks (vlib.py): c1-c8 must accept
every bystander and second-group member the package placed in eval_10..12 of seeds 0-2 (others = everyone else in
the scene). Also a check-by-check sanity table. Writes out/selftest.txt."""
from __future__ import annotations

import vlib as V


def main():
    lines = []
    total = refused = 0
    for seed in (0, 1, 2):
        bases = V.load_bases(seed)
        for row in (10, 11, 12):
            b = bases[row]
            for p in b.nonmembers:
                fails = b.checks("B", p, b.people)
                total += 1
                refused += bool(fails)
                lines.append(f"seed {seed} {b.scene_id} {p.id} (group {p.group}): {'refused ' + str(fails) if fails else 'accepted'}")
    lines.append(f"{total} package-placed people tested, {refused} refused")
    text = "\n".join(lines)
    print(text)
    (V.HERE / "out" / "selftest.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
