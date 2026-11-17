"""Allowed and skipped records of every run split.py reads (realistic arms of s16b, s19, s20, s21, s22, s23)."""
import collections, sys
from pathlib import Path
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
t = collections.Counter(); nruns = 0
for st in ("s16b", "s19", "s20", "s21", "s22", "s23"):
    for arm in ("real", "staffed", "plain"):
        for ref in H.runs(stage=st, arm=arm):
            for _ in H.records(ref, tally=t):
                pass
            nruns += 1
line = f"{nruns} runs (realistic arms of s16b, s19, s20, s21, s22, s23): {dict(t)}"
print(line)
(Path(__file__).resolve().parent / "tally_out.txt").write_text(line + "\n", encoding="utf-8")
