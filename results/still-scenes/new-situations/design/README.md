# New situations: the design

**Question.** How are the two new tests built: a waiting line that turns a corner, and one person using a machine?

**Result.** Six fresh generator seeds (600-605), each with a bent line of 4 people (turning after 2), a bent line of 6
(turning after 3), a cash machine and a ticket machine: 12 layouts per situation, two requests each, 3 trials. The bent
line keeps the existing waiting-line rule (0.5-1.2 m behind the last person, along the second leg); machine use is a
new rule kept hidden during development, with a looser 0.8 m variant reported beside it.

**Files.**
- `E4_DESIGN.md`: the design (scenes, success areas, what each test checks).
- `configs/`: the method configs (`cot` pointing, `fsc` pointing with worked examples, `con` the constraint method).
- `scenes/`: the 24 scene files (`seed60N/seed60N_0K_*/scene.json`, no pictures).
- The plan, scorer and results are one folder up (`../README.md`).

**Reproduce.** The scenes were built with the placeholder renderer (`sgi.synth`); the runs with `../run_e4.sh MODEL`,
which calls `python -m sgi.run --config configs/<method>.yaml --scenes scenes/seed60N --final-test` and needs model
calls.

**Data.** `results/logs/research-data.tar.gz` (the scene files).
