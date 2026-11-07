# Learned models

**Question.** Can a small learned model of where the robot may stand beat the hand-written solver?

**Result.** Not on these records. The soft density matches the solver only when it is as simple as a Gaussian control (pooled mean success 0.670 against 0.760); the flow-matching pilot is killed under its own criteria (56.8 percent top-1 against 76.0 for the solver); the zero-call gates give 22 of 30 required configurations (gate 1 not met) and gate 2 met in 2 of 3 folds.

**Files.**
- `soft-density/` : soft-density place model fitted by leave-one-layout-out (I04).
- `flow-model-gates/` : the zero-call gates that decide whether a flow model is worth training (C5).
- `flow-model-pilot/` : the flow-matching pilot itself and its verification.

**Reproduce.** See each subfolder.

**Data.** `logs/still-runs.tar.gz`, `logs/research-data.tar.gz`, `logs/bulk.tar.gz`. Uses the step-0 replay harness in `results/still-scenes/analysis-scripts/harness/` and the development scenes in `data/dev_scenes/` (release dev-scenes-v1).
