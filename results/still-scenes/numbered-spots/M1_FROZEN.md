# Marks probe M1: does a geometric proposer contain the right target? (frozen before scoring, 1 Nov 2026)

## Question
Numbered target marks (I11 Stage 1) replace "the model clicks a pixel" with "the model picks a numbered candidate". This only helps if the candidates contain the right target. Wrong target is the largest failure class of C: 111 of 233 failures at the new wording (`results/still-scenes/replay-harness/sets-and-definitions/README.md` Table 2c). M1 asks two things:
- **Coverage:** how often does a zero-call, truth-free proposer contain a candidate that the scorer counts as the right target, and for which target kinds?
- **Ceiling:** how much of the oracle-target headroom would a perfect pick among these candidates recover?

## Hypothesis
H1: depth components alone (the frozen I11 G1 join: floor removed, 3D gap 0.10 m, fragments under 25 px dropped) contain the right target for nearly every target instance, because every target is a physical thing or a person that stands out of the floor.

Prediction, written before scoring, to compare surprises against:
- objects and people: high;
- place targets that are part of a larger structure (a counter run, a door in a wall): lower;
- objects resting on furniture (a parcel on a table): merged, as I11 found (3 of 3, 10 of 10);
- ceiling: about half to two thirds of the oracle-target gain.

## Proposer P0 (no truth, no model)
- **Candidates.** Each G1 component of class object or structure (`results/still-scenes/perception/lifting/i11lib.py` `Comps`) is one candidate. Fragments, floor and invalid pixels are not.
- **The mark pixel.** The component's most interior pixel. The mask is eroded repeatedly by its 4-neighbourhood until the next erosion would empty it. Of the pixels left, the one nearest the mask's pixel centroid is taken (lower row, then lower column, on a tie).
- **The lift.** `i11lib.comp_lift` at that pixel: the median depth of the component's own pixels in the 5x5 window.
- **The order.** Candidates are ordered by pixel count, largest first. A cap N keeps the first N.

## Sets
- **Coverage.** All 68 whitelisted development scenes (`research_harness.distinct_scenes`), each through the first listed run that evaluates it, under that run's recorded thresholds.
  - The unit is a target instance: (scene, episode, target id), for every episode target id.
  - It is reported split into realistic and placeholder, and by target kind (place, object, person).
- **Ceiling.** The 18 new-wording C runs (s21 Qwen low; s23 GLM, DeepSeek, Haiku, Sonnet low, Gemini low; arms real, staffed, plain).
  - For each record with a lifted target, the record is replayed with each candidate's target through `i11lib.replay_success`. That is the same path as I11's replays and I01's R6.
  - The best candidate counts: a success if any candidate's replay succeeds.
  - Records with no target are left as recorded.

## Measures
- coverage at N = 10, 15, 20 and all;
- candidates per image (median, 90th percentile, max);
- per instance: whether it is covered, by which component class, and how many candidates match;
- ceiling successes per model against recorded C and against R6 (the oracle-target replay: +11, +14, +12, +1, +7, +26 = +71 at the new wording; `results/still-scenes/overview/phase-notes/phase1.md` Table 1.4).

## Pass bars (judged once, at N = 20)
- **H1 holds:** realistic pooled coverage is at least 0.95, and every target kind with at least 5 realistic instances is at least 0.90.
- **Ceiling gate:** the pooled six-model ceiling gain over recorded C is at least +36 records, half of R6's +71.
- **Both pass:** the next probe is mark reading (live, small).
- **Either fails:** the next probe redesigns the proposer, starting from the uncovered instances. No live call until a proposer passes.

## Not blind
- The designer knows I11's component findings and the 111 wrong-target records.
- P0 is I11's frozen component definition. Only the mark pixel and the order are new.
- The whitelisted development scenes are design data. Nothing held out is read.

## Rules
- Zero model calls, CPU only.
- Runs and scenes are read only through `research_harness`.
- No file under `results/still-scenes/scene-seeds/phase2/` is opened.
- Outputs go only in this folder.
