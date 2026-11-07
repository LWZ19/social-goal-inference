# Marks probe M4: do marked first answers put the target on the counter in D1's moving scenes? (frozen before any call, 2 Nov 2026)

## Why
D1's failures start from first answers whose target point is not the counter the line serves: on a person mid-line, on the last person, or at the wrong end. The question is whether M2's marked plan call ("number plus point") moves the target onto the counter more often. It runs on plan calls only, before any CT episode.

## Design
- **Frames.** The first frames of D1's 5 evaluation moving scenarios (cafe, cafe_line2_moving, cafe_line6_moving, room_coffee_cart, sweet_spot_line2_moving) × seeds 0–2: 15 frames. They are taken from `moving/runs/dsv41/<scenario>/seed<k>/C1/001/`; the first frame is shared by every arm. Fresh seeds 10–12 are not touched; they are reserved for CTs.
- **Arms.** U is the pipeline plan call (M2's A0). M is M2's marked call (A2) with the "number plus point" rule. Trials 20, 21 and 22, for fresh answers.
- **Marks.** Depth components only: M1's proposer, below 2.5 m, at most 8, numbering shuffled per frame and trial. The sim frames carry no detector or tracked people, so there are no separate person marks; people show up as their own components when they stand apart.
- **Coverage precheck** (zero calls): a mark lies within 1 m of the counter point in 12 of the 15 frames. In the 3 cafe frames the counter, about 6 m away and small, is not among the 8 largest components (nearest mark 1.76–3.47 m). There, M can at best fall back to the click.
- **Code.** `m4_driver.py` and `m4_score.py`. They import M2's frozen driver and library unchanged.

## Measure and decision (judged once, per model)
- **Hit:** the first navigate subtask's grounded target (U by its click, M by the "both" rule) lies within 1.0 m horizontally of the scenario's slot-0 face point.
- **Counts:** hits out of 45, gain and harm, the median distance, and the U arm's trial-to-trial flips as its redraw floor.
- **Run CT episodes with marked first answers for a model only if** M's hits ≥ U's hits + 5.

## Prediction (written before any call; inference)
- **No gain for most models.** The M2 Gemini run showed that line-scene target errors are mostly interpretation errors: the counter against "the person at the end of the line". Marks flip these both ways at the redraw rate.
- **A possible exception is GLM,** whose M2 gains were identity errors.
- **DeepSeek may lose,** through label attraction (M2P).

## Size
90 plan calls per model. Proposed: GLM and DeepSeek only, 180 free calls, at most 13 a minute.
