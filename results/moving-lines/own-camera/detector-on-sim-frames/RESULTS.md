# E1: faster people-finding

## E1a: the detector on Habitat sim frames (zero calls; one 16 s GPU pass, peak 1.3 GB)
- **Setup.** A1's frozen RT-DETRv2 + ViTPose (tau 0.25, F5 facing) on the 13 `keep/sim_bench/frozen/*/001` frames, scored against `truth.json` (`e1a/e1a_score.py`, `e1a/e1a_out.txt`).
- **Recall:** 75 of 75 visible people (1.000), including 25 in concert_crowded.
- **False people:** 1 in 13 images (0.08 per image).
- **Median position error:** 0.077 m.
- **Facing:** 34 of 34 queue members within 45° of their slot's face direction.
- **Caveats.**
  - Visibility uses a depth proxy (the point 1.1 m up, within 12 m, not occluded with a 0.5 m tolerance).
- **Reading (inference).** The sim's humanoids are easy for the detector, much easier than the realistic counter staff. A detector people source is technically sound in the sim.

## E1b: detector people in place of the VLM people call (frozen `E1_FROZEN.md`, sha256 4cd5485c)

**Results:** Sonnet, DeepSeek and Qwen are complete. GLM is partial: the run stopped at 317 of 450 calls (105 people-matter records); it is left there, as the driver cannot resume.

| Model | People-matter records | Recorded C | A0 (VLM people, fresh) | D (detector people) | Redraw floor, rec→A0 (rw / wr) | D vs A0 (gain / harm) | Judged non-inferior |
|---|---|---|---|---|---|---|---|
| Sonnet 5.5 low | 143 | 134 | 124 | **132** | 11 / 1 | 11 / 3 (layout p 0.31) | yes |
| DeepSeek V4.1 (1017) | 151 | 106 | 127 | **127** | 8 / 29 | 8 / 8 | yes |
| Qwen 3.8 low | 144 | 131 | 131 | **131** | 2 / 2 | 2 / 2 | yes |
| GLM-5.3 (partial, 105 records) | 105 | 58 | 63 | **73** | 12 / 17 | 11 / 1 (layout p 0.125) | yes |

- **Speed.** D drops the people call. The model-call time per people-matter goal step (people plus constraints) falls from 9.0 to 5.4 s for Sonnet and from 12.7 to 7.6 s for DeepSeek. For Qwen, about 21 s of people call is saved per record (constraints 36.2 s against 35.4 s). For GLM, about 29 s is saved, but its gateway times (constraints 70–83 s) are too noisy to read. The detector adds 0.06 s, and it is one call fewer.
- **Parse failures.** Sonnet had 3 in A0 and 1 in D; they count as failures.
- **DeepSeek's floor carries the version change.** Its recorded (old version) C scores 106, while the fresh A0 on 1017 scores 127. So "recorded → A0" there is mostly a model upgrade, and D against A0 (same version, same trial) is the clean comparison: equal.
- **Reading (inference).** On the realistic development scenes, the detector replaces the people call with no detectable loss in accuracy (Sonnet slightly better; DeepSeek and Qwen equal; GLM's 105 partial records 11/1 in D's favour) and saves about 40% of the people-and-constraints model time.
- **Caveats.**
  - These scenes are design data for I09's detector (A1 was fitted on I03's split, and its development check was not blind).
  - On unseen layouts A1 had 0.242 false people per image; fix 2 reaches 0.036 on training seeds, but its confirmation never ran.
  - Sim frames: 75/75 (E1a).
- **Setup.** The static development records, with the recorded plan, context and target held fixed.
- **Arms.** A0 (VLM people plus constraints) against D (A1 detector people plus constraints) at fresh trials.
- **Judged.** D non-inferior within A0's redraw floor.

## E1d: detector tracking for CT (frozen `e1d/E1D_FROZEN.md`)

- **Detector service.** A1 frozen, behind `POST /detect`. Its output is identical to E1a's batch pass. Warm round trip: median 133 ms (90th percentile 221 ms). About 1.3 GB of GPU.
- **Tracker.** One-to-one association within 0.75 m; dropped after 3 in-view misses; out-of-view or occluded tracks kept as stale. 6 unit tests pass.
- **Arms.** CTd against CT2 (oracle tracks), on the same cached answers and fresh seeds.
- **Prediction.** Keep-up mostly preserved; losses from staleness.

## E1c: detector people at answer time for CT (frozen `results/moving-lines/people-finder/first-check/E1C_FROZEN.md`; Sonnet, seeds 10–12, 15 episodes)

Post hoc, zero calls: the scripts and outputs are `results/moving-lines/people-finder/first-check/cte_*.py` and `cte_*_out.txt`.

- **Latency: as predicted.**
  - The first answer comes 2.4 s sooner by the median (10.6 against 13.0 s), and 1.4–7.5 s sooner per episode.
  - There is no people call; both arms add the same ~0.4–0.5 s on top of their call times.
  - **Correction (2 Nov):** the sim pauses for the request-frame detection, so the 0.16 s warm detection is not in
    CTe's latency. Counting it, the gain is **about 2.2 s** (about 17%). The cold first frame (1.1 s) is a one-off
    service warm-up and is reported separately. Periodic tracking every 0.5 s runs asynchronously, so its ~0.2 s lag is in the results.
  - First in line, strict: CTe is earlier in 8 episodes, equal in 3 and later in 2 (of the 13 where both arms get there), 1.5 s earlier by the median. Where the times are equal, the line's own movement seems to set the arrival time.
- **Gains beyond latency: not separable from the spread between nearly identical arms.**
  - Five arms share CTd's exact first answer (same cached calls, same pose) and nearly its method: CT2, CT2m, CT2-vac, CT2-rat and CTd. On the same 15 episodes they range over:
    - share strict 0.25–0.30;
    - end in line 0–4;
    - cut-in samples 69–210;
    - back within 10 s 14/26–21/31.
  - CTe (0.32, 3, 86, 20/32) sits at the edge of that band, and CTd at the bottom of it.
  - CTe against CT2: +0.051 (sign-flip p 0.36).
  - 57 of CTd's 210 cut-ins come from one episode (cafe_slots s11), where CT2 had 0 with the same first answer.
  - The mean per-episode |difference| between nearly identical arms is 0.06–0.11; CTe against CTd is 0.12.
- **The E1b mechanism ("better-placed people give better constraints") is not what happened.**
  - **People were well placed in both arms.** Every VLM person in CTd's 15 answers is within 0.5 m of a true person (median error 0.09–0.18 m). The detector does slightly better (0.04–0.13 m), finds one more person in 6 episodes, and has 1 false person.
  - **Only one episode's goal changed in substance:** room_coffee_cart_slots s10. It accounts for 0.022 of the +0.070.
    - The two people lists agree within 0.07 m.
    - CTd's cached constraints call returned no usable constraint; CTe's fresh call returned come_after.
    - So it is a constraints redraw, not a difference in people quality.
- **What does change systematically is the constraint mix.**
  - CTe's answers include keep_out_of_shared_space in 12 of 15 episodes, CTd's in 3 of 15 (9 episodes gain it, none lose it).
  - On cafe_line2_moving it moves the goal 0.3 m (heading 98° against 86°).
  - The cause is the people text: positions only with no descriptions, plus the extra people. Detector people change what the model asks for.
  - Whether that helps is not shown: of the 9 episodes that gained it, 4 had fewer cut-ins, 3 about equal and 2 slightly more.
  - E1b's records do not keep the constraint content, so the same shift there cannot be checked.
- **Reading (inference).** CTe matches CTd's quality, with the first answer about 2.2 s (about 17%) sooner once detection is counted and no people call. The other gains are within the spread of nearly identical arms.
- **Direct redraw floor (Sonnet repeats, 2 Nov; `results/moving-lines/people-finder/first-check/cte_floor.py`, `cte_floor_out.txt`).**
  - CT and CT2 were rerun on the same 15 cells with the same cached first answer (the same pose in 15 of 15), so only the sim's timing differs.
  - CT, repeat against original: mean +0.005, mean |d| 0.096. Totals moved: end in line 2→0, cut-ins 135→102.
  - CT2, repeat against original: mean +0.040, mean |d| 0.052. Share 0.265→0.305, end in line 4→3, cut-ins 143→147.
  - CTe against CTd: +0.070, mean |d| 0.124.
  - With CT2's repeat, the band of arms that share CTd's answer runs from 0.247 to 0.305. CTe's 0.316 is just above it.
  - **Reading:** no gain beyond latency is shown at n = 15. A small effect cannot be ruled out.
- **Source of the shared-space shift: tested with zero calls.**
  - CT's own first answers (VLM people) on seeds 10–18 contain keep_out_of_shared_space in 13 of 44 (30%). By scenario: 3/8, 2/9, 2/9, 5/9, 1/9.
  - Detector people (CTe, seeds 10–12): 12 of 15 (80%).
  - Exact test stratified by scenario: one-sided p 0.0009 (pooled Fisher 0.0009).
  - Seeds 13–18 are other frames of the same scenarios, so this is a base rate, not a same-frame redraw.
  - **Reading:** the detector's people text (positions only, more people, no descriptions) makes Sonnet ask for shared space much more often. No same-frame redraw (about 15 Sonnet calls) is needed for an observation-level claim.
