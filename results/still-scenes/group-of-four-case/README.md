# The group-of-four case: a hidden target person

**Question.** Why does the constraint method miss on one development layout, `conversation_group_4` ("go to the
person in red"), and does snapping the click onto the detected person fix it on new scenes?

**Result.** Every model names and clicks the right woman, but a man's arm in front of her catches the depth lift, so
the target lands on him: the miss is in our grounding, not in the models. Snapping the click to the detected person
(only when the lift lands more than 0.5 m from them) fixes 24 development successes and breaks none of 1,512. On 24 new
scenes built to contain the case it fixes 11 and breaks none of 432 (free models), but the frozen bar fails (an 8.6
point gap against the bar of 10), so the claim is not made.

**Files.**
- This analysis (development records): `g4_targets.py` → `g4_targets_out.txt` (where each run put its target),
  `vote_tally.py` → `vote_tally_out.txt` (voting over draws), `snap_sweep.py`, `snap_sweep_v2.py` and their
  `*_out.txt` (the snap rule, then with its 0.5 m condition), `snap_success.py`, `snap_all_success.py`,
  `snap_all_success_v2.py` and their `*_out.txt` (full success with the snap), `occlusion.py` →
  `occlusion_out.txt` (how much of each person target is hidden).
- The frozen test: `SNAP_FROZEN.md` with `SNAP_AMENDMENT_1.md` to `_3.md`; `DEVIATIONS.md` (the scene recipe and
  what happened); `snap_score.py` → `snap_score_out.txt`; `anchor_gap.py` → `anchor_gap_out.txt` (how near an
  occluder's edge comes to the target's chest).
- `snapset/`: how the 24 new scenes were chosen and built: `bearing_sweep.py`, `bearing_exact.py`, `scan.py`,
  `scan_v2.py`, `check_v2.py`, `select.py`, `select_v2.py` (each with its `*_out.txt` or `.csv`),
  `render_subset*.py`, the detector pass `snap_detect.py` / `run_detect.sh` → `snap_people.json`, and
  `plan_only.yaml`, the config of the plan-only runs.
- The same test on Claude and Gemini: `results/still-scenes/depth-fix/paid-models/`.

**Reproduce.** From the repository root, with `G=results/still-scenes/group-of-four-case`:

```bash
PYTHONPATH=src python $G/g4_targets.py
PYTHONPATH=src python $G/vote_tally.py
PYTHONPATH=src python $G/occlusion.py data/dev_scenes/realistic_final/real/scenes/real_eval_1[012]_*
PYTHONPATH=src python $G/snapset/bearing_sweep.py      # also bearing_exact.py, select.py, select_v2.py [--bearings literal]
PYTHONPATH=src python $G/snap_score.py --runs runs/20261102-191349_snap_plan_soclaas_{dsv41,glm53,qwen38_low} \
    --scenes $G/snapset/run_scenes_v2 --people $G/snapset/snap_people.json
```

The first four lines reproduce their outputs from the repository alone (`select.py` and `select_v2.py` rewrite their
`.csv` files, with the same bytes). `snap_score.py` and `anchor_gap.py $G/snapset/run_scenes_v2/*` also read the new
scenes' depth images, which are not in the repository; with them both reproduce their outputs exactly.
`snap_sweep*.py` and `snap_*success*.py` first check the sha256 of the detector's people file
(`results/still-scenes/people-finder/early-detector/phase1c/a1/eval_people.json`); the copy here names its scenes by
relative paths, so its hash differs and the check stops them. With the original file for the check, and its scene
names pointed at `data/dev_scenes/`, all five reproduce their outputs exactly.

**Data.** `results/logs/still-runs.tar.gz` (`runs/`); the development scenes in `data/dev_scenes/` (see
`results/README.md`); `results/logs/research-data.tar.gz` (`snap_people.json`, `snapset/real_v2/`,
`snapset/run_scenes_v2/`).

## Notes

**conversation_group_4: why C misses there, and what would fix it (zero calls, 2 Nov 2026)**

### Answer

**The models aren't getting anything wrong about people. The pipeline lifts the click onto the wrong person.**
- The task is "go to the person in red" (or "give this to Sam", Sam = the person wearing red).
- Every model names her correctly in every missed record ("woman in red top", "person wearing red"), and clicks her.
- But she stands behind a man in a dark suit (p3, back to the robot), whose arm crosses the middle of her torso in the
  picture (`g4_real_crop.png`).
- The lift reads the median depth of a 5x5 window at the click (`src/sgi/geometry.py:77`). A click on her torso centre
  gets his sleeve: 3.48 m against her 4.40 m. The target lands 0.3 m from him and 1.06 m from her, and the scorer
  matches the nearest person (`src/sgi/evaluate.py:38`).

So "target_correct = false" here is a grounding failure in our code, upstream of the people and constraints steps.
That is why E1b's arms A0 and D and the recorded run miss the same records: E1b injected the recorded target point.

### Evidence (`g4_targets_out.txt`, `vote_tally_out.txt`)

All six models click the same woman, inside one 15 px column. Success depends only on the click's height relative
to the arm:

| Model | Typical click y (px) | Target right (18 records) |
|---|---|---|
| Gemini | ~183 (upper chest, above the arm) | 18 |
| Haiku | ~230 (hips, below the arm) | 17 |
| GLM | 152-221, spread | 15 |
| Sonnet | 200-210 (on the arm) | 11 |
| Qwen | 202-216 | 10 |
| DeepSeek | spread; 3 clicks off her body to the floor or the far wall | 9 |

- Of the 28 con misses: 23 lift to p3 (the occluder), 4 to the background, and 1 to the floor near p1.
- Counts are identical for con, cot and fsc, which share the plan's target.
- **Voting doesn't fix it.** Majority of 3 draws gives Sonnet 0.61 -> 0.67, Qwen 0.56 -> 0.67, DeepSeek
  0.50 -> 0.67. Taking the median click of 3 is worse (2/6, 2/6, 3/6), because the median sits on the boundary.

### A fix, tested post hoc on all development records (`snap_sweep.py`, `snap_all_success*.py`)

**Person snap.** If the click lies inside a detected person's box, take the person whose box centre is nearest the
click horizontally (relative to box width), at their detected position, 1.1 m up.
- The detector is I09 A1's frozen detector, the same one as E1b's D arm.
- The rule is the same for every model, situation and target kind. Records: all six models' con runs, 1,512 in all.

| | Fired on | Targets fixed | Successes fixed | Successes broken |
|---|---|---|---|---|
| Snap whenever the click is in a box | 506 | 27 | 24 | 4 (DeepSeek; target moved 0.17-0.27 m, out of the region) |
| Snap only when the depth lift is more than 0.5 m from the clicked person (post hoc, chosen after the 4 breaks) | 32 | 27 | 24 | 0 |

- Fixed successes by model (both rules): Sonnet +7, Qwen +8, DeepSeek +6, GLM +2, Haiku +1, Gemini 0.
- Person targets afterwards: Sonnet 72/72, Qwen 72/72, Haiku 72/72, Gemini 72/72, GLM 70/72, DeepSeek 69/72.
- 25 of the 27 target fixes are in conversation_group_4.

### What this is and isn't

- It is not a weak-model hack. Sonnet gains as much as Qwen, and the failure is in our lift, not in any model.
- It is still a grounding fix, not a social-reasoning method. It fixes one scene's occlusion, found on that scene,
  with no held-out test. Claim only "person targets partly hidden by another person fail at the lift; a detector snap
  removes them on development data with no breaks".
- A test would need new scenes where a person target is partly hidden by another person. It would be frozen before
  the scenes are run: free models, plan call only, and the target scored with and without the snap.
- **For "beat C":** static C has almost no social headroom here. Its misses in this scene are all lift misses,
  shared by R, E and C alike.

### Follow-up (same day)

- `occlusion.py` measures, from scene truth, how much of a person target a nearer person hides. conversation_group_4's
  target is 35% hidden, and every other development person target is 0%. So the development set has exactly one
  occluded person target, which explains why 25 of the 27 fixes sit there.

### Frozen test result (snap_score_out.txt, scorer 265bb538)

- **Primary 8:** the depth lift is right in 125/144 and the snap in 136/144 (11 fixed, 0 broken). Far 4: 72/72 both
  ways. Set A: 103 → 104 of 108, none broken. Set B: unchanged.
- **H1 FAILS** (8.6 points against a bar of 10), so "not needed" applies and H2 is not judged (its p would be 0.125;
  only 4 layouts changed). H3 passes. The claim is NOT MADE.
- **Post hoc:** the failure needs the occluder's edge almost touching the chest point.
  - With the edge within 0.05 body half-widths (s412, s406; the development case is 0.04), the lift is right in
    26/36 and the snap in 35/36.
  - At 0.19-0.41, the lift is right in 99/108, the same as the unhidden controls.
- **The snap fired 12 times in 432 new records** (11 fixed, 0 broken), and 32 times in 1,512 development records (24
  successes fixed, 0 broken).
- Reading: a safe grounding fix with a small, narrow effect, not a general gain over C.
