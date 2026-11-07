# Marks probe M2: do numbered marks harm the free models' target choice? (frozen before any call, 1 Nov 2026)

## Why this probe
- **What M1, M1b and M1c showed.** Marks can only add the cases where the model clicks the wrong object: 54 of the snap ceiling's 98 gains.
  - The free models hold about 15 (GLM 9, DeepSeek 5, Qwen 1).
  - The 43 near misses are reachable by a zero-call snap, so marks add nothing there.
- **What marks can harm.** They can break a right choice, lose the location information in the click (M1b: 109 successes lost when one mark per object replaced the click), distract the request's decomposition, and bind correct reasoning to the wrong number.

## Hypothesis
H2: on the free models, marked plan calls with the "number plus point" rule do not harm target choice. Right→wrong flips stay at no more than 2% of the unmarked arm's right answers, per model, and plan decomposition stays level.

## Arms (paired, same stage, fresh trial indices 3, 4 and 5; s21 and s23 used 0–2)
- **A0: the plan call exactly as the pipeline makes it.**
  - It uses the repository's `prompts/`, with plan.md sha256 `00f2fae8…3c38` (the s21/s23 text).
  - Same strategy (cot), same entry, full-size image.
  - It measures drift since s21/s23, including DeepSeek's change to version 1017.
- **A2: the same call on a marked image.**
  - It uses `prompts_marks/`, where only plan.md's point_format block differs (sha256 `102494e0…430c`; the diff is in the appendix).
  - The schema adds `target_mark` per subtask, restricted to the numbers shown, or null. The gateway models never see schema descriptions, so everything they need about marks is in the prompt text.
- **Order.** A0 then A2 for each (scene, episode, trial). Model by model, over the 42 realistic development scenes × 2 episodes = 84 episodes.

## The marks (truth-free; `m2lib.py` sha256 `8c26635a…`)
- **Candidates.**
  - **People:** I09 **A1**'s stored detector people for the scene (`phase1c/i09/a1/eval_people.json`; A1 frozen, not fix 2), the first 6 by score, marked at their torso pixel.
  - **Components:** then the largest G1 depth components (as M1) whose mark point lies below 2.5 m and more than 0.5 m from every detected person.
  - At most 8 marks in all.
- **Coverage precheck** (`m2_score.py --precheck`, zero calls): 84 of 84 realistic target instances have a mark that, picked, gives the right target. That covers 54 places, 24 people and 6 objects.
- **Numbering.** 1..n shuffled per (scene, trial), seeded. This separates choosing the right thing from choosing a number (binding).
- **Style.** One style for every model: a white rounded label with a 2 px black border and a black DejaVu Sans Bold number at 15 px, centred on the mark pixel. Frozen examples: `m2_example_12.png` (sha256 `7a38aecc…`) and `m2_example_08.png` (sha256 `8cb70970…`), both at trial 3.

## The resolution rule ("number plus point", `m2lib.resolve`, rule 'both')
- **With a number:**
  - If the click lies within 15 px of the numbered candidate, keep the click.
    - A click just outside a component is moved to its nearest pixel.
    - A click in a person's box whose lift lands more than 0.6 m from the detected person is replaced by the person's mark.
  - Otherwise, use the candidate's own mark point.
- **Without a number:** the click, as now.
- **Reported from the same answers:** 'number' (the mark point alone) and 'point' (the click alone).

## Scoring (`m2_score.py` sha256 `74bd4713…`; zero calls)
- **Target right.** `sgi.evaluate.match_target` on the first navigate subtask's grounded target, under the matching s21/s23 C run's recorded thresholds.
- **Gain and harm per model and rule.** Gain is A0 wrong → A2 right; harm is A0 right → A2 wrong.
- **Replayed success.** The new target replaces the target of the C record with the same scene, episode and trial − 3, keeping its recorded people and constraints (`i11lib.replay_success`). This is labelled as a replay.
- **Plan exact match** (`decomposition_accuracy`), for both arms.
- **Salience and binding:**
  - the histogram of chosen numbers;
  - the share of picks that are the mark nearest the image centre, or the largest component;
  - binding conflicts: the click lies on another candidate but not the numbered one.

## Judged bars, per free model, rule 'both' (judged once, after all calls)
- **Harm:** right→wrong ≤ 2% of A0-right answers.
- **Plan:** A0 plan-exact minus A2 plan-exact ≤ 2% of the scored pairs.
- **Any model fails either:** stop before Gemini. The failure is then studied (binding, salience, location) before any redesign.
- **Gain:** reported, not judged. The free models hold too few wrong-object cases for a test: about 15 against a paired noise of 11–21 per model.

- **Smoke of x-test-3 first:**
  - `--models soclaas_qwen38_next_low --trials 3 --episodes 3`: 6 calls.
  - Check that the mark field parses. x-test-3 joins the run only if it does.
- **Main run:**
  - Qwen 3.8 low, GLM-5.3 and DeepSeek V4.1: 84 episodes × 3 trials × 2 arms = 504 calls per model, 1,512 in all (2,016 with x-test-3).
  - At most 40 calls a minute over all M2 processes.
  - Qwen with thinking answers in about 20–30 s per call, so its process takes about 3–4 h. GLM and DeepSeek take under 1 h each.

## Not blind
- The designer knows M1's results and the recorded answers' failures.
- The development scenes are design data. Nothing held out is read.

## Appendix: the A2 prompt change (prompts/plan.md → prompts_marks/plan.md, point_format block only)
Old:
```text
The image that follows is the robot's current camera view. For each navigate subtask, set "target_point" to a point on its target, well inside the target's visible outline, if the target is visible in the image; set it to null if it is not visible. Set it to null for every act subtask. {point_instruction}
```
New:
```text
The image that follows is the robot's current camera view. Some people and things in it carry a small white label with a black number; a label sits on the person or thing it names, and the numbers only name them. For each navigate subtask, set "target_mark" to the number on the label of its target, or to null if its target carries no label or is not visible; and set "target_point" to a point on its target, well inside the target's visible outline, if the target is visible in the image, or to null if it is not visible. Choose the point where it would be on the target without the labels; do not point at a label. Set both to null for every act subtask. {point_instruction}
```

## Pins (sha256)
- `m2_driver.py` 452e310c23c1f0fad8a8f224edb27d4a14598578fc1e7e31ad4b3148d7483e98
- `m2lib.py` 8c26635af7e3651373995949a7f904dfe2fc422aab1d6a556d9003e7d212bf4c
- `m2_score.py` 74bd4713b32566b5404adc890723291acdc0d025d9820126f610b681c20da99e
- `m1_coverage.py` 5d05a90544538b52b3016f44db3fd2e0e1ea827c73e5e87d430041c33bdaaab1
- `prompts_marks/plan.md` 102494e09ab5ccb1734949c2fb908b7c28ad30154dba28ea40606623909f430c
- `m2_example_12.png` 7a38aecc2d1f77a4dcb25b3f2fcca6913a3b80819f748d55b5cbb567497c2740
- `m2_example_08.png` 8cb709702667eeb11d41b7d13944f9b872d39df40099df33c2faa380a654a46b
