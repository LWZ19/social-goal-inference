# Person snap: a frozen test on new scenes (frozen 2026-11-02, before any such scene exists)

## Why

- On the development scenes, C's misses on person targets are lift misses. The models click the right person, but
  the 5×5 median depth at the click reads a nearer person who hides part of them (`README.md`).
- A detector "person snap" fixes 24 successes and breaks 0 over 1,512 records. But that is post hoc, and the
  development set holds only one occluded person target: `occlusion_out.txt` shows 35% of conversation_group_4's
  target hidden, and 0% for every other person target.
- So the evidence is one layout. This note tests the fixed rule once, on new layouts built to contain the case.

## The rule (fixed; the code is `snap_sweep_v2.py`'s `snap` plus its 0.5 m condition)

- Detector: I09 A1, as frozen (rtdetr_v2_r50vd, tau 0.25, `eval_people.json` settings, F5 facing). It runs on each
  new picture: zero model calls, GPU about 60 ms a picture, announced before use.
- Candidates: detected people whose box contains the clicked pixel.
- If there is a candidate, take the one minimising |u − box centre u| / box width. The target's world point becomes
  that detection's (x, y, 1.1 m), but only if the depth-lifted point is more than 0.5 m from that (x, y). Otherwise
  the depth lift is kept.
- The same rule applies to every model, situation and target kind. Nothing in it may change after the first plan call
  on a new scene.

- **Occluded set (primary):** 12 layouts. Each has a person target with 15–70% of its silhouette hidden by a nearer
  person, by `occlusion.py` on scene truth. That is the cylinder model used before any model call.
  - Bins are fixed at 15–40% and 40–70%, at least 5 layouts each.
  - Situations are varied, not all conversation groups: groups, a line, an audience.
- **Controls (harm):**
  - 6 layouts with an unhidden person target (under 5% hidden);
  - 6 layouts with an object target that a person stands within 1 m of in the picture (the snap must leave these
    alone).
- **Episodes:** two per layout, as in the development sets: a direct reference ("go to the person in …"), and a
  high-level request with the name in the context. The descriptions are by clothing or a visible item, with no big
  text signs.

## Calls

- Plan stage only, as recorded runs make it (it gives the target point).
- Models: Qwen 3.8, GLM-5.3 and DeepSeek V4.1 (free). Three trials each.

## Measures (zero extra calls: the snap is applied to the same clicks)

- **target_correct with the depth lift against with the snap**, paired per record, scored by `sgi.evaluate.match_target`.
- **Fixed and broken**, per model, per occlusion bin, and on each control set.
- **Detector recall of the target person**, per bin. The target is detected if a detection lies within 0.5 m of it.
- **Where the baseline misses land:** the occluder, the background or another person.

## Bars (judged once, after all calls)

- **H1 (the failure exists on new scenes):** baseline target_correct on the occluded set is at least 10 points below
  the unhidden controls, pooled over the three models. If H1 fails, the models click visible parts and the snap has
  nothing to fix. That is reported as "not needed", and H2 is not judged.
- **H2 (the snap fixes it):** on the occluded set, fixed − broken > 0, with an exact two-sided sign-flip test over
  layouts (layout net change, pooled models) of p ≤ 0.05.
- **H3 (no harm):** on the controls, broken records ≤ 2 of 216 (about 1%), and none on object targets.
- **The claim "the snap beats C's lift on person targets"** is made only with H1, H2 and H3.

## Predictions (inference)

- **P1:** H1 holds. The baseline misses about a third of occluded-set records, mostly lifted onto the occluder.
- **P2:** the snap fixes at least half of the baseline misses in the 15–40% bin. It fixes fewer in 40–70%, where the
  detector more often misses the hidden target (recall below 0.8 there). Either no box contains the click, or only
  the occluder's does.
- **P3:** H3 holds. The 0.5 m condition leaves correct clicks alone, and object clicks are seldom inside a person's
  box with a lift more than 0.5 m from them.
- **P4:** the gain is not specific to one model. Each of the three free models has fixed ≥ broken.

## Not covered

- The full C pipeline's success. Only the target is scored, because pose success also depends on the people and
  constraints calls. The development replay showed target fixes became successes in 24 of 27 cases.
- Real photographs.
- Targets hidden by furniture rather than people. The rule only knows people.
