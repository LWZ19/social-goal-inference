# I09 live check (phase 2): frozen protocol

---

Does the C pipeline keep A2's gain when the constraints call itself sees the detector's numbered people, with no VLM
people call at all? A2 (`results/still-scenes/overview/phase-notes/phase1c.md` section 2) replayed recorded constraints that were written against the
VLM's own people, re-addressed to the detector's people. Here the model writes its constraints against the detector's
people: they are numbered on the image and listed in the request exactly as the package numbers and lists any people
source's people.

- **Configuration.** The recorded C configuration (`config.yaml`, through `H.recorded_config`) of these runs, at trial
  t0 only, with ONE change: the people source is the frozen A1 detector people instead of the VLM people call.

  | Model | Runs (realistic arms real, staffed, plain; method con) |
  |---|---|
  | GLM-5.3 (`soclaas_glm53`, gateway id x-test-1, reasoning none) | `20261011-160500_s19_real_glm53_con`, `20261011-160352_s19_staffed_glm53_con`, `20261011-160452_s19_plain_glm53_con` |
  | DeepSeek V4.1 Flash (`soclaas_dsv41`, gateway id x-test-2, reasoning none) | `20261013-191337_s20_real_dsv41_con`, `20261013-191345_s20_staffed_dsv41_con`, `20261013-191402_s20_plain_dsv41_con` |

  Qwen sits at its ceiling and is left out to stay within the cap.
- **Records.** Every t0 record of those runs: 3 arms x 14 scenes x 2 episodes = 84 per model (no_one 36, waiting_line
  30, conversation_group 18). The scenes are the 42 realistic development scenes (E-real) A1 was evaluated on; this is
  NOT an unseen-layout test (seeds 200–219 are that, separately).
- **Expected calls [builder].** Plan, locate and context calls are rebuilt exactly as recorded and are meant to be
  answered by the cache. The constraints call is new in every goal step, because its numbered image and people lines
  change. A goal step happens when the context answer says people matter; that does not depend on the people source.
  Recorded t0 goal steps (records with a people call; each also made a constraints call): **GLM 48** (real 17,
  staffed 16, plain 15), **DeepSeek 54** (18, 20, 16): **102** in all, under the cap of 120 and under 60 per model. None
  of the 102 recorded t0 constraints calls needed a validation or transient retry.

- A class with OraclePeople's interface (`sgi.run.OraclePeople`): `__call__(observation) -> PeopleFound`, with
  `calls=[]`, `call_time_s=0.0` and the geometry time measured.
- For the observation's scene: the frozen detector people of `results/still-scenes/people-finder/early-detector/phase1c/a1/eval_people.json` (sha256
  `ffa595453c94b444c8dd030d053d37ac103441a7e74f07e3f172836b3d0c24b3`, checked on every load; RT-DETRv2 at score 0.25,
  band 1.00–2.30 m, flat test off, d_dup 0.50 m, d_push 0.20 m, facing F5 = A1's frozen choice with its one fix,
  unchanged), in descending score, each as `FoundPerson(xy=(x, y), yaw=yaw, pixel=projection of (x, y,
  person_mark_height_m), label=None)`, the pixel built exactly as OraclePeople builds it (`project(camera, (x, y,
  1.1))[:2]`). A person whose point does not project (behind the camera) is left out, as in OraclePeople. Before
  freezing: all 119 E-real detector people project and have a yaw.
- Scene matching: by the harness fingerprint (`H.fingerprint`, sha256 of the whole scene folder) of the scene being
  run against the fingerprint of the folder that keys `eval_people.json`; a scene without an entry is an error,
  never an empty list. The source also checks at every call that the observation is the scene's own
  (`observation.sha256()`).
- Then the package does what it does for every people source: `number_people` (merge within 0.25 m, cut to 40,
  number left to right), `numbered_image`, `people_lines`, `build_constraints_request`, the constraints call, the
  solver. No people call is made: the people stage's client fails the run if it is ever asked.
- **Substitution [builder].** The driver runs the package's own `run_episode`. The one hook replaced is
  `sgi.run._given_inferrer`, which for `run.people: oracle` returns `pipeline.constraint_inferrer(OraclePeople(truth,
  th), trial=trial)`; the driver's replacement returns `pipeline.constraint_inferrer(DetectorPeople(...),
  trial=trial)` for the scene being run, so the trial still goes into the constraints call's cache key.

All at t0, on the same 84 records per model; counted before this note.

| Model | Recorded C t0 (lines of 30) | A2 D replay t0 (lines) | I01 P t0 (lines) |
|---|---|---|---|
| GLM | 64 (14) | 69 (19) | 69 (19) |
| DeepSeek | 65 (25) | 71 (28) | 70 (27) |

Sources: recorded C = the records' `score.success` (also column `C_rec` of the A2 rows); D replay = column `D` of
`phase1c/i09/a2/rows_old_{glm53,dsv41}.jsonl` at trial 0; I01 P = column `P` of the same files, which A2 checked
record for record against `phase1/i01_repairs/rows_old_*` (and that file gives 69 and 70 at t0 too).

Each is reported per model with its n. **[builder]** definitions:

1. **Successes** over the 84 t0 records: `score.success` as the package (202c165) scores the D-live record in the run.
   A record that still has no answer (`vlm_error`) after one re-run of the same command counts as a failure, and their
   number is reported.
2. **Line successes** over the 30 t0 waiting_line records. Also by situation: no_one (36), conversation_group (18).
3. **Flips and breaks** (record level): flip = baseline failure and D-live success; break = baseline success and
   D-live failure. Against recorded C t0, against the D replay t0, and (secondary) against I01 P t0. With the
   effective n: the layouts and scenes the flips come from.
4. **Constraint usability**, over D-live goal steps with a constraints answer, against recorded C t0's goal steps:
   counts of `ConstraintUse.status` used / unusable / superseded / cut; goal steps with flag `no_usable_constraint`;
   the record measures `constraints_unusable` and `exclusions_broken`; constraints per goal step.
5. **Numbers named that exist in the list**: every number in any constraint's `people` field of a goal step; the share
   that are in the numbered list shown (1..N). For recorded C the list is the VLM's own numbered people. Also the
   share of goal steps naming at least one number not in the list.
6. **Latency of the goal step**, per goal step:
   D-live = `latency_total_s` of its constraints call + `timings_s.geometry`;
   recorded C = `latency_total_s` of its people call + of its constraints call + `timings_s.geometry`.
   Medians and p90 per model, also paired on records that are goal steps in both, and the call parts alone. A
   secondary line adds A1's warm detector-plus-pose runtime as a constant (median 55.6 ms; the verifier's 61.9 ms),
   which is not measured live (the frozen people are read from a file). Recorded latencies come from the calls first
   made on 11–13 Oct (a cached call carries the latency of the call that was stored); D-live's are new; the gateway's
   load differs, so this is descriptive.
7. Descriptive extras (no role in the reading): detector people per goal step against the VLM's list size; the record
   measures `people_found_share`, `people_unmatched`, `members_precision`, `members_recall` (the package computes them
   because `run.people` reads `vlm`); validation and transient retries; calls made.

On successes over the 84 t0 records per model:
- **Consistent with A2** if D-live is at least recorded C t0 for both models AND at least the D replay t0 minus 3 for
  both: GLM >= max(64, 69 − 3) = **66** and DeepSeek >= max(65, 71 − 3) = **68**.
- **Contradicts A2** if D-live is below recorded C t0 for both: GLM <= 63 and DeepSeek <= 64.
- **Mixed** otherwise.

- **[builder] A call** is one request sent to the provider's back end: every attempt counts, the validation retry
  and transient retries included. It is counted before it is sent.
- **Hard stop.** The driver refuses to send the 121st call in total over both models, or the 61st of one model. The
  count is kept in a ledger file in the output folder, shared by every invocation that uses that folder, and
  persists across invocations and re-runs; a lock file keeps two invocations from running at once. The flags
  `--max-calls` and `--max-calls-per-model` may lower the caps, never raise them above 120 and 60. On the stop, the run
  keeps what it finished (results.jsonl, summary.csv, `config.yaml` with `incomplete: true` and `stopped_by`).
- **Count-only mode.** The model clients are replaced: a cache hit returns the cached answer exactly as the package's
  client does; a miss is counted and recorded (model, arm, scene, episode, stage, cache key) and answered locally as
  `vlm_error`, never sent. No back end is built and no key is read. Because a miss stops that record, a plan, locate
  or context miss hides the calls after it; the report gives a lower bound and an upper bound (plus 3 for a plan miss,
  2 for a locate miss, 1 for a context miss).
- **vlm_error.** The same command run again repeats only the calls that got no answer (the cache resumes); every
  repeat counts against the cap. One such re-run at most before analysis.

## 8. Checks before the live run (the build's acceptance; zero calls)

- (a) The mock model end to end on at least 6 episodes across lines, groups and nobody-near scenes, with the 202c165
  code and the recorded entries: the people stage makes no call; the constraints request carries the detector's
  numbered people (numbered image and people lines saved and looked at for 2 scenes); nothing sent contains a scene
  id, target id, situation name or forbidden-region kind (the rule of `tests/test_leakage.py`, applied to the requests
  built).
- (b) Count-only against an empty temporary cache of the builder's own: every call counted as a miss, nothing sent.
- (c) The hard stop fires at the cap with a fake client, across invocations sharing a ledger.

## 9. Not blind, and what is fixed

- The builder knows A2's result and mechanism (dropping detections no detector supports), the baselines of section 4,
  and that the expected direction at t0 is up (D replay +5 GLM, +6 DeepSeek over recorded C). The detector setting is
  A1's frozen choice with its one fix, unchanged; nothing is tuned on any layout, and no prompt, schema, threshold or
  solver rule is changed. No situation-specific code: DetectorPeople knows only a scene's fingerprint and its list.
- The people answer of the recorded runs is not used anywhere in D-live.

## 10. Outputs

- Run folders `<out>/<timestamp>_s24_i09_<arm>_<model>_con/` in the package's format (config.yaml, prompts/,
  results.jsonl, summary.csv, viz/), one per arm and model; the ledger `<out>/ledger.jsonl`; per invocation a check
  file with the cache counts by stage and the comparison of the plan and context answers with the recorded ones.
- The analysis of section 5 is run later, offline, from those folders and the baseline files named in section 4.
