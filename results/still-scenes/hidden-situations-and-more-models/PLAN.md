# E3: more models (decision, 6 Nov 2026). Descriptive only: nothing here changes a frozen result.

1. Older models, still scenes (the 17 Oct six-model protocol, s23): three realistic sets x 14 layouts x 2 episodes x
   3 trials = 252 per method; methods pointing (cot), pointing with examples (few_shot_cot), constraint method.
   Same prompts (identical to the s23 snapshot) and code (HEAD d50a7b3; replay_solver.py reproduces 716 of 716
   s23 solver steps and every verdict, so the solver change since f1a488a does not touch still scenes).
   Models: Gemini 3 Flash (gemini-3-flash-preview), Gemini 3.1 Flash Lite, Gemini 3.5 Flash Lite, Gemini 3.6 Flash,
   Claude Haiku 4.5, Sonnet 4.6, Sonnet 5. Sonnet 4.5 is not offered on the key (models list, 6 Nov).
   Thinking as their newer counterparts: Gemini low, Haiku off, Sonnet effort low. Point format calibrated per model
   with sgi.smoke --calibrate before any run. Order per set: pointing first, so the other two methods reuse its
   plan and context answers through the cache, as s23 did.
2. Latest models, hidden situations (lift, photo): all 3 methods, as the 11-13 Oct runs for Qwen, GLM, DeepSeek.
- 2026-11-06 00:35: repro check exact (s24check_real_haiku55_con: 84 records, 80 = 80 successes, 0 differing, 0 fresh calls). Gemini calibration: xy_1000 for all four (calib_*.txt). Gemini runs started.
- 2026-11-06 00:36: Claude calibration (calib_claude_*.txt): Haiku 4.5 points in xy_1000 (64.9 px against 161.4 in pixels, the image centre's 161.0), so its entry is set to xy_1000 in every E3 config before any run; Sonnet 4.6 (64.4 px) and Sonnet 5 (52.0 px) in xy_pixel, as Sonnet 5.5.
- 2026-11-06 01:07: bug: queue.sh's simulator track cd'd into moving/ and then wrote logs by a relative path, so every command failed at the redirect and nothing ran. Relaunched as sim_track.sh with absolute paths.
- **Results so far (descriptive).** Older Gemini, of 252 (pointing / with examples / constraint method): 3.6 Flash
  123/142/232; 3 Flash 119/132/176; 3.5 Flash Lite 102/101/215; 3.1 Flash Lite 131/122/227 (17 Oct Gemini 3.8:
  164/173/214). Sonnet 4.6: 166/162/162. Gemini 3.8 hidden situations (pointing / examples / ours): lift 15/20/28 of
  36, photo 6/3/0 of 36 (DeepSeek 13 Oct: lift 8/15/31, photo 1/1/0).
- **Sonnet 4.6 diagnosed (no setup fault).** No call errors; context decisions as Sonnet 5.5 (constraints source 141
  vs 143). Lines 23/90 (Sonnet 5.5 87/90): its people call matches a median 0.71 of people to true people with one
  unmatched point an image (Sonnet 5.5 and Haiku 5.5: 1.00, 0), and the people it names in come_after are line
  members at a median precision of 0.50 (1.00). Empty scenes 85/108 with every target right: placed points miss the
  region by a median 0.23 m, the same 85/108 with pointing. Images are 640x480 (no provider downscaling). So its
  weaker pointing at people, not its reasoning, drives the loss; the people finder would remove that dependence.
- **All still-scene and hidden-situation runs done (6 Nov 02:16), 54 runs complete, no errors** (results_all_out.txt).
  Still scenes of 252 (pointing / examples / ours): Sonnet 5 139/144/194; Haiku 4.5 24/26/57. Hidden situations of
  36 (pointing / examples / ours): Sonnet 5.5 lift 33/31/36, photo 2/2/4; Haiku 5.5 lift 7/7/32, photo 0/1/9;
  Gemini 3.8 lift 15/20/28, photo 6/3/0.
- **Haiku 4.5 diagnosed (no setup fault).** Its raw points are on the 0-1000 scale as calibrated (y up to 850, which
  pixels on a 640x480 image cannot reach); read that way they lie a median 76 px from Haiku 5.5's targets for the
  same records (259 px if read as pixels). They are coarse (most common (500,400) x12, multiples of 10-50), so it
  names the right target in only 156 of 252 (Haiku 5.5: 251) and places outside the region often. A capability
  limit of the older model, as with Sonnet 4.6's people pointing.
- 2026-11-06 03:24: simulator track split for speed: sim_track.sh stopped by PID (Haiku headline 90/90 done, Haiku CT2 70/72 kept; 2 half-finished CT2 folders removed). Relaunched as sim_model.sh per model in parallel: claude_haiku55 (ports 8970+) and gemini_fast (ports 8990+), 2 workers each; finished episodes are skipped.
- **Moving-line headline, Haiku 5.5 (moving_headline_haiku_out.txt; strict share, seeds 10-14, the five headline
  scenarios):** C1 0.01 (n 15), R1 0.01, CT 0.16, CT2 0.30 (n 25 each); CT2 against R1 19 better / 2 worse / 4 same.
  On the same 25 cells DeepSeek R1 0.04, CT2 0.27 (19/3/3) and Sonnet 5.5 R1 0.06, CT2 0.29 (21/2/2).
- **Moving-line headline, Gemini 3.8 (moving_headline_gemini_out.txt):** C1 0.01 (n 15), R1 0.09, CT 0.15, CT2 0.21
  (n 25); CT2 against R1 18 better / 6 worse / 1 same.
- **Final test, Haiku 5.5 (confirm_score_haiku55_out.txt; check_mech faults none):** strict share from the first
  service, CT3F 0.232, CT4PJ 0.216, CT4P 0.170, CT4 0.164 (n 55: 17 of 72 first answers are no_target, every call
  ok, the model's own outcome; DeepSeek had 5). Same order as DeepSeek and Sonnet.
- **Final test, Gemini 3.8 (confirm_score_gemini_out.txt; check_mech faults none):** CT3F 0.231, CT4PJ 0.193, CT4
  0.145, CT4P 0.132 (n 71; 1 no_target). Same order as the other models.
- **E3 complete (6 Nov ~10:00).** Progress page updated (version 9): older-model chart, hidden-situation rows for the
  latest models, moving-line headline and final-test rows for Haiku 5.5 and Gemini 3.8, timeline.
  CT4PJ > CT4P repeats on four models (DeepSeek 0.208/0.159/0.107 frozen; Sonnet 0.246/0.174/0.124; Haiku
  0.232/0.216/0.170, n 55; Gemini 0.231/0.193/0.132). The joined-line rule helps by about +0.05 and the hand-built
- 2026-11-06 13:09: E3b started after the E4 build finished: the 7 older entries are in configs/default.yaml (same settings as the E3 config: calibrated point formats, Claude paced at 25 a minute). Two lanes: Claude (sonnet46 -> sonnet5 -> haiku45, ports 8970+), Gemini (36 -> 3 -> 3.5 Lite -> 3.1 Lite, ports 8990+), 2 workers each.
- 2026-11-06 17:03: E3b Claude lane resumed (Sonnet 4.6 from 37/90, then Sonnet 5, Haiku 4.5), ports 8970+; run_moving now survives a stuck episode. Gemini lane resumes when GPU allows.
