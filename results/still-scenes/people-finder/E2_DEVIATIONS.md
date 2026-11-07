
5. **2026-11-04 02:05, renders done.** Seeds 500-504 each exit 0, 72 images a seed, no errors in the logs.
6. **Assembly:** seeds 500-504 all ok (assemble_log.txt), 40 scenes in run_scenes/. Seeds 505-509 not rendered.
   - manifest.csv sha256 6052e279daae8a525abf84695ef3a9b107df23697d0d7409e91f304e297ddaee
7. **Detector pass** (run_detect.sh, as group4's; service on 8899, stopped after by its pid): 40 scenes in 15 s, exit 0.
   - e2_people.json sha256 7d3693c55b9ee1b493d844bd4f534c2ea697778d85328bb4b790d3d55e32cde9
   - The GPU had about 3.7 GB in use by another job at the time.
9. **Smoke** (2 units a free model, --limit 2, out runs_smoke/): mechanics only, its results not read. Its calls are
   cached, so the full run reuses those draws for the same units.
11. **Smoke checked (Qwen, DeepSeek), mechanics only:** 2 units each, 7 calls a unit, the order rotates, every call ok,
    no vlm_error. GLM's smoke was still running.
12. **02:19, full run started for Qwen (4 shards, 6 a minute each) and DeepSeek (3 shards, 10 a minute each)**,
    out runs/e2_<model>/, logs runs_<model>_<i>.log. GLM follows its smoke with 4 shards at 6 a minute: 78 a minute in
    all, under the shared 85.
13. **GLM smoke checked** (mechanics only, from the records): 2 units, 7 calls each, order rotates, all ok, no
    vlm_error. The smoke command's own output (log tails) was not read.
14. **02:28, GLM full run started:** 4 shards at 6 a minute, out runs/e2_soclaas_glm53/.
15. **04:39, Qwen and DeepSeek finished:** 160 of 160 units each, exit 0. Mechanics (`check_e2_mech.py`, no score
    fields read; outputs mech_<model>.txt):
    - DeepSeek: 157 people-matter units; every arm ok on all 157; 1727 listed calls all ok; 3 transient retries;
      orders 54/53/50; no vlm_error; meta HEAD d50a7b3, empty diff, people 7d3693c5.
    - Qwen: 159 people-matter units; every arm ok on all 159; 1749 listed calls all ok; no retries; orders
      54/52/53; no vlm_error; same meta.
16. **08:4x, GLM finished its first pass:** 160 of 160 units, exit 0. Mechanics (mech_soclaas_glm53.txt): 154
    people-matter units; 38 units with a vlm_error (arm errors A0 18, A0b 24, D 10; 52 failed calls of 1694 listed;
    182 transient retries); orders 53/50/51; meta HEAD d50a7b3, empty diff, people 7d3693c5. The driver logs no
    error text, so the error kind is not known.
17. **08:43, GLM resume (§4: --resume re-runs units written with a vlm_error),** the same 4 shards at 6 a minute.
    Units still failing after it are left out of every arm and counted, as §5 says.
21. **20:52, pass 2 done:** 30 -> 25 units with a vlm_error (5 cleared). Every remaining failure is a constraints call
    that gives up after about 550 s; the same units fail again, so it looks unit-specific, not an outage.
22. **20:52, pass 3 started** under item 19's rule (pass 2 cleared some), same 4 shards, HEAD d50a7b3, empty diff.
23. **Pass 3 done:** 25 -> 24 units with a vlm_error (1 cleared).
24. **22:33, pass 4 started** under item 19's rule, same 4 shards, HEAD d50a7b3, empty diff.
25. **Pass 4 done:** 24 -> 22 units with a vlm_error (2 cleared). Pass 4 was launched detached (nohup) by mistake, so a
    watcher waited on it; no effect on the run.
26. **2026-11-05 00:07, pass 5 started** under item 19's rule, same 4 shards, HEAD d50a7b3, empty diff.
27. **Pass 5 done:** 22 -> 19 units with a vlm_error (3 cleared).
28. **2026-11-05 01:32, pass 6 started** under item 19's rule, same 4 shards, HEAD d50a7b3, empty diff.
29. **Pass 6 done:** 19 -> 18 units with a vlm_error (1 cleared).
30. **2026-11-05 02:47, pass 7 started** under item 19's rule, same 4 shards, HEAD d50a7b3, empty diff.
31. **Pass 7 done:** 18 -> 18 units with a vlm_error (none cleared; the first pass in a row with none).
32. **2026-11-05 04:01, pass 8 started** under item 19's rule; if it clears none, retrying stops and GLM is scored.
33. **Pass 8 done:** 18 -> 14 units with a vlm_error (4 cleared; the no-clear streak resets).
34. **2026-11-05 05:01, pass 9 started** under item 19's rule, same 4 shards, HEAD d50a7b3, empty diff.
35. **Pass 9 done:** 14 -> 13 units with a vlm_error (1 cleared).
36. **2026-11-05 05:57, pass 10 started** under item 19's rule, same 4 shards, HEAD d50a7b3, empty diff.
37. **Pass 10 done:** 13 -> 13 units with a vlm_error (none cleared; the first pass in a row with none).
38. **2026-11-05 06:44, pass 11 started** under item 19's rule; if it clears none, retrying stops and GLM is scored.
39. **Pass 11 done:** 13 -> 11 units with a vlm_error (2 cleared; the no-clear streak resets).
40. **2026-11-05 07:25, pass 12 started** under item 19's rule, same 4 shards, HEAD d50a7b3, empty diff.
41. **Pass 12 done:** 11 -> 11 units with a vlm_error (none cleared; the first pass in a row with none).
42. **2026-11-05 08:12, pass 13 started** under item 19's rule; if it clears none, retrying stops and GLM is scored.
43. **Pass 13 done:** 11 -> 11 (none cleared; the second pass in a row with none), so retrying stops by item 19's rule.
44. **GLM final mechanics** (`mech_soclaas_glm53_final.txt` ea3e2f63, latest row per unit, no score fields read):
    160 distinct units over 382 record lines; 11 units still with a vlm_error, left out of every arm (8 conversation
    group, 3 waiting line); 149 clean units, 143 people-matter, every arm ok on all 143; orders 53/44/46; meta HEAD
    d50a7b3, empty diff, people 7d3693c5. 11 of 160 is within the 16-unit limit, so GLM is not "partial".
