# I03 Stage A: incidents after evaluation scoring began (17 Oct 2026)

`DEVIATIONS.md` (sha256 in `DEVIATIONS.sha256`) was written before the fit and is not edited. This file records what happened during the evaluation pass. It has its own hash in `DEVIATIONS_POST_EVAL.sha256`.

**P1. The first evaluation pass stopped in the VLM part.** `stageA/evaluate.py` scored the 68 evaluation scenes with the frozen detector and wrote `stageA/rows.jsonl` (sha256 1feae3a0…c120d750, recorded in `stageA/ROWS.sha256`). It then stopped with "unknown set" on the first placeholder VLM run, whose scenes folder is the link folder `keep/placeholder_first14`. The per-scene log of that pass is `stageA/evaluate_pass1.log`.
- The fix: `set_of` also maps `/placeholder_first14/` to the placeholder set, and `evaluate.py` now reads the blob rows back from `rows.jsonl` instead of scoring them again.
- The second pass therefore scored only the VLM lists, for the first time, and wrote the tables. `rows.jsonl` has the same hash after the second pass.
- No threshold, detector, scorer or exemption changed between the two passes.

**P2. Post hoc descriptions.** Two remarks in the returned summary are descriptions made after scoring, and they are labelled post hoc there:
- the 6 E-real false bodies all come from one layout (eval_08_waiting_line_4) in its three arms;
- the 13 misses are 7 staff positions, each repeated across the real and staffed arms.
