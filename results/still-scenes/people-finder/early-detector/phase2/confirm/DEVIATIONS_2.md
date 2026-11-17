# I09 unseen-layout confirmation: the rulings on the departures (DEVIATIONS_2)

Written by the research lead on 18 Oct 2026, after scoring (the one detector pass ran 04:45:34–04:46:32 and scoring
04:46:43–04:47:33). These rulings are therefore post hoc. None changes a number or the verdict (neither), and none
was chosen with a view to the verdict: under every alternative reading S3 still fails.

   44/196 = 0.224, still above 0.10. Future render checks must log every generate command in the deviations file.
3. **The verifier's own detector pass** (FROZEN section 9 step 7 said the optional verifier runs no detector on a
   test image). Accepted. Its dispatch asked for it, it ran once, after scoring had finished, only to compare raw
   outputs, and it was bit-identical (2,791 boxes, 994 pose sets, maximum difference 0;
   `../verify_i09_confirm/v_order.txt`).
4. **The 3 h clock and generator failures** (the two open questions of `FROZEN.md`). The note's choices stand: the
   clock counts generate-command wall time only, and no generator failure occurred in seeds 200–209, so the second
   question did not arise.
