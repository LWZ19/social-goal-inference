# E1d: detector tracking in place of the sim's true positions, for CT (frozen before any episode, 2 Nov 2026)

## Why
D1's main threat to validity is oracle tracking: CT re-solves on the sim's true positions. E1a showed that A1's detector is near perfect on Habitat frames (75/75, 0.08 false people per image, 0.077 m error). That makes a tracker built only from the robot's own camera feasible.

- **`detector_service.py`**, in the torch sidecar, read-only, offline.
  - A1's frozen RT-DETRv2 + ViTPose (infer.py, sha256 asserted) behind `POST /detect` on 127.0.0.1:8899.
  - Its answer has the fields of A1's raw pass line.
  - **Smoke test:** its boxes and keypoints for cafe_line6 and concert_crowded are identical to E1a's batch pass.
  - **Warm round trip over the 13 sim frames:** median 133 ms (90th percentile 221 ms); detector 37 ms, pose 92 ms. Pose runs on every box scoring at least 0.10, as A1's frozen pass does.
  - **GPU:** one instance, peak about 1.3 GB allocated (2.1 GB reserved).
- **`tracker.py`**, in the sgi env (`test_tracker.py`: 6 unit tests pass, no GPU).
  - **People:** `people_from_raw` runs A1's deployed path (`pipeline.run_image`, A1's frozen setting, F5 facing), using the frame's depth and camera.
  - **Association:** nearest pairs first, one to one, within 0.75 m. A leftover detection starts a new track with a new integer id.
  - **Miss and drop:** a track counts a miss when unmatched while it should be seen (its point 1.1 m up projects into the image within 12 m, and the depth there is no nearer than its distance minus 0.5 m). Three misses in a row drop it.
  - **Stale:** an unmatched track that is out of view or occluded keeps its last position and yaw, and is marked stale.
  - Matched tracks take the new position and yaw, with no smoothing.
  - **Cadence:** every re-solve frame, i.e. 2 Hz, the frame the robot sees at that moment.

- At each re-solve, the sim posts its robot camera's rgb to `/detect`.
- The resolve side runs `people_from_raw` and `Tracker.update` with that frame's depth and camera, and passes `Tracker.as_people()` to `Resolver` in place of the true tracks.
- The first match of the VLM's numbers to track ids stays as now: one to one (Hungarian), 0.75 m gate, at the first re-solve.
- Nothing else in CT2 changes.

## Arms and scoring
- **CTd** (CT2 with detector tracking) against **CT2** (oracle tracking), on the same cached answers and the same fresh seeds as the CT2 runs (10–19).
- Scored by SCORING.md, SCORING_STRICT.md and CT2's 0.6 m-first scoring, all frozen.
- **Reported:**
  - in line at the end;
  - share in line;
  - keep-up per service event;
  - reached the front;
  - lost place;
  - cut-ins;
  - goal left the line;
  - the first-match rate;
  - per frame, tracks against visible true people, id switches against the nearest true person, and the share of stale tracks among the waiting set.

## Prediction (inference, written before any episode)
- **Keep-up mostly preserved.** The robot looks up the line, and the waiting people are in view.
- **Losses come from staleness.** The person ahead goes out of view when the robot turns, or a served person leaves behind an occluder. The 0.75 m miss rule then lags by about 1.5 s.
- **First-match rate** close to the oracle's.
- **No judged bar.** This step measures how much of CT2 survives without oracle tracks.

## Pins
See `E1D.sha256`.
