# E1c: detector people at answer time for CT, in place of the people call (frozen before any episode, 2 Nov 2026)

## Why
- **E1b, static scenes:** detector people in place of the VLM people call were non-inferior (Sonnet 132 against 124 of 143; DeepSeek 127 against 127 of 151). They cut the people-and-constraints model time by about 40% (Sonnet 9.0 → 5.4 s).
- **CTd:** its first answer still comes from the VLM people call, so the people call's latency stays in the time to the first answer, which is the largest D1 failure class.
- **E1c removes it,** with the rest of CTd unchanged.

- **The first answer.**
  - The request frame's detector tracker snapshot (E1d's service and `tracker.py`, unchanged) supplies its live, non-stale tracks as the people of the first answer.
  - They go into /goal's `people` field and the constraint method takes them as its people source, so there is no people call.
  - The plan and context calls are as in C1, from its cache for the same frame and trial. Only the constraints call is new, because its people text differs.
- **Detector people in the numbered image** follow E1b's DetectorPeople convention:
  - each person's world point 1.1 m up (`person_mark_height_m`) is projected into the request frame;
  - a person whose point does not project (behind the camera, or outside the image) is left out;
  - yaw is the track's (F5);
  - people are numbered 1..n in ascending track id.
- **First match.** The answer's numbers are the track ids by construction (number k is the k-th track id), so the first match is that mapping, not a Hungarian match.
- **After the answer:** exactly CTd, i.e. CT2 with detector tracks every 0.5 s, stale tracks kept.
- **Pairing:** CTe against CTd on the same scenarios × Sonnet seeds 10–12 (15 episodes each), scored by the three frozen scorings.

## Measures
- the first answer's latency (sim time from the request to the delivery);
- time from the request to first in line;
- in line at the end;
- share in line;
- keep-up per service event;
- reached the front;
- lost place;
- cut-ins;
- whether the first answer has a usable place in turn (turn_m ≤ 0.3 at the first solve).

## Prediction (inference, written before any episode)
- **The first answer arrives about 3.6 s sooner** (E1b's Sonnet people-call time), with about the same answer quality.
- **The time to first in line is earlier by a similar amount.**
- **Keep-up and the other measures are about equal to CTd's.**
- **No judged bar.** It is a measurement, reported beside the projection from E1b timings.

## Calls
