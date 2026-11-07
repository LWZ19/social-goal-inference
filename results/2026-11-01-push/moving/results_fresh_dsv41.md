# Fresh seeds: 255 episodes; models ['soclaas_dsv41']

## soclaas_dsv41: evaluation scenarios, seeds [10, 11, 12, 13, 14]

| Arm | Episodes | Share (frozen) | Share (first 0.6 m) | Share (strict) | End in line (strict) | Median first in line, strict (s) | Back in line within 10 s of a service | Cut-in samples |
|---|---|---|---|---|---|---|---|---|
| C1 | 15 | 0.02 | 0.02 | 0.02 | 0 | 28.97 | 4/14 | 122 |
| R1 | 25 | 0.16 | 0.06 | 0.04 | 1 | 18.97 | 4/13 | 504 |
| Rm | 25 | 0.17 | 0.08 | 0.04 | 1 | 26.47 | 5/12 | 541 |
| CT | 25 | 0.23 | 0.22 | 0.20 | 4 | 30.97 | 23/34 | 242 |
| CT2 | 25 | 0.31 | 0.29 | 0.27 | 5 | 31.97 | 36/45 | 226 |
| CT2m | 25 | 0.31 | 0.28 | 0.26 | 6 | 33.97 | 35/42 | 205 |
| CT2-rat | 25 | 0.25 | 0.22 | 0.21 | 2 | 34.97 | 30/45 | 231 |
| CT2-nora | 25 | 0.28 | 0.27 | 0.26 | 5 | 30.97 | 33/38 | 294 |
| CT2-norv | 25 | 0.30 | 0.29 | 0.28 | 6 | 29.97 | 36/39 | 221 |
| CT2-none | 25 | 0.22 | 0.20 | 0.18 | 3 | 32.97 | 23/32 | 244 |
| CTd | 15 | 0.28 | 0.25 | 0.25 | 2 | 29.97 | 19/25 | 199 |

## soclaas_dsv41: paired by (model, scenario, seed); share in line under the strict scoring

| Comparison | Which | Pairs | Wins / losses / ties | Mean difference | Sign p | Same, frozen scoring (W/L/T) |
|---|---|---|---|---|---|---|
| CT2 vs CT | primary (CT2_FROZEN) | 25 | 18 / 3 / 4 | +0.078 | 0.001 | 17 / 3 / 5 |
| CT2 vs R1 |  | 25 | 19 / 3 / 3 | +0.233 | 0.001 | 15 / 7 / 3 |
| CT2m vs CT2 | latency (CT2m_FROZEN) | 25 | 7 / 10 / 8 | -0.012 | 0.629 | 11 / 8 / 6 |
| Rm vs R1 | latency, R (CT2m_FROZEN) | 25 | 1 / 2 / 22 | +0.005 | 1.000 | 1 / 1 / 23 |
| CT2m vs Rm | headline across | 25 | 19 / 1 / 5 | +0.217 | 0.000 | 15 / 6 / 4 |
| CT2 vs CT2-rat | ablation | 25 | 13 / 4 / 8 | +0.063 | 0.049 | 15 / 6 / 4 |
| CTd vs CT2 | tracking (E1D_FROZEN) | 15 | 5 / 4 / 6 | -0.017 | 1.000 | 6 / 4 / 5 |
| CT2 vs CT2-nora | post hoc ablation: readmit (CT2_NORA_POSTHOC) | 25 | 9 / 8 / 8 | +0.018 | 1.000 | 9 / 8 / 8 |
| CT2-nora vs CT |  | 25 | 17 / 5 / 3 | +0.060 | 0.017 | 17 / 5 / 3 |
| CT2 vs CT2-norv | post hoc: readmit + vacate (CT2_NORV_POSTHOC) | 25 | 9 / 11 / 5 | -0.007 | 0.824 | 10 / 10 / 5 |
| CT2-norv vs CT |  | 25 | 18 / 2 / 5 | +0.085 | 0.000 | 14 / 2 / 9 |
| CT2-none vs CT | sanity: every part off (CT2_NONE_RAT_POSTHOC N1) | 25 | 7 / 6 / 12 | -0.011 | 1.000 | 7 / 10 / 8 |
| CT2-rat vs CT | R2 | 25 | 11 / 7 / 7 | +0.015 | 0.481 | 12 / 9 / 4 |
| CT2 vs CT2-rat | ratchet, all cells | 25 | 13 / 4 / 8 | +0.063 | 0.049 | 15 / 6 / 4 |

## CTd: people tracked from the robot's own camera (E1D_FROZEN.md)

| Model | Episodes | Frames | Errors | Median ms | Live tracks matched to a true person (0.5 m) | Id switches / live track-frames | Stale share of tracks | First-match rate |
|---|---|---|---|---|---|---|---|---|
| soclaas_dsv41 | 15 | 2499 | 0 | 173 | 4394/4734 = 0.93 | 107/4734 | 0.67 | 0.9697802197802198 |
