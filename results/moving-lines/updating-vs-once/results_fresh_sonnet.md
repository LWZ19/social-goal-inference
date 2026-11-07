# Fresh seeds: 355 episodes; models ['claude_sonnet55_low']

## claude_sonnet55_low: evaluation scenarios, seeds [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

| Arm | Episodes | Share (frozen) | Share (first 0.6 m) | Share (strict) | End in line (strict) | Median first in line, strict (s) | Back in line within 10 s of a service | Cut-in samples |
|---|---|---|---|---|---|---|---|---|
| C1 | 15 | 0.02 | 0.02 | 0.02 | 0 | 28.97 | 6/27 | 13 |
| R1 | 50 | 0.14 | 0.13 | 0.03 | 2 | 14.97 | 4/12 | 162 |
| Rm | 50 | 0.14 | 0.11 | 0.01 | 0 | 17.97 | 2/12 | 210 |
| CT | 50 | 0.35 | 0.23 | 0.22 | 5 | 22.97 | 55/94 | 348 |
| CT2 | 50 | 0.40 | 0.31 | 0.31 | 11 | 22.97 | 62/97 | 481 |
| CT2m | 50 | 0.39 | 0.31 | 0.31 | 10 | 22.97 | 70/107 | 477 |
| CT2-vac | 15 | 0.47 | 0.29 | 0.29 | 4 | 22.97 | 20/29 | 69 |
| CT2-rat | 15 | 0.40 | 0.26 | 0.25 | 3 | 23.97 | 17/29 | 89 |
| CTs | 12 | 0.29 | 0.22 | 0.22 | 1 | 24.97 | 14/21 | 107 |
| CTd | 15 | 0.32 | 0.25 | 0.25 | 0 | 22.97 | 14/26 | 210 |
| CTe | 15 | 0.47 | 0.32 | 0.32 | 3 | 21.47 | 20/32 | 86 |

## claude_sonnet55_low: paired by (model, scenario, seed); share in line under the strict scoring

| Comparison | Which | Pairs | Wins / losses / ties | Mean difference | Sign p | Same, frozen scoring (W/L/T) |
|---|---|---|---|---|---|---|
| CT2 vs CT | primary (CT2_FROZEN) | 50 | 32 / 8 / 10 | +0.082 | 0.000 | 29 / 16 / 5 |
| CT2 vs R1 |  | 50 | 43 / 2 / 5 | +0.273 | 0.000 | 41 / 6 / 3 |
| CT2m vs CT2 | latency (CT2m_FROZEN) | 50 | 12 / 27 / 11 | +0.002 | 0.024 | 15 / 26 / 9 |
| Rm vs R1 | latency, R (CT2m_FROZEN) | 50 | 2 / 3 / 45 | -0.019 | 1.000 | 12 / 4 / 34 |
| CT2m vs Rm | headline across | 50 | 43 / 2 / 5 | +0.295 | 0.000 | 40 / 8 / 2 |
| CT2 vs CT2-vac | ablation | 15 | 5 / 5 / 5 | -0.021 | 1.000 | 3 / 7 / 5 |
| CT2 vs CT2-rat | ablation | 15 | 7 / 5 / 3 | +0.018 | 0.774 | 6 / 6 / 3 |
| CTs vs CT | CTs_FROZEN | 12 | 4 / 5 / 3 | +0.008 | 1.000 | 3 / 6 / 3 |
| CTd vs CT2 | tracking (E1D_FROZEN) | 15 | 5 / 6 / 4 | -0.018 | 1.000 | 7 / 6 / 2 |
| CTe vs CTd | detector people at answer (E1C_FROZEN) | 15 | 9 / 4 / 2 | +0.070 | 0.267 | 10 / 3 / 2 |

## CTd: people tracked from the robot's own camera (E1D_FROZEN.md)

| Model | Episodes | Frames | Errors | Median ms | Live tracks matched to a true person (0.5 m) | Id switches / live track-frames | Stale share of tracks | First-match rate |
|---|---|---|---|---|---|---|---|---|
| claude_sonnet55_low | 15 | 2518 | 0 | 162 | 3622/4073 = 0.89 | 93/4073 | 0.71 | 1.0 |

## Break-it case cafe_line6_stepout

| Arm | Episodes | Share (frozen) | Share (first 0.6 m) | Share (strict) | End in line (strict) | Median first in line, strict (s) | Back in line within 10 s of a service | Cut-in samples |
|---|---|---|---|---|---|---|---|---|
| C1 | 3 | 0.04 | 0.04 | 0.04 | 0 | 46.97 | 2/10 | 0 |
| CT | 3 | 0.39 | 0.39 | 0.38 | 0 | 22.97 | 7/12 | 5 |
| CT2 | 3 | 0.60 | 0.60 | 0.60 | 3 | 22.97 | 12/12 | 0 |

## Break-it case cafe_line_bent_moving

| Arm | Episodes | Share (frozen) | Share (first 0.6 m) | Share (strict) | End in line (strict) | Median first in line, strict (s) | Back in line within 10 s of a service | Cut-in samples |
|---|---|---|---|---|---|---|---|---|
| C1 | 3 | 0.00 | 0.00 | 0.00 | 0 | - | 0/0 | 0 |
| CT | 3 | 0.37 | 0.37 | 0.11 | 1 | 57.47 | 3/8 | 9 |
| CT2 | 3 | 0.46 | 0.46 | 0.46 | 2 | 24.97 | 12/12 | 6 |
