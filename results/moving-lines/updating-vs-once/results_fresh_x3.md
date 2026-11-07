# Fresh seeds: 60 episodes; models ['soclaas_qwen38_next_low']

## soclaas_qwen38_next_low: evaluation scenarios, seeds [10, 11, 12]

| Arm | Episodes | Share (frozen) | Share (first 0.6 m) | Share (strict) | End in line (strict) | Median first in line, strict (s) | Back in line within 10 s of a service | Cut-in samples |
|---|---|---|---|---|---|---|---|---|
| R1 | 15 | 0.18 | 0.12 | 0.04 | 0 | 33.97 | 3/17 | 33 |
| Rm | 15 | 0.21 | 0.15 | 0.05 | 0 | 41.97 | 4/17 | 40 |
| CT2 | 15 | 0.14 | 0.13 | 0.12 | 4 | 54.47 | 6/8 | 56 |
| CT2m | 15 | 0.14 | 0.13 | 0.11 | 4 | 54.47 | 6/7 | 93 |

## soclaas_qwen38_next_low: paired by (model, scenario, seed); share in line under the strict scoring

| Comparison | Which | Pairs | Wins / losses / ties | Mean difference | Sign p | Same, frozen scoring (W/L/T) |
|---|---|---|---|---|---|---|
| CT2 vs R1 |  | 15 | 9 / 0 / 6 | +0.076 | 0.004 | 6 / 5 / 4 |
| CT2m vs CT2 | latency (CT2m_FROZEN) | 15 | 3 / 5 / 7 | -0.010 | 0.727 | 3 / 6 / 6 |
| Rm vs R1 | latency, R (CT2m_FROZEN) | 15 | 1 / 1 / 13 | +0.014 | 1.000 | 4 / 1 / 10 |
| CT2m vs Rm | headline across | 15 | 7 / 1 / 7 | +0.052 | 0.070 | 4 / 6 / 5 |
