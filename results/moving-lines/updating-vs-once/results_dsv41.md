# Moving people: 201 episodes, models ['soclaas_dsv41']

## Evaluation scenarios (all but cafe_busy and the break-it cases)

| Arm | Episodes | In line at end | Mean share in line | Median first in line (s) | Never | Any cut-in | Lost place | Calls / ep | Median latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 15 | 4 | 0.20 | 18 | 9 | 6 | 5 | 3.0 | 11.8 |
| Rq | 15 | 1 | 0.16 | 26 | 4 | 11 | 19 | 20.6 | 5.3 |
| Rq0 | 15 | 4 | 0.14 | 25 | 3 | 15 | 21 | 30.4 | 5.8 |
| RT | 15 | 3 | 0.09 | 31 | 6 | 11 | 15 | 3.0 | 11.8 |
| RTw | 15 | 2 | 0.08 | 31 | 6 | 10 | 19 | 3.0 | 11.8 |
| C1 | 15 | 0 | 0.03 | 32 | 6 | 4 | 12 | 4.0 | 20.6 |
| Cq | 15 | 5 | 0.15 | 30 | 4 | 8 | 21 | 20.8 | 10.3 |
| CTv | 15 | 4 | 0.17 | 25 | 6 | 6 | 13 | 4.0 | 20.3 |
| CT | 15 | 4 | 0.18 | 26 | 5 | 6 | 16 | 4.0 | 20.8 |

## CT against each arm, paired by (model, scenario, seed)

| Other arm | Pairs | CT share wins / losses / ties | Mean share difference | End: CT only / other only |
|---|---|---|---|---|
| R1 | 15 | 7 / 4 / 4 | -0.02 | 2 / 2 |
| Rq | 15 | 7 / 6 / 2 | +0.02 | 3 / 0 |
| Rq0 | 15 | 8 / 5 / 2 | +0.04 | 2 / 2 |
| RT | 15 | 9 / 3 / 3 | +0.10 | 2 / 1 |
| RTw | 15 | 7 / 3 / 5 | +0.10 | 2 / 0 |
| C1 | 15 | 9 / 4 / 2 | +0.16 | 4 / 0 |
| Cq | 15 | 8 / 5 / 2 | +0.03 | 2 / 3 |
| CTv | 15 | 6 / 1 / 8 | +0.02 | 0 / 0 |

## Share in line by scenario (mean over seeds)

| Scenario | R1 | Rq | Rq0 | RT | RTw | C1 | Cq | CTv | CT0 | CT | CTo | CTt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cafe | 0.00 (0/3) | 0.13 (0/3) | 0.19 (1/3) | 0.01 (0/3) | 0.01 (0/3) | 0.00 (0/3) | 0.02 (0/3) | 0.00 (0/3) | - | 0.07 (0/3) | - | - |
| cafe_busy (dev) | 0.08 (0/3) | 0.10 (0/3) | - | - | - | 0.01 (0/3) | 0.07 (0/3) | - | - | 0.08 (0/3) | - | - |
| cafe_line2_fast | 0.58 (2/3) | 0.52 (2/2) | - | 0.27 (0/3) | - | 0.14 (0/3) | - | 0.20 (1/3) | - | 0.20 (1/3) | - | - |
| cafe_line2_moving | 0.55 (2/3) | 0.28 (1/3) | 0.16 (2/3) | 0.26 (2/3) | 0.21 (1/3) | 0.07 (0/3) | 0.47 (3/3) | 0.26 (1/3) | - | 0.26 (1/3) | - | - |
| cafe_line6_moving | 0.00 (0/3) | 0.29 (0/3) | 0.21 (1/3) | 0.00 (0/3) | 0.00 (0/3) | 0.02 (0/3) | 0.11 (1/3) | 0.08 (1/3) | - | 0.08 (1/3) | - | - |
| cafe_line6_stepout | 0.00 (0/3) | 0.02 (0/2) | - | 0.01 (0/3) | - | 0.02 (0/3) | - | 0.11 (0/3) | - | 0.13 (0/3) | - | - |
| cafe_line_bent_moving | 0.00 (0/3) | 0.00 (0/2) | - | 0.00 (0/3) | - | 0.00 (0/3) | - | 0.03 (0/3) | - | 0.04 (0/3) | - | - |
| room_coffee_cart | 0.17 (1/3) | 0.05 (0/3) | 0.12 (0/3) | 0.12 (1/3) | 0.15 (1/3) | 0.01 (0/3) | 0.04 (1/3) | 0.43 (2/3) | - | 0.45 (2/3) | - | - |
| sweet_spot_line2_moving | 0.27 (1/3) | 0.05 (0/3) | 0.04 (0/3) | 0.04 (0/3) | 0.04 (0/3) | 0.03 (0/3) | 0.13 (0/3) | 0.06 (0/3) | - | 0.06 (0/3) | - | - |

## Once the robot was first in line (Q5)

| Arm | Episodes ever first | In line while first (share of those samples) | In line at end |
|---|---|---|---|
| R1 | 3 of 15 | 1.00 | 3 |
| Rq | 3 of 15 | 0.48 | 1 |
| Rq0 | 1 of 15 | 0.55 | 1 |
| RT | 2 of 15 | 0.33 | 1 |
| RTw | 2 of 15 | 0.11 | 0 |
| C1 | 7 of 15 | 0.00 | 0 |
| Cq | 7 of 15 | 0.77 | 4 |
| CTv | 6 of 15 | 0.79 | 3 |
| CT | 6 of 15 | 0.80 | 3 |

## Staleness of answers: how far the line's people moved between the frame sent and the answer

| Arm | Answers | Median mean move (m) | Median max move (m) | Median latency (s) |
|---|---|---|---|---|
| Rq0 | 192 | 0.48 | 0.69 | 5.8 |
| RT | 15 | 0.00 | 0.00 | 11.8 |
| RTw | 15 | 0.00 | 0.00 | 11.8 |
| CTv | 15 | 0.13 | 0.13 | 20.3 |

## First-match rate of the model's numbered people to tracks (CT, CTv)

- soclaas_dsv41: mean 0.94 over 30 episodes, min 0.75

## Evaluation scenarios: post hoc measures (Q9; not frozen)

| Arm | Episodes | Reached the front in turn | Median time to front (s) | Keep-up after services | Share in line while someone is ahead | Goal left the line |
|---|---|---|---|---|---|---|
| R1 | 15 | 1 | 34 | 0/7 | 0.11 | 0 |
| Rq | 15 | 1 | 34 | 3/11 | 0.13 | 10 |
| Rq0 | 15 | 1 | 32 | 2/7 | 0.14 | 11 |
| RT | 15 | 0 | - | 6/7 | 0.08 | 12 |
| RTw | 15 | 0 | - | 4/6 | 0.08 | 11 |
| C1 | 15 | 0 | - | 0/16 | 0.03 | 0 |
| Cq | 15 | 0 | - | 2/15 | 0.08 | 6 |
| CTv | 15 | 2 | 87 | 14/14 | 0.12 | 4 |
| CT | 15 | 2 | 87 | 15/16 | 0.14 | 7 |

## Evaluation scenarios: initial correctness x maintenance (Q10; post hoc)

| Arm | Episodes | Settled | In line at first settle | ...and at the end | Keep-up from in line | First answer has a place in turn |
|---|---|---|---|---|---|---|
| R1 | 15 | 15 | 3 | 2 of 3 | 1/5 | - |
| Rq | 15 | 15 | 5 | 1 of 5 | 5/9 | - |
| Rq0 | 15 | 15 | 6 | 3 of 6 | 6/9 | - |
| RT | 15 | 11 | 1 | 0 of 1 | 2/3 | - |
| RTw | 15 | 13 | 2 | 0 of 2 | 3/5 | - |
| C1 | 15 | 15 | 4 | 0 of 4 | 0/5 | 8 of 15 |
| Cq | 15 | 15 | 4 | 2 of 4 | 0/7 | 8 of 15 |
| CTv | 15 | 15 | 5 | 2 of 5 | 9/9 | 8 of 15 |
| CT | 15 | 15 | 7 | 3 of 7 | 8/9 | 8 of 15 |

## Development scenario cafe_busy

| Arm | Episodes | In line at end | Mean share in line | Median first in line (s) | Never | Any cut-in | Lost place | Calls / ep | Median latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 3 | 0 | 0.08 | 47 | 1 | 2 | 2 | 3.3 | 15.1 |
| Rq | 3 | 0 | 0.10 | 37 | 0 | 2 | 3 | 21.3 | 5.5 |
| C1 | 3 | 0 | 0.01 | 42 | 1 | 2 | 2 | 4.3 | 19.4 |
| Cq | 3 | 0 | 0.07 | 39 | 0 | 3 | 5 | 23.3 | 10.9 |
| CT | 3 | 0 | 0.08 | 39 | 0 | 3 | 6 | 4.3 | 19.2 |

## Break-it case cafe_line2_fast

| Arm | Episodes | In line at end | Mean share in line | Median first in line (s) | Never | Any cut-in | Lost place | Calls / ep | Median latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 3 | 2 | 0.58 | 15 | 0 | 1 | 2 | 3.0 | 10.4 |
| Rq | 2 | 2 | 0.52 | 27 | 0 | 1 | 2 | 18.0 | 6.8 |
| RT | 3 | 0 | 0.27 | 14 | 0 | 3 | 8 | 3.0 | 10.4 |
| C1 | 3 | 0 | 0.14 | 29 | 0 | 2 | 8 | 4.0 | 20.4 |
| CTv | 3 | 1 | 0.20 | 45 | 0 | 3 | 4 | 4.0 | 20.4 |
| CT | 3 | 1 | 0.20 | 45 | 0 | 3 | 4 | 4.0 | 20.3 |

## Break-it case cafe_line6_stepout

| Arm | Episodes | In line at end | Mean share in line | Median first in line (s) | Never | Any cut-in | Lost place | Calls / ep | Median latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 3 | 0 | 0.00 | - | 3 | 2 | 0 | 3.0 | 11.1 |
| Rq | 2 | 0 | 0.02 | 34 | 0 | 2 | 3 | 22.0 | 7.1 |
| RT | 3 | 0 | 0.01 | 41 | 2 | 3 | 2 | 3.0 | 11.1 |
| C1 | 3 | 0 | 0.02 | 33 | 0 | 1 | 3 | 4.0 | 23.2 |
| CTv | 3 | 0 | 0.11 | 26 | 1 | 2 | 4 | 4.0 | 23.2 |
| CT | 3 | 0 | 0.13 | 33 | 2 | 1 | 1 | 4.0 | 23.2 |

## Break-it case cafe_line_bent_moving

| Arm | Episodes | In line at end | Mean share in line | Median first in line (s) | Never | Any cut-in | Lost place | Calls / ep | Median latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 3 | 0 | 0.00 | - | 3 | 3 | 0 | 3.0 | 6.7 |
| Rq | 2 | 0 | 0.00 | - | 2 | 2 | 0 | 20.0 | 5.2 |
| RT | 3 | 0 | 0.00 | - | 3 | 3 | 0 | 3.0 | 6.8 |
| C1 | 3 | 0 | 0.00 | - | 3 | 0 | 0 | 4.0 | 13.2 |
| CTv | 3 | 0 | 0.03 | 36 | 0 | 3 | 5 | 4.0 | 13.2 |
| CT | 3 | 0 | 0.04 | 36 | 0 | 2 | 6 | 4.0 | 13.2 |
