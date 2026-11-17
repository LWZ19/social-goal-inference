# Round 5, check 1: QB as specified fails; an away-only variant is logged as exploratory (3 Nov)

**QB v1 (ROUND5.md 13eecdfd) fails check 1, so it stops here, as fixed** (`b_falsefire_out.txt`). Share of walking
bouts that break a step-up bound:

| Who | design cells | pilot layouts |
|---|---|---|
| members who stay | 0.202 (path 233, away 14) | 0.205 (path 46, away 17) |
| ex-members | 0.818 (away 697) | 0.819 (away 120) |
| joiners | 0.788 | 0.841 |
| never members | 0.644 | 0.585 |

The bar was at most 5% of member bouts. The path bound (2.0 m) is what fails: members do walk more than 2 m in one
bout. So "a step-up is short" is false in these scenes, even though the median member bout is 0.7-0.9 m.

**QB v2, exploratory, chosen after seeing check 1's data:** only the away bound (net change in distance to the target
over the bout > +0.5 m) triggers `H_LEAVE`.
- On check 1's own data, member false fires are 14/1224 (1.1%) on the design cells and 17/307 (5.5%) on the pilot
  layouts, so it is above 5% on the pilot.
- Ex-members fire 81% and 77%.

Because the bound was picked after seeing this data, v2 is not a candidate under ROUND5.md. Its check 2 below is
exploration and is labelled as such. Any closed-loop use would need its own predictions written first, and in the
end a test on fresh layouts.

## Check 2 for QB v2 (exploratory; `c_eval_pilot_out.txt`, pilot layouts, robot idle)

| Variant | recall | false ex-member ticks | false walker ticks | far | cut | empty | proxy goal in line |
|---|---|---|---|---|---|---|---|
| W0 | 0.825 | 1,494 | 528 | 0.340 | 0.304 | 0.014 | 0.232 |
| QB v2 | 0.716 | 828 | 245 | 0.261 | 0.383 | 0.038 | 0.269 |

QB v2 roughly halves the false ex-members and walkers, lowers far, and has the best proxy goal of any arm (CT4 0.241).
But it loses recall (−0.11), so cuts and empties rise. ROUND5.md's bars would fail on recall and empty.

**Why recall drops** (`d_recall_out.txt`, `d2_fires_out.txt`):
- 1,048 member-ticks are lost. 545 had a broken bout in the 10 s before, 433 were members for under 10 s, and 981
  are people the answer named.
- QB v2's broken bouts, by status when the bound broke:

  | never a member | joiner on the way to the tail | member (false fire) | ex-member |
  |---|---|---|---|
  | 168 | 143 | 22 | 5 |

- Joining means walking to the tail, which from the front is walking away from the counter. QB v2 puts in the
  leaving half of the process (walking away means leaving) without the arriving half (stopping at the tail means
  joining). So joiners are pushed to about 0 on the way, and W0's slow join hazard (0.05/s) takes many seconds to
  bring them back.

**QB v3 (exploratory, chosen after this diagnosis):** QB v2 plus arrival at the tail. For 2 s after a walking bout
ends, if the person stands where the line as currently believed would put its next member, the join hazard is
`H_JOIN_TAIL` = 1.0/s, the same rate as `H_LEAVE`.
- "Where the line would put its next member" uses W0's static residual against the roster at P ≥ 0.5 (without
  them), below `R0` (0.8 m, the existing neutral point).
- Otherwise joining is W0's.
- New parameters: `STOP_WIN_S` 2.0 and `H_JOIN_TAIL` 1.0, set a priori.

## Check 2 for QB v3 (exploratory; `c_eval_pilot_out.txt`)

| Variant | recall | false ex-member | false walker | false other bystander | right | cut | far | empty | proxy |
|---|---|---|---|---|---|---|---|---|---|
| W0 | 0.825 | 1,494 | 528 | 75 | 0.342 | 0.304 | 0.340 | 0.014 | 0.232 |
| QB v3 | 0.840 | 1,026 | 794 | 445 | 0.319 | 0.289 | 0.383 | 0.009 | 0.229 |

Recall comes back, and empties and cuts fall. But walkers and bystanders who were not named are now let in at the
tail: far rises and the proxy falls. Bystanders fidget (0.35 m every 5-10 s), so each fidget ends a tiny bout. v3's
tail join fires for 2 s after any bout, without W0's own arrival condition (moved ARRIVE_M 1.0 m within ARRIVE_S
20 s). That condition was left out by oversight.

**QB v4 (exploratory, the last variant of this round):** v3 with the tail join also requiring W0's existing arrival
test (`_arrived`). No new parameter. Whatever v4 shows, round 5 stops after it and is written up.

## Check 2 for QB v4 (exploratory; `c_eval_pilot_out.txt`, pilot layouts, robot idle)

| Variant | recall | false ex-member | false walker | false other bystander | right | cut | far | empty | proxy |
|---|---|---|---|---|---|---|---|---|---|
| CT4 | 0.856 | 1,300 | 762 | 73 | 0.400 | 0.263 | 0.323 | 0.014 | 0.241 |
| W0 | 0.825 | 1,494 | 528 | 75 | 0.342 | 0.304 | 0.340 | 0.014 | 0.232 |
| QB v4 | 0.774 | 940 | 576 | 94 | 0.384 | 0.300 | 0.288 | 0.027 | 0.283 |

Against W0, the decision measures improve:
- right +0.04;
- far −0.05;
- proxy goal in line +0.05, the best of any arm and above CT4's 0.241.

The belief measures get worse: recall −0.05, empty +0.013, walkers +9%, and ex-members −37% (the bar was 40%). So
v4 would fail ROUND5.md's bars on recall, walkers, empties and the ex-member margin. Those bars were written for v1,
but they are the standard I hold v4 to as well. So v4 is not a candidate.

## Round 5 stops here (variants logged: v1 failed check 1; v2, v3 and v4 exploratory)

What it shows:
- **Leaving (walking away from the counter) is a strong, specific signal.** 77-82% of ex-member bouts break the away
  bound; 1-5% of member bouts do. It halves false ex-members and walkers (v2).
- **Joining cannot be told apart by position or kinematics in these scenes.** The "ours" bystanders stand exactly
  where the next member would, so a fast join at the tail admits them (v3). Requiring a real arrival (v4) restores
  most of the balance, but recall still sits below W0's.
- **The separating evidence for joiners is the one the belief already has:** co-motion at service. Process structure
  speeds up the leaving side, but on the joining side the information arrives only at service events.
- **Offline, v4 gives the best proxy goal in line of any belief.** But it fails its belief bars, and offline replay
  has not been shown to predict closed-loop time in line. Nothing here goes to the sim before the confirmatory
  result. If anything does go later, it would be v4 with predictions written first, and a pilot on design layouts
  before any fresh test.

The design-cell run of the same variants is in `c_eval_design_out.txt` (added when it finishes).

## Design cells × 4 conditions (`c_eval_design_out.txt`; 129 cells, pooled)

| Variant | recall | right | cut | far | empty |
|---|---|---|---|---|---|
| W0 | 0.782 | 0.352 | 0.164 | 0.453 | 0.031 |
| QB v4 | 0.728 | 0.358 | 0.197 | 0.396 | 0.050 |

The pilot's gain in right share does not hold on the design cells (+0.006). v4 trades far (−0.057) for cut (+0.033)
and empty (+0.019), with lower recall (−0.054). It swaps one error for another rather than removing one. This
confirms it is not a candidate.

v2 and v3 show the same tradeoff more sharply:
- v2: far 0.313, cut 0.254, empty 0.095;
- v3: far 0.500, false other bystanders 45,344.

The process-structured leave hazard moves errors between kinds. It does not lower the total, because the joining
side has no evidence that separates joiners from the "ours" bystanders before a service.
