# Plain-language results for the progress page (7 Nov)

**Terms.**
- "Pointing": the model marks a spot in the image.
- "The constraint method": the model says who to stand behind and how, and a solver places the robot.
- "Time in line": the share of an episode the robot stands correctly in the queue.

## Still scenes

- **Fresh still scenes (GLM still running).**
  - On 70 new layouts no prompt was tuned on, the constraint method beat pointing on all five finished models, each
    clearly (p < 0.0001). For example, Haiku 260 against 151 successes of 280, and Sonnet 257 against 198.
  - the analysis is as large as on the layouts we developed on. It comes mostly from waiting lines: pointing 4-52 of 100,
- **New situations: a bent waiting line, and someone using a machine (GLM still running).**
  - The constraint method won on all 12 layouts in both situations, on all five finished models (p 0.0005, the
    strongest result possible with 12 layouts).
  - Bent lines: pointing succeeded 0-4 times of 72 per model, against 40-66.
  - Machine use: 0-20 against 31-66.
  - Pointing's misses are mostly 0.3-1.7 m from the right spot.
- **How much results move between repeat runs (DeepSeek; GLM still running).**
  - In three repeats of the same runs, the constraint method's lead over pointing stayed between +65 and +93 of 252
    each time.
  - That is far more than the repeat-to-repeat spread (at most 34).
- **People from the camera's detector, still scenes.** Using the detector's people instead of asking the model was
  faster and more accurate over the three free models: 41 more successes, p 0.0001. Qwen's time per scene fell from
  107 s to 83 s.

## Moving scenes

- **Who to stand behind, against following a marked spot.** Keeping the model's "who to stand behind" answer and
  updating it as people move beat following the person beside the model's marked spot.
  - Time in line was +0.20 on average: 87 of 125 scenes better, 31 worse (p < 0.0001).
  - The marked-spot follower fails because that person is usually served and walks away (117 of 140 episodes).
  - It holds on Sonnet, DeepSeek and Haiku, not on Gemini.
- **Older models.** Updating one answer as people move beat asking once on 5 of 7 older models: Sonnet 4.6 and 5, and
  three Gemini Flash versions (p 0.013 or lower). It did not on Haiku 4.5, where 10 of 25 answers were unusable, or on
  Gemini 3 Flash.
- **The best moving method on fresh layouts: not shown.**
  - The free gateway was about 6 times slower than on 2-3 Nov, so the first answer reached the robot in time in only
    39 of 72 layouts.
  - Where it did, the best method was ahead (9 layouts better, 2 worse, p 0.065), the same direction as before.
- **Knowing the exact serving spot.** If the robot knew exactly where the counter serves, the best method's time in
  line would rise by 0.08 on average (46 layouts better, 14 worse, four models). The gain is mostly at the front of the
  line. So observing the serving spot is the first thing to add next.
- **Bystanders, on a school model (Qwen).**
  - Using how people move stops the robot lining up behind someone who is only standing nearby. Over 60 scenes, time
    stuck behind bystanders fell from 2,065 s to 259 s, and time in line rose (25 scenes better, 4 worse).
  - Read with 240 s episodes, since Qwen's answers take about 2 minutes.
- **The camera detector's people in moving scenes: not shown.** On DeepSeek, giving the model the detector's people
  lowered time in line (0.10 against 0.29) and did not make the first answer earlier. On Qwen both versions were too
  slow to compare. This differs from still scenes, where the detector helped.
- **Following people from the robot's own camera.** Tracking people from the robot's camera cost no measurable time
  in line against the sim's true positions: +0.004 on average, and at worst −0.023. It helps in some scenes and hurts
  in others.
- **Updating one answer, against asking the model again and again.** Updating one answer as people move beat
  re-asking every few seconds on Sonnet: +0.31 and +0.23 time in line against the two re-asking versions. DeepSeek
  points the same way; Qwen was too slow to test.
- **The school models' speed.**
  - Qwen: given 240 s so its roughly 2-minute answers arrive, updating one answer beat asking once in 19 of 25 scenes,
    against 1.
  - GLM: still too slow even at 240 s. Its answer arrived in time in only 1 of 25, so it can't be judged on moving
    scenes at the gateway's current speed.
