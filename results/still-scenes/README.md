# Still scenes

Realistic 3D scenes rendered in Blender; each try is scored against hand-written regions of good spots. Each folder
has a README with its question, result and files.

| Folder | Question |
|---|---|
| [`method-comparison/`](method-comparison/) | Does the constraint method beat pointing, layout by layout? |
| [`fresh-scenes/`](fresh-scenes/) | Does it hold on 70 new scenes never used in development? |
| [`new-situations/`](new-situations/) | A line that bends, and someone using a machine: does it carry over? |
| [`hidden-situations-and-more-models/`](hidden-situations-and-more-models/) | Situations kept hidden until the end, and older and newer models. |
| [`people-finder/`](people-finder/) | Can a small people detector replace one VLM question? |
| [`numbered-spots/`](numbered-spots/) | Picking from numbered spots instead of pointing. |
| [`group-of-four-case/`](group-of-four-case/) | Why the method misses one conversation-group layout. |
| [`run-to-run-noise/`](run-to-run-noise/) | How much do results move between repeated runs? |
| [`depth-fix/`](depth-fix/) | Moving a target that lands on someone in front onto the right person. |
| [`perception/`](perception/) | Finding people and lifting points to 3D. |
| [`facing/`](facing/) | How well the models judge which way people face. |
| [`metrics-and-statistics/`](metrics-and-statistics/) | How the scores and comparisons are computed. |
| [`learned-models/`](learned-models/) | A learned density of good spots and a trained spot model. |
| [`crowds/`](crowds/) | Dense crowds, and a facing gate for joining a line. |
| [`confidence-signal/`](confidence-signal/) | Can disagreement between perception sources flag a wrong answer? |
| [`combining-models/`](combining-models/) | Voting between models, and passing hard cases to a stronger one. |
| [`speed/`](speed/) | Where the time goes in an answer. |
| [`thinking-levels/`](thinking-levels/) | Does more model thinking help? |
| [`other-checks/`](other-checks/) | Smaller checks. |
| [`replay-harness/`](replay-harness/) | The exact offline replay of recorded runs that the analyses use. |
| [`overview/`](overview/) | The October analysis report, its re-run, and the notes of each phase. |
| [`analysis-scripts/`](analysis-scripts/), [`scene-seeds/`](scene-seeds/) | Shared analysis scripts, and the seeds of the scenes built for training and tests. |
