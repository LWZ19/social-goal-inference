# Moving lines

Habitat 3.0 with Spot and animated people: the front person is served, walks off and rejoins at the back. The score
is the share of time the robot spends in the right spot over a 92-second run. Each folder has a README with its
question, result and files.

| Folder | Question |
|---|---|
| [`design-and-scoring/`](design-and-scoring/) | How the runs work and how they are scored; the run script. |
| [`updating-vs-once/`](updating-vs-once/) | Does re-solving the spot as people move beat deciding once? |
| [`school-models/`](school-models/) | The same on the school's slower free models. |
| [`re-asking/`](re-asking/) | Does re-solving beat asking the VLM again and again? |
| [`roles-vs-marked-spot/`](roles-vs-marked-spot/) | Do the VLM's roles carry the gain, or would following its marked spot do? |
| [`own-camera/`](own-camera/) | Does tracking people with the robot's own camera cost time in line? |
| [`people-finder/`](people-finder/) | Giving the VLM the detector's people instead of asking it who is there. |
| [`bystanders/`](bystanders/) | Ignoring people who stand near the line without queuing. |
| [`best-version-and-running-guess/`](best-version-and-running-guess/) | The best rule-based version against a running guess of who waits. |
| [`best-version-new-layouts/`](best-version-new-layouts/) | The best version on fresh layouts. |
| [`test-robots/`](test-robots/) | Robots given part of the true answer, to show what each missing piece costs. |
| [`front-of-line-fix/`](front-of-line-fix/) | A rule-based fix for the front of the line. |
| [`bumps/`](bumps/) | How close the robot comes to people. |
| [`cut-ins-and-decomposition/`](cut-ins-and-decomposition/) | Does updating cut in more often, and where does its gain come from? |
| [`walking-while-thinking/`](walking-while-thinking/), [`settled-tracks/`](settled-tracks/) | Walking before the answer arrives; re-solving only on settled tracks. |
| [`test-layouts/`](test-layouts/) | How the fresh test layouts were generated and chosen. |
| [`failure-analysis/`](failure-analysis/), [`questions/`](questions/) | Where the time out of line goes, and follow-up questions. |
