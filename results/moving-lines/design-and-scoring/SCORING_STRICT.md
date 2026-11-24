# How moving-line runs are scored

Each run lasts 92 s. From the request to the end of the run, the simulator checks the robot once a second against the
line as it stands at that moment. **Time in the right spot** is the share of these samples in which the robot is in line
under the rule below.

## The line

The line runs along the queue's slot path, from slot 0 (served next) backwards. A point's `along` is its distance along
that path, and its `side` is its distance from it. The members are the people in the queue at that moment.

A member is **ahead of the robot** if they joined the queue before the robot did. A served person leaves the queue. If
they come back, they join again, so someone who rejoins after the robot joined is behind it.

## In line

**When someone is ahead of the robot**, let L be the member ahead of it who is farthest back along the line. The sample
is in line when all of these hold:

- the robot is not ahead of L along the line
- the robot is 0.5 to 1.2 m from L
- the robot is within 0.6 m of the line (`side` <= 0.6 m)
- the robot's heading is within 45 degrees of facing up the line
- the robot is at least 0.55 m from every person

**When nobody is ahead of the robot**, the sample is in line when all of these hold:

- the robot is within 0.6 m of slot 0
- the robot faces within 45 degrees of the bearing to the serving point
- the robot is at least 0.55 m from every person

## Cutting in

A sample counts as a cut-in when the robot is within 1.0 m of the line and more than 0.25 m ahead along it of a member
who is ahead of it in queue order.

## Comparing methods

Methods are paired by scenario, seed and model. Results are reported as the mean share, as wins, losses and ties, and
with a sign test.

In the outputs this score is the `strict` share, and the results use it. The outputs also print two looser scores:

- `primary`, also labelled `looser` (field `share_frozen`). It does not check the distance from the line. When nobody is
  ahead, it accepts a robot within 1.6 m of the front slot.
- `first 0.6 m`. It is the same score, with 0.6 m when nobody is ahead.
