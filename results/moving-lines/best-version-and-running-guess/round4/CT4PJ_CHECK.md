# CT4PJ: J1 failed on its premise; the port is right (analysis, 3 Nov; J2-J5 not computed by anyone yet)

- **The port is right** (`g_ct4pj_port.py`, output `g_ct4pj_port_out.txt`, reading only the join and "behind"
  fields):
  - On the 1,577 re-solves where the arm logs joined, I rebuilt "behind" from the tracks and the geometry Resolver:
    1,570 match exactly. The other 7 differ by one person each, at the boundary, because the robot's position is
    interpolated between 1-s samples.
  - The solver's last is never a logged "behind" id.
  - the rebuild of the join test from the logs agrees on 3,441 of 3,534 re-solves. The first join is
    the same in 22 of 24 episodes; the other 2 sit at the 0.35 m radius, within interpolation error.
- **J1's 56 violations:**
  - 50 are on cafe_1031, where the arm never joined. The sim had the robot in the queue, but the robot was mostly
    1.8-4 m from the previous re-solve's goal, so the Resolver's join test cannot fire.
  - 6 are on cafe_1081 with the arm joined. That layout has no path, so "behind" is distance to the target point,
    as CT4PJ.md specifies (CT3F has the same kind: 10 ticks).
- **The premise that failed:** J1 assumed the arm's join test fires whenever the sim has the robot in the queue. That
  holds for CT3F. For the role belief it does not: on cafe_1031 the goal stays far from where the robot stands, so
  the commitment rule never engages. That is a result in itself: commitment cannot help if the goal never comes to
  the robot.
