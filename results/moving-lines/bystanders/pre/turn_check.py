"""Zero calls, real solver code: does sgi.solver._turn take a bystander as the last (the robot's place goes behind
them) at the scenario's nominal places? This rule runs inside every solve, before and regardless of the resolver's
admission gate (Resolver.record), so a gate in the resolver alone cannot stop it.
World frame; numbers: 1.. members front to back, 11.. bystanders. Target point = slot 0's face point.
    .venv/bin/python turn_check.py   (main checkout venv)"""
import numpy as np
from sgi import solver as S
from sgi.config import Thresholds
th = Thresholds()
CASES = {
    'cafe_line2': dict(target=(5.0, -0.5), slots=[(5.0, -1.45), (5.0, -2.15), (5.0, -2.85)],
                       bys={11: (4.2, -2.6), 12: (5.85, -3.45), 13: (6.6, -3.7)}, robot=(8.0, -2.6)),
    'room_cart': dict(target=(5.0, -1.0), slots=[(5.0, -2.5), (5.0, -3.2), (5.0, -3.9), (5.0, -4.6)],
                      bys={11: (4.2, -4.35), 12: (4.0, -5.15), 13: (3.4, -5.5)}, robot=(8.0, -4.0)),
    'sweet_spot_line2': dict(target=(15.8, 5.2), slots=[(15.8, 4.35), (15.8, 3.65), (15.8, 2.95)],
                             bys={11: (16.55, 3.1), 12: (16.9, 2.2), 13: (17.5, 2.6)}, robot=(13.0, 1.4)),
}
for name, c in CASES.items():
    for n in range(1, len(c['slots']) + 1):
        for line in (None, np.array([0.0, -1.0])):
            bodies = {i + 1: S._Body(i + 1, np.array(xy), None) for i, xy in enumerate(c['slots'][:n])}
            bodies.update({k: S._Body(k, np.array(xy), None) for k, xy in c['bys'].items()})
            for apart in (set(), {12, 13}):
                turn = S._turn(list(range(1, n + 1)), apart, bodies, None, np.array(c['target']), np.array(c['robot']),
                               th, line)
                print(f'{name:16} line of {n}  direction={"remembered" if line is not None else "from people"}  '
                      f'pair exempt={bool(apart)}:  joined {turn.joined or "-"}  place {np.round(turn.place, 2)}')
