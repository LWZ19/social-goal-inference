"""QB (round 5): W0's belief (ct5.CT5 with W4's options) with the queue's kinematics in the leave hazard. During a
walking bout (ticks at over WALK_MPS, from the position 1 s before its first walking tick), once the bout's path
exceeds BOUT_PATH_M or its net change in distance to the target exceeds +BOUT_AWAY_M, the leave hazard is H_LEAVE
until the bout ends; otherwise the leave hazard is H_STILL (a step-up is not a way of leaving). Joining as W0
(H_MOVE after arriving and stopping). v1: BOUT_PATH_M 2.0 (failed check 1); v2 (exploratory): BOUT_PATH_M inf.
Zero calls; offline research code."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "round3"))
from ct5 import CT5  # noqa: E402

QB_DEFAULTS = dict(BOUT_PATH_M=2.0, BOUT_AWAY_M=0.5, H_LEAVE=1.0)
W4 = dict(STATIC_WIN_S=5.0, WALKERS_STATIC=True, LR_EXIT=1.0)


class QB(CT5):
    def __init__(self, cell, params=None):
        params = {**W4, **QB_DEFAULTS, **(params or {})}
        super().__init__(cell, params)
        self.bout = {}                       # id: dict(a=start xy, last=xy, path=m, broken=bool)
        self.log_broken = []

    def _update_bouts(self, t, pos, stale):
        p = self.p
        for i, xy in pos.items():
            if i in stale:
                continue
            walking = self._speed(i, t) > p["WALK_MPS"]
            if not walking:
                self.bout.pop(i, None)
                continue
            b = self.bout.get(i)
            if b is None:
                a = self._at(i, t - 1.0) or xy
                b = self.bout[i] = dict(a=a, last=a, path=0.0, broken=False)
            b["path"] += math.dist(b["last"], xy)
            b["last"] = xy
            net = math.dist(xy, self.tgt) - math.dist(b["a"], self.tgt)
            if not b["broken"] and (b["path"] > p["BOUT_PATH_M"] or net > p["BOUT_AWAY_M"]):
                b["broken"] = True
                self.log_broken.append((t, i))

    def step(self, t, tracks, dt, stale=frozenset()):
        p = self.p
        pos = {tr.id: tr.xy for tr in tracks if tr.id in self.k}
        for i, xy in pos.items():                       # history first, as CT5.step does (it appends again; harmless)
            self.hist[i].append((t, xy))
            self.hist[i] = [h for h in self.hist[i] if t - h[0] <= 30.0]
        self._update_bouts(t, pos, stale)
        b = self.b.reshape((2,) * self.n)
        for i in self.ids:
            k = self.k[i]
            walking = self._speed(i, t) > p["WALK_MPS"]
            h_nw = (p["H_MOVE"] if (self._arrived(i, t) and not walking) else p["H_STILL"]) * dt
            broken = i in self.bout and self.bout[i]["broken"]
            h_wn = min((p["H_LEAVE"] if broken else p["H_STILL"]) * dt, 1.0)
            ax = self.n - 1 - k
            b0, b1 = np.take(b, 0, axis=ax), np.take(b, 1, axis=ax)
            n0, n1 = (1 - h_nw) * b0 + h_wn * b1, h_nw * b0 + (1 - h_wn) * b1
            b = np.stack([n0, n1], axis=ax)
        self.b = b.reshape(-1)
        saved = (p["H_MOVE"], p["H_STILL"])
        p["H_MOVE"], p["H_STILL"] = 0.0, 0.0             # CT5.step's own transitions become the identity
        try:
            super().step(t, tracks, dt, stale)
        finally:
            p["H_MOVE"], p["H_STILL"] = saved


QB3_DEFAULTS = dict(BOUT_PATH_M=float("inf"), STOP_WIN_S=2.0, H_JOIN_TAIL=1.0, TAIL_NEEDS_ARRIVAL=False)


class QB3(QB):
    """QB v3 (exploratory): v2 (away bound only) plus arrival at the tail. For STOP_WIN_S after a walking bout ends,
    the join hazard into each set W is H_JOIN_TAIL where the person's static residual given W (CT5.resid_vec: one
    spacing behind W's member just ahead, along the chain) is below R0; elsewhere W0's join rule. Joint transitions
    over the 2^n sets, person by person."""

    def __init__(self, cell, params=None):
        super().__init__(cell, {**QB3_DEFAULTS, **(params or {})})
        self.stopped_at = {}
        self.log_tail_join = []

    def step(self, t, tracks, dt, stale=frozenset()):
        p = self.p
        pos = {tr.id: tr.xy for tr in tracks if tr.id in self.k}
        for i, xy in pos.items():
            self.hist[i].append((t, xy))
            self.hist[i] = [h for h in self.hist[i] if t - h[0] <= 30.0]
        was = set(self.bout)
        self._update_bouts(t, pos, stale)
        for i in was - set(self.bout):                  # a bout ended this tick: the person stopped
            if i in pos and i not in stale:
                self.stopped_at[i] = t
        stand = {i: self._standing(i, t, xy) for i, xy in pos.items()}
        dist = {i: float(np.hypot(*(np.asarray(xy) - self.tgt))) for i, xy in stand.items()}
        b = self.b.copy()
        allidx = np.arange(len(b))
        for i in self.ids:
            k = self.k[i]
            walking = self._speed(i, t) > p["WALK_MPS"]
            i0 = allidx[~self.bits[:, k]]
            i1 = i0 + (1 << k)
            base = (p["H_MOVE"] if (self._arrived(i, t) and not walking) else p["H_STILL"]) * dt
            j = np.full(len(i0), base)
            if (i in pos and i not in stale and not walking and i in self.stopped_at
                    and t - self.stopped_at[i] <= p["STOP_WIN_S"] + 1e-9
                    and (not p.get("TAIL_NEEDS_ARRIVAL") or self._arrived(i, t))):
                r = self.resid_vec(i, stand, dist)[i0]
                tail = r < p["R0"]
                j = np.where(tail, min(p["H_JOIN_TAIL"] * dt, 1.0), j)
                if tail.any():
                    self.log_tail_join.append((t, i))
            broken = i in self.bout and self.bout[i]["broken"]
            lv = min((p["H_LEAVE"] if broken else p["H_STILL"]) * dt, 1.0)
            b0, b1 = b[i0].copy(), b[i1].copy()
            b[i0] = (1 - j) * b0 + lv * b1
            b[i1] = j * b0 + (1 - lv) * b1
        self.b = b
        saved = (p["H_MOVE"], p["H_STILL"])
        p["H_MOVE"], p["H_STILL"] = 0.0, 0.0
        try:
            CT5.step(self, t, tracks, dt, stale)
        finally:
            p["H_MOVE"], p["H_STILL"] = saved
