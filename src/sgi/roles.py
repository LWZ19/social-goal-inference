"""Who waits, as one estimate kept over time: an option for re-solving.

The constraint method's answer names the people who wait ("come after 4, 5, 6"). Here that answer is taken as
evidence, not as the roster: a belief over which tracked people wait their turn at the anchor (the answer's target
point), exact over every subset of the people tracked at the photo (at most 2^n sets; scenes hold a handful of
people). Each tick of the tracks updates it. The evidence all comes from what waiting in turn means for where people
stand and how they move, with nothing about any one kind of place:

- the answer, fused at the photo: P(named | waits) and P(named | not);
- membership changes with a hazard that is high only while a person moves (joining and leaving a line are walks);
- static, "would I stand there if I were waiting?": a member stands one spacing behind the member just ahead, or near
  the anchor if first; a snapshot of the current configuration, used once at decision time, never accumulated;
- co-motion: when a still person steps up (still, then nearer the anchor, then still), every still person behind
  them should follow if both wait; the step itself counts towards the stepper waiting;
- a track marked stale gives no evidence.

The decision: the roster is everyone with P(waits) >= 0.5; the solver gets `come_after` = the roster and everyone
else as exempt from its place-in-turn rule, along `dir`.

Two further options are off by default:
- `STATIC_WIN_S`: the static snapshot read as the mean log ratio over that many seconds (one observation of a
  configuration that persists, still counted once at decision time), not a single instant;
- `decide`: whom to come after, by least expected cost over the belief's sets, with costs fixed in advance:
  coming ahead of a person who waits 1 each, coming after a person who does not wait 0.25 each, moving 0.125 a
  metre.

The class takes the answer and the tracks at the photo; every parameter has the default given in `DEFAULTS`.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from sgi.resolve import Track
from sgi.types import Constraint, DetectedPerson

DEFAULTS = dict(P0=0.3, P_NAMED_W=0.85, P_NAMED_N=0.15, H_MOVE=0.05, H_STILL=0.0001, ARRIVE_M=1.0, ARRIVE_S=20.0,
                WALK_MPS=0.3, SIG_S=0.35, R0=0.8, TAU_S=4.0, Q=0.85, B=0.05, LR_STEP=3.0, T_FOLLOW=5.0,
                STEP_MIN=0.5, STEP_MAX=1.5, FOLLOW_M=0.25, NEAR_M=3.0, GAP_LO=0.4, GAP_HI=1.4, FRONT_LO=0.3,
                FRONT_HI=1.6, LR_STATIC_MAX=5.0, FOLLOW_FRAC=0.5, PERSIST_S=2.0, STAND_S=0.0, STALE_DECIDE_S=1e9,
                WALKERS_STATIC=True, LR_BACK=1.0, STATIC_WIN_S=0.0)

C_AHEAD, C_BEHIND, C_MOVE_PER_M = 1.0, 0.25, 0.125      # in service intervals
UNSET = object()                                         # `decide` has made no choice yet


def _named(people: Sequence[DetectedPerson], constraints: Sequence[Constraint], at: Sequence[Track]) -> set[int]:
    """The track ids of the people a come_after names: the track nearest each, within 0.5 m, at the photo."""
    where = {p.number: p.xy for p in people}
    out = set()
    for c in constraints:
        if c.kind == "come_after":
            for n in c.people:
                if n in where and at:
                    tr = min(at, key=lambda tr: math.dist(tr.xy, where[n]))
                    if math.dist(tr.xy, where[n]) <= 0.5:
                        out.add(tr.id)
    return out


def _direction(target_xy, at: Sequence[Track], named: set[int], robot_start_xy) -> np.ndarray:
    """The line's direction, front to back: from the first named person to the last, or from the anchor to the one
    named, or towards the robot's start."""
    tgt = np.array(target_xy, dtype=np.float64)
    pts = sorted((np.array(tr.xy, dtype=np.float64) for tr in at if tr.id in named),
                 key=lambda p: float(np.hypot(*(p - tgt))))
    if len(pts) >= 2 and float(np.hypot(*(pts[-1] - pts[0]))) >= 0.5:
        d = pts[-1] - pts[0]
    elif pts:
        d = pts[0] - tgt
    else:
        d = np.array(robot_start_xy[:2], dtype=np.float64) - tgt
    n = float(np.hypot(*d))
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


class RoleBelief:
    """The belief over who waits, over the people tracked at the photo (track ids)."""

    def __init__(self, people: Sequence[DetectedPerson], constraints: Sequence[Constraint], target_xy,
                 at_photo: Sequence[Track], robot_start_xy, params: dict | None = None):
        self.p = dict(DEFAULTS, **(params or {}))
        at = list(at_photo)
        self.ids = sorted(tr.id for tr in at)
        self.k = {i: n for n, i in enumerate(self.ids)}
        self.n = len(self.ids)
        self.named = _named(people, constraints, at)
        self.dir = _direction(target_xy, at, self.named, robot_start_xy)
        self.tgt = np.array(target_xy, dtype=np.float64)
        bits = ((np.arange(2 ** self.n)[:, None] >> np.arange(self.n)[None, :]) & 1).astype(bool)
        self.bits = bits                                            # (S, n): bit i set = person i waits
        p = self.p
        prior = np.where(bits, p["P0"], 1 - p["P0"]).prod(axis=1)
        named = np.array([i in self.named for i in self.ids])
        lik = np.where(bits, np.where(named, p["P_NAMED_W"], 1 - p["P_NAMED_W"]),
                       np.where(named, p["P_NAMED_N"], 1 - p["P_NAMED_N"])).prod(axis=1)
        self.b = prior * lik
        self.b /= self.b.sum()
        self.hist: dict[int, list] = {i: [] for i in self.ids}       # (t, xy)
        self.pending: list = []                                       # (t_check, t_a, j, {i: before}, mid, lead)
        self.last_step: dict[int, float] = {}
        self.last_back: dict[int, float] = {}
        self.seen_at: dict[int, float] = {}
        self.static = 1.0
        self.static_hist: list = []                                   # (t, log static), for STATIC_WIN_S
        self.dist_now: dict[int, float] = {}                          # each present person's distance from the anchor

    # -- helpers
    def _proj(self, xy):
        return float((np.asarray(xy) - self.tgt) @ self.dir)        # along the line, from the anchor backwards

    def _at(self, i, t):
        best = None
        for ts, xy in self.hist[i]:
            if ts <= t + 1e-9:
                best = xy
            else:
                break
        return best

    def _speed(self, i, t):
        a, b = self._at(i, t - 1.0), self._at(i, t)
        return 0.0 if a is None or b is None else math.dist(a, b) / 1.0

    def _arrived(self, i, t):
        a, b = self._at(i, t - self.p["ARRIVE_S"]), self._at(i, t)
        return a is not None and b is not None and math.dist(a, b) >= self.p["ARRIVE_M"]

    def _still_before(self, i, t, span=2.0, tol=0.2):
        pts = [xy for ts, xy in self.hist[i] if t - span - 3.0 - 1e-9 <= ts <= t - 3.0 + 1e-9]
        return len(pts) >= 2 and max(math.dist(pts[0], q) for q in pts) <= tol

    def _standing(self, i, t, now_xy):
        if self._speed(i, t) > self.p["WALK_MPS"] or self.p["STAND_S"] <= 0:
            return now_xy
        pts = [xy for ts, xy in self.hist[i] if t - ts <= self.p["STAND_S"] + 1e-9]
        if not pts:
            return now_xy
        return (float(np.median([q[0] for q in pts])), float(np.median([q[1] for q in pts])))

    def _factor_bit(self, i, f_w, f_n):
        k = self.k[i]
        self.b *= np.where(self.bits[:, k], f_w, f_n)

    # -- one tick
    def step(self, t: float, tracks: Sequence[Track], dt: float, stale=frozenset()) -> None:
        """Update with the tracks at time `t`, `dt` seconds after the last update. Tracks not seen at the photo are
        left out; `stale` track ids give no evidence."""
        p = self.p
        pos = {tr.id: tuple(tr.xy) for tr in tracks if tr.id in self.k}
        for i in pos:
            if i not in stale:
                self.seen_at[i] = t
        for i, xy in pos.items():
            self.hist[i].append((t, xy))
            self.hist[i] = [h for h in self.hist[i] if t - h[0] <= 30.0]
        # transitions: per-person 2x2, a high hazard only while moving
        b = self.b.reshape((2,) * self.n)                           # axis order: bit n-1 ... bit 0
        for i in self.ids:
            k = self.k[i]
            walking = self._speed(i, t) > p["WALK_MPS"]
            h_nw = (p["H_MOVE"] if (self._arrived(i, t) and not walking) else p["H_STILL"]) * dt
            h_wn = (p["H_MOVE"] if walking else p["H_STILL"]) * dt
            ax = self.n - 1 - k
            b0, b1 = np.take(b, 0, axis=ax), np.take(b, 1, axis=ax)
            n0, n1 = (1 - h_nw) * b0 + h_wn * b1, h_nw * b0 + (1 - h_wn) * b1
            b = np.stack([n0, n1], axis=ax)
        self.b = b.reshape(-1)
        # static evidence, joint: each person's residual given who else waits ahead of them; the predicted place
        # continues the chain one spacing behind the member just ahead
        static = np.ones_like(self.b)
        stand = {i: self._standing(i, t, xy) for i, xy in pos.items()}
        dist = {i: float(np.hypot(*(np.asarray(xy) - self.tgt))) for i, xy in stand.items()}
        self.dist_now = dist
        idx = list(self.ids)
        dj = np.array([dist.get(j, np.inf) for j in idx])
        xyj = np.array([stand.get(j, (np.nan, np.nan)) for j in idx], dtype=np.float64)
        for i in pos:
            if i in stale or (not p["WALKERS_STATIC"] and self._speed(i, t) > p["WALK_MPS"]):
                continue
            k = self.k[i]
            ahead = (dj < dist[i] - 1e-6)
            ahead[k] = False
            mask = self.bits & ahead[None, :]
            has = mask.any(axis=1)
            dm = np.where(mask, dj[None, :], -np.inf)
            pi = dm.argmax(axis=1)                                  # the member just ahead
            ahead2 = mask & (dj[None, :] < dj[pi][:, None] - 1e-6)
            has2 = ahead2.any(axis=1)
            ppi = np.where(ahead2, dj[None, :], -np.inf).argmax(axis=1)
            P = xyj[pi]
            PP = np.where(has2[:, None], xyj[ppi], self.tgt[None, :])
            u = P - PP
            un = np.linalg.norm(u, axis=1, keepdims=True)
            u = np.where(un > 1e-6, u / np.maximum(un, 1e-9), self.dir[None, :])
            d = np.asarray(stand[i]) - P
            along = (d * u).sum(axis=1)
            lat = np.abs(d[:, 0] * u[:, 1] - d[:, 1] * u[:, 0])
            long_res = np.maximum(0.0, np.maximum(p["GAP_LO"] - along, along - p["GAP_HI"]))
            r_behind = np.sqrt(lat ** 2 + long_res ** 2)
            fr = dist[i]
            r_front = max(0.0, p["FRONT_LO"] - fr, fr - p["FRONT_HI"])
            r = np.where(has, r_behind, r_front)
            ll_w = -0.5 * (r / p["SIG_S"]) ** 2
            ll_n = -0.5 * (p["R0"] / p["SIG_S"]) ** 2
            f = np.exp(np.clip(ll_w - ll_n, -math.log(p["LR_STATIC_MAX"]), math.log(p["LR_STATIC_MAX"])))
            static *= np.where(self.bits[:, k], f, 1.0)
        if p["STATIC_WIN_S"] > 0:
            self.static_hist.append((t, np.log(static)))
            self.static_hist = [h for h in self.static_hist if t - h[0] <= p["STATIC_WIN_S"] + 1e-9]
            static = np.exp(np.mean([h[1] for h in self.static_hist], axis=0))
        self.static = static
        # dynamic evidence: steps up (still, then nearer the anchor, then still), steps back, and who follows
        for j in pos:
            if j in stale:
                continue
            a, bb, cc = self._at(j, t - 5.0), self._at(j, t - 2.0), pos[j]
            if a is None or bb is None or not self._still_before(j, t - 2.0):
                continue
            if math.dist(bb, cc) > 0.2:
                continue
            da, db = float(np.hypot(*(np.asarray(a) - self.tgt))), float(np.hypot(*(np.asarray(bb) - self.tgt)))
            adv = da - db
            t_a = t - 2.0
            if -adv >= p["STEP_MIN"] and math.dist(a, bb) <= p["STEP_MAX"] and t_a - self.last_back.get(j, -1e9) > 5.0:
                self.last_back[j] = t_a
                self._factor_bit(j, 1.0 / p["LR_BACK"], 1.0)
            if p["STEP_MIN"] <= adv and math.dist(a, bb) <= p["STEP_MAX"] and t_a - self.last_step.get(j, -1e9) > 5.0:
                self.last_step[j] = t_a
                self._factor_bit(j, 1.0, 1.0 / p["LR_STEP"])
                before = {}
                for i in pos:
                    if i == j or i in stale:
                        continue
                    ai = self._at(i, t_a - 3.0)
                    if ai is None:
                        continue
                    di = float(np.hypot(*(np.asarray(ai) - self.tgt)))
                    if di <= da or math.dist(ai, a) > p["NEAR_M"] or not self._still_before(i, t_a):
                        continue
                    before[i] = di
                if before:
                    self.pending.append((t_a + p["T_FOLLOW"], t_a, j, before, None, adv))
        keep = []
        for item in self.pending:
            t_check, t_a, j, before, mid, lead = item
            if t + 1e-9 < t_check:
                keep.append(item)
                continue
            if mid is None:                                         # first look; persistence PERSIST_S later
                now = {i: (float(np.hypot(*(np.asarray(pos[i]) - self.tgt))) if i in pos else None) for i in before}
                keep.append((t_check + p["PERSIST_S"], t_a, j, before, now, lead))
                continue
            jj = self.k[j]
            for i, pb in before.items():
                if i in stale or i not in pos or mid.get(i) is None:
                    continue
                now_i = float(np.hypot(*(np.asarray(pos[i]) - self.tgt)))
                need = max(p["FOLLOW_M"], p["FOLLOW_FRAC"] * lead)
                followed = (pb - mid[i] >= need) and (pb - now_i >= need)
                ki = self.k[i]
                both = self.bits[:, ki] & self.bits[:, jj]
                fq = p["Q"] if followed else 1 - p["Q"]
                fb = p["B"] if followed else 1 - p["B"]
                self.b *= np.where(both, fq, fb)
        self.pending = keep
        self.b /= self.b.sum()

    def posterior(self) -> np.ndarray:
        """P of each set, with the static snapshot of the last update."""
        post = self.b * self.static
        return post / post.sum()

    def marginals(self) -> dict[int, float]:
        """P(waits) for each track id tracked at the photo, with the static snapshot of the last update."""
        post = self.posterior()
        return {i: float(post[self.bits[:, self.k[i]]].sum()) for i in self.ids}

    def roster(self) -> set[int]:
        """The track ids taken as waiting: P(waits) >= 0.5."""
        return {i for i, v in self.marginals().items() if v >= 0.5}


def decide(belief: RoleBelief, present, current=UNSET, c_ahead: float = C_AHEAD, c_behind: float = C_BEHIND,
           c_move: float = C_MOVE_PER_M):
    """Whom to come after (a track id, or None to go first), by least expected cost over the belief's sets: for a set
    W, coming after j costs `c_ahead` for each member of W farther from the anchor than j, and `c_behind` for each
    non-member between W's last and j (j included). `current` is kept unless its expected cost exceeds the best one's
    plus `c_move` times the difference in their distances from the anchor; a fresh choice when there is none yet
    (UNSET) or its person is not present. Returns (choice, expected cost by candidate)."""
    post = belief.posterior()
    ids = [i for i in belief.ids if i in present and i in belief.dist_now]
    if not ids:
        return None, {}
    cols = np.array([belief.k[i] for i in ids])
    d = np.array([belief.dist_now[i] for i in ids])
    W = belief.bits[:, cols]                                        # (S, m): which present people wait in each set
    last_d = np.where(W, d[None, :], -np.inf).max(axis=1)           # -inf when nobody waits
    cost = {}
    for c, dj in [(None, 0.0)] + list(zip(ids, d)):
        ahead = (W & (d[None, :] > dj + 1e-9)).sum(axis=1)
        behind = ((~W) & (d[None, :] > last_d[:, None] + 1e-9) & (d[None, :] <= dj + 1e-9)).sum(axis=1)
        cost[c] = float(post @ (c_ahead * ahead + c_behind * behind))
    best = min(cost, key=cost.get)
    if current is not UNSET and current in cost:
        where = {None: 0.0, **{i: belief.dist_now[i] for i in ids}}
        if cost[current] <= cost[best] + c_move * abs(where[best] - where[current]):
            return current, cost
    return best, cost
