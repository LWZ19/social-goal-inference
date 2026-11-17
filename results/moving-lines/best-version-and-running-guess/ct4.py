"""CT4-min, offline (phase 1b research code; nothing here enters src/): one estimator for who waits.

State: the set W of the scene's tracked people who wait their turn at the anchor (the answer's target point); exact
over all subsets (scenes hold at most 8 people, so at most 256 sets). Each person's membership changes with a hazard
that is high only while they move (joining and leaving a line are walks; CT3's join gate as a transition prior).

Evidence, all derived from what `come_after` means, none from a scene type:
- the answer, fused at the photo time t0: P(named | waits) = P_NAMED_W, P(named | not) = P_NAMED_N;
- static, "would I stand there if I were waiting?": a still member of W stands one spacing behind the member of W
  just ahead (0.4-1.4 m, continuing the chain's direction from the member or anchor ahead of that one), or, if first,
  0.3-1.6 m from the anchor; residual r against a soft threshold (Gaussian SIG_S, neutral at R0), the likelihood ratio
  bounded to [1/LR_STATIC_MAX, LR_STATIC_MAX]. It is a snapshot of the current configuration, multiplied in once at
  decision time and never accumulated (an unchanged scene seen again carries no new information);
- dynamic, co-motion: when a still person j steps up (still, then STEP_MIN-1.5 m nearer the anchor, then still),
  every still person i behind j and within 3 m should follow (nearer the anchor by at least half j's step, and at
  least FOLLOW_M, at +T_FOLLOW s and still PERSIST_S later) if both wait; P(follow | i and j wait) = Q, otherwise B; the step itself counts LR_STEP for j waiting;
- a step back from the anchor (still, STEP_MIN-1.5 m farther, still) counts 1/LR_STEP against waiting;
- missing data: a stale track (cell._stale, set by stress S4) gives no evidence of any kind.
The static snapshot reads where a still person stands (the median of the last STAND_S s), where a walker is now.
Decision: roster = {P(waits) >= 0.5}, less anyone whose track has been stale longer than STALE_DECIDE_S; the
solver's place-in-turn rule gets everyone else as exempt.
All parameters are set a priori here (none fitted on bystander cells); `params` overrides them for sensitivity runs.
"""
from __future__ import annotations

import math

import numpy as np

from replay_lib import TH, dump_tracks, robot_at
from sgi import solver as S

DEFAULTS = dict(P0=0.3, P_NAMED_W=0.85, P_NAMED_N=0.15, H_MOVE=0.05, H_STILL=0.0001, ARRIVE_M=1.0, ARRIVE_S=20.0,
                WALK_MPS=0.3, SIG_S=0.35, R0=0.8, TAU_S=4.0, Q=0.85, B=0.05, LR_STEP=3.0, T_FOLLOW=5.0,
                STEP_MIN=0.5, STEP_MAX=1.5, FOLLOW_M=0.25, NEAR_M=3.0, GAP_LO=0.4, GAP_HI=1.4, FRONT_LO=0.3,
                FRONT_HI=1.6, LR_STATIC_MAX=5.0, FOLLOW_FRAC=0.5, PERSIST_S=2.0, STAND_S=0.0, STALE_DECIDE_S=1e9, WALKERS_STATIC=True, LR_BACK=1.0)


def _named(cell, at):
    where = {p.number: p.xy for p in cell.people}
    out = set()
    for c in cell.constraints:
        if c.kind == "come_after":
            for n in c.people:
                if n in where and at:
                    tr = min(at, key=lambda tr: math.dist(tr.xy, where[n]))
                    if math.dist(tr.xy, where[n]) <= 0.5:
                        out.add(tr.id)
    return out


def _direction(cell, at, named):
    tgt = np.array(cell.target)
    pts = sorted((np.array(tr.xy) for tr in at if tr.id in named), key=lambda p: float(np.hypot(*(p - tgt))))
    if len(pts) >= 2 and float(np.hypot(*(pts[-1] - pts[0]))) >= 0.5:
        d = pts[-1] - pts[0]
    elif pts:
        d = pts[0] - tgt
    else:
        d = np.array(cell.episode["samples"][0]["robot"][:2]) - tgt
    n = float(np.hypot(*d))
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


class CT4:
    def __init__(self, cell, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.cell = cell
        at = dump_tracks(cell.dump, cell.asked_t)
        self.ids = sorted(tr.id for tr in at)
        self.k = {i: n for n, i in enumerate(self.ids)}
        self.n = len(self.ids)
        self.named = _named(cell, at)
        self.dir = _direction(cell, at, self.named)
        self.tgt = np.array(cell.target, dtype=np.float64)
        bits = ((np.arange(2 ** self.n)[:, None] >> np.arange(self.n)[None, :]) & 1).astype(bool)
        self.bits = bits                                            # (S, n): bit i set = person i waits
        p = self.p
        prior = np.where(bits, p["P0"], 1 - p["P0"]).prod(axis=1)
        named = np.array([i in self.named for i in self.ids])
        lik = np.where(bits, np.where(named, p["P_NAMED_W"], 1 - p["P_NAMED_W"]),
                       np.where(named, p["P_NAMED_N"], 1 - p["P_NAMED_N"])).prod(axis=1)
        self.b = prior * lik
        self.b /= self.b.sum()
        self.hist = {i: [] for i in self.ids}                      # (t, xy)
        self.pending = []                                           # (t_check, t_a, j, {i: proj_before})
        self.last_step = {}
        self.last_back = {}
        self.seen_at = {}
        self.was_on = set()

    # -- helpers
    def _proj(self, xy):
        return float((np.asarray(xy) - self.tgt) @ self.dir)        # along the line, from the anchor backwards

    def _at(self, i, t):
        h = self.hist[i]
        best = None
        for ts, xy in h:
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
        """Where a person stands: the median of their positions over the last STAND_S seconds while they are still;
        where they are, while they walk."""
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
    def step(self, t, tracks, dt, stale=frozenset()):
        p = self.p
        pos = {tr.id: tr.xy for tr in tracks if tr.id in self.k}
        for i in pos:
            if i not in stale:
                self.seen_at[i] = t
        for i, xy in pos.items():
            self.hist[i].append((t, xy))
            self.hist[i] = [h for h in self.hist[i] if t - h[0] <= 30.0]
        # transitions: per-person 2x2, high hazard while moving
        b = self.b.reshape((2,) * self.n)                        # axis order: bit n-1 ... bit 0
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
        proj = {i: self._proj(xy) for i, xy in pos.items()}
        lat = {i: abs(float((np.asarray(xy) - self.tgt) @ np.array([-self.dir[1], self.dir[0]]))) for i, xy in pos.items()}
        # static evidence, joint: each still person's residual given who else waits ahead of them. Ahead = nearer the
        # anchor; the predicted place continues the chain: one spacing behind the member just ahead, along the
        # direction from the member ahead of them (or the anchor) to that member.
        static = np.ones_like(self.b)
        stand = {i: self._standing(i, t, xy) for i, xy in pos.items()}
        dist = {i: float(np.hypot(*(np.asarray(xy) - self.tgt))) for i, xy in stand.items()}
        idx = [j for j in self.ids]
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
            # the member ahead of them, or the anchor
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
        self.static = static                    # a snapshot of the current configuration: used once, at decision time
        # dynamic evidence: steps up (still, then 0.4-1.5 m towards the front, then still again), and who follows
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
                self.last_back[j] = t_a                             # a step back from the front: not how one waits
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
            if mid is None:                                         # first look; persistence 3 s later
                now = {i: (float(np.hypot(*(np.asarray(pos[i]) - self.tgt))) if i in pos else None) for i in before}
                keep.append((t_check + p["PERSIST_S"], t_a, j, before, now, lead))
                continue
            jj = self.k[j]
            for i, pb in before.items():
                if i in stale or i not in pos or mid.get(i) is None:
                    continue
                now_i = float(np.hypot(*(np.asarray(pos[i]) - self.tgt)))
                need = max(p["FOLLOW_M"], p["FOLLOW_FRAC"] * lead)       # take up a real part of the space
                followed = (pb - mid[i] >= need) and (pb - now_i >= need)
                ki = self.k[i]
                both = self.bits[:, ki] & self.bits[:, jj]
                fq = p["Q"] if followed else 1 - p["Q"]
                fb = p["B"] if followed else 1 - p["B"]
                self.b *= np.where(both, fq, fb)
        self.pending = keep
        self.b /= self.b.sum()

    def marginals(self):
        post = self.b * getattr(self, "static", 1.0)
        post = post / post.sum()
        return {i: float(post[self.bits[:, self.k[i]]].sum()) for i in self.ids}


def replay_ct4(cell, params=None, stress=None, every_s=0.5, end_t=None):
    """Run CT4-min from the photo (the answer fused at t0; decisions logged from delivery), in replay_lib's tick
    format, with the solver's place-in-turn rule given the roster and everyone else exempt."""
    e = cell.episode
    end = end_t if end_t is not None else e["samples"][-1]["t"]
    t0 = cell.asked_t
    n = int(math.floor((end - t0) / every_s)) + 1
    ticks = [(t0 + k * every_s, dump_tracks(cell.dump, t0 + k * every_s), None) for k in range(n)]
    cell._stale = {}
    if stress is not None:
        ticks = stress(cell, ticks)
    est = CT4(cell, params)
    out = []
    dropped_on = set()
    for t, tracks, _ in ticks:
        est.step(t, tracks, every_s, cell._stale.get(round(t, 3), frozenset()))
        if t + 1e-9 < cell.delivered_t:
            m = est.marginals()
            est.was_on |= {i for i, v in m.items() if v >= 0.5}
            continue
        m = est.marginals()
        roster = {i for i, v in m.items() if v >= 0.5}
        est.was_on |= roster
        st = cell._stale.get(round(t, 3), frozenset())
        usable = {i for i in roster if i not in st or t - est.seen_at.get(i, -1e9) <= est.p["STALE_DECIDE_S"]}
        dropped_on = {i for i in est.was_on if m[i] < 0.5}
        robot = robot_at(e, t)
        last = None
        if usable:
            bodies = {tr.id + 1: S._Body(tr.id + 1, np.array(tr.xy, dtype=np.float64), tr.yaw) for tr in tracks}
            waiting = sorted(i + 1 for i in usable if i + 1 in bodies)
            if waiting:
                apart = set(bodies) - set(waiting)
                turn = S._turn(waiting, apart, bodies, None, est.tgt, np.array(robot, dtype=np.float64), TH, est.dir)
                last = turn.last - 1
        out.append(dict(t=t, last=last, joined=[], vacated=False, roster=roster, nonfollowers=set(dropped_on),
                        logged=None, marginals=m))
    return out
