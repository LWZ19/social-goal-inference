"""CT5, offline (round 3 research code; nothing here enters src/): CT4-min's belief, changed where POSTHOC_1 points,
plus a decision that uses the whole belief. ROUND3.md sections 2-4.

Belief (as ct4.CT4 except):
- E1: the static snapshot reads still people only (a walker's current position is not where they stand);
- E2: exit evidence, phrased by the target: someone who came within REACH_M of the target point and is now walking
  away (EXIT_M farther than their nearest point there, distance rising) counts LR_EXIT against waiting, once per exit;
- the static likelihood ratio may come from a table over the residual (`STATIC_TABLE`: [(r_hi, lr), ...]) instead of
  the hand-set Gaussian, for the calibration by counting (b_calibrate.py);
- hooks log follow checks, exits and residuals for counting;
- STATIC_WIN_S > 0: the snapshot is the mean log ratio over that window (one observation of a persisting
  configuration, still counted once at decision time), instead of a single instant.

Decision (`decide`): exact expected cost over the posterior's sets, ROUND3.md section 2's costs, fixed before this
code ran: cutting in 1 per waiter cut, standing behind a non-waiter 0.25 each, moving 0.125 per metre.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ct4 import CT4, DEFAULTS as CT4_DEFAULTS  # noqa: E402

C_CUT, C_FAR, C_MOVE_PER_M = 1.0, 0.25, 0.125          # ROUND3.md section 2 (frozen before any decision code ran)

DEFAULTS = dict(CT4_DEFAULTS, WALKERS_STATIC=False, REACH_M=1.6, EXIT_M=0.5, LR_EXIT=10.0, STATIC_TABLE=None,
                STATIC_WIN_S=0.0)


class CT5(CT4):
    def __init__(self, cell, params=None):
        super().__init__(cell, dict(DEFAULTS, **(params or {})))
        self.exited = set()
        self.near_min = {}
        self.log_follow = []          # (t, i, j, followed)
        self.log_exit = []            # (t, i)
        self.log_static = []          # (t, i, r_vec) for still people, when self.keep_static
        self.keep_static = False
        self.static_hist = []         # (t, log static vector), for STATIC_WIN_S

    def _static_lr(self, r):
        p = self.p
        tab = p["STATIC_TABLE"]
        if tab is None:
            ll_w = -0.5 * (r / p["SIG_S"]) ** 2
            ll_n = -0.5 * (p["R0"] / p["SIG_S"]) ** 2
            return np.exp(np.clip(ll_w - ll_n, -math.log(p["LR_STATIC_MAX"]), math.log(p["LR_STATIC_MAX"])))
        edges = np.array([hi for hi, _ in tab])
        lrs = np.array([lr for _, lr in tab])
        return lrs[np.minimum(np.searchsorted(edges, r, side="left"), len(lrs) - 1)]

    def resid_vec(self, i, stand, dist):
        """Residual of person i's standing place, for every set (as ct4.CT4.step computes it)."""
        p = self.p
        idx = self.ids
        dj = np.array([dist.get(j, np.inf) for j in idx])
        xyj = np.array([stand.get(j, (np.nan, np.nan)) for j in idx], dtype=np.float64)
        k = self.k[i]
        ahead = (dj < dist[i] - 1e-6)
        ahead[k] = False
        mask = self.bits & ahead[None, :]
        has = mask.any(axis=1)
        dm = np.where(mask, dj[None, :], -np.inf)
        pi = dm.argmax(axis=1)
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
        return np.where(has, r_behind, r_front)

    def step(self, t, tracks, dt, stale=frozenset()):
        p = self.p
        pos = {tr.id: tr.xy for tr in tracks if tr.id in self.k}
        for i in pos:
            if i not in stale:
                self.seen_at[i] = t
        for i, xy in pos.items():
            self.hist[i].append((t, xy))
            self.hist[i] = [h for h in self.hist[i] if t - h[0] <= 30.0]
        b = self.b.reshape((2,) * self.n)
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
        # E2: exits from the target point
        for i, xy in pos.items():
            if i in stale:
                continue
            d = float(np.hypot(*(np.asarray(xy) - self.tgt)))
            if d <= p["REACH_M"]:
                self.near_min[i] = min(self.near_min.get(i, d), d)
                self.exited.discard(i)
            if i in self.near_min and i not in self.exited:
                prev = self._at(i, t - 1.0)
                rising = prev is not None and d > float(np.hypot(*(np.asarray(prev) - self.tgt))) + 0.1
                if d >= self.near_min[i] + p["EXIT_M"] and rising and self._speed(i, t) > p["WALK_MPS"]:
                    self.exited.add(i)
                    self.near_min.pop(i, None)
                    self.log_exit.append((t, i))
                    self._factor_bit(i, 1.0 / p["LR_EXIT"], 1.0)
        # static snapshot (E1: still people only)
        static = np.ones_like(self.b)
        stand = {i: self._standing(i, t, xy) for i, xy in pos.items()}
        dist = {i: float(np.hypot(*(np.asarray(xy) - self.tgt))) for i, xy in stand.items()}
        self.dist_now = dist
        for i in pos:
            if i in stale or (not p["WALKERS_STATIC"] and self._speed(i, t) > p["WALK_MPS"]):
                continue
            r = self.resid_vec(i, stand, dist)
            if self.keep_static:
                self.log_static.append((t, i, r))
            static *= np.where(self.bits[:, self.k[i]], self._static_lr(r), 1.0)
        if p["STATIC_WIN_S"] > 0:
            # one observation of a persisting configuration: the mean log ratio over the last STATIC_WIN_S seconds,
            # per set (a window, not an accumulation: it still counts once, at decision time)
            self.static_hist.append((t, np.log(static)))
            self.static_hist = [h for h in self.static_hist if t - h[0] <= p["STATIC_WIN_S"] + 1e-9]
            static = np.exp(np.mean([h[1] for h in self.static_hist], axis=0))
        self.static = static
        # co-motion, as ct4.CT4.step, with follow checks logged
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
            if mid is None:
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
                self.log_follow.append((t, i, j, followed))
                ki = self.k[i]
                both = self.bits[:, ki] & self.bits[:, jj]
                self.b *= np.where(both, p["Q"] if followed else 1 - p["Q"], p["B"] if followed else 1 - p["B"])
        self.pending = keep
        self.b /= self.b.sum()

    def posterior(self):
        post = self.b * getattr(self, "static", 1.0)
        return post / post.sum()


UNSET = object()


def decide(est, present, current=UNSET, c_cut=C_CUT, c_far=C_FAR, c_move=C_MOVE_PER_M):
    """The person to come after (None = first), by least expected cost over the posterior's sets, keeping `current`
    unless switching saves more than the move costs (no current choice yet: UNSET; a current person no longer
    tracked: a fresh choice). Returns (choice, expected costs by candidate)."""
    post = est.posterior()
    ids = [i for i in est.ids if i in present and i in est.dist_now]
    if not ids:
        return None, {}
    cols = np.array([est.k[i] for i in ids])
    d = np.array([est.dist_now[i] for i in ids])
    W = est.bits[:, cols]                                           # (S, m) membership of present people
    last_d = np.where(W, d[None, :], -np.inf).max(axis=1)           # -inf when nobody waits
    cost = {}
    for c, dj in [(None, 0.0)] + list(zip(ids, d)):
        cut = (W & (d[None, :] > dj + 1e-9)).sum(axis=1)
        far = ((~W) & (d[None, :] > last_d[:, None] + 1e-9) & (d[None, :] <= dj + 1e-9)).sum(axis=1)
        cost[c] = float(post @ (c_cut * cut + c_far * far))
    best = min(cost, key=cost.get)
    if current is not UNSET and current in cost:
        dpos = {None: 0.0, **{i: est.dist_now[i] for i in ids}}
        move = c_move * abs(dpos[best] - dpos[current])
        if cost[current] <= cost[best] + move:
            return current, cost
    return best, cost
