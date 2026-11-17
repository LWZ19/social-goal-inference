"""Phase 1b variants of the resolver's role logic (offline research code; nothing here enters src/).

- GroupFollow: CT3's follow check, but an advance tests every waiting person behind the advancer, not only the next
  one (ml_reasoning idea 2's group test).
- AlongLine: advances measured along the remembered line (towards its front) instead of towards the target point.
- Anchored: when nobody the answer named still waits, re-derive the waiting people from the anchor (the target
  point): the still person nearest it at the front gap the named people showed at the answer, then still people
  behind a member within 1.5 x their spacing; never anyone the follow check dropped (far_analogies idea 1, inside the
  resolver, with the follow check's memory).
Combine by multiple inheritance, e.g. class CT3Fix(Anchored, GroupFollow, Resolver).
"""
from __future__ import annotations

import math

from sgi.resolve import Resolver, _ADVANCE_EVERY_S, _ADVANCE_WITHIN_S, _EPS, _apart


class GroupFollow(Resolver):
    def _follow_up(self, t: float) -> None:
        active = {n: self._history[n][-1][2] for n in self._waiting - self.dropped - self.gone
                  if n in self._history and abs(self._history[n][-1][0] - t) <= _EPS}
        for q, gap in active.items():
            if (self._gap_since(q, t - _ADVANCE_WITHIN_S) - gap >= self._advance_m - _EPS
                    and t - self._advanced_at.get(q, -1e9) > _ADVANCE_EVERY_S):
                self._advanced_at[q] = t
                for p in [n for n in active if n != q and active[n] > gap + _EPS]:
                    if not any(d[1] == p for d in self._due):
                        self._due.append((t + self._co_motion_s, p, self._gap_since(p, t - _ADVANCE_WITHIN_S)))
        due, self._due = [d for d in self._due if d[0] <= t + _EPS], [d for d in self._due if d[0] > t + _EPS]
        for when, p, before in due:
            if self._co_motion_rejoin and self._readmitted_at.get(p, -1e9) > when - self._co_motion_s + _EPS:
                continue
            if p in active and before - active[p] < self._follow_m - _EPS:
                self.dropped.add(p)
                self._left_at[p] = t
                self.non_followers.add(p)
                self.non_follower_t[p] = t


class Anchored(Resolver):
    """Re-derive from the anchor when the named waiting people have all gone."""
    STILL_M, STILL_S, SPACING_X, FRONT_TOL = 0.10, 1.0, 1.5, 0.5

    def __init__(self, people, constraints, target_xy, at_answer, **kw):
        super().__init__(people, constraints, target_xy, at_answer, **kw)
        named = [self._at_answer[n] for n in self._waiting if n in self._at_answer]
        gaps = sorted(_apart(xy, self._target) for xy in named)
        self._front_gap = gaps[0] if gaps else None
        pts = sorted(named, key=lambda xy: _apart(xy, self._target))
        sp = [_apart(a, b) for a, b in zip(pts, pts[1:])]
        self._spacing = (sum(sp) / len(sp)) if sp else 0.8
        self.rederived = 0

    def _gate_ok(self, n):
        return n in self._at_answer or self._moved(n)

    def _base_ok(self, n):
        return True

    def _is_still(self, number: int, t: float) -> bool:
        h = [x for x in self._history.get(number, []) if t - x[0] <= self.STILL_S + _EPS]
        return len(h) >= 2 and all(_apart(x[1], h[-1][1]) <= self.STILL_M + _EPS for x in h)

    def inputs(self, tracks, robot_xy, t):
        people, constraints = super().inputs(tracks, robot_xy, t)
        if self._front_gap is None or (self._waiting - self.dropped - self.gone):
            return people, constraints
        where = {p.number: p.xy for p in people}
        pool = [n for n in where if n not in self.non_followers and self._is_still(n, t)]
        base = [n for n in pool if self._base_ok(n) and abs(_apart(where[n], self._target) - self._front_gap) <= self.FRONT_TOL + _EPS]
        if not base:
            return people, constraints
        chain = [min(base, key=lambda n: _apart(where[n], self._target))]
        while True:
            last = chain[-1]
            nxt = [n for n in pool if n not in chain and _apart(where[n], where[last]) <= self.SPACING_X * self._spacing
                   and _apart(where[n], self._target) > _apart(where[last], self._target) + _EPS
                   and (self._co_motion_s is None or self._gate_ok(n))]
            if not nxt:
                break
            chain.append(min(nxt, key=lambda n: _apart(where[n], where[last])))
        for n in chain:
            self.dropped.discard(n)
            self.gone.discard(n)
            self._closest.pop(n, None)
            if n not in self._waiting:
                self._waiting.add(n)
                self._joiners.append(n)
        self.rederived += 1
        left = self.dropped | self.gone
        out, first = [], True
        for c in self._constraints:
            named = c.people + (self._joiners if c.kind == "come_after" and first else [])
            first = first and c.kind != "come_after"
            out.append(c.model_copy(update={"people": [n for n in named if n not in left]}))
        return people, out


class CT3Group(GroupFollow, Resolver):
    pass


class CT3Fix(Anchored, GroupFollow, Resolver):
    pass


class CT3Anchored(Anchored, Resolver):
    pass


class AnchoredGated(Anchored):
    """Anchored, but only the people the answer named skip the join gate: anyone else must have moved as a joiner
    would (CT3's `_moved`) before the re-derivation takes them."""

    def _gate_ok(self, n):
        return n in self._named or self._moved(n)

    def _base_ok(self, n):
        return self._gate_ok(n)


class CT3FixGated(AnchoredGated, GroupFollow, Resolver):
    pass
