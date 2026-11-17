"""I03 phase 1b: the shared people scorer (FROZEN.md section 4). I09 A1 reuses it unchanged.

    score_people(people, scene, th=None, *, exempt_xy=(), facing="full") -> dict

people: a list of (x, y, yaw or None), world metres and radians; detection number = index + 1.
scene: any object with .observation and .truth (H.scene(), H.example(), or FreshScene(load_scene(folder))).
th: sgi.config.Thresholds; default configs/default.yaml. Asserted: person_radius_m 0.25, visibility_min_samples 3,
visibility_depth_tolerance_m 0.3 (and grid_resolution_m 0.05, which the blob detector uses).

Definitions (FROZEN.md section 4; "within d" is a distance <= d, as the package's matching has it):
- visible: a person of truth.people that passes sgi.gt.derive.is_visible under th.
- matching: sgi.run._matched_people itself (greedy, nearest first, one to one, within 2 x person_radius_m).
- recall: matched / visible, None when nobody is visible.
- false: an unmatched detection not within 0.5 m of an exempt_xy point; exempt ones are counted apart;
  near_hidden: the false detections within 0.5 m of a true person who fails is_visible.
- merge (per detection): >= 2 visible true people within 0.5 m of it, at most one of whom is matched to any detection.
- neighbour pair: two visible true people with centres < 0.8 m apart; merged when not both matched and one detection
  lies within 0.5 m of both. Merged share = merged pairs / neighbour pairs.
- xy error: the distance within each matched pair (median reported, list returned for pooling).
- facing: on matched pairs with a known yaw; e = |wrap(yaw_det - yaw_true)| ("full") or min(e, 180 - e) ("axis");
  within 45 when e <= 45 deg. Known, within-45 and unknown counts; the axis version is always reported too.
- missed people: id, role, group, lower_hidden (>= 2 of the samples at z = 0.3, 0.5, 0.7 m project into the image onto
  finite positive depth nearer than the sample by more than 0.3 m), role class.
- staff behind counters: missed people with a role who are lower_hidden.
- role class (tabulation only): staff (has a role); line member (group at a target of a waiting_line episode); group
  member (group at a target of a conversation_group, audience or group_photo episode); other.

Returns counts, so callers pool by summing; medians over the pooled lists. No model, no network, numpy only.
"""
from __future__ import annotations

import math
import statistics
from functools import lru_cache

import numpy as np

from sgi.config import Thresholds, load_config
from sgi.geometry import in_image, project, wrap_angle
from sgi.gt.derive import is_visible
from sgi.run import _matched_people
from sgi.types import DetectedPerson

NEAR_M = 0.5                    # 2 x person_radius_m: the matching width, and every other "within 0.5 m"
NEIGHBOUR_M = 0.8               # a neighbour pair: centres closer than this
FACING_TOL_DEG = 45.0
LOWER_Z = (0.3, 0.5, 0.7)       # lower_hidden samples
LOWER_TOL_M = 0.3
LOWER_MIN = 2
LINE_SITS = ("waiting_line",)
GROUP_SITS = ("conversation_group", "audience", "group_photo")


@lru_cache(maxsize=1)
def default_thresholds() -> Thresholds:
    return load_config().thresholds


def check_thresholds(th: Thresholds) -> None:
    assert th.person_radius_m == 0.25, th.person_radius_m
    assert th.visibility_min_samples == 3, th.visibility_min_samples
    assert th.visibility_depth_tolerance_m == 0.3, th.visibility_depth_tolerance_m
    assert th.grid_resolution_m == 0.05, th.grid_resolution_m


class FreshScene:
    """A scene loaded with sgi.scene.load_scene, shaped as the scorer reads it (.observation, .truth, .inputs)."""

    def __init__(self, loaded):
        self.inputs, self.truth = loaded
        self.observation = self.inputs.observation


def lower_hidden(person, obs) -> bool:
    cam, depth = obs.camera, obs.depth
    hits = 0
    for z in LOWER_Z:
        p = project(cam, (person.x, person.y, z))
        if p is None or not in_image(cam, p[0], p[1]):
            continue
        d = float(depth[math.floor(p[1]), math.floor(p[0])])
        if math.isfinite(d) and d > 0 and d < p[2] - LOWER_TOL_M:
            hits += 1
    return hits >= LOWER_MIN


def role_classes(truth) -> dict[str, str]:
    """person id -> staff / line / group / other, from the truth's episodes (tabulation only)."""
    sits_of_group: dict[str, set] = {}
    for ep in truth.episodes.values():
        g = ep.group_at_target
        ids = list(g.values()) if isinstance(g, dict) else [g]
        for gid in ids:
            if gid is not None:
                sits_of_group.setdefault(gid, set()).add(ep.situation)
    out = {}
    for p in truth.people:
        if p.role:
            out[p.id] = "staff"
            continue
        sits = sits_of_group.get(p.group, set()) if p.group is not None else set()
        out[p.id] = "line" if sits & set(LINE_SITS) else "group" if sits & set(GROUP_SITS) else "other"
    return out


_TRUTH_CACHE: dict = {}


def truth_info(scene, th: Thresholds) -> dict:
    """Visible and hidden people, roles and lower_hidden of a scene, computed once per (truth, thresholds)."""
    key = (id(scene.truth), id(scene.observation), th.model_dump_json())
    hit = _TRUTH_CACHE.get(key)
    if hit is not None and hit["truth"] is scene.truth:
        return hit
    obs, truth = scene.observation, scene.truth
    vis, hid = [], []
    for p in truth.people:
        (vis if is_visible(p, obs, th) else hid).append(p)
    info = {"truth": truth, "visible": vis, "hidden": hid, "roles": role_classes(truth),
            "lower_hidden": {p.id: lower_hidden(p, obs) for p in vis}}
    _TRUTH_CACHE[key] = info
    while len(_TRUTH_CACHE) > 8:
        _TRUTH_CACHE.pop(next(iter(_TRUTH_CACHE)))
    return info


def _err_deg(a: float, b: float) -> float:
    return abs(math.degrees(wrap_angle(a - b)))


def score_people(people, scene, th: Thresholds | None = None, *, exempt_xy=(), facing: str = "full") -> dict:
    if facing not in ("full", "axis"):
        raise ValueError(facing)
    th = default_thresholds() if th is None else th
    check_thresholds(th)
    info = truth_info(scene, th)
    vis, hid, roles, lh = info["visible"], info["hidden"], info["roles"], info["lower_hidden"]
    people = [(float(p[0]), float(p[1]), None if p[2] is None else float(p[2])) for p in people]
    dets = [DetectedPerson(number=i + 1, xy=(x, y), yaw=yaw, pixel=(0.0, 0.0)) for i, (x, y, yaw) in enumerate(people)]
    matched = _matched_people(dets, vis, th)                     # number -> true id
    width = 2 * th.person_radius_m
    assert abs(width - NEAR_M) < 1e-12
    by_id = {p.id: p for p in vis}
    matched_ids = set(matched.values())
    exempt = [tuple(map(float, e[:2])) for e in exempt_xy]

    false_n = exempt_n = near_hidden = 0
    false_numbers, exempt_numbers = [], []
    for d in dets:
        if d.number in matched:
            continue
        if any(math.dist(d.xy, e) <= NEAR_M for e in exempt):
            exempt_n += 1
            exempt_numbers.append(d.number)
            continue
        false_n += 1
        false_numbers.append(d.number)
        if any(math.dist(d.xy, (p.x, p.y)) <= NEAR_M for p in hid):
            near_hidden += 1

    merges = 0
    for d in dets:
        near = [p for p in vis if math.dist(d.xy, (p.x, p.y)) <= NEAR_M]
        if len(near) >= 2 and sum(p.id in matched_ids for p in near) <= 1:
            merges += 1

    pairs = []
    for i in range(len(vis)):
        for j in range(i + 1, len(vis)):
            a, b = vis[i], vis[j]
            gap = math.dist((a.x, a.y), (b.x, b.y))
            if gap >= NEIGHBOUR_M:
                continue
            both = a.id in matched_ids and b.id in matched_ids
            shared = any(math.dist(d.xy, (a.x, a.y)) <= NEAR_M and math.dist(d.xy, (b.x, b.y)) <= NEAR_M for d in dets)
            pairs.append({"a": a.id, "b": b.id, "gap": round(gap, 4), "merged": bool(not both and shared)})

    errs, fac_rows = [], []
    known = unknown = w45_full = w45_axis = 0
    for d in dets:
        tid = matched.get(d.number)
        if tid is None:
            continue
        p = by_id[tid]
        errs.append(math.dist(d.xy, (p.x, p.y)))
        if d.yaw is None:
            unknown += 1
            continue
        known += 1
        e = _err_deg(d.yaw, p.yaw)
        ea = min(e, 180.0 - e)
        w45_full += e <= FACING_TOL_DEG
        w45_axis += ea <= FACING_TOL_DEG
        fac_rows.append({"id": tid, "err_full": round(e, 3), "err_axis": round(ea, 3)})

    misses = [{"id": p.id, "role": p.role, "group": p.group, "lower_hidden": bool(lh[p.id]), "role_class": roles[p.id],
               "staff_behind": bool(p.role and lh[p.id])} for p in vis if p.id not in matched_ids]
    per_person = [{"id": p.id, "role_class": roles[p.id], "matched": p.id in matched_ids} for p in vis]
    return {
        "n_det": len(dets), "n_visible": len(vis), "n_hidden_people": len(hid), "matched": len(matched),
        "recall": (len(matched) / len(vis)) if vis else None,
        "false": false_n, "exempt": exempt_n, "false_raw": false_n + exempt_n, "near_hidden": near_hidden,
        "false_numbers": false_numbers, "exempt_numbers": exempt_numbers,
        "merges": merges, "neighbour_pairs": len(pairs), "merged_pairs": sum(p["merged"] for p in pairs),
        "pairs": pairs,
        "xy_err": errs, "xy_err_median": statistics.median(errs) if errs else None,
        "facing": {"mode": facing, "matched": len(matched), "known": known, "unknown": unknown,
                   "within45_full": w45_full, "within45_axis": w45_axis,
                   "within45": w45_full if facing == "full" else w45_axis},
        "facing_rows": fac_rows,
        "misses": misses, "staff_behind": sum(m["staff_behind"] for m in misses),
        "people": per_person, "matches": {int(k): v for k, v in matched.items()},
    }


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def exempt_points(truth) -> list[tuple[float, float]]:
    """Stage A's exemption: the anchors of every target of truth.targets that has no person_id."""
    return [(t.anchor[0], t.anchor[1]) for t in truth.targets.values() if t.person_id is None]


POOL_KEYS = ("n_det", "n_visible", "n_hidden_people", "matched", "false", "exempt", "false_raw", "near_hidden",
             "merges", "neighbour_pairs", "merged_pairs", "staff_behind")


def pool(results) -> dict:
    """Sum counts over scored scenes; pool lists."""
    out = {k: 0 for k in POOL_KEYS}
    out.update(scenes=0, xy_err=[], facing={"matched": 0, "known": 0, "unknown": 0, "within45_full": 0,
                                            "within45_axis": 0}, misses=[], people=[])
    for r in results:
        out["scenes"] += 1
        for k in POOL_KEYS:
            out[k] += r[k]
        out["xy_err"] += r["xy_err"]
        for k in out["facing"]:
            out["facing"][k] += r["facing"][k]
        out["misses"] += r["misses"]
        out["people"] += r["people"]
    out["recall"] = out["matched"] / out["n_visible"] if out["n_visible"] else None
    out["fb_per_scene"] = out["false"] / out["scenes"] if out["scenes"] else None
    out["merged_share"] = out["merged_pairs"] / out["neighbour_pairs"] if out["neighbour_pairs"] else None
    out["xy_err_median"] = statistics.median(out["xy_err"]) if out["xy_err"] else None
    return out
