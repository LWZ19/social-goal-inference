"""Independent verifier, I06 build: summarise the replay rows (out/replay_seed*.jsonl) into out/replay_summary.txt."""
from __future__ import annotations

import collections
import json

import vlib as V


def main():
    rows = [json.loads(l) for s in (0, 1, 2) for l in (V.HERE / "out" / f"replay_seed{s}.jsonl").read_text().splitlines() if l]
    L = []
    say = L.append
    say(f"replayed planned variants: {len(rows)} (270 expected); in manifest {sum(r['in_manifest'] for r in rows)}; "
        f"stream key ok {sum(r['manifest_stream_ok'] for r in rows)}; scene id ok {sum(r['manifest_scene_id_ok'] for r in rows)}")
    say(f"attempt outcomes equal to the manifest's (every attempt): {sum(r['reasons_match'] for r in rows)}/{len(rows)}")
    mism = [r["scene_id"] + f" replay {r['replay_reasons']} manifest {r['manifest_reasons']}" for r in rows if not r["reasons_match"]]
    for m in mism:
        say(f"   MISMATCH {m}")
    oc = collections.Counter(r["replay_reasons"][-1] if r["replay_reasons"][-1] in ("accepted", "no_spot") else "scene_check"
                             for r in rows)
    say(f"replay final outcomes: {dict(oc)}; replay attempt outcomes: "
        f"{dict(collections.Counter(x for r in rows for x in r['replay_reasons']))}")
    sc = [r for r in rows if r["replay_reasons"][-1] not in ("accepted", "no_spot")]
    say(f"scene-check-rejected variants: {len(sc)}; last attempt's people equal the manifest's last_attempt_added: "
        f"{sum(bool(r.get('last_attempt_match')) for r in sc)}/{len(sc)}")
    kept = [r for r in rows if r["replay_reasons"][-1] == "accepted" and "geometry" in r]
    say(f"kept: {len(kept)}")
    say(f"  replayed people equal stored (max |dx|,|dy|,|dyaw|): max {max(r['people_max_diff'] for r in kept):.2e}; "
        f"ids/groups/roles equal {sum(r['people_ids_groups_match'] for r in kept)}; shirts equal {sum(r['shirts_match'] for r in kept)}")
    for n in ("rgb.png", "depth.npy", "scene.json"):
        say(f"  re-rendered {n} byte-identical to the build's: {sum(r['bytes_equal'][n] for r in kept)}/{len(kept)}")
    say(f"  manifest sha256 equals files: {sum(r['manifest_sha_ok'] for r in kept)}/{len(kept)}")
    G = [r["geometry"] for r in kept]
    for k in ("base_people_unchanged", "ids_continue", "roles_none", "non_members", "groups_ok", "camera_same", "robot_same",
              "targets_same", "bounds_same", "obstacles_same", "l1_invariant", "group_at_target_same"):
        say(f"  {k}: {sum(bool(g[k]) for g in G)}/{len(G)}")
    say(f"  situations: {dict(collections.Counter(s for g in G for s in g['situations']))}")
    say(f"  final-configuration checks c1-c8 (family's set, others = everyone else) failed by any added person: "
        f"{sum(any(v for v in g['check_fails'].values()) for g in G)}/{len(G)}")
    for r in kept:
        bad = {k: v for k, v in r["geometry"]["check_fails"].items() if v}
        if bad:
            say(f"     {r['scene_id']}: {bad}")
    say(f"  people per variant: {min(g['n_people'] for g in G)}-{max(g['n_people'] for g in G)}; representative point moved "
        f"vs my base: {sum(g['rep_moved_m'] > 1e-9 for g in G)}, max {max(g['rep_moved_m'] for g in G):.3f} m")
    by = collections.defaultdict(list)
    for r in kept:
        by[(r["family"], r["level"])].append(r)
    say("\nFamily definitions (stored people vs my base; jitter allowance 0.05 m per axis, 0.15 rad):")
    for (f, lv), rs in sorted(by.items()):
        g = [r["geometry"] for r in rs]
        nm = [x for q in g for x in q["nearest_member_m"]]
        no = [x for q in g for x in q["nearest_other_m"]]
        line = f"  {f}{lv} n={len(rs)}: added/variant {sorted({q['n_added'] for q in g})}; nearest member {min(nm):.3f}-{max(nm):.3f}; nearest other {min(no):.3f}"
        if f == "A":
            line += (f"; jitter max {max(q['jitter_xy_max'] for q in g):.4f}; yaw dev max {max(q['yaw_dev'] for q in g):.4f}; "
                     f"gap along axis {min(q['gap_along_axis_m'] for q in g):.3f}-{max(q['gap_along_axis_m'] for q in g):.3f}; "
                     f"facing off line {min(q['facing_off_line_deg'] for q in g):.1f}-{max(q['facing_off_line_deg'] for q in g):.1f} deg")
        elif f in ("B", "C"):
            pm = [x for q in g for x in q["pair_m"]]
            line += (f"; pairs {min(pm):.3f}-{max(pm):.3f}; nominal gap err max {max(q['nominal_gap_err'] for q in g):.1e}; "
                     f"jitter max {max(q['jitter_xy_max'] for q in g):.4f}; outermost root ok {sum(q['outermost_root_ok'] for q in g)}/{len(g)}; "
                     f"on bearing err max {max(q['on_bearing_err'] for q in g):.1e}; spacing to base non-members min "
                     f"{min((q['spacing_nonmembers_min_m'] for q in g if q['spacing_nonmembers_min_m'] is not None), default=float('nan')):.3f}")
            if f == "C":
                line += (f"; yaw dev from facing nominal centre max {max(q['yaw_dev_face_centre'] for q in g):.4f}; "
                         f"facing jittered centroid max {max(q['facing_centroid_dev_max'] for q in g):.3f} rad; radius "
                         f"{min(x for q in g for x in q['radius_stored']):.3f}-{max(x for q in g for x in q['radius_stored']):.3f}")
        elif f == "D":
            line += (f"; pairwise added min {min(q['pairwise_added_min_m'] for q in g if q['pairwise_added_min_m'] is not None):.3f}; "
                     f"to base people min {min(q['to_base_min_m'] for q in g):.3f}")
        elif f == "E":
            line += (f"; jitter max {max(q['jitter_xy_max'] for q in g):.4f}; off the camera-rep line max "
                     f"{max(q['perp_off_line_m'] for q in g):.4f}; along from camera {min(q['along_from_cam_m'] for q in g):.3f}-"
                     f"{max(q['along_from_cam_m'] for q in g):.3f}; rep - along {min(q['cam_to_rep_m'] - q['along_from_cam_m'] for q in g):.3f}-"
                     f"{max(q['cam_to_rep_m'] - q['along_from_cam_m'] for q in g):.3f}")
        say(line)
    text = "\n".join(L)
    print(text)
    (V.HERE / "out" / "replay_summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
