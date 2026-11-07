import sys
from pathlib import Path
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
from sgi.scene import load_scene
from sgi.config import DEFAULT_CONFIG, load_config
from sgi.gt.derive import prepare
FRESH = Path('results/2026-10-17-beyond-vlm/phase1b/fresh_seeds')
cfg = load_config(DEFAULT_CONFIG)
for split, sid in [('eval','eval_05_no_one_person'),('eval','eval_08_waiting_line_4'),('eval','eval_12_conversation_group_4'),('eval','eval_15_audience_5'),('eval','eval_00_no_one_place')]:
    f = FRESH/'seed100'/split/sid
    H.guard_scene(f)
    inputs, truth = load_scene(f)
    print('==', sid, 'robot', inputs.observation.robot)
    for t in truth.targets.values(): print('  target', t.id, t.kind, t.anchor, t.front_yaw, t.person_id, 'fp', t.footprint is not None)
    for p in truth.people: print('  person', p.id, round(p.x,3), round(p.y,3), round(p.yaw,3), p.group, p.role)
    print('  bounds', truth.bounds[:3])
    ets = prepare(inputs, truth, cfg)
    for eid, et in ets.items():
        g = et.gt
        print('  ep', eid, g.situation, g.target_ids, len(g.acceptable_regions or []), [r.target_id for r in g.acceptable_regions or []])
    print('  raw eps', {k: (v.situation, len(v.acceptable_regions or [])) for k, v in truth.episodes.items()})
