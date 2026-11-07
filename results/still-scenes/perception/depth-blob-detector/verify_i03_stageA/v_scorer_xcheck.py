"""Verifier: cross-check the frozen people_scorer.score_people against the verifier's own scorer on random detection
sets that exercise matching ties, exemption, near-hidden, merges and merged pairs (fit scenes + a few eval scenes),
plus the step-2 self-check (true people as detections) with the verifier's scorer on all 289 fit scenes."""
import random, sys, math
from pathlib import Path
sys.path.insert(0, 'docs/model_notes/tools')
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1b/i03')
import research_harness as H
import people_scorer as PS            # frozen scorer, imported only to compare numbers
import v_core as C, v_fit as F
from sgi.config import load_config

class Sc:
    def __init__(self, obs, truth): self.observation, self.truth = obs, truth

th = load_config().thresholds
rng = random.Random(7)
scenes = F.fit_scenes()
selfcheck_ok = 0; nd = 0; ntest = 0
for i, (dom, st, ld) in enumerate(scenes):
    obs, truth = F.load(ld)
    T = C.Truth(obs, truth, th)
    r = C.score([(p.x, p.y, p.yaw) for p in T.vis], T)
    selfcheck_ok += (r['matched'] == len(T.vis) and r['false'] == 0 and r['merges'] == 0 and r['w45_full'] == len(T.vis))
    if i % 7 and dom != 'F-real':
        continue
    sc = Sc(obs, truth)
    for trial in range(15):
        dets = []
        for p in truth.people:
            if rng.random() < 0.7:
                dets.append((p.x + rng.gauss(0, 0.25), p.y + rng.gauss(0, 0.25), rng.choice([None, rng.uniform(-math.pi, math.pi)])))
        ppl = list(truth.people)
        for a in range(len(ppl)):
            for b in range(a + 1, len(ppl)):
                if rng.random() < 0.3:
                    dets.append(((ppl[a].x + ppl[b].x) / 2, (ppl[a].y + ppl[b].y) / 2, None))
        for t in truth.targets.values():
            if rng.random() < 0.5:
                dets.append((t.anchor[0] + rng.uniform(-0.4, 0.4), t.anchor[1] + rng.uniform(-0.4, 0.4), 0.3))
        for _ in range(rng.randint(0, 3)):
            dets.append((rng.uniform(-8, 8), rng.uniform(-8, 8), None))
        rng.shuffle(dets)
        mine = C.score(dets, T)
        theirs = PS.score_people(dets, sc, th, exempt_xy=PS.exempt_points(truth), facing='full')
        a = (mine['matched'], mine['false'], mine['exempt'], mine['near_hidden'], mine['merges'], mine['npairs'], mine['mpairs'],
             mine['known'], mine['unknown'], mine['w45_full'], mine['w45_axis'], sorted(m['id'] for m in mine['misses']),
             sum(m['staff_behind'] for m in mine['misses']), mine['matches'])
        b = (theirs['matched'], theirs['false'], theirs['exempt'], theirs['near_hidden'], theirs['merges'], theirs['neighbour_pairs'],
             theirs['merged_pairs'], theirs['facing']['known'], theirs['facing']['unknown'], theirs['facing']['within45_full'],
             theirs['facing']['within45_axis'], sorted(m['id'] for m in theirs['misses']), theirs['staff_behind'], theirs['matches'])
        ntest += 1
        if a != b:
            nd += 1
            if nd < 6: print('DIFF', ld[2], a, b)
print('self-check (true people as detections, verifier scorer):', selfcheck_ok, 'of', len(scenes))
print('random cross-check vs frozen scorer:', ntest - nd, 'of', ntest, 'agree')
