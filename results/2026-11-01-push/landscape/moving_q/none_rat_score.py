"""Scorer for CT2_NONE_RAT_POSTHOC.md (sha256 f4e88d7e), zero calls. Judges N1, R1, R2 exactly as written."""
import statistics, sys
from pathlib import Path
M = Path('results/2026-11-01-push/moving')
sys.path.insert(0, str(M)); sys.path.insert(0, str(M / 'analysis')); sys.path.insert(0, str(Path(__file__).parent))
import summarise_fresh as S, failures as F
from ct2_attrib import grp
from ct2_attrib2 import load
R1 = {'Sonnet': 179, 'DeepSeek': 60}; R2 = {'Sonnet': 0.041, 'DeepSeek': 0.039}
def sh(e): return S.measures(e)['share_strict']
for root, label in ((M / 'runs/sonnet55_low', 'Sonnet'), (M / 'runs/dsv41', 'DeepSeek')):
    eps = load(root)
    cells = sorted(k[:2] for k in eps if k[2] == 'CT2' and k[:2] + ('CT',) in eps)
    print(f'\n## {label}: {len(cells)} cells')
    none = [c for c in cells if c + ('CT2-none',) in eps]
    if none:
        d = [sh(eps[c + ('CT2-none',)]) - sh(eps[c + ('CT',)]) for c in none]
        w, l = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
        print(f'  CT2-none present {len(none)}: vs CT mean {statistics.mean(d):+.3f} mean|d| {statistics.mean(abs(x) for x in d):.3f} W/L/T {w}/{l}/{len(d)-w-l}'
              f' -> N1 {"PASS" if abs(statistics.mean(d)) <= 0.04 else "FAIL"}{"" if len(none) == len(cells) else " (partial)"}')
    rat = [c for c in cells if c + ('CT2-rat',) in eps]
    if rat:
        rep = sum(grp(x) == 'replace rule' for c in rat for x in F.classify(eps[c + ('CT2-rat',)]))
        rep_ct = sum(grp(x) == 'replace rule' for c in rat for x in F.classify(eps[c + ('CT',)]))
        kept = statistics.mean(sh(eps[c + ('CT2-rat',)]) - sh(eps[c + ('CT',)]) for c in rat)
        gain = statistics.mean(sh(eps[c + ('CT2',)]) - sh(eps[c + ('CT',)]) for c in rat)
        d = [sh(eps[c + ('CT2-rat',)]) - sh(eps[c + ('CT2',)]) for c in rat]
        print(f'  CT2-rat present {len(rat)}: replace rule {rep} (CT {rep_ct}) -> R1 {"PASS" if rep >= R1[label] else "FAIL"}; '
              f'CT2-rat - CT {kept:+.3f} (CT2 - CT {gain:+.3f}) -> R2 {"PASS" if kept >= R2[label] else "FAIL"}; '
              f'CT2-rat - CT2 mean {statistics.mean(d):+.3f} W/L/T {sum(x>1e-9 for x in d)}/{sum(x<-1e-9 for x in d)}/{sum(abs(x)<=1e-9 for x in d)}'
              f'{"" if len(rat) == len(cells) else " (partial)"}')
