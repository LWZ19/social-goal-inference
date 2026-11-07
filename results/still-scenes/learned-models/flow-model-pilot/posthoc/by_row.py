"""POST HOC (written after tables.txt was read; decides nothing): development records by scene row (label used only
to tabulate), flow top-1 against C (setting b) and P (setting a), and the top pose's components (setting b)."""
import collections, json, glob
from pathlib import Path
B = Path(__file__).resolve().parents[1]
out = []
for st, ref in (('b', 'C_top1'), ('a', 'P_top1')):
    rows = [json.loads(x) for f in sorted(glob.glob(str(B / f'rows_{st}_*.jsonl'))) if 'loso' not in f
            for x in open(f)]
    g = collections.defaultdict(list)
    for r in rows:
        row = r['scene'].split('_', 3)[-1] if r['scene'].startswith(('real_', 'staffed_', 'plain_')) else r['scene'].split('_', 2)[-1]
        g[row].append(r)
    out.append(f'setting ({st}): row, n, {ref[0]} top-1, flow top-1, flow top in region / heading ok / reachable, fallbacks, valid share')
    for row in sorted(g):
        rr = g[row]
        out.append(f"  {row:26s} n {len(rr):3d}  {ref[0]} {sum(r[ref] for r in rr):3d}  flow {sum(r['top1'] for r in rr):3d}  "
                   f"inreg {sum(r['top_in_region'] for r in rr):3d} head {sum(r['top_heading_ok'] for r in rr):3d} "
                   f"reach {sum(r['top_reach'] for r in rr):3d}  fb {sum(r['fallback'] for r in rr):3d}  "
                   f"valid {sum(r['valid_share'] for r in rr) / len(rr):.2f}")
txt = 'POST HOC, decides nothing.\n' + '\n'.join(out) + '\n'
(B / 'posthoc' / 'by_row.txt').write_text(txt)
print(txt)
