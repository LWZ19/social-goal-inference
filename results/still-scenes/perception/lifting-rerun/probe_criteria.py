"""Per-model tallies from lifting_records.tsv (written by lifting.py): lift classes, oracle fixes and full successes
of the C wrong targets on the realistic sets, for the I11 first-probe criteria. Output: probe_criteria_out.txt."""
import csv, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
lines = []
rows = [r for r in csv.DictReader(open(HERE / 'lifting_records.tsv'), delimiter='\t') if r['method'] == 'con' and r['arm'] in ('real', 'staffed', 'plain')]
lines.append(f'C realistic wrong-target rows {len(rows)}')
for ms in (['glm53', 'dsv41'], ['qwen38_low'], ['qwen38_low', 'glm53', 'dsv41']):
    rs = [r for r in rows if r['model'] in ms and r['has_point'] == 'True']
    allr = [r for r in rows if r['model'] in ms]
    lines.append(f"{ms} all {len(allr)} with point {len(rs)} lift classes {sum(r['lift_class'] != 'choice error' for r in rs)} "
                 f"oracle fixes {sum(r['fix_oracle'] == 'True' for r in rs)} fix->full success {sum(r['fix_success'] == 'True' for r in rs)} "
                 f"{dict(collections.Counter(r['lift_class'] for r in rs))}")
for m in ('qwen38_low', 'glm53', 'dsv41'):
    rs = [r for r in rows if r['model'] == m and r['has_point'] == 'True']
    lines.append(f"{m} (n, oracle fix, full success) " + str({c: (sum(r['class'] == c for r in rs), sum(r['class'] == c and r['fix_oracle'] == 'True' for r in rs), sum(r['class'] == c and r['fix_success'] == 'True' for r in rs)) for c in ('person', 'parcel', 'place')}))
print('\n'.join(lines))
(HERE / 'probe_criteria_out.txt').write_text('\n'.join(lines) + '\n')
