"""Reproduce rerun/lifting (lifting.py and frontsnap.py, unchanged) with their outputs redirected into
phase1b/i11/repro_lifting/ (DEVIATIONS.md D4), then diff against the original outputs."""
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i11lib as I  # noqa: E402
import frontsnap as FS  # noqa: E402

OUT = I.OUTDIR / 'repro_lifting'
ORIG = I.RD / 'rerun/lifting'


def main():
    for line in I.check_pins():
        print(line)
    OUT.mkdir(exist_ok=True)
    I.L.OUT = OUT
    I.L.LOG.clear()
    I.L.main()
    FS.OUT = OUT
    FS.LOG.clear()
    FS.main()
    report = []
    for name in ('lifting_out.txt', 'frontsnap_out.txt', 'lifting_records.tsv'):
        a = (ORIG / name).read_text().splitlines()
        b = (OUT / name).read_text().splitlines()
        diff = list(difflib.unified_diff(a, b, 'original/' + name, 'repro/' + name, lineterm='', n=0))
        report.append(f'{name}: {len(a)} lines original, {len(b)} reproduced, '
                      f'{"IDENTICAL" if not diff else str(len(diff)) + " diff lines"}')
        report.extend(diff[:200])
    (OUT / 'diff.txt').write_text('\n'.join(report) + '\n')
    print('\n'.join(report[:60]))


if __name__ == '__main__':
    main()
