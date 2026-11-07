"""I06 repair probe: shared paths, start-up checks and small helpers. Zero model or network calls; no .env is read.
Every script calls start() first: it asserts the sha256 of FROZEN.md and of src/sgi/solver.py, that sgi is imported
from the worktree's src/, and runs the static check on rules.py."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

WT = paths.REPO
RD = paths.STILL
HERE = Path(__file__).resolve().parents[1]
CODE = HERE / 'code'
OUT = HERE / 'out'
WORK = HERE / 'work'
LOGS = HERE / 'logs'
I06_SCORE = HERE.parent / 'score'
I01 = RD / 'replay-harness/repairs'
A2 = RD / 'people-finder/early-detector/phase1c/a2'

FROZEN_SHA = '02432f62320c0ebb0924c0f0e5b4b2089d508eb4a85fd28cf56641c25e27560f'
SOLVER_SHA = '10ece70f521eff2b1224267fa0b91bfac546d5740f85622073989a16c6c5d800'
FORBIDDEN_TOKENS = ('situation', 'role', 'group')

for p in (str(CODE), str(I06_SCORE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now() -> str:
    return time.strftime('%F %T %z')


def static_check() -> None:
    text = (CODE / 'rules.py').read_text().lower()
    bad = [t for t in FORBIDDEN_TOKENS if t in text]
    assert not bad, f'static check failed: rules.py contains {bad}'


def start(log=sys.stdout) -> dict:
    import sgi
    from sgi import solver as S
    assert sha256(HERE / 'FROZEN.md') == FROZEN_SHA, 'FROZEN.md hash'
    assert Path(sgi.__file__).resolve().is_relative_to(WT / 'src'), sgi.__file__
    assert sha256(Path(S.__file__)) == SOLVER_SHA, 'solver.py hash'
    static_check()
    info = {'frozen_sha256': FROZEN_SHA, 'solver_sha256': SOLVER_SHA, 'sgi': sgi.__file__,
            'rules_sha256': sha256(CODE / 'rules.py'), 'started': now()}
    print(f'{now()} start: {info}', file=log, flush=True)
    return info


def i06_hashes() -> dict:
    return {name: sha256(I06_SCORE / name) for name in ('lib.py', 'noise.py', 'noise_model.json', 'arms.py',
                                                         'noisy.py', 'rows_oracle.jsonl')}


def free_mb() -> int:
    out = subprocess.run(['free', '-m'], capture_output=True, text=True).stdout
    return int(next(l for l in out.splitlines() if l.startswith('Mem:')).split()[6])


def wait_memory(log=sys.stdout, need=3000):
    while True:
        avail = free_mb()
        if avail >= need:
            print(f'{now()} free -m available {avail} MB', file=log, flush=True)
            return avail
        print(f'{now()} waiting for memory: {avail} MB available', file=log, flush=True)
        time.sleep(120)


def xy_list(xy):
    return None if xy is None else [float(xy[0]), float(xy[1])]


def xy_close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.dist(a, b) <= tol


def write_jsonl(path: Path, rows) -> None:
    with open(path, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')


def read_jsonl(path: Path) -> list:
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def absorbed(trace_entry) -> list:
    """The people a call admitted, each with the current last at that time, yaw known and f . u (from the trace)."""
    if not trace_entry:
        return []
    joined = trace_entry.get('joined', [])
    # the admitted candidate that joined at each iteration is the nearest admitted one: those in `joined`
    got = []
    for k, n in enumerate(joined):
        it = trace_entry['iterations'][k]
        cand = [c for c in it if c['admitted'] and c['number'] == n]
        assert len(cand) == 1, (trace_entry, n)
        got.append(cand[0])
    return got


def gated(trace_entry) -> list:
    """Candidates the package's test accepted that the gate turned away (repair rules only)."""
    if not trace_entry:
        return []
    return [c for it in trace_entry['iterations'] for c in it if not c['admitted']]
