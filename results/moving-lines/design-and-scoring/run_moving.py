"""Run the moving-people arms for one model: groups of (scenario, seed), each group's arms in order on one worker,
so that C1, CT and Cq share their first answer through the cache (the arms are paired), and R1, Rq theirs.

    .venv/bin/python run_moving.py --model ENTRY --out DIR [--workers 3] [--scenarios a b ...] [--seeds 0 1 2]
                                   [--arms C1 CT Cq R1 Rq] [--max-rpm 25]

Each worker starts its own two services (constraints and direct_marking) and stops them by PID at the end. An
episode whose episode.json exists is skipped, so a run can be resumed. A relative --out names a run folder under
results/data/moving/ (paths.MOVING_RUNS), where every run folder lives; --jobs is read relative to the working directory.
"""
import argparse
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

REPO = paths.REPO
SIM_PY = Path.home() / "miniforge3/envs/sgi-habitat/bin/python"
VENV_PY = REPO / ".venv/bin/python"


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)    # as HTTPServer: only a listener blocks
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def wait_up(port: int, proc: subprocess.Popen, seconds: float = 60.0) -> None:
    """Waits for our own service: a health answer while `proc` has died means another process holds the port."""
    end = time.time() + seconds
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"service on {port} (pid {proc.pid}) exited with {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
        except OSError:
            time.sleep(0.5)
            continue
        time.sleep(1.0)                                # a bind failure exits within a second or so
        if proc.poll() is not None:
            raise RuntimeError(f"port {port} answers, but our service (pid {proc.pid}) exited: a stale service")
        return
    raise RuntimeError(f"service on {port} did not start")


def worker(w: int, jobs: list, lock: threading.Lock, args, log) -> None:
    ports = {"C": args.port_base + 2 * w, "R": args.port_base + 1 + 2 * w}
    serves = {}
    busy = [port for port in ports.values() if not port_free(port)]
    if busy:
        raise RuntimeError(f"w{w}: ports {busy} are in use; stop the processes holding them first")
    for kind, port in ports.items():
        goal = "constraints" if kind == "C" else "direct_marking"
        f = open(args.out / f"serve_w{w}_{kind}.log", "a")
        serves[kind] = subprocess.Popen([str(VENV_PY), "-m", "sgi.serve", "--model", args.model, "--goal", goal,
                                         "--port", str(port), "--max-rpm", str(args.max_rpm),
                                         *(f"--threshold={pair}" for pair in args.threshold)],
                                        cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        log(f"w{w} serve {kind} pid {serves[kind].pid} port {port}")
    try:
        for kind, port in ports.items():
            wait_up(port, serves[kind])
        while True:
            with lock:
                if not jobs:
                    return
                scenario, seed = jobs.pop(0)
            for arm in args.arms:
                out = args.out / scenario / f"seed{seed}" / arm
                if (out / "episode.json").exists():
                    continue
                cmd = [str(SIM_PY), "-m", args.module, "--scenario", f"sim/scenarios/{scenario}.yaml",
                       "--arm", arm, "--seed", str(seed), "--out", str(out), "--label", args.model,
                       "--sgi-url", f"http://127.0.0.1:{ports['R' if arm.startswith('R') else 'C']}", "--seconds", str(args.seconds)]
                if scenario in args.record:
                    cmd += ["--record", str(out / "video.mp4")]
                out.mkdir(parents=True, exist_ok=True)
                # A service answers one request at a time, and a slow first answer can outlive its own episode: wait
                # here, outside the episode's time limit, until the service is free, so the wait delays the next
                # episode's start and never its clock or its limit.
                port = ports['R' if arm.startswith('R') else 'C']
                for _ in range(60):                       # up to an hour
                    try:
                        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=60).read()
                        break
                    except OSError:
                        continue
                started = time.time()
                with open(out / "sim.log", "w") as f:
                    try:
                        code = subprocess.call(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                                               env={**os.environ, "PYTHONPATH": str(REPO / "sim")},
                                               timeout=args.seconds * 4 + 120)
                    except subprocess.TimeoutExpired:     # one stuck episode must not end this worker's other jobs
                        code = "timeout"                  # no episode.json: a later pass runs it again
                tail = (out / "sim.log").read_text().strip().splitlines()[-1:] or [""]
                log(f"w{w} {scenario} seed{seed} {arm}: exit {code} in {time.time() - started:.0f} s | {tail[0][:160]}")
    finally:
        for kind, proc in serves.items():
            proc.terminate()
            proc.wait(timeout=10)
            log(f"w{w} serve {kind} pid {proc.pid} stopped")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--scenarios", nargs="+", default=["cafe", "room_coffee_cart", "cafe_line2_moving",
                                                        "cafe_line6_moving", "sweet_spot_line2_moving", "cafe_busy"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=["C1", "CT", "Cq", "R1", "Rq"])
    ap.add_argument("--max-rpm", type=int, default=25)
    ap.add_argument("--seconds", type=float, default=92.0)
    ap.add_argument("--record", nargs="*", default=[], help="scenarios to record video for")
    ap.add_argument("--module", default="sgi_habitat.experiment", help="or sgi_habitat.experiment_controls")
    ap.add_argument("--port-base", type=int, default=8800, help="services on this port and up, two a worker")
    ap.add_argument("--threshold", action="append", default=[], metavar="KEY=VALUE",
                    help="passed to every sgi.serve as --threshold (e.g. join_spacing_ratio=1.5)")
    ap.add_argument("--jobs", type=Path, help="a file of 'scenario seed' lines, in place of --scenarios x --seeds")
    args = ap.parse_args()
    if not args.out.is_absolute():                 # run folders live in results/data/moving/
        args.out = paths.MOVING_RUNS / args.out
    args.out = args.out.resolve()                  # the sim runs from the repository root
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, seed) for seed in args.seeds for s in args.scenarios]
    if args.jobs is not None:                      # e.g. the sweep: each layout with its own seed
        jobs = [(line.split()[0], int(line.split()[1])) for line in args.jobs.read_text().splitlines() if line.strip()]
    lock, log_lock = threading.Lock(), threading.Lock()
    logf = open(args.out / "run.log", "a")

    def log(msg):
        with log_lock:
            line = f"{time.strftime('%H:%M:%S')} {msg}"
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()

    log(f"model {args.model}: {len(jobs)} groups x {len(args.arms)} arms, {args.workers} workers")
    threads = [threading.Thread(target=worker, args=(w, jobs, lock, args, log)) for w in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
