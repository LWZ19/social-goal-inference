"""C5 flow pilot: the torch sidecar (FROZEN.md sections 4-5, 6(ii) latency). Never imports sgi; installs nothing.

Run only as:
  HF_HUB_OFFLINE=1 python -I flow.py MODE ...

  train NAME LEFT_OUT CAP_MIN    LEFT_OUT in {none, waiting_line, conversation_group, audience}; trains on
                                 data/train.npz, validation loss on data/val.npz, keeps the lowest-validation-loss
                                 checkpoint: models/NAME/best.npz, curve.json, env.json
  val                            64 samples per validation unit (main on all, each LOSO model on its left-out
                                 situation): samples/val_<model>.npz
  dev                            development records (data/dev.npz): main model on settings (b) and (a), timed
                                 (10 untimed warm-ups, then each record once); each LOSO model on its left-out
                                 situation's records in setting (b): samples/dev_<setting>_<model>.npz, latency.json

It reads and writes only .npz and .json files under phase2/flow_pilot/ (paths guarded). GPU: one process; a job
starts only when nvidia-smi shows >= 3000 MiB free, else it polls every 5 min and logs the wait; nothing is killed.
"""
import json
import math
import os
import subprocess
import sys
import threading
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = 'results/still-scenes/learned-models/flow-model-pilot'
FREE_MIB = 3000
POLL_S = 300
N_SAMPLES = 64
STEPS = 16
SCALE = 3.2
DENS_SIGMA = 0.15
VALID_SLACK = 0.001 + 1e-9
SITS = ('waiting_line', 'conversation_group', 'audience')
MODELS = ('main', 'loso_waiting_line', 'loso_conversation_group', 'loso_audience')


def log(*a):
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), *a, flush=True)


def guard(path, write=False):
    real = os.path.realpath(path)
    assert real.startswith(BASE + '/'), f"refused: {path} is outside {BASE}"
    assert real.endswith('.npz') or real.endswith('.json'), f"refused: {path} is not .npz or .json"
    low = real.lower()
    for bad in ('heldout', 'held_out', 's18', 's20_held', 'eval_16', 'eval_17', 'eval_18', 'eval_19', 'group_photo',
                'lift_wait', '.env', '.cache'):
        assert bad not in low, f"refused: {bad} in {path}"
    if write:
        os.makedirs(os.path.dirname(real), exist_ok=True)
    return real


def load_npz(path):
    return np.load(guard(path), allow_pickle=False)


def save_json(path, obj):
    with open(guard(path, write=True), 'w') as f:
        json.dump(obj, f, indent=1)


def smi(query):
    out = subprocess.run(['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout.strip().split('\n')[0]
    return [v.strip() for v in out.split(',')]


def wait_for_gpu():
    while True:
        free, used, total = (int(v) for v in smi('memory.free,memory.used,memory.total'))
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                               '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
        log(f"nvidia-smi before start: free {free} MiB, used {used} of {total} MiB; compute apps: {apps or 'none'}")
        if free >= FREE_MIB:
            return dict(free_mib=free, used_mib=used, total_mib=total, apps=apps)
        log(f"GPU has < {FREE_MIB} MiB free; waiting {POLL_S} s")
        time.sleep(POLL_S)


class SmiPoll:
    """Polls nvidia-smi memory.used and utilization.gpu every 0.5 s."""

    def __init__(self):
        self.mem, self.util = [], []
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop:
            try:
                m, u = smi('memory.used,utilization.gpu')
                self.mem.append(int(m))
                self.util.append(int(u))
            except Exception:
                pass
            time.sleep(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        self._t.join(timeout=3)

    def summary(self):
        return dict(n=len(self.mem), mem_used_max_mib=max(self.mem, default=None),
                    mem_used_mean_mib=float(np.mean(self.mem)) if self.mem else None,
                    util_max_pct=max(self.util, default=None),
                    util_mean_pct=float(np.mean(self.util)) if self.util else None)


def env_info():
    name, driver = smi('name,driver_version')
    return dict(gpu=name, driver=driver, torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), numpy=np.__version__, python=sys.version.split()[0],
                cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
                threads=torch.get_num_threads())


# ------------------------------------------------------------------------------------------------------ the model
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch = (7, 32, 64, 128, 256)
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1), nn.GroupNorm(8, b), nn.SiLU()]
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(4096, 256)

    def forward(self, r):
        return self.fc(self.net(r).flatten(1))


class TimeEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('freq', torch.exp(-math.log(1e4) * torch.arange(64, dtype=torch.float32) / 64))
        self.mlp = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 256))

    def forward(self, t):
        a = 1000.0 * t[:, None] * self.freq[None, :]
        return self.mlp(torch.cat([a.sin(), a.cos()], dim=1))


class FiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(256, 256)
        self.ln = nn.LayerNorm(256)
        self.film = nn.Linear(256, 512)

    def forward(self, h, c):
        g, b = self.film(F.silu(c)).chunk(2, dim=1)
        return F.silu(self.ln(self.lin(h)) * (1 + g) + b)


class Flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.temb = TimeEmb()
        self.inp = nn.Linear(4, 256)
        self.layers = nn.ModuleList([FiLM() for _ in range(3)])
        self.out = nn.Linear(256, 4)

    def velocity(self, x, t, e):
        c = e + self.temb(t)
        h = self.inp(x)
        for layer in self.layers:
            h = layer(h, c)
        return self.out(h)

    def forward(self, x, t, r):
        return self.velocity(x, t, self.enc(r))


def state_to_npz(model, path):
    np.savez(guard(path, write=True), **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})


def model_from_npz(path, dev):
    z = load_npz(path)
    m = Flow()
    m.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files})
    return m.to(dev).eval()


# ------------------------------------------------------------------------------------------------------- training
def poses(npz, left_out):
    keep = npz['has_label'] & (npz['situation'] != left_out)
    units = np.flatnonzero(keep)
    lab = npz['labels'][units].reshape(-1, 4)
    unit_of = np.repeat(np.arange(len(units)), npz['labels'].shape[1])
    return units, lab, unit_of


@torch.no_grad()
def val_loss(model, R, X, U, Z, T):
    model.eval()
    tot, n = 0.0, 0
    for lo in range(0, len(X), 4096):
        x, z, t = X[lo:lo + 4096], Z[lo:lo + 4096], T[lo:lo + 4096]
        xt = (1 - t[:, None]) * z + t[:, None] * x
        v = model(xt, t, R[U[lo:lo + 4096]])
        tot += float(((v - (x - z)) ** 2).mean(dim=1).sum())
        n += len(x)
    model.train()
    return tot / n


def train(name, left_out, cap_min):
    assert left_out in ('none',) + SITS
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)
    gpu0 = wait_for_gpu()
    dev = torch.device('cuda')
    torch.manual_seed(0)
    tr, va = load_npz(f'{BASE}/data/train.npz'), load_npz(f'{BASE}/data/val.npz')
    tu, tx, tuo = poses(tr, left_out)
    vu, vx, vuo = poses(va, left_out)
    R = torch.from_numpy(tr['rasters'][tu]).to(dev)
    X = torch.from_numpy(tx).to(dev)
    U = torch.from_numpy(tuo).to(dev)
    RV = torch.from_numpy(va['rasters'][vu]).to(dev)
    XV = torch.from_numpy(vx).to(dev)
    UV = torch.from_numpy(vuo).to(dev)
    g = torch.Generator().manual_seed(1)
    ZV = torch.randn(len(vx), 4, generator=g).to(dev)
    TV = torch.rand(len(vx), generator=g).to(dev)
    sit_tr = {str(s): int((tr['situation'][tu] == s).sum()) for s in np.unique(tr['situation'][tu])}
    sit_va = {str(s): int((va['situation'][vu] == s).sum()) for s in np.unique(va['situation'][vu])}
    log(f"{name}: left out {left_out}; train units {len(tu)} {sit_tr}, poses {len(tx)}; val units {len(vu)} {sit_va}, "
        f"poses {len(vx)}")
    model = Flow().to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    n = len(tx)
    spe = math.ceil(n / 256)
    total = 30 * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    torch.cuda.reset_peak_memory_stats()
    curve = dict(epoch=[], train_loss=[], val_loss=[], lr_end=[], elapsed_s=[], steps=[])
    step_loss = []
    best, best_epoch = math.inf, None
    cap = cap_min * 60.0
    stopped = 'epochs'
    t0 = time.monotonic()
    with SmiPoll() as poll:
        for epoch in range(1, 31):
            perm = torch.randperm(n).to(dev)
            run, cnt = 0.0, 0
            out_of_time = False
            for b in range(spe):
                if time.monotonic() - t0 > cap:
                    out_of_time = True
                    break
                idx = perm[b * 256:(b + 1) * 256]
                x = X[idx]
                r = R[U[idx]]
                z = torch.randn_like(x)
                t = torch.rand(len(x), device=dev)
                xt = (1 - t[:, None]) * z + t[:, None] * x
                loss = ((model(xt, t, r) - (x - z)) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                lv = loss.item()
                run += lv * len(x)
                cnt += len(x)
                if b % 50 == 0:
                    step_loss.append([(epoch - 1) * spe + b, lv])
            if out_of_time:
                stopped = f'wall-time cap {cap_min} min during epoch {epoch} (no validation for the partial epoch)'
                break
            vl = val_loss(model, RV, XV, UV, ZV, TV)
            el = time.monotonic() - t0
            curve['epoch'].append(epoch)
            curve['train_loss'].append(run / max(cnt, 1))
            curve['val_loss'].append(vl)
            curve['lr_end'].append(sched.get_last_lr()[0])
            curve['elapsed_s'].append(el)
            curve['steps'].append(epoch * spe)
            mark = ''
            if vl < best:
                best, best_epoch = vl, epoch
                state_to_npz(model, f'{BASE}/models/{name}/best.npz')
                mark = ' (kept)'
            log(f"{name} epoch {epoch:2d}: train {run / max(cnt, 1):.5f} val {vl:.5f} lr {sched.get_last_lr()[0]:.2e} "
                f"{el:7.1f} s{mark}")
            if time.monotonic() - t0 > cap:
                stopped = f'wall-time cap {cap_min} min after epoch {epoch}'
                break
    wall = time.monotonic() - t0
    curve['step_loss'] = step_loss
    env = dict(env_info(), name=name, left_out=left_out, cap_min=cap_min, stopped=stopped, best_epoch=best_epoch,
               best_val_loss=best, wall_s=wall, steps_per_epoch=spe, schedule_total_steps=total,
               train_units=int(len(tu)), train_poses=int(n), val_units=int(len(vu)), val_poses=int(len(vx)),
               train_units_by_situation=sit_tr, val_units_by_situation=sit_va,
               peak_mem_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
               peak_mem_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20, nvidia_smi=poll.summary(),
               gpu_before=gpu0, params=sum(p.numel() for p in model.parameters()))
    save_json(f'{BASE}/models/{name}/curve.json', curve)
    save_json(f'{BASE}/models/{name}/env.json', env)
    log(f"{name}: done, best epoch {best_epoch} val {best:.5f}, wall {wall:.1f} s, stopped by {stopped}, "
        f"peak alloc {env['peak_mem_allocated_mib']:.0f} MiB")


# -------------------------------------------------------------------------------------------------------- sampling
@torch.no_grad()
def sample(model, raster, key, dev):
    """64 samples, 16 Euler steps; z from a CPU generator seeded with crc32(key). Returns (64, 4) numpy."""
    g = torch.Generator().manual_seed(zlib.crc32(key.encode()))
    z = torch.randn(N_SAMPLES, 4, generator=g)
    x = z.to(dev)
    e = model.enc(torch.from_numpy(raster)[None].to(dev)).expand(N_SAMPLES, -1)
    for k in range(STEPS):
        t = torch.full((N_SAMPLES,), k / STEPS, device=dev)
        x = x + model.velocity(x, t, e) / STEPS
    return x.cpu().numpy().astype(np.float64)


def to_world(x, frame):
    ox, oy, yaw = frame
    c, s = math.cos(yaw), math.sin(yaw)
    px, py = SCALE * x[:, 0], SCALE * x[:, 1]
    wx, wy = ox + c * px - s * py, oy + s * px + c * py
    head = np.arctan2(x[:, 3], x[:, 2]) + yaw
    head = math.pi - np.mod(math.pi - head, 2 * math.pi)          # wrap into (-pi, pi]
    return np.stack([wx, wy], axis=1), head, np.stack([px, py], axis=1)


def density(p):
    d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * DENS_SIGMA ** 2)).sum(1)


def project(pw, head, Fw, ex, place_ex, rr, places_fb):
    """Snap, validity, density over valid snapped positions, top pose and hit@5 places (FROZEN.md 5, 6(ii))."""
    d = np.hypot(pw[:, None, 0] - Fw[None, :, 0], pw[:, None, 1] - Fw[None, :, 1])
    snap = np.argmin(d, axis=1)
    sp = Fw[snap]
    valid = ex[snap] <= place_ex + VALID_SLACK
    dens = np.full(len(pw), np.nan)
    if not valid.any():
        return dict(snap=snap, sp=sp, valid=valid, dens=dens, fallback=True, top=places_fb[0].copy(),
                    top_idx=-1, places=places_fb[:5].copy(), n_places=len(places_fb[:5]))
    vi = np.flatnonzero(valid)
    dv = density(sp[vi])
    dens[vi] = dv
    order = vi[np.lexsort((vi, -dv))]                               # densest first, ties to the lowest index
    top = int(order[0])
    chosen = [top]
    for i in order[1:]:
        if len(chosen) == 5:
            break
        if all(math.hypot(sp[i, 0] - sp[j, 0], sp[i, 1] - sp[j, 1]) >= 2 * rr - 1e-9 for j in chosen):
            chosen.append(int(i))
    places = np.array([[sp[i, 0], sp[i, 1], head[i]] for i in chosen])
    return dict(snap=snap, sp=sp, valid=valid, dens=dens, fallback=False,
                top=np.array([sp[top, 0], sp[top, 1], head[top]]), top_idx=top, places=places, n_places=len(chosen))


def val_mode():
    torch.set_num_threads(2)
    wait_for_gpu()
    dev = torch.device('cuda')
    va = load_npz(f'{BASE}/data/val.npz')
    keys, sit, frames, rasters = va['keys'], va['situation'], va['frame'], va['rasters']
    for name in MODELS:
        model = model_from_npz(f'{BASE}/models/{name}/best.npz', dev)
        sel = np.arange(len(keys)) if name == 'main' else np.flatnonzero(sit == name[len('loso_'):])
        out = dict(keys=keys[sel], raw_world=[], head=[], raw_raster=[], dens=[], top=[], x=[])
        for i in sel:
            x = sample(model, rasters[i], str(keys[i]), dev)
            pw, hd, pr = to_world(x, frames[i])
            dn = density(pr)
            top = int(np.lexsort((np.arange(len(dn)), -dn))[0])
            out['raw_world'].append(pw)
            out['head'].append(hd)
            out['raw_raster'].append(pr)
            out['dens'].append(dn)
            out['top'].append(top)
            out['x'].append(x)
        np.savez(guard(f'{BASE}/samples/val_{name}.npz', write=True),
                 **{k: (np.array(v) if k != 'keys' else v) for k, v in out.items()})
        log(f"val samples {name}: {len(sel)} units")


def dev_pass(model, dv, setting, idx, dev, timed):
    F_all, ex_all, off = dv[f'F_{setting}'], dv[f'ex_{setting}'], dv[f'off_{setting}']
    rasters, keys, frames = dv[f'raster_{setting}'], dv['keys'], dv['frame']
    pex, places, rrs = dv[f'place_ex_{setting}'], dv[f'places_{setting}'], dv['robot_radius']
    cols = dict(keys=[], raw_world=[], head=[], snap=[], sp=[], valid=[], dens=[], fallback=[], top=[], top_idx=[],
                places=[], n_places=[], raw_top=[], raw_top_idx=[], latency_s=[])
    arrays = [(rasters[i], F_all[off[i]:off[i + 1]], ex_all[off[i]:off[i + 1]]) for i in range(len(keys))]
    if timed:
        for i in idx[:10]:                                          # untimed warm-ups
            r, Fw, ex = arrays[i]
            x = sample(model, r, str(keys[i]), dev)
            pw, hd, _ = to_world(x, frames[i])
            project(pw, hd, Fw, ex, pex[i], rrs[i], places[i])
    for i in idx:
        r, Fw, ex = arrays[i]
        t0 = time.perf_counter()
        x = sample(model, r, str(keys[i]), dev)
        pw, hd, _ = to_world(x, frames[i])
        pr = project(pw, hd, Fw, ex, pex[i], rrs[i], places[i])
        lat = time.perf_counter() - t0
        dn = density(pw)
        rt = int(np.lexsort((np.arange(len(dn)), -dn))[0])
        pl = np.full((5, 3), np.nan)
        pl[:pr['n_places']] = pr['places']
        for k, v in (('keys', keys[i]), ('raw_world', pw), ('head', hd), ('snap', pr['snap']), ('sp', pr['sp']),
                     ('valid', pr['valid']), ('dens', pr['dens']), ('fallback', pr['fallback']), ('top', pr['top']),
                     ('top_idx', pr['top_idx']), ('places', pl), ('n_places', pr['n_places']),
                     ('raw_top', np.array([pw[rt, 0], pw[rt, 1], hd[rt]])), ('raw_top_idx', rt),
                     ('latency_s', lat if timed else np.nan)):
            cols[k].append(v)
    return {k: np.array(v) for k, v in cols.items()}


def dev_mode():
    torch.set_num_threads(2)
    gpu0 = wait_for_gpu()
    dev = torch.device('cuda')
    dv = load_npz(f'{BASE}/data/dev.npz')
    dv = {k: dv[k] for k in dv.files}
    keys, sit = dv['keys'], dv['situation']
    allidx = np.arange(len(keys))
    lat = dict(env=env_info(), gpu_before=gpu0, n_records=int(len(keys)), warmups=10)
    main = model_from_npz(f'{BASE}/models/main/best.npz', dev)
    for setting in ('b', 'a'):
        torch.cuda.reset_peak_memory_stats()
        with SmiPoll() as poll:
            out = dev_pass(main, dv, setting, allidx, dev, timed=True)
        np.savez(guard(f'{BASE}/samples/dev_{setting}_main.npz', write=True), **out)
        ls = out['latency_s']
        lat[f'main_{setting}'] = dict(median_s=float(np.median(ls)), p90_s=float(np.percentile(ls, 90)),
                                      mean_s=float(np.mean(ls)), max_s=float(np.max(ls)), n=int(len(ls)),
                                      peak_mem_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                                      nvidia_smi=poll.summary(), fallbacks=int(out['fallback'].sum()))
        log(f"dev {setting} main: median {np.median(ls) * 1000:.1f} ms, p90 {np.percentile(ls, 90) * 1000:.1f} ms, "
            f"fallbacks {int(out['fallback'].sum())}")
    for name in MODELS[1:]:
        model = model_from_npz(f'{BASE}/models/{name}/best.npz', dev)
        idx = np.flatnonzero(sit == name[len('loso_'):])
        out = dev_pass(model, dv, 'b', idx, dev, timed=False)
        np.savez(guard(f'{BASE}/samples/dev_b_{name}.npz', write=True), **out)
        log(f"dev b {name}: {len(idx)} records, fallbacks {int(out['fallback'].sum())}")
    save_json(f'{BASE}/samples/latency.json', lat)


def main(argv):
    mode = argv[0]
    log(f"flow.py {' '.join(argv)}; HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}; isolated={sys.flags.isolated}")
    assert 'sgi' not in sys.modules
    if mode == 'train':
        train(argv[1], argv[2], float(argv[3]))
    elif mode == 'val':
        val_mode()
    elif mode == 'dev':
        dev_mode()
    else:
        raise SystemExit(f'unknown mode {mode}')
    assert 'sgi' not in sys.modules


if __name__ == '__main__':
    main(sys.argv[1:])
