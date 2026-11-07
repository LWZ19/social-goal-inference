"""Verifier's own torch sidecar for the C5 flow pilot. Never imports sgi; installs nothing; reads and writes only
.npz/.json under phase2/flow_pilot/ (guarded). Model class written from FROZEN.md section 4 (plus DEVIATIONS.md
clarification 5 for the parameter layout the checkpoints were saved with); loaded with strict=True.

Run only as:  HF_HUB_OFFLINE=1 python -I -B v_sidecar.py MODE
  val     64 samples per validation unit from MY rebuilt rasters (verify/data/vseed190-199): main on all units,
          each LOSO model on its left-out situation; two seed sets: 'proto' = crc32(unit key) (FROZEN 5) and
          'alt' = crc32('verify-alt/' + key). Writes verify/samples/val_<model>_<seedset>.npz (raw x only).
  dev     MY rebuilt development arrays (verify/data/vdev_*): main on settings b and a, each LOSO model on its
          left-out situation in setting b; both seed sets; projection in numpy (snap, validity, density, top,
          hit@5 places). Writes verify/samples/dev_<setting>_<model>_<seedset>.npz.
  lat     warm latency re-measure on a subset (every 3rd record, n 183), settings b and a, main model, proto
          seeds: 10 untimed warm-ups, then each record once. Writes verify/samples/latency.json.
GPU: one process; starts only with >= 3000 MiB free (nvidia-smi), else polls every 5 min and logs; kills nothing.
"""
import json
import math
import os
import subprocess
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = 'results/2026-10-17-beyond-vlm/phase2/flow_pilot'
VER = BASE + '/verify'
MODELS = ('main', 'loso_waiting_line', 'loso_conversation_group', 'loso_audience')


def log(*a):
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), *a, flush=True)


def safe(path, write=False):
    real = os.path.realpath(path)
    assert real.startswith(BASE + '/'), real
    assert real.endswith('.npz') or real.endswith('.json'), real
    for bad in ('heldout', 'held_out', 's18', 's20_held', 'eval_16', 'eval_17', 'eval_18', 'eval_19', '.env', '.cache',
                'group_photo', 'lift_wait'):
        assert bad not in real.lower(), real
    if write:
        assert real.startswith(VER + '/'), real
    return real


def smi(q):
    out = subprocess.run(['nvidia-smi', f'--query-gpu={q}', '--format=csv,noheader,nounits'], capture_output=True,
                         text=True, check=True).stdout.strip().split('\n')[0]
    return [v.strip() for v in out.split(',')]


def gpu_gate():
    while True:
        free, used, total = (int(v) for v in smi('memory.free,memory.used,memory.total'))
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                              capture_output=True, text=True).stdout.strip()
        log(f'nvidia-smi: free {free} MiB, used {used}/{total} MiB; apps: {apps or "none"}')
        if free >= 3000:
            return dict(free_mib=free, used_mib=used, total_mib=total, apps=apps)
        log('less than 3000 MiB free; waiting 300 s')
        time.sleep(300)


def env():
    name, drv = smi('name,driver_version')
    return dict(gpu=name, driver=drv, torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                numpy=np.__version__, python=sys.version.split()[0])


# ------------------------------------------------------------------------------------------- my model (FROZEN 4)
def conv_block(cin, cout):
    return [nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, cout), nn.SiLU()]


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(*conv_block(7, 32), *conv_block(32, 64), *conv_block(64, 128), *conv_block(128, 256))
        self.fc = nn.Linear(256 * 4 * 4, 256)

    def forward(self, r):
        h = self.net(r)
        assert h.shape[1:] == (256, 4, 4)
        return self.fc(h.reshape(h.shape[0], -1))


class TEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('freq', torch.exp(-math.log(10000.0) * torch.arange(64).float() / 64.0))
        self.mlp = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 256))

    def forward(self, t):
        ang = (1000.0 * t).unsqueeze(1) * self.freq.unsqueeze(0)
        return self.mlp(torch.cat([torch.sin(ang), torch.cos(ang)], 1))


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(256, 256)
        self.ln = nn.LayerNorm(256)
        self.film = nn.Linear(256, 512)

    def forward(self, h, c):
        gb = self.film(F.silu(c))
        gamma, beta = gb[:, :256], gb[:, 256:]
        return F.silu(self.ln(self.lin(h)) * (1.0 + gamma) + beta)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Enc()
        self.temb = TEmb()
        self.inp = nn.Linear(4, 256)
        self.layers = nn.ModuleList([Layer(), Layer(), Layer()])
        self.out = nn.Linear(256, 4)

    def vel(self, x, t, e):
        c = e + self.temb(t)
        h = self.inp(x)
        for L in self.layers:
            h = L(h, c)
        return self.out(h)


def load_model(name, dev):
    z = np.load(safe(f'{BASE}/models/{name}/best.npz'), allow_pickle=False)
    m = Net()
    sd = {k: torch.from_numpy(z[k].copy()) for k in z.files}
    assert torch.allclose(sd['temb.freq'], m.temb.freq), 'time frequencies differ from FROZEN 4'
    m.load_state_dict(sd, strict=True)
    n = sum(p.numel() for p in m.parameters())
    assert n == 2134148, n
    return m.to(dev).eval()


@torch.no_grad()
def draw(model, raster, seed, dev):
    g = torch.Generator().manual_seed(int(seed))
    x = torch.randn(64, 4, generator=g).to(dev)
    e = model.enc(torch.from_numpy(np.ascontiguousarray(raster)).unsqueeze(0).to(dev)).expand(64, -1)
    for k in range(16):
        x = x + model.vel(x, torch.full((64,), k / 16.0, device=dev), e) / 16.0
    return x.cpu().numpy().astype(np.float64)


def seed_of(key, seedset):
    return zlib.crc32(key.encode()) if seedset == 'proto' else zlib.crc32(('verify-alt/' + key).encode())


# ------------------------------------------------------------------------------------- my projection (FROZEN 5)
def world_of(x, frame):
    ox, oy, yaw = frame
    px, py = 3.2 * x[:, 0], 3.2 * x[:, 1]
    c, s = math.cos(yaw), math.sin(yaw)
    w = np.stack([ox + c * px - s * py, oy + s * px + c * py], 1)
    h = np.arctan2(x[:, 3], x[:, 2]) + yaw
    h = math.pi - np.mod(math.pi - h, 2 * math.pi)
    return w, h


def dens(p):
    d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * 0.15 * 0.15)).sum(1)


def projection(w, h, Fw, ex, place_ex, rr, fb):
    d2 = (w[:, 0:1] - Fw[None, :, 0]) ** 2 + (w[:, 1:2] - Fw[None, :, 1]) ** 2
    snap = np.argmin(d2, 1)
    sp = Fw[snap]
    valid = ex[snap] <= place_ex + 0.001 + 1e-9
    if not valid.any():
        return snap, sp, valid, True, np.array(fb[0], float), np.array(fb[:5], float)
    vi = np.flatnonzero(valid)
    dv = dens(sp[vi])
    order = sorted(range(len(vi)), key=lambda j: (-dv[j], vi[j]))
    seq = [int(vi[j]) for j in order]
    chosen = [seq[0]]
    for i in seq[1:]:
        if len(chosen) == 5:
            break
        if min(math.hypot(sp[i, 0] - sp[j, 0], sp[i, 1] - sp[j, 1]) for j in chosen) >= 2 * rr - 1e-9:
            chosen.append(i)
    places = np.array([[sp[i, 0], sp[i, 1], h[i]] for i in chosen])
    return snap, sp, valid, False, places[0].copy(), places


# ------------------------------------------------------------------------------------------------------ modes
def load_val():
    keys, R, sits = [], [], []
    for k in range(190, 200):
        z = np.load(safe(f'{VER}/data/vseed{k}.npz'))
        info = json.load(open(safe(f'{VER}/data/vseed{k}.json')))
        assert [u['key'] for u in info['units']] == [str(x) for x in z['keys']]
        keys += [str(x) for x in z['keys']]
        R.append(z['rasters'])
        sits += [u['situation'] for u in info['units']]
    return keys, np.concatenate(R), np.array(sits)


def load_dev():
    parts = []
    for name in sorted(os.listdir(VER + '/data')):
        if name.startswith('vdev_') and name.endswith('.npz'):
            parts.append(name[:-4])
    D = {}
    rows = []
    for p in parts:
        zf = np.load(safe(f'{VER}/data/{p}.npz'))
        z = {k: zf[k] for k in zf.files}                    # load each array once (an NpzFile re-reads per access)
        meta = json.load(open(safe(f'{VER}/data/{p}.json')))
        for j, m in enumerate(meta):
            assert m['key'] == str(z['keys'][j])
            rec = dict(meta=m, frame=z['frame'][j], rr=float(z['robot_radius'][j]))
            for s in ('a', 'b'):
                o = z[f'off_{s}']
                rec[f'raster_{s}'] = z[f'raster_{s}'][j].copy()
                rec[f'F_{s}'] = z[f'F_{s}'][o[j]:o[j + 1]].copy()
                rec[f'ex_{s}'] = z[f'ex_{s}'][o[j]:o[j + 1]].copy()
                rec[f'place_ex_{s}'] = float(z[f'place_ex_{s}'][j])
                rec[f'places_{s}'] = z[f'places_{s}'][j]
            rows.append(rec)
    rows.sort(key=lambda r: r['meta']['idx'])
    assert [r['meta']['idx'] for r in rows] == list(range(549)), 'dev arrays incomplete'
    return rows


def mode_val(dev):
    keys, R, sits = load_val()
    for name in MODELS:
        m = load_model(name, dev)
        sel = list(range(len(keys))) if name == 'main' else [i for i in range(len(keys)) if sits[i] == name[5:]]
        for ss in ('proto', 'alt'):
            X = np.stack([draw(m, R[i], seed_of(keys[i], ss), dev) for i in sel])
            np.savez(safe(f'{VER}/samples/val_{name}_{ss}.npz', write=True), keys=np.array([keys[i] for i in sel]), x=X)
            log(f'val {name} {ss}: {len(sel)} units')


def dev_pass(m, rows, setting, sel, ss, dev):
    out = {k: [] for k in ('keys', 'x', 'world', 'head', 'snap', 'sp', 'valid', 'fallback', 'top', 'places', 'n_places',
                           'raw_top')}
    for i in sel:
        r = rows[i]
        key = r['meta']['key']
        x = draw(m, r[f'raster_{setting}'], seed_of(key, ss), dev)
        w, h = world_of(x, r['frame'])
        snap, sp, valid, fb, top, places = projection(w, h, r[f'F_{setting}'], r[f'ex_{setting}'],
                                                      r[f'place_ex_{setting}'], r['rr'], r[f'places_{setting}'])
        dn = dens(w)
        rt = int(sorted(range(64), key=lambda j: (-dn[j], j))[0])
        pl = np.full((5, 3), np.nan)
        pl[:len(places)] = places
        for k, v in (('keys', key), ('x', x), ('world', w), ('head', h), ('snap', snap), ('sp', sp), ('valid', valid),
                     ('fallback', fb), ('top', top), ('places', pl), ('n_places', len(places)),
                     ('raw_top', np.array([w[rt, 0], w[rt, 1], h[rt]]))):
            out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}


def mode_dev(dev):
    rows = load_dev()
    allidx = list(range(len(rows)))
    for name in MODELS:
        m = load_model(name, dev)
        for setting in (('b', 'a') if name == 'main' else ('b',)):
            sel = allidx if name == 'main' else [i for i in allidx if rows[i]['meta']['sit'] == name[5:]]
            for ss in ('proto', 'alt'):
                o = dev_pass(m, rows, setting, sel, ss, dev)
                np.savez(safe(f'{VER}/samples/dev_{setting}_{name}_{ss}.npz', write=True), **o)
                log(f'dev {setting} {name} {ss}: {len(sel)} records, fallbacks {int(o["fallback"].sum())}')


def mode_lat(dev):
    rows = load_dev()
    m = load_model('main', dev)
    sub = list(range(0, len(rows), 3))
    res = dict(env=env(), n=len(sub), warmups=10, subset='every 3rd record (idx % 3 == 0)')
    for setting in ('b', 'a'):
        arr = [(rows[i][f'raster_{setting}'], rows[i][f'F_{setting}'], rows[i][f'ex_{setting}'],
                rows[i][f'place_ex_{setting}'], rows[i]['rr'], rows[i][f'places_{setting}'], rows[i]['frame'],
                rows[i]['meta']['key']) for i in sub]
        for a in arr[:10]:
            x = draw(m, a[0], seed_of(a[7], 'proto'), dev)
            w, h = world_of(x, a[6])
            projection(w, h, a[1], a[2], a[3], a[4], a[5])
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for a in arr:
            t0 = time.perf_counter()
            x = draw(m, a[0], seed_of(a[7], 'proto'), dev)
            w, h = world_of(x, a[6])
            projection(w, h, a[1], a[2], a[3], a[4], a[5])
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts)
        res[setting] = dict(n=len(ts), median_s=float(np.median(ts)), p90_s=float(np.percentile(ts, 90)),
                            mean_s=float(ts.mean()), max_s=float(ts.max()),
                            peak_alloc_mib=torch.cuda.max_memory_allocated() / 2 ** 20)
        log(f'latency {setting}: median {1000 * np.median(ts):.1f} ms p90 {1000 * np.percentile(ts, 90):.1f} ms n {len(ts)}')
    json.dump(res, open(safe(f'{VER}/samples/latency.json', write=True), 'w'), indent=1)


def main(mode):
    log(f'v_sidecar {mode}; HF_HUB_OFFLINE={os.environ.get("HF_HUB_OFFLINE")}; isolated={sys.flags.isolated}; '
        f'dont_write_bytecode={sys.flags.dont_write_bytecode}')
    assert 'sgi' not in sys.modules
    torch.manual_seed(0)
    torch.set_num_threads(2)
    g0 = gpu_gate()
    dev = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    {'val': mode_val, 'dev': mode_dev, 'lat': mode_lat}[mode](dev)
    info = dict(env(), mode=mode, gpu_before=g0, wall_s=time.monotonic() - t0,
                peak_alloc_mib=torch.cuda.max_memory_allocated() / 2 ** 20)
    json.dump(info, open(safe(f'{VER}/samples/env_{mode}.json', write=True), 'w'), indent=1)
    log(json.dumps(info))
    assert 'sgi' not in sys.modules


if __name__ == '__main__':
    main(sys.argv[1])
