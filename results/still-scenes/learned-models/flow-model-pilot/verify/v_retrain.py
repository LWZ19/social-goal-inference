"""Verifier's provenance check (torch sidecar): retrain the MAIN model with my own loop, written from FROZEN.md
section 4 and DEVIATIONS.md clarification 6 (rng order), on data/train.npz (seeds 110-189 only; checked in the
project venv) with data/val.npz (seeds 190-199) used only for the validation loss and the checkpoint choice, and
compare the result with the saved models/main/best.npz. Reads only .npz under phase2/flow_pilot/, writes only
verify/samples/retrain_main*.{npz,json}. Never imports sgi, installs nothing.

    HF_HUB_OFFLINE=1 python -I -B v_retrain.py
"""
import json
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, 'results/still-scenes/learned-models/flow-model-pilot/verify')
import v_sidecar as S  # noqa: E402  (my own sidecar module: model class, guards, GPU gate)


def main():
    S.log(f'v_retrain main; isolated={sys.flags.isolated}')
    assert 'sgi' not in sys.modules
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)
    g0 = S.gpu_gate()
    dev = torch.device('cuda')
    torch.manual_seed(0)
    tr = np.load(S.safe(f'{S.BASE}/data/train.npz'))
    va = np.load(S.safe(f'{S.BASE}/data/val.npz'))
    keys_tr = [str(k) for k in tr['keys']]
    keys_va = [str(k) for k in va['keys']]
    assert all(110 <= int(k.split('/')[0][4:]) <= 189 for k in keys_tr)
    assert all(190 <= int(k.split('/')[0][4:]) <= 199 for k in keys_va)

    def label_poses(z):
        units = np.flatnonzero(z['has_label'])
        lab = z['labels'][units].reshape(-1, 4)
        owner = np.repeat(np.arange(len(units)), z['labels'].shape[1])
        return units, lab, owner
    tu, tx, to = label_poses(tr)
    vu, vx, vo = label_poses(va)
    R = torch.from_numpy(tr['rasters'][tu]).to(dev)
    X = torch.from_numpy(tx).to(dev)
    U = torch.from_numpy(to).to(dev)
    RV = torch.from_numpy(va['rasters'][vu]).to(dev)
    XV = torch.from_numpy(vx).to(dev)
    UV = torch.from_numpy(vo).to(dev)
    gv = torch.Generator().manual_seed(1)
    ZV = torch.randn(len(vx), 4, generator=gv).to(dev)
    TV = torch.rand(len(vx), generator=gv).to(dev)
    S.log(f'train units {len(tu)}, poses {len(tx)}; val units {len(vu)}, poses {len(vx)}')
    net = S.Net().to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    n = len(tx)
    spe = math.ceil(n / 256)
    total = 30 * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    def fwd(x, t, r):
        return net.vel(x, t, net.enc(r))

    @torch.no_grad()
    def vloss():
        net.eval()
        tot = 0.0
        for lo in range(0, len(XV), 4096):
            x, z, t = XV[lo:lo + 4096], ZV[lo:lo + 4096], TV[lo:lo + 4096]
            xt = (1 - t[:, None]) * z + t[:, None] * x
            v = fwd(xt, t, RV[UV[lo:lo + 4096]])
            tot += float(((v - (x - z)) ** 2).mean(dim=1).sum())
        net.train()
        return tot / len(XV)
    torch.cuda.reset_peak_memory_stats()
    best, best_ep, best_state, curve = math.inf, None, None, []
    t0 = time.monotonic()
    for ep in range(1, 31):
        perm = torch.randperm(n).to(dev)
        run = 0.0
        for b in range(spe):
            if time.monotonic() - t0 > 40 * 60:
                raise SystemExit('cap reached')
            idx = perm[b * 256:(b + 1) * 256]
            x = X[idx]
            r = R[U[idx]]
            z = torch.randn_like(x)
            t = torch.rand(len(x), device=dev)
            xt = (1 - t[:, None]) * z + t[:, None] * x
            loss = ((fwd(xt, t, r) - (x - z)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * len(x)
        vl = vloss()
        curve.append(dict(epoch=ep, train=run / n, val=vl, t=time.monotonic() - t0))
        S.log(f'epoch {ep:2d} train {run / n:.5f} val {vl:.5f}')
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
    theirs = np.load(S.safe(f'{S.BASE}/models/main/best.npz'))
    diff = {k: float(np.abs(best_state[k].astype(np.float64) - theirs[k]).max()) for k in theirs.files}
    their_curve = json.load(open(S.safe(f'{S.BASE}/models/main/curve.json')))
    res = dict(env=S.env(), gpu_before=g0, best_epoch=best_ep, best_val=best, curve=curve,
               their_best_epoch=int(np.argmin(their_curve['val_loss'])) + 1, their_val_loss=their_curve['val_loss'],
               their_train_loss=their_curve['train_loss'],
               max_abs_weight_diff=max(diff.values()), bit_identical=all(v == 0 for v in diff.values()),
               max_val_loss_diff=max(abs(a['val'] - b) for a, b in zip(curve, their_curve['val_loss'])),
               peak_alloc_mib=torch.cuda.max_memory_allocated() / 2 ** 20, wall_s=time.monotonic() - t0)
    np.savez(S.safe(f'{S.VER}/samples/retrain_main_best.npz', write=True), **best_state)
    json.dump(res, open(S.safe(f'{S.VER}/samples/retrain_main.json', write=True), 'w'), indent=1)
    S.log(f'best epoch {best_ep} val {best:.5f}; max |w - theirs| {res["max_abs_weight_diff"]:.3e}; '
          f'bit identical {res["bit_identical"]}; max val-loss diff {res["max_val_loss_diff"]:.2e}')
    assert 'sgi' not in sys.modules


if __name__ == '__main__':
    main()
