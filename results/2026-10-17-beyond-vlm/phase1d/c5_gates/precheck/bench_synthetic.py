"""C5 gates, pre-freeze: timing of the Gaussian-product log-KDE kernel on SYNTHETIC random numbers only (no scene,
no feature, no density of any data), to size the frozen subsampling. numpy only."""
import time

import numpy as np


def log_kde(xe, xt, logw, h, chunk=2048):
    """log sum_i w_i exp(-|xe - xt_i|^2 / (2 h^2)) for every row of xe (constants dropped)."""
    a = xt / h
    a2 = 0.5 * np.einsum('ij,ij->i', a, a)
    out = np.empty(len(xe))
    for lo in range(0, len(xe), chunk):
        b = xe[lo:lo + chunk] / h
        b2 = 0.5 * np.einsum('ij,ij->i', b, b)
        e = b @ a.T
        e -= a2[None, :]
        e += logw[None, :]
        e -= b2[:, None]
        m = e.max(axis=1)
        np.exp(e - m[:, None], out=e)
        out[lo:lo + chunk] = m + np.log(e.sum(axis=1))
    return out


rng = np.random.default_rng(0)
for d, n_e, n_t in ((9, 20000, 60000), (6, 20000, 60000), (9, 20000, 30000)):
    xe, xt = rng.normal(size=(n_e, d)), rng.normal(size=(n_t, d))
    logw = np.full(n_t, -np.log(n_t))
    t0 = time.monotonic()
    log_kde(xe, xt, logw, 0.4)
    dt = time.monotonic() - t0
    print(f'd {d} eval {n_e} train {n_t}: {dt:.2f} s = {dt / (n_e * n_t) * 1e9:.2f} ns per pair')
