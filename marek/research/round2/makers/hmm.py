"""Grid HMM (forward-backward) for the hidden fair price S in round 2, from maker inserts.

State: S on an absolute grid of step H ticks, tracked in a moving window of N points.
Dynamics between consecutive maker inserts: S += c*sign*sqrt(filled) for each taker in between (optional),
plus Gaussian diffusion with sd sigma*sqrt(dt) (optionally a mixture with a wider component).
Emission of a maker insert (round-1 rule): bid = min(floor(S - D), ask - 1), ask = max(ceil(S + D), bid + 1),
D = d0 + Exp(scale).
    bid b < clip:  P = F(S-b) - F(S-b-1);  b == clip: P = F(S-clip)      (F = cdf of D)
    ask a > clip:  P = F(a-S) - F(a-S-1);  a == clip: P = F(clip-S)
Takers are not used as observations (their price rule is checked afterwards).

run(params, t0, t1) returns per-insert smoothed mean / sd / 5-95% quantiles of S and the total log-likelihood.
"""
from __future__ import annotations

import numpy as np
from prep import load

H = 0.02
N = 600  # window = 12 ticks


def inputs(t0_s=0.0, t1_s=1e9):
    o = load()
    o = o[(o.t_us >= t0_s * 1e6) & (o.t_us < t1_s * 1e6)]
    g = o[~o.ioc]
    f = o[o.ioc]
    clip = np.where(g.sg == 1, g.ask - 1, g.bid + 1)
    return g, f, clip


def _kernel_cache(sigma, lam, jsd):
    """Transition over dt: (1 - lam*dt) * N(0, sigma^2 dt) + lam*dt * N(0, sigma^2 dt + jsd^2)  (grid units)."""
    cache = {}

    def k(dt_s):
        key = int(dt_s * 1e5)  # 10 us buckets
        if key in cache:
            return cache[key]
        d = max(key, 1) / 1e5
        sd = max(sigma * np.sqrt(d) / H, 0.3)
        sd2 = np.sqrt(sd ** 2 + (jsd / H) ** 2) if lam > 0 else sd
        half = int(min(np.ceil(4 * sd2), N // 2 - 1))
        x = np.arange(-half, half + 1)
        w = np.exp(-0.5 * (x / sd) ** 2); w /= w.sum()
        if lam > 0:
            p = min(lam * d, 0.5)
            w2 = np.exp(-0.5 * (x / sd2) ** 2)
            w = (1 - p) * w + p * w2 / w2.sum()
        cache[key] = w
        return w
    return k


def run(sigma=3.0, d0=0.05, scale=0.45, c=0.0, lam=0.0, jsd=2.0, t0_s=0.0, t1_s=1e9,
        smooth=True, store_post=False, taker_off=1.0, taker_eps=0.02, use_takers=True):
    o = load()
    o = o[(o.t_us >= t0_s * 1e6) & (o.t_us < t1_s * 1e6)]
    t = o.t_us.to_numpy()
    sg = o.sg.to_numpy()
    px = o.price.to_numpy().astype(np.int64)
    ioc = o.ioc.to_numpy()
    cl = np.where(sg == 1, o.ask - 1, o.bid + 1).astype(float)
    filled = o.filled.to_numpy()
    K = len(o)
    # impact jump applied AFTER taker k (grid points); accumulate fractional remainder
    jt = np.where(ioc, sg * np.sqrt(filled) * c, 0.0)
    cum = np.cumsum(jt)
    cg = np.rint(cum / H).astype(np.int64)
    jafter = np.diff(np.r_[0, cg])  # jump after order k
    jbefore = np.r_[0, jafter[:-1]]  # jump between k-1 and k
    dt = np.diff(np.r_[t[0], t]) / 1e6
    kern = _kernel_cache(sigma, lam, jsd)
    inv = 1.0 / scale
    xs = np.arange(N)

    def F(x):
        return np.where(x > d0, -np.expm1(-(x - d0) * inv), 0.0)

    def emis(k, base):
        s = (base + xs) * H
        if ioc[k]:
            if not use_takers:
                return np.ones(N)
            centre = px[k] - sg[k] * taker_off
            return np.where(np.abs(s - centre) <= 0.5, 1.0, taker_eps)
        if sg[k] == 1:
            b = px[k]
            if not np.isnan(cl[k]) and b >= cl[k]:
                return F(s - cl[k])
            return F(s - b) - F(s - b - 1)
        a = px[k]
        if not np.isnan(cl[k]) and a <= cl[k]:
            return F(cl[k] - s)
        return F(a - s) - F(a - s - 1)

    alpha = np.empty((K, N), dtype=np.float32)
    base = np.empty(K, dtype=np.int64)
    pre = np.empty(K, dtype=np.int64)
    a = np.ones(N) / N
    cur_base = int(round(px[0] / H)) - N // 2
    ll = 0.0
    nreset = 0
    for k in range(K):
        if k > 0 and dt[k] > 0:
            a = np.convolve(a, kern(dt[k]), 'same')
        pb = cur_base + jbefore[k]
        pre[k] = pb
        m = (a * xs).sum() / a.sum()
        shift = int(round(m - N / 2))
        if abs(shift) > N // 8:
            nb = pb + shift
            a = _shift(a, pb, nb)
        else:
            nb = pb
        e = emis(k, nb)
        a = a * e
        z = a.sum()
        if z <= 1e-200 or not np.isfinite(z):
            a = e + 1e-12
            z = a.sum()
            ll += -50.0
            nreset += 1
        else:
            ll += np.log(z)
        a = a / z
        alpha[k] = a
        base[k] = nb
        cur_base = nb
    out = {'t_us': t, 'll': ll, 'K': K, 'nreset': nreset, 'ioc': ioc, 'sg': sg, 'px': px}
    s_all = (base[:, None] + xs[None, :]) * H
    out['mean_f'] = (s_all * alpha).sum(1)
    del s_all
    if not smooth:
        return out
    beta = np.ones(N)
    mean = np.empty(K); sd = np.empty(K); q05 = np.empty(K); q95 = np.empty(K)
    post = np.empty((K, N), dtype=np.float32) if store_post else None
    for k in range(K - 1, -1, -1):
        p = alpha[k] * beta
        z = p.sum()
        if z <= 0 or not np.isfinite(z):
            p = alpha[k].astype(float)
            z = p.sum()
        p = p / z
        s = (base[k] + xs) * H
        mu = (p * s).sum()
        mean[k] = mu
        sd[k] = np.sqrt(max((p * (s - mu) ** 2).sum(), 0))
        cdf = np.cumsum(p)
        q05[k] = s[min(np.searchsorted(cdf, 0.05), N - 1)]
        q95[k] = s[min(np.searchsorted(cdf, 0.95), N - 1)]
        if store_post:
            post[k] = p
        if k == 0:
            break
        m = emis(k, base[k]) * beta
        m = _shift(m, base[k], pre[k])
        if dt[k] > 0:
            m = np.convolve(m, kern(dt[k]), 'same')
        mx = m.max()
        beta = m / mx if mx > 0 else np.ones(N)
    out.update(mean=mean, sd=sd, q05=q05, q95=q95, base=base, post=post)
    return out


def _shift(a, from_base, to_base):
    """Re-index array a (defined on from_base + i) onto to_base + i, zero fill."""
    d = to_base - from_base
    if d == 0:
        return a
    outa = np.zeros_like(a)
    if d > 0:
        if d < len(a):
            outa[:-d] = a[d:]
    else:
        d = -d
        if d < len(a):
            outa[d:] = a[:-d]
    return outa


if __name__ == '__main__':
    import time
    import sys
    t0 = time.time()
    r = run(t0_s=0, t1_s=100, smooth=True)
    print('K', r['K'], 'll', r['ll'], 'per obs', r['ll'] / r['K'], time.time() - t0)
    print('sd median', np.median(r['sd']), 'q05-q95 width median', np.median(r['q95'] - r['q05']))
