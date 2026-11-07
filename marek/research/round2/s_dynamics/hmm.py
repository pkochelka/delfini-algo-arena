"""Grid HMM filter / smoother for the hidden fair price S on a moving window.

State: S on a grid of step h around a moving centre `cen` (a robust running median of maker
prices, so 100-tick jumps stay inside the +-H*h window). Transition between consecutive inserts
i-1 -> i (dt = t_i - t_{i-1}), applied in this order:
  A  previous taker's impact (deterministic shift imp[i-1], linear interpolation) + recentre
  G  Gaussian diffusion, var = sig2[i] * dt + jv[i-1]  (jv: extra per-taker impact noise variance)
  R  'reset' jump: prob 1-exp(-lamr[i] dt), new S uniform on the window   (rare big jumps)
  L  Laplace jump: prob 1-exp(-lamj[i] dt), size ~ Laplace(scale bj[i])     (small jumps)
Observations (per insert):
  maker bid p : P(floor(S-D) = p) ; clipped (p == book ask - 1): P(floor(S-D) >= p)
  maker ask p : P(ceil(S+D)  = p) ; clipped (p == book bid + 1): P(ceil(S+D)  <= p)
  D = d0 + Exp(theta[i])
  taker buy p : (1-eps) 1[S in (p-off-.5, p-off+.5]] + eps * Exp(tau) density of (p-S) - (off+.5)
All parameter arrays are per insert.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _Fd(z, d0, th):
    if z <= d0:
        return 0.0
    return 1.0 - math.exp(-(z - d0) / th)


@njit(cache=True)
def _lik(x, kind, p, d0, th, off, eps, tau, floor_):
    # kind: 0 bid, 1 ask, 2 bid clipped, 3 ask clipped, 4 taker buy, 5 taker sell, 6 none
    if kind == 0:
        v = _Fd(x - p, d0, th) - _Fd(x - p - 1.0, d0, th)
    elif kind == 1:
        v = _Fd(p - x, d0, th) - _Fd(p - 1.0 - x, d0, th)
    elif kind == 2:
        v = _Fd(x - p, d0, th)
    elif kind == 3:
        v = _Fd(p - x, d0, th)
    elif kind == 4:
        z = p - off - x
        v = (1 - eps) if (z > -0.5 and z <= 0.5) else 0.0
        if z > 0.5:
            v += eps / tau * math.exp(-(z - 0.5) / tau)
    elif kind == 5:
        z = x - (p + off)
        v = (1 - eps) if (z >= -0.5 and z < 0.5) else 0.0
        if z < -0.5:
            v += eps / tau * math.exp((z + 0.5) / tau)
    else:
        v = 1.0
    return v + floor_


@njit(cache=True)
def _shift(src, dst, k0, fr, N):
    for j in range(N):
        dst[j] = 0.0
    for j in range(N):
        v = src[j]
        if v == 0.0:
            continue
        a = j + k0
        if 0 <= a < N:
            dst[a] += v * (1 - fr)
        if 0 <= a + 1 < N:
            dst[a + 1] += v * fr


@njit(cache=True)
def _shiftT(src, dst, k0, fr, N):
    for j in range(N):
        a = j + k0
        acc = 0.0
        if 0 <= a < N:
            acc += (1 - fr) * src[a]
        if 0 <= a + 1 < N:
            acc += fr * src[a + 1]
        dst[j] = acc


@njit(cache=True)
def _gauss_kernel(var, kern, N):
    w = int(math.ceil(5 * math.sqrt(var))) + 1
    if w > N:
        w = N
    ks = 0.0
    for k in range(-w, w + 1):
        kv = math.exp(-0.5 * k * k / var)
        kern[k + w] = kv
        ks += kv
    for k in range(2 * w + 1):
        kern[k] /= ks
    return w


@njit(cache=True)
def _conv(src, dst, kern, w, N):
    # symmetric kernel -> same op for transpose
    for j in range(N):
        lo = max(-w, -j)
        hi = min(w, N - 1 - j)
        acc = 0.0
        for k in range(lo, hi + 1):
            acc += kern[k + w] * src[j + k]
        dst[j] = acc


@njit(cache=True)
def _laplace(src, dst, a, f, b, N):
    # dst = K src, K(j,k) = (1-a)/(1+a) a^|j-k|  (symmetric)
    acc = 0.0
    for j in range(N):
        acc = src[j] + a * acc
        f[j] = acc
    acc = 0.0
    for j in range(N - 1, -1, -1):
        acc = src[j] + a * acc
        b[j] = acc
    c = (1 - a) / (1 + a)
    for j in range(N):
        dst[j] = c * (f[j] + b[j] - src[j])


@njit(cache=True)
def forward(t, kind, price, cen, imp, jv, sig2, lamr, lamj, bj, d0, theta, off, eps, tau, floor_, h, H,
            store, post_out):
    n = len(t)
    N = 2 * H + 1
    p = np.full(N, 1.0 / N)
    tmp = np.zeros(N)
    t2 = np.zeros(N)
    f1 = np.zeros(N)
    f2 = np.zeros(N)
    ll = np.zeros(n)
    kern = np.zeros(2 * N + 1)
    for i in range(n):
        if i > 0:
            tot = imp[i - 1] / h - (cen[i] - cen[i - 1])
            fl = math.floor(tot)
            _shift(p, tmp, int(fl), tot - fl, N)
            s = tmp.sum()
            if s < 1e-300:
                for j in range(N):
                    p[j] = 1.0 / N
            else:
                for j in range(N):
                    p[j] = tmp[j] / s
            dt = t[i] - t[i - 1]
            var = (sig2[i] * dt + jv[i - 1]) / (h * h)
            if var > 1e-4:
                w = _gauss_kernel(var, kern, N)
                _conv(p, tmp, kern, w, N)
                s = tmp.sum()
                if s < 1e-300:
                    s = 1e-300
                for j in range(N):
                    p[j] = tmp[j] / s
            q = 1.0 - math.exp(-lamr[i] * dt)
            if q > 0:
                for j in range(N):
                    p[j] = (1 - q) * p[j] + q / N
            q = 1.0 - math.exp(-lamj[i] * dt)
            if q > 0:
                a = math.exp(-h / bj[i])
                _laplace(p, tmp, a, f1, f2, N)
                for j in range(N):
                    p[j] = (1 - q) * p[j] + q * tmp[j]
        c0 = cen[i] * h
        s = 0.0
        for j in range(N):
            x = c0 + (j - H) * h
            v = p[j] * _lik(x, kind[i], price[i], d0[i], theta[i], off[i], eps[i], tau[i], floor_)
            tmp[j] = v
            s += v
        if s <= 0:
            for j in range(N):
                p[j] = 1.0 / N
            ll[i] = -50.0
        else:
            ll[i] = math.log(s)
            for j in range(N):
                p[j] = tmp[j] / s
        if store:
            for j in range(N):
                post_out[i, j] = p[j]
    return ll


@njit(cache=True)
def backward(t, cen, imp, jv, sig2, lamr, lamj, bj, h, H, filt, sm):
    """smooth_i = filt_i * M_i^T (smooth_{i+1} / M_i filt_i), M_i the forward transition i -> i+1."""
    n = filt.shape[0]
    N = 2 * H + 1
    pA = np.zeros(N)
    pG = np.zeros(N)
    pR = np.zeros(N)
    pred = np.zeros(N)
    r = np.zeros(N)
    tmp = np.zeros(N)
    f1 = np.zeros(N)
    f2 = np.zeros(N)
    kern = np.zeros(2 * N + 1)
    for j in range(N):
        sm[n - 1, j] = filt[n - 1, j]
    for i in range(n - 2, -1, -1):
        tot = imp[i] / h - (cen[i + 1] - cen[i])
        fl = math.floor(tot)
        k0 = int(fl)
        fr = tot - fl
        dt = t[i + 1] - t[i]
        _shift(filt[i], pA, k0, fr, N)
        sA = pA.sum()
        if sA < 1e-300:
            sA = 1e-300
        for j in range(N):
            pA[j] /= sA
        var = (sig2[i + 1] * dt + jv[i]) / (h * h)
        w = 0
        sG = 1.0
        if var > 1e-4:
            w = _gauss_kernel(var, kern, N)
            _conv(pA, pG, kern, w, N)
            sG = pG.sum()
            if sG < 1e-300:
                sG = 1e-300
            for j in range(N):
                pG[j] /= sG
        else:
            for j in range(N):
                pG[j] = pA[j]
        qr = 1.0 - math.exp(-lamr[i + 1] * dt)
        for j in range(N):
            pR[j] = (1 - qr) * pG[j] + qr / N
        qj = 1.0 - math.exp(-lamj[i + 1] * dt)
        a = math.exp(-h / bj[i + 1])
        if qj > 0:
            _laplace(pR, tmp, a, f1, f2, N)
            for j in range(N):
                pred[j] = (1 - qj) * pR[j] + qj * tmp[j]
        else:
            for j in range(N):
                pred[j] = pR[j]
        for j in range(N):
            r[j] = sm[i + 1, j] / pred[j] if pred[j] > 1e-300 else 0.0
        # L^T
        if qj > 0:
            _laplace(r, tmp, a, f1, f2, N)
            for j in range(N):
                r[j] = (1 - qj) * r[j] + qj * tmp[j]
        # R^T
        mr = r.sum() / N
        for j in range(N):
            r[j] = (1 - qr) * r[j] + qr * mr
        # G^T (normalised)
        if w > 0:
            _conv(r, tmp, kern, w, N)
            for j in range(N):
                r[j] = tmp[j] / sG
        # A^T
        _shiftT(r, tmp, k0, fr, N)
        s = 0.0
        for j in range(N):
            v = filt[i, j] * tmp[j] / sA
            sm[i, j] = v
            s += v
        if s > 0:
            for j in range(N):
                sm[i, j] /= s
        else:
            for j in range(N):
                sm[i, j] = filt[i, j]


def summarize(P, cen, h, H, q=(0.005, 0.5, 0.995)):
    N = 2 * H + 1
    x = (np.arange(N) - H) * h
    mean = P @ x + cen * h
    c = np.cumsum(P, axis=1)
    out = [cen * h + (np.minimum((c < qq).sum(axis=1), N - 1) - H) * h for qq in q]
    sd = np.sqrt(np.maximum(P @ (x * x) - (mean - cen * h) ** 2, 0))
    return mean, sd, out
