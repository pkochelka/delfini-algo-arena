"""Thin wrapper around hmm.py: build per-event parameter arrays, run filter / smoother."""
import numpy as np
import prep, hmm

import os
d = prep.build(os.environ.get('TAPE', 'data/round2.csv'))
h = prep.H_STEP
H = 240
DEFAULT = dict(sig2=0.4 ** 2, lamr=0.003, lamj=0.0, bj=1.5, d0=0.05, theta=0.5, off=1.0, eps=0.05, tau=2.0,
               c=0.026, e=0.5, jv0=0.0, jvq=0.0, jvn=0.0, jvm=0.0)
ARR = ['sig2', 'lamr', 'lamj', 'bj', 'd0', 'theta', 'off', 'eps', 'tau']


def _arrs(sl, kw):
    N = sl.stop - sl.start
    out = {}
    for k in ARR:
        v = kw[k]
        v = np.asarray(v, float)
        out[k] = np.full(N, float(v)) if v.ndim == 0 else (v[sl] if len(v) == len(d['t']) else v).astype(float)
    return out


def _s(v, sl):
    v = np.asarray(v, float)
    return v[sl] if v.ndim else v


def impact(sl, c, e):
    c = _s(c, sl); e = _s(e, sl)
    return np.where(d['ioc'][sl], c * d['sign'][sl] * d['filled'][sl] ** e, 0.0)


def jvar(sl, jv0, jvq, jvn=0.0, jvm=0.0):
    # extra variance added after event: filled taker jv0 + jvq*filled; unfilled taker jvn; maker insert jvm
    f = d['filled'][sl]; ioc = d['ioc'][sl]
    jv0 = _s(jv0, sl); jvq = _s(jvq, sl)
    return np.where(ioc & (f > 0), jv0 + jvq * f, np.where(ioc, jvn, jvm))


def loglik(sl, floor_=1e-6, **kw):
    kw = {**DEFAULT, **kw}
    a = _arrs(sl, kw)
    imp = impact(sl, kw['c'], kw['e'])
    jv = jvar(sl, kw['jv0'], kw['jvq'], kw['jvn'], kw['jvm'])
    return hmm.forward(d['t'][sl], d['kind'][sl], d['price'][sl], d['cen'][sl], imp, jv, a['sig2'], a['lamr'], a['lamj'],
                       a['bj'], a['d0'], a['theta'], a['off'], a['eps'], a['tau'], floor_, h, H, False,
                       np.zeros((1, 1)))


def smooth(sl, floor_=1e-6, **kw):
    kw = {**DEFAULT, **kw}
    a = _arrs(sl, kw)
    imp = impact(sl, kw['c'], kw['e'])
    jv = jvar(sl, kw['jv0'], kw['jvq'], kw['jvn'], kw['jvm'])
    t = d['t'][sl]
    N = len(t)
    post = np.zeros((N, 2 * H + 1))
    ll = hmm.forward(t, d['kind'][sl], d['price'][sl], d['cen'][sl], imp, jv, a['sig2'], a['lamr'], a['lamj'], a['bj'],
                     a['d0'], a['theta'], a['off'], a['eps'], a['tau'], floor_, h, H, True, post)
    fmean, fsd, fq = hmm.summarize(post, d['cen'][sl], h, H)
    sm = np.zeros_like(post)
    hmm.backward(t, d['cen'][sl], imp, jv, a['sig2'], a['lamr'], a['lamj'], a['bj'], h, H, post, sm)
    del post
    mean, sd, qs = hmm.summarize(sm, d['cen'][sl], h, H)
    return dict(ll=ll, mean=mean, sd=sd, lo=qs[0], med=qs[1], hi=qs[2], fmean=fmean, fsd=fsd, flo=fq[0], fhi=fq[2],
                imp=imp)


def seg(a, b):
    i = np.where((d['t'] >= a) & (d['t'] < b))[0]
    return slice(int(i[0]), int(i[-1]) + 1)
