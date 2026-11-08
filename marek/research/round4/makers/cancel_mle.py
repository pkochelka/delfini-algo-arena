"""Maker cancel model, direct MLE on round 3 (base makers only: ladder chains excluded) with the reconstructed S.

Model: a cancel *decision* arrives with hazard h(t) = alpha_state + beta_state * max(0, mv(t)) [+ optional terms],
mv = adverse S move since insert (S down for a bid, up for an ask). The cancel row is written a delay
Delta ~ Exp(b) after the decision (optionally Delta = L0 + Exp(b) or Gamma(2, b)), whether or not the order filled.
Every maker order has a cancel row (99.99 %), so the likelihood uses all orders, filled or not, with the full S path
from insert to the cancel row:
    L_i = sum_j  P(decision in step j) * f_delay(T_i - s_j)
State = state at insert (calm / vol / drought / post), from states.py.

Also prints model-free hazard tables (cancel-row intensity among live orders) by mv bin x state, by distance, by age.
"""
from __future__ import annotations

import sys
import json
import numpy as np
import pandas as pd
from scipy import optimize
from s_load import with_s
import states

DT = 0.0025  # s
o = with_s()
T_ALL = o.t_us.to_numpy()
S_ALL = o.S.to_numpy()


def S_fn(t_us):
    k = np.clip(np.searchsorted(T_ALL, t_us, 'right') - 1, 0, len(T_ALL) - 1)
    return S_ALL[k]


def build_paths(n=120_000, seed=0, max_life=5.0):
    g = o[(~o.ioc) & (~o.lad) & o.cancel_us.notna()]
    g = g.sample(min(n, len(g)), random_state=seed)
    life = (g.cancel_us.to_numpy() - g.t_us.to_numpy()) / 1e6
    keep = life < max_life
    g, life = g[keep], life[keep]
    nstep = np.floor(life / DT).astype(int) + 1  # steps covering [0, life]
    rep = np.repeat(np.arange(len(g)), nstep)
    j = np.arange(nstep.sum()) - np.repeat(np.cumsum(nstep) - nstep, nstep)
    s = j * DT  # step start (s since insert)
    t0 = g.t_us.to_numpy()
    sg = g.sg.to_numpy().astype(float)
    s_ins = g.S.to_numpy()
    S_t = S_fn(t0[rep] + (s * 1e6).astype(np.int64))
    mv = -sg[rep] * (S_t - s_ins[rep])
    dist_now = sg[rep] * (S_t - g.price.to_numpy()[rep])  # >0: order on the safe side of S, distance in ticks
    filled_t = np.where(g.end.to_numpy() == 'filled', (g.end_t_us.to_numpy() - t0) / 1e6, np.inf)
    st = states.state(t0)
    ds_ins = None
    rem = life[rep] - s  # time from step start to cancel row (>0)
    width = np.minimum(DT, rem)  # last step partial
    return dict(g=g, life=life, nstep=nstep, rep=rep, j=j, s=s, mv=mv, dist=dist_now, rem=rem, width=width,
                st=st, ds_ins=ds_ins, filled_t=filled_t, starts=np.cumsum(nstep) - nstep)


def nll(theta, P, form, delay):
    states_ = ['calm', 'vol', 'drought', 'post']
    k = 0
    a = {}; bt = {}
    for s_ in states_:
        a[s_] = np.exp(theta[k]); bt[s_] = np.exp(theta[k + 1]); k += 2
    b = np.exp(theta[k]); k += 1
    L0 = 0.0
    if delay == 'shifted':
        L0 = np.exp(theta[k]); k += 1
    gamma_d = 0.0
    if form == 'dist':
        gamma_d = theta[k]; k += 1
    st = P['st'][P['rep']]
    al = np.select([st == s_ for s_ in states_], [a[s_] for s_ in states_])
    be = np.select([st == s_ for s_ in states_], [bt[s_] for s_ in states_])
    h = al + be * np.maximum(P['mv'], 0)
    if form == 'dist':
        h = h * np.exp(gamma_d * np.clip(P['dist'], -3, 5))
    hw = h * P['width']
    Hc = np.cumsum(hw)
    Hstart = Hc - hw
    first = P['starts']
    Hbase = np.repeat(Hstart[first], P['nstep'])
    Hprev = Hstart - Hbase
    pdec = np.exp(-Hprev) * -np.expm1(-hw)
    tau = P['rem'] - P['width'] / 2  # decision at step mid -> delay until the row
    if delay == 'exp':
        fd = b * np.exp(-b * np.maximum(tau, 0))
    elif delay == 'shifted':
        x = tau - L0
        fd = np.where(x > 0, b * np.exp(-b * np.clip(x, 0, None)), 0.0) + 1e-6
    else:  # gamma2
        x = np.maximum(tau, 0)
        fd = b * b * x * np.exp(-b * x) + 1e-12
    Li = np.add.reduceat(pdec * fd, first)
    return -np.log(np.maximum(Li, 1e-300)).sum()


def fit(P, form='lin', delay='exp'):
    x0 = list(np.log([1.8, 3.0] * 4)) + [np.log(30.0)]
    if delay == 'shifted':
        x0 += [np.log(0.003)]
    if form == 'dist':
        x0 += [0.0]
    bnds = [(-6, 7)] * len(x0)
    if form == 'dist':
        bnds[-1] = (-3, 3)
    r = optimize.minimize(nll, x0, args=(P, form, delay), method='L-BFGS-B', bounds=bnds, options={'maxiter': 300})
    th = r.x
    out = {}
    k = 0
    for s_ in ['calm', 'vol', 'drought', 'post']:
        out[f'alpha_{s_}'] = float(np.exp(th[k])); out[f'beta_{s_}'] = float(np.exp(th[k + 1])); k += 2
    out['b'] = float(np.exp(th[k])); k += 1
    if delay == 'shifted':
        out['L0_s'] = float(np.exp(th[k])); k += 1
    if form == 'dist':
        out['gamma_dist'] = float(th[k])
    out['nll'] = float(r.fun); out['n'] = int(len(P['nstep'])); out['form'] = form; out['delay'] = delay
    return out


def hazard_tables(P):
    """Model-free: cancel-row intensity per live order-second, exposure ends at the row (fills included)."""
    canc = P['j'] == (P['nstep'][P['rep']] - 1)
    w = P['width']
    st = P['st'][P['rep']]
    old = P['s'] > 0.05
    df = pd.DataFrame({'c': canc, 'w': w, 'st': st, 'mv': P['mv'], 'dist': P['dist'], 's': P['s'], 'old': old})
    edges = [-50, -0.5, 0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 50]
    d = df[df.old]
    t1 = d.groupby([pd.cut(d.mv, edges), 'st'], observed=True).apply(lambda x: x.c.sum() / x.w.sum()).unstack().round(2)
    n1 = d.groupby([pd.cut(d.mv, edges), 'st'], observed=True).w.sum().unstack().round(0)
    t2 = df.groupby([pd.cut(df.s, [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]), 'st'], observed=True).apply(
        lambda x: x.c.sum() / x.w.sum()).unstack().round(2)
    dc = d[(d.mv.abs() < 0.1)]
    t3 = dc.groupby([pd.cut(dc.dist, [-5, 0, 0.25, 0.5, 1, 1.5, 2, 3, 5, 50]), 'st'], observed=True).apply(
        lambda x: x.c.sum() / x.w.sum()).unstack().round(2)
    return t1, n1, t2, t3


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120_000
    P = build_paths(n)
    t1, n1, t2, t3 = hazard_tables(P)
    pd.set_option('display.width', 200)
    print('cancel-row hazard (/s) by mv (age>50ms) x state\n', t1, '\nexposure (order-s)\n', n1)
    print('hazard by age x state\n', t2)
    print('hazard by current distance from S (|mv|<0.1, age>50ms)\n', t3)
    res = []
    for form, delay in [('lin', 'exp')]:
        r = fit(P, form, delay)
        print(json.dumps(r))
        res.append(r)
    json.dump(res, open('cancel_mle.json', 'w'), indent=1)
