"""Calm passive fills by taker size and by our fill size; holding time; exact PnL attribution with step positions."""
import sys, pickle, csv, pathlib, numpy as np, pandas as pd
R3 = pathlib.Path(__file__).resolve().parents[2]
fill = pd.read_pickle(sys.argv[1] + '.fills.pkl')
tk = pd.read_pickle(R3 / 'flow/cache/takers.pkl')
tk['t_us'] = (tk.t * 1e6).round().astype(np.int64)
tk = tk[['t_us', 'vol', 'side', 'px']].rename(columns={'vol': 'tvol', 'side': 'tside', 'px': 'tpx'})
f = fill.merge(tk.drop_duplicates('t_us'), left_on='t', right_on='t_us', how='left')
c = f[(f.g == 0) & (~f.ioc.astype(bool))].copy()
print('matched taker', c.tvol.notna().mean())
# total filled by that taker in our run against us = sum over fills at same t
c['ourq_evt'] = c.groupby('t').q.transform('sum')
c['tb'] = pd.cut(c.tvol, [0, 30, 60, 120, 250, 500, 1000, 2000, 1e9])
print('\n=== calm passive by taker order size ===')
for k, d in c.groupby('tb', observed=True):
    w = d.q
    print(f"{str(k):18s} n={len(d):5d} lots={w.sum():7d} share={w.sum()/c.q.sum():.3f} " + ' '.join(f"{x}={np.average(d[x], weights=w):+.3f}" for x in ['e0', 'm1', 'm5', 'm30', 'm60']) + f" m1*lots={np.sum(d.m1*w):8.0f}")
c['qb'] = pd.cut(c.ourq_evt, [0, 20, 50, 100, 150, 200, 1e9])
print('\n=== calm passive by our filled qty in that taker event ===')
for k, d in c.groupby('qb', observed=True):
    w = d.q
    print(f"{str(k):14s} n={len(d):5d} lots={w.sum():7d} " + ' '.join(f"{x}={np.average(d[x], weights=w):+.3f}" for x in ['e0', 'm1', 'm5', 'm30', 'm60']) + f" m1*lots={np.sum(d.m1*w):8.0f}")
# pos before
c['pb'] = c.pos_after - c.side * c.q
c['pbb'] = pd.cut(c.side * c.pb, [-201, -99, -50, -1, 0, 50, 99, 201])  # side*pos: negative = reducing
print('\n=== calm passive by side*pos_before (neg = fill reduces |pos|) ===')
for k, d in c.groupby('pbb', observed=True):
    w = d.q
    print(f"{str(k):14s} lots={w.sum():7d} " + ' '.join(f"{x}={np.average(d[x], weights=w):+.3f}" for x in ['e0', 'm1', 'm5', 'm30', 'm60']) + f" m1*lots={np.sum(d.m1*w):8.0f}")
