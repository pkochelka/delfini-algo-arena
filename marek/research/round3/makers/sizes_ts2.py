"""Relate A-band excess (110-250 lots) and B-band excess (25-55) to spread per 10 s window."""
import numpy as np, pandas as pd
from prep import load
o = load(); m = o[~o.ioc.astype(bool)].copy()
W = 10
m['w'] = (m.t_us // (W*1_000_000)).astype(int)
v = m.volume
m['A'] = v.between(110, 250); m['B'] = v.between(25, 55); m['small'] = v <= 10
g = m.groupby('w').agg(n=('volume','size'), A=('A','sum'), B=('B','sum'), small=('small','sum'))
# base ratios from round 2 lognormal: A/small and B/small
o2 = pd.read_pickle('../../round2/makers/cache/orders.pkl'); R = o2[~o2.ioc.astype(bool)].volume
rA = R.between(110,250).mean()/ (R<=10).mean(); rB = R.between(25,55).mean()/(R<=10).mean()
print('r2 ratios A/small %.3f B/small %.3f' % (rA, rB))
g['xA'] = (g.A - rA*g.small)/W; g['xB'] = (g.B - rB*g.small)/W
q = pd.read_pickle('cache/quotes.pkl')
q = q[(q.bid>0)&(q.ask>0)]
dt = np.diff(np.append(q.t_us.to_numpy(), 6000_000_000))
sp = (q.ask-q.bid).to_numpy()
w = (q.t_us//(W*1_000_000)).astype(int).to_numpy()
g['spread'] = pd.Series(np.bincount(w, sp*dt, minlength=len(g))[:len(g)] / np.bincount(w, dt, minlength=len(g))[:len(g)])
tr = pd.read_pickle('cache/trades.pkl')
print(tr.columns.tolist())
g['xA_on'] = g.xA > 3; g['xB_on'] = g.xB > 8
print(g[['xA','xB','spread']].corr())
print(g.groupby(['xB_on', pd.cut(g.spread,[0,1.05,1.1,1.2,1.4,2,5])]).xA.describe())
g.to_csv('cache/sizes_ts10b.csv')
import sys
np.set_printoptions(linewidth=250)
print('xA series (x10 /s):'); print(np.round(g.xA.to_numpy()).astype(int))
print('xB series:'); print(np.round(g.xB.to_numpy()).astype(int))
print('spread:'); print(np.round(g.spread.to_numpy()*10).astype(int))
