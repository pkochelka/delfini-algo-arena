"""Per short window (25 ms): maker-insert envelope (max bid, min ask) and taker offsets vs its centre, by regime."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); hi = np.load(C / 'hmm_taker_gamma2.npy')[:, 1] > 0.5
ev = pd.read_csv(D / 'shock_events.csv')
ins = r[r.op == 0].copy(); w = 0.025
ins['b'] = (ins.t / w).astype(int)
mk = ins[~ins.ioc]
env = pd.DataFrame({'mb': mk[mk.side == 0].groupby('b').px.max(), 'ma': mk[mk.side == 1].groupby('b').px.min()}).dropna()
env['W'] = env.ma - env.mb; env['M'] = (env.ma + env.mb) / 2
sec = (env.index.to_numpy() * w).astype(int).clip(0, 5999); env['hi'] = hi[sec]
sh = np.zeros(len(env), bool); tt = env.index.to_numpy() * w
for o in ev.onset: sh |= (tt > o - 0.5) & (tt < o + 15)
env['shock'] = sh
print('maker envelope width (min ask - max bid within 25 ms) by regime:')
print(pd.crosstab([env.shock, env.hi], env.W.clip(-2, 8), normalize='index').round(3).to_string())
tk = ins[ins.ioc].join(env, on='b', how='inner')
tk['off'] = np.where(tk.side == 0, 1, -1) * (tk.px - tk.M)
print('taker offset sg*(p - envelope centre) by regime:')
print(pd.crosstab([tk.shock, tk.hi_y if 'hi_y' in tk else tk.hi], tk.off.clip(-3, 7), normalize='index').round(3).to_string())
print(tk.groupby([tk.shock, tk.hi]).off.agg(['mean', 'median', 'count']).round(3).to_string())
# offset vs taker size (does the limit include the taker's own impact?)
tk['sq'] = np.sqrt(tk.vol)
print('taker offset by sqrt(size) bin and regime (no shock):')
X = tk[~tk.shock]
print(X.groupby([X.hi, pd.cut(X.sq, [0, 3, 6, 10, 15, 25, 1000])], observed=True).off.agg(['mean', 'median', 'count']).round(3).to_string())
for hh in [False, True]:
    Y = X[X.hi == hh]; Y = Y[Y.off.abs() < 6]
    print('regime', hh, 'slope off ~ sqrt(vol):', np.round(np.polyfit(Y.sq, Y.off, 1), 4))
