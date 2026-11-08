"""Agent wake rate: after the same-side touch moves 1 tick TOWARD the ladder (bid down / ask up) in calm, new level
touch-7 is empty for every agent. Count maker inserts at exactly that price (same side) in the following window, by delay,
stopping the window at the next same-side touch change."""
import numpy as np, pandas as pd
o = pd.read_pickle('cache/chains.pkl')
g = np.load('cache/regime_sz.npz')['post'][:, 1]
m = o[~o.ioc.astype(bool)]
q = pd.read_pickle('cache/quotes.pkl')
res = {}
for side, col in ((1, 'bid'), (-1, 'ask')):
    x = q[['t_us', col]].dropna()
    x = x[x[col].diff() != 0]
    tt = x.t_us.to_numpy(); px = x[col].to_numpy()
    mv = np.r_[0, np.diff(px)] * side          # -1 = moved toward ladder (bid down)
    nxt = np.r_[tt[1:], 6e9]
    ms = m[m.sg == side]; mt = ms.t_us.to_numpy(); mp = ms.price.to_numpy(); mL = ms.L.to_numpy(); mv_ = ms.volume.to_numpy()
    for kind, sel in (('toward1', mv == -1), ('away1', mv == 1)):
        for reg in ('calm', 'vol'):
            idx = np.flatnonzero(sel & ((g[(tt // 1_000_000).astype(int)] > .5) == (reg == 'vol')))
            delays = []; expo = []; sizes = []
            for i in idx:
                lvl = px[i] - side * (7 if kind == 'toward1' else 2)
                t0, t1 = tt[i], min(nxt[i], tt[i] + 3e6)
                j0, j1 = np.searchsorted(mt, t0), np.searchsorted(mt, t1)
                hit = np.flatnonzero(mp[j0:j1] == lvl)
                delays += list((mt[j0:j1][hit] - t0) / 1e3); sizes += list(mv_[j0:j1][hit]); expo.append((t1 - t0) / 1e3)
            delays = np.array(delays); expo = np.array(expo)
            bins = np.array([0, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 3000])
            E = np.array([np.clip(expo - a, 0, b - a).sum() / 1e3 for a, b in zip(bins[:-1], bins[1:])])
            C = np.histogram(delays, bins)[0]
            print(f'side {side} {kind} {reg}: events {len(idx)}, inserts at new level {len(delays)}; median size {np.median(sizes) if len(sizes) else 0}')
            print('   rate (/s per event) by delay ms:', ' '.join(f'{a}-:{c/e:.2f}' for a, c, e in zip(bins[:-1], C, E)))
