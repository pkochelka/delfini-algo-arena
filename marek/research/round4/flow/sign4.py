"""Non-TWAP, ex-shock takers, by s_dynamics regime: sign-pair corr vs time separation; P(same as previous) vs gap and
vs lag-index at fixed gap (latent-in-time vs copy-previous)."""
import numpy as np, pandas as pd
from pathlib import Path
C = Path(__file__).parent / 'cache'
tk = pd.read_parquet(C / 'takers.parquet')
keep = (~tk.tw) & ((tk.shock_dt < -1) | (tk.shock_dt > 8))
tk = tk[keep].reset_index(drop=True)
t = tk.t.to_numpy(); s = tk.sg.to_numpy().astype(float); reg = tk.reg.to_numpy()
edges = np.array([0, .05, .1, .25, .5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
for R in [0, 1]:
    m = reg == R; num = np.zeros(len(edges) - 1); den = np.zeros(len(edges) - 1)
    for i in np.flatnonzero(m):
        j = np.searchsorted(t, t[i] + edges[-1]); jj = np.arange(i + 1, j); jj = jj[reg[jj] == R]
        d = t[jj] - t[i]; b = np.searchsorted(edges, d, 'right') - 1
        num += np.bincount(b, s[i] * s[jj], len(edges) - 1); den += np.bincount(b, None, len(edges) - 1)
    print(f'regime {R}: n={m.sum()}  pair corr vs dt')
    print('  ' + '  '.join(f'{a:g}-{b:g}:{c:+.3f}' for a, b, c in zip(edges[:-1], edges[1:], num / den)))
    # consecutive: P(same) vs gap
    i = np.flatnonzero(m[1:] & m[:-1]) + 1; g = t[i] - t[i - 1]; same = s[i] == s[i - 1]
    qs = np.quantile(g, np.linspace(0, 1, 9))
    print('  P(same|gap):', [(round(qs[k], 3), round(same[(g >= qs[k]) & (g <= qs[k + 1])].mean(), 3)) for k in range(8)])
    # lag-2 at fixed total separation: is corr(s_i, s_{i+2}) at separation d equal to pair corr at d? (latent-in-time => yes)
    for L in [1, 2, 4]:
        i = np.arange(len(t) - L); ok = m[i] & m[i + L]; d = t[i + L] - t[i]
        for a, b in [(0, .25), (.25, .75), (.75, 2)]:
            mm = ok & (d >= a) & (d < b); print(f'   lag {L} sep {a}-{b}: corr {np.mean(s[i][mm] * s[i + L][mm]):+.3f} n={mm.sum()}')
