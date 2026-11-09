"""Is age-binned signed taker flow informative about future S beyond what r4_best's filter knows (mean, kv, Y, F - mid)?
Target: S(t+h) - F(t) (exact S from s_path). Base regressors: kv, Y, F - mid. Added: signed kick-weighted flow
sum_{age bin} sg * filled^0.65 per age bin. Fit on one half, test on the other (both ways). Shock windows excluded.
uv run --with pyarrow python marek/research/round4/impact_peak/tools/resid.py [cache.npz]"""
import sys, pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parents[1]
z = np.load(sys.argv[1] if len(sys.argv) > 1 else HERE / 'cache/r4_best_round4.npz')['x']
t, F, sd, pv, kv, Y, turb, pos, bb, ba, shk = z.T
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet'); st = sp.t_us.values * 1e-6; ss = sp.s_hat.values
tk = pd.read_parquet(R4 / 'flow/cache/takers_s.parquet'); tk = tk[tk.filled > 0]
tt = tk.t.values; kw = tk.sg.values * tk.filled.values ** 0.65; twm = tk.tw.values.astype(bool)
ck = np.concatenate([[0], np.cumsum(kw)])
def flow(t, a0, a1):  # sum over takers with t - a1 < tt <= t - a0
    return ck[np.searchsorted(tt, t - a0, 'right')] - ck[np.searchsorted(tt, t - a1, 'right')]
def S_at(q):
    return ss[np.clip(np.searchsorted(st, q, 'right') - 1, 0, len(ss) - 1)]
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
sub = np.arange(0, len(t), 10)  # 100 ms grid
t, F, pv, kv, Y, turb, bb, ba, shk = (x[sub] for x in (t, F, pv, kv, Y, turb, bb, ba, shk))
ok = (t > 30) & (t < 6000 - 30) & (shk == 0)
for o in on:
    ok &= ~((t > o - 31) & (t < o + 50))
mid = 0.5 * (bb + ba)
BINS = [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 30)]
FL = np.stack([flow(t, a, b) for a, b in BINS], 1)
base = np.stack([np.ones_like(t), kv, Y, F - mid], 1)
half = t < 3000
def fit(X, y, m):
    b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None); return b
def oos(X, y, m):
    r = []
    for a in (half, ~half):
        b = fit(X, y, m & a); e = y[m & ~a] - X[m & ~a] @ b; r.append(1 - e.var() / y[m & ~a].var())
    return np.mean(r)
for reg, rm in (('volatile', turb > 0.5), ('calm', turb < 0.5)):
    m = ok & rm
    print(f'\n### {reg} n={m.sum()}  (coef of flow age bins, kick units; t-stat w/ overlap correction sqrt(h/0.1))')
    print('h   R2base  R2+flow ' + ' '.join(f'{a}-{b}s'.rjust(13) for a, b in BINS))
    for h in (0.5, 1, 2, 5, 10, 20):
        y = S_at(t + h) - F
        X = np.hstack([base, FL])
        b = fit(X, y, m); e = y[m] - X[m] @ b
        cov = np.linalg.inv(X[m].T @ X[m]) * e.var() * max(1, h / 0.1)
        ts = b / np.sqrt(np.diag(cov))
        cells = ' '.join(f'{b[4+i]:+.3f}({ts[4+i]:+.1f})'.rjust(13) for i in range(len(BINS)))
        print(f'{h:<4} {oos(base, y, m):6.3f} {oos(X, y, m):6.3f} {cells}')
    # mean target given base: what does base say
    print('base coefs (h=5):', np.round(fit(base, S_at(t + 5) - F, m), 3))
