"""Is there predictable structure in S at 2-30 s beyond known state (KF V, KF Y, S-mid)?
Regress S(t+h)-S(t) on [V, Y, S-mid] (base) and + past S moves over 2/5/10/20/30 s (ext). Half-split OOS R^2."""
import pathlib, numpy as np
H = pathlib.Path(__file__).resolve().parents[1]
z = np.load(H / 'cache/grid.npz'); dt = 0.05
g, M, S, KF, vol, shock = z['g'], z['M'], z['S'], z['KF'], z['vol'], z['shock']
n = len(g); cs = np.concatenate([[0], np.cumsum(shock)])
idx = np.arange(int(30 / dt), n - int(30 / dt), 10)  # every 0.5 s
def past(h): m = int(h / dt); return S[idx] - S[idx - m]
def fut(h): m = int(h / dt); return S[idx + m] - S[idx]
clean = (cs[idx + int(30 / dt) + 1] - cs[idx - int(30 / dt)]) == 0
V, Y, G = KF[idx, 2], KF[idx, 1], S[idx] - M[idx]
P = {h: past(h) for h in (2, 5, 10, 20, 30)}
half = idx < n // 2
def fit(X, y, m):
    X = np.column_stack([np.ones(len(y)), X])
    out = []
    for tr in (half, ~half):
        a, b = m & tr, m & ~tr
        beta = np.linalg.lstsq(X[a], y[a], rcond=None)[0]
        r = y[b] - X[b] @ beta; out.append(1 - r.var() / y[b].var())
    beta = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
    return np.mean(out), beta[1:]
for rg, rm in (('calm', ~vol[idx]), ('vol', vol[idx])):
    m = clean & rm
    print(f'\n== {rg} (n={m.sum()})')
    for h in (2, 5, 10, 20, 30):
        y = fut(h)
        r0, b0 = fit(np.column_stack([V, Y, G]), y, m)
        Xe = np.column_stack([V, Y, G] + [P[k] for k in (2, 5, 10, 20, 30)])
        r1, b1 = fit(Xe, y, m)
        rp, bp = fit(np.column_stack([P[k] for k in (2, 5, 10, 20, 30)]), y, m)
        print(f'h={h:>2}s base R2 {r0:6.3f} [V {b0[0]:5.2f} Y {b0[1]:5.2f} S-mid {b0[2]:5.2f}]  +past R2 {r1:6.3f}  '
              f'past b ' + ' '.join(f'{x:5.2f}' for x in b1[3:]) + f' | past-only R2 {rp:6.3f} b ' + ' '.join(f'{x:5.2f}' for x in bp))
