"""Volatility signature sigma(h) = sqrt(E[dX^2]/h) by regime, overlapping increments inside pure regime runs.
Series: mid, S (s_hat), KF F, KF Y, S - int(V_smooth) (kick part), int(V_smooth) (drift part)."""
import pathlib, numpy as np
H = pathlib.Path(__file__).resolve().parents[1]
z = np.load(H / 'cache/grid.npz'); dt = 0.05
g, M, S, KF, KS, vol, shock = z['g'], z['M'], z['S'], z['KF'], z['KS'], z['vol'], z['shock']
lab = np.where(shock, 2, vol.astype(int))
n = len(g)
# run end index (inclusive) for each index
runend = np.empty(n, int); e = n - 1
for k in range(n - 1, -1, -1):
    if k < n - 1 and lab[k] != lab[k + 1]: e = k
    runend[k] = e
Dv = np.cumsum(KS[:, 2] * dt)  # integrated smoothed velocity
ser = {'mid': M, 'S': S, 'F_kf': KF[:, 0], 'Y_kf': KF[:, 1], 'S-intV': S - Dv, 'intV': Dv}
hs = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60]
def sig(X, h, mask_lab=None, pure=True):
    m = int(round(h / dt)); a = np.arange(n - m)
    ok = np.ones(len(a), bool)
    if mask_lab is not None:
        ok &= (lab[a] == mask_lab)
        if pure: ok &= (a + m <= runend[a])
    d = X[a + m] - X[a]
    return np.sqrt(np.mean(d[ok] ** 2) / h), ok.sum() * dt / h
print('signature sigma (ticks/sqrt s); N = effective independent samples')
for name, X in ser.items():
    print(f'\n== {name}')
    print('%6s %8s %8s %8s %8s %7s' % ('h', 'all', 'calm', 'vol', 'all-noshk', 'Nvol'))
    for h in hs:
        a_, _ = sig(X, h)
        c_, _ = sig(X, h, 0); v_, nv = sig(X, h, 1)
        m = int(round(h / dt)); a = np.arange(n - m)
        ok = ~shock[a] & ~shock[a + m] & ((a + m <= runend[a]) | (lab[a] != 2))
        # no-shock: neither endpoint in shock window and no shock window in between
        cs = np.cumsum(shock); ok = (cs[a + m] - cs[a]) == 0
        ok &= ~shock[a]
        d = X[a + m] - X[a]; ns = np.sqrt(np.mean(d[ok] ** 2) / h)
        print('%6.2f %8.3f %8.3f %8.3f %8.3f %7.0f' % (h, a_, c_, v_, ns, nv))
