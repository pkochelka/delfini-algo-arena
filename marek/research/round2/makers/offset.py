"""Maker placement: MLE of D = d0 + Gamma(k, scale) (k=1: exponential), with the no-cross clip,
integrating over the S posterior (Gaussian, mean S, sd S_sd) by Gauss-Hermite quadrature.
Splits by side, size, 600 s segment, local volatility and clip status."""
import numpy as np, pandas as pd
from scipy import optimize, stats
from s_load import with_s

o = with_s()
g = o[~o.ioc & o.bid.notna() & o.ask.notna()].copy()
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(7); gh_w = gh_w / gh_w.sum()


def prep(g):
    b = (g.sg == 1).to_numpy()
    p = g.price.to_numpy().astype(float)
    lim = np.where(b, g.ask - 1, g.bid + 1).astype(float)
    clipped = p == lim
    S = g.S.to_numpy()[:, None] + g.S_sd.to_numpy()[:, None] * gh_x[None, :]
    y = np.where(b[:, None], S - p[:, None], p[:, None] - S)
    ylim = np.where(b[:, None], S - lim[:, None], lim[:, None] - S)
    return y, ylim, clipped


def nll(th, y, ylim, clipped, shape_free):
    d0, sc = th[0], th[1]
    k = th[2] if shape_free else 1.0
    if sc <= 0 or d0 < 0 or k <= 0:
        return 1e18
    F = lambda z: stats.gamma.cdf(np.maximum(z - d0, 0), k, scale=sc) if shape_free else np.where(z > d0, -np.expm1(-(z - d0) / sc), 0.0)
    pr = np.where(clipped[:, None], F(ylim), F(y) - F(y - 1))
    pr = (pr * gh_w).sum(1)
    return -np.log(np.maximum(pr, 1e-300)).sum()


def fit(g, shape_free=False):
    y, ylim, c = prep(g)
    x0 = [0.05, 0.45] + ([1.0] if shape_free else [])
    r = optimize.minimize(nll, x0, args=(y, ylim, c, shape_free), method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-3, 'maxiter': 400})
    return r.x, r.fun / len(g), c.mean()


if __name__ == '__main__':
    sub = g.sample(200_000, random_state=1)
    x, l, cs = fit(sub); print('exp: d0 %.4f scale %.4f  nll/obs %.5f clipped %.3f' % (x[0], x[1], l, cs))
    x, l, cs = fit(sub, True); print('gamma: d0 %.4f scale %.4f k %.4f  nll/obs %.5f' % (x[0], x[1], x[2], l))
    # precise S only
    good = sub[sub.S_sd < 0.1]
    x, l, cs = fit(good); print('exp, S_sd<0.1 (%d): d0 %.4f scale %.4f' % (len(good), x[0], x[1]))
    rows = []
    sub = g.sample(400_000, random_state=2)
    # local vol: |S change| over the previous 1 s, from the order sequence
    t = o.t_us.to_numpy(); S = o.S.to_numpy()
    k = np.searchsorted(t, sub.t_us.to_numpy() - 1_000_000)
    sub['vol1s'] = np.abs(sub.S.to_numpy() - S[k])
    groups = {'side': sub.sg, 'size': pd.cut(sub.volume, [0, 10, 100, 1000, 1e6]),
              'seg600': sub.t_us // 600_000_000, 'absdS_1s': pd.qcut(sub.vol1s, 5),
              'spread_before': (sub.ask - sub.bid).clip(upper=4)}
    for name, key in groups.items():
        for val, gg in sub.groupby(key, observed=True):
            if len(gg) < 3000:
                continue
            x, l, cs = fit(gg)
            rows.append({'split': name, 'value': str(val), 'n': len(gg), 'd0': round(x[0], 4), 'scale': round(x[1], 4), 'clipped': round(cs, 3)})
    df = pd.DataFrame(rows); print(df.to_string()); df.to_csv('offset_splits.csv', index=False)
