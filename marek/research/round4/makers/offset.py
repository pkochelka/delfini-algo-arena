"""Base-maker placement per state: censored MLE of D = d0 + Exp(scale) under the round-1 clip rule
(bid = min(floor(S - D), ask - 1)), S integrated over its posterior (7-pt Gauss-Hermite), plus a ladder-singleton
component: with weight w the price is uniform on same-side touch -2..-7. Chain ladder orders are excluded.
Also the one-multiplier test D_state = m * D_calm and splits (side, size, regime x size-band)."""
import numpy as np, pandas as pd
from scipy import optimize
from s_load import with_s

o = with_s()
g0 = o[~o.ioc & ~o.lad & o.bid.notna() & o.ask.notna()].copy()
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(7); gh_w = gh_w / gh_w.sum()


def prep(g):
    b = (g.sg == 1).to_numpy(); p = g.price.to_numpy().astype(float)
    lim = np.where(b, g.ask - 1, g.bid + 1).astype(float); clipped = p == lim
    S = g.S.to_numpy()[:, None] + np.maximum(g.S_sd.to_numpy(), 0.02)[:, None] * gh_x[None, :]
    y = np.where(b[:, None], S - p[:, None], p[:, None] - S)
    ylim = np.where(b[:, None], S - lim[:, None], lim[:, None] - S)
    rel = np.where(b, p - g.bid, g.ask - p)               # 0 at touch, -k behind
    lad_ok = ((rel <= -2) & (rel >= -7)).astype(float)
    return y, ylim, clipped, lad_ok


def nll(th, y, ylim, clipped, lad_ok, mix=True):
    d0, sc = th[0], th[1]
    w = 1 / (1 + np.exp(-th[2])) if mix else 0.0
    if sc <= 0 or d0 < 0:
        return 1e18
    F = lambda z: np.where(z > d0, -np.expm1(-(z - d0) / sc), 0.0)
    pr = np.where(clipped[:, None], F(ylim), F(y) - F(y - 1))
    pr = (pr * gh_w).sum(1)
    pr = (1 - w) * pr + w * lad_ok / 6
    return -np.log(np.maximum(pr, 1e-300)).sum()


def fit(g, mix=True):
    a = prep(g)
    r = optimize.minimize(nll, [0.05, 0.45, -3.5], args=(*a, mix), method='Nelder-Mead',
                          options={'xatol': 1e-4, 'fatol': 1e-3, 'maxiter': 600})
    return r.x, r.fun / len(g), a[2].mean()


if __name__ == '__main__':
    rows = []; calm = None
    for s_ in ['calm', 'vol', 'drought', 'post']:
        gg = g0[g0.st == s_]; gg = gg.sample(min(len(gg), 100000), random_state=0)
        x, l, cs = fit(gg)
        r = dict(state=s_, n_all=int((g0.st == s_).sum()), d0=x[0], scale=x[1], mean_D=x[0] + x[1], w_ladder=1 / (1 + np.exp(-x[2])),
                 nll=l, clipped=cs, S_sd_med=gg.S_sd.median())
        if s_ == 'calm':
            calm = x
        else:
            a = prep(gg)
            f = lambda th: nll([calm[0] * th[0], calm[1] * th[0], th[1]], *a) / len(gg)
            rm = optimize.minimize(f, [2.0, -3.5], method='Nelder-Mead')
            r['m_one_mult'] = rm.x[0]; r['dnll_one_mult'] = rm.fun - l
        rows.append(r); print(r, flush=True)
    good = g0[(g0.st == 'calm') & (g0.S_sd < 0.06)].sample(100000, random_state=3)
    x, l, cs = fit(good); rows.append(dict(state='calm, S_sd<0.06', n_all=len(good), d0=x[0], scale=x[1], mean_D=x[0] + x[1], w_ladder=1 / (1 + np.exp(-x[2])), nll=l, clipped=cs))
    print(rows[-1])
    pd.DataFrame(rows).round(4).to_csv('offset_by_state.csv', index=False)
    # splits
    sp = []
    for s_ in ['calm', 'vol']:
        gs = g0[g0.st == s_].sample(150000, random_state=4)
        for name, key in {'side': gs.sg, 'size': pd.cut(gs.volume, [0, 10, 20, 70, 320, 1e6]),
                          'seg1000': gs.t_us // 1_000_000_000, 'spread_before': (gs.ask - gs.bid).clip(upper=4)}.items():
            for val, gg in gs.groupby(key, observed=True):
                if len(gg) < 3000: continue
                x, l, cs = fit(gg)
                sp.append(dict(state=s_, split=name, value=str(val), n=len(gg), d0=round(x[0], 4), scale=round(x[1], 4), w_ladder=round(1 / (1 + np.exp(-x[2])), 4), clipped=round(cs, 3)))
    df = pd.DataFrame(sp); pd.set_option('display.width', 200); print(df.to_string()); df.to_csv('offset_splits.csv', index=False)
