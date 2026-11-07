"""Volatility regime label from the S path.
Robust vol (1.4826*median|dS_100ms|*sqrt(10), ticks/sqrt(s)) over a centred 3 s window, on a 100 ms grid.
hot = rv3 > 1.1. Calm rv ~0.45, hot ~2-3 (bimodal). Writes cache/regime.npz; regime_at(t_us) helper."""
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).parent


def build():
    from s_load import with_s
    o = with_s()
    t = o.t_us.to_numpy(); S = o.S.to_numpy()
    grid = np.arange(0, 6000_000_000 + 1, 100_000)
    Sg = S[np.clip(np.searchsorted(t, grid, 'right') - 1, 0, None)]
    d = np.abs(np.diff(Sg))
    rv3 = pd.Series(d).rolling(30, center=True, min_periods=5).median().to_numpy() * 1.4826 * np.sqrt(10)
    # trailing (causal) version, usable live
    rv3c = pd.Series(d).rolling(30, min_periods=5).median().to_numpy() * 1.4826 * np.sqrt(10)
    np.savez(HERE / 'cache/regime.npz', grid=grid[1:], rv3=rv3, rv3c=rv3c)


def regime_at(t_us, causal=False):
    z = np.load(HERE / 'cache/regime.npz')
    k = np.clip(np.searchsorted(z['grid'], t_us, 'right') - 1, 0, len(z['grid']) - 1)
    rv = (z['rv3c'] if causal else z['rv3'])[k]
    return rv, rv > 1.1


if __name__ == '__main__':
    build()
    z = np.load(HERE / 'cache/regime.npz'); rv = z['rv3']
    print('rv3 hist', np.histogram(rv[~np.isnan(rv)], [0, .3, .5, .7, .9, 1.1, 1.5, 2, 3, 5, 50])[0])
    hot = rv > 1.1
    print('hot share', np.nanmean(hot))
    st = np.flatnonzero(hot & ~np.r_[False, hot[:-1]]); en = np.flatnonzero(hot & ~np.r_[hot[1:], False])
    dur = (en - st + 1) / 10
    print('hot spells', len(st), 'duration s quantiles 10/50/90', np.percentile(dur, [10, 50, 90]).round(1), 'max', dur.max(), 'total', dur.sum())
    print('long spells (>20s) start,dur', [(round(a / 10), round(b, 1)) for a, b in zip(st, dur) if b > 20])
    print('agreement causal vs centred', np.nanmean((z['rv3c'] > 1.1) == hot))
