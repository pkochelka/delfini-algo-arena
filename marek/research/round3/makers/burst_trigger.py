"""What triggers a ladder burst? Time from the latest same-side / opposite-side touch change, latest trade, latest
same-side burst, compared with random times (in the same regime)."""
import numpy as np, pandas as pd
o = pd.read_pickle('cache/chains.pkl')
g = np.load('cache/regime_sz.npz')['post']
o['vol'] = g[(o.t_us//1_000_000).to_numpy(), 1] > .5
q = pd.read_pickle('cache/quotes.pkl'); tr = pd.read_pickle('cache/trades.pkl')
st = o[(o.L >= 2) & (o.L <= 6) & (o.k == 0) & ~o.ioc.astype(bool)]
rng = np.random.default_rng(0)
def last_before(ev_t, t):
    j = np.searchsorted(ev_t, t) - 1
    return np.where(j >= 0, t - ev_t[np.maximum(j, 0)], np.nan)
qt = q.t_us.to_numpy()
bch = qt[np.r_[True, np.diff(q.bid.to_numpy()) != 0]]; ach = qt[np.r_[True, np.diff(q.ask.to_numpy()) != 0]]
bup = qt[np.r_[False, np.diff(q.bid.to_numpy()) > 0]]; bdn = qt[np.r_[False, np.diff(q.bid.to_numpy()) < 0]]
aup = qt[np.r_[False, np.diff(q.ask.to_numpy()) > 0]]; adn = qt[np.r_[False, np.diff(q.ask.to_numpy()) < 0]]
tt = np.unique(tr.t_us.to_numpy())
bins = [0, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1e9]
for v in (True, False):
    s = st[st.vol == v]
    # random comparison times: uniform within the regime seconds
    secs = np.flatnonzero((g[:, 1] > .5) == v)
    rt = (rng.choice(secs, 50000) * 1e6 + rng.uniform(0, 1e6, 50000)).astype(np.int64)
    rs = rng.choice([1, -1], 50000)
    print('=== VOL' if v else '=== CALM', len(s))
    for name, fb, fa in [('same-side touch change', bch, ach), ('opp-side touch change', ach, bch),
                         ('same-side touch moved AWAY (bid down/ask up)', bdn, aup), ('same-side touch moved TOWARD (bid up/ask down)', bup, adn),
                         ('trade', tt, tt)]:
        x = np.where(s.sg > 0, last_before(fb, s.t_us.to_numpy()), last_before(fa, s.t_us.to_numpy()))
        y = np.where(rs > 0, last_before(fb, rt), last_before(fa, rt))
        hx = np.histogram(x, bins)[0] / len(x); hy = np.histogram(y, bins)[0] / len(y)
        print(f' {name}: ratio burst/random per bin (us):', ' '.join(f'{int(a)}-:{a_/b_:.2f}' for a, a_, b_ in zip(bins[:-1], hx, hy)))
