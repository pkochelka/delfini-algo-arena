"""Book vs S: sd(mid - S), gap half-life, stale-quote episodes (ask < S - thr or bid > S + thr), by state,
and the stale book at the shock jumps.

S(t) = posterior mean at the latest insert <= t (step function).
"""
import numpy as np, pandas as pd
from pathlib import Path
from s_load import with_s
import states

HERE = Path(__file__).parent
o = with_s()
T_ALL = o.t_us.to_numpy(); S_ALL = o.S.to_numpy()
q = pd.read_pickle(HERE / 'cache/quotes.pkl')
t = q.t_us.to_numpy()
k = np.clip(np.searchsorted(T_ALL, t, 'right') - 1, 0, None)
S = S_ALL[k]
dur = np.diff(np.r_[t, t[-1]]) / 1e6
bid, ask = q.bid.to_numpy(), q.ask.to_numpy()
bq, aq = q.bid_qty.to_numpy(), q.ask_qty.to_numpy()
mid = (bid + ask) / 2
st = states.state(t)
two = ~np.isnan(mid)
pd.set_option('display.width', 220)

# 1. per-state book summary
rows = []
for s_ in ['calm', 'vol', 'drought', 'post', 'all']:
    m = two & ((st == s_) if s_ != 'all' else True)
    w = dur[m]; W = w.sum()
    x = (mid - S)[m]
    sp = (ask - bid)[m]
    rows.append({'state': s_, 'time_s': W, 'sd_mid_minus_S': np.sqrt((w * x ** 2).sum() / W),
                 'mean_abs': (w * np.abs(x)).sum() / W,
                 'sp1': (w * (sp == 1)).sum() / W, 'sp2': (w * (sp == 2)).sum() / W, 'sp3+': (w * (sp >= 3)).sum() / W,
                 'mean_spread': (w * sp).sum() / W, 'touch_qty': (w * (bq[m] + aq[m]) / 2).sum() / W,
                 'S_minus_bid_mean': (w * (S[m] - bid[m])).sum() / W, 'ask_minus_S_mean': (w * (ask[m] - S[m])).sum() / W})
print(pd.DataFrame(rows).round(3).to_string())
pd.DataFrame(rows).to_csv(HERE / 'book_by_state.csv', index=False)

# 2. gap half-life: ACF of mid - S on a 10 ms grid, per state (calm/vol, outside shocks)
grid = np.arange(t[0], t[-1], 10_000)
kg = np.searchsorted(t, grid, 'right') - 1
xg = (mid - S)[kg]
sg_ = st[kg]
for s_ in ['calm', 'vol']:
    ok = (sg_ == s_) & ~np.isnan(xg)
    xx = np.where(ok, xg - np.nanmean(xg[ok]), np.nan)
    ac = []
    for lag in [1, 2, 5, 10, 20, 30, 50, 100, 200]:
        a, b = xx[:-lag], xx[lag:]
        m = ~np.isnan(a) & ~np.isnan(b)
        ac.append((lag * 10, round(np.corrcoef(a[m], b[m])[0, 1], 3)))
    print(s_, 'ACF(mid-S) lag_ms:', ac)

# 2b. response to S jumps: after |dS| > 0.5 within 10 ms, how the gap closes
dS = np.diff(S_ALL[np.clip(np.searchsorted(T_ALL, grid, 'right') - 1, 0, None)])
for s_ in ['calm', 'vol']:
    ev = np.flatnonzero((np.abs(dS) > 0.5) & (sg_[1:] == s_)) + 1
    ev = ev[ev < len(grid) - 120]
    sgn = np.sign(dS[ev - 1])
    prof = [(lag * 10, round(np.nanmean(-sgn * xg[ev + lag]), 3)) for lag in [0, 1, 2, 3, 5, 10, 20, 30, 50, 100]]
    print(s_, 'after S jump >0.5 (n=%d): E[sign * (S - mid)] at lag ms' % len(ev), prof)

# 3. stale-quote episodes
def episodes(cond, extra_qty, edge):
    on = cond & two
    st_i = np.flatnonzero(on & ~np.r_[False, on[:-1]])
    en_i = np.flatnonzero(on & ~np.r_[on[1:], False])
    d_ms = (t[np.minimum(en_i + 1, len(t) - 1)] - t[st_i]) / 1e3
    qty0 = extra_qty[st_i]
    e0 = edge[st_i]
    emax = np.array([edge[a:b + 1].max() for a, b in zip(st_i, en_i)]) if len(st_i) < 400000 else e0
    return st_i, d_ms, qty0, e0, emax

rows = []
for thr in [0.25, 0.5, 1.0, 2.0]:
    for side, cond, qty, edge in [('ask<S-thr', ask < S - thr, aq, S - ask), ('bid>S+thr', bid > S + thr, bq, bid - S)]:
        si, d, q0, e0, emax = episodes(cond, qty, edge)
        sst = st[si]
        for s_ in ['calm', 'vol', 'drought', 'post', 'all']:
            m = (sst == s_) if s_ != 'all' else np.ones(len(si), bool)
            T = dur[(st == s_) if s_ != 'all' else slice(None)].sum()
            if m.sum() == 0:
                continue
            rows.append({'thr': thr, 'side': side, 'state': s_, 'n': int(m.sum()), 'per_s': m.sum() / T,
                         'time_share': (dur * (cond & two))[(st == s_) if s_ != 'all' else slice(None)].sum() / T,
                         'dur_ms_p10': np.percentile(d[m], 10), 'dur_ms_p50': np.median(d[m]), 'dur_ms_p90': np.percentile(d[m], 90),
                         'share_gt5ms': (d[m] > 5).mean(), 'touch_qty_p50': np.median(q0[m]), 'edge0_mean': e0[m].mean(),
                         'edge_max_mean': emax[m].mean()})
ep = pd.DataFrame(rows)
print(ep.round(3).to_string())
ep.to_csv(HERE / 'stale_episodes.csv', index=False)

# 4. what starts a stale episode: the S change just before (taker impact vs exogenous)
o_t = o.t_us.to_numpy()
for thr in [0.5]:
    for side, cond in [('ask<S-thr', ask < S - thr), ('bid>S+thr', bid > S + thr)]:
        on = cond & two
        si = np.flatnonzero(on & ~np.r_[False, on[:-1]])
        kk = np.clip(np.searchsorted(o_t, t[si], 'right') - 1, 0, None)
        trig_ioc = o.ioc.to_numpy()[kk - 1] | o.ioc.to_numpy()[kk]
        print(side, thr, 'share of episode starts within 1 order of a taker', trig_ioc.mean().round(3))

# 5. shocks: the stale book at the jump
ons = states.ONSETS_US
rows = []
for on in ons:
    tj = on + 3_000_000
    a = np.searchsorted(t, tj - 200_000); b = np.searchsorted(t, tj + 1_000_000)
    pre = slice(np.searchsorted(t, tj - 100_000), np.searchsorted(t, tj))
    S_before = np.median(S[pre]); S_after = np.median(S[np.searchsorted(t, tj + 100_000):np.searchsorted(t, tj + 300_000)])
    J = S_after - S_before
    # time until the touch on the stale side is no longer beyond S_after by > 0.5
    seg = slice(np.searchsorted(t, tj - 50_000), b)
    if J > 0:
        stale = ask[seg] < S_after - 0.5; qty = aq[seg]
    else:
        stale = bid[seg] > S_after + 0.5; qty = bq[seg]
    tt = t[seg]
    idx = np.flatnonzero(stale)
    first = (tt[idx[0]] - tj) / 1e3 if len(idx) else np.nan
    last = (tt[idx[-1]] - tj) / 1e3 if len(idx) else np.nan
    rows.append({'onset_s': on / 1e6, 'J': J, 'first_stale_ms': first, 'last_stale_ms': last,
                 'touch_qty_at_jump': qty[np.searchsorted(tt, tj)] if len(tt) else np.nan,
                 'spread_before': np.median((ask - bid)[pre])})
sh = pd.DataFrame(rows)
print(sh.round(2).to_string())
sh.to_csv(HERE / 'shock_stale.csv', index=False)
