"""Regime transitions: detection lag of turb vs the s_dynamics volatile segments, PnL around each start / end, and the
position at the start vs the move that follows.
uv run --with pyarrow python marek/research/round4/edge/tools/trans.py cache/r4_volsig_round4.npz"""
import sys, pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]
z = np.load(sys.argv[1])
x, curve = z['x'], z['curve']
t, F, sd, pv, turb, pos, bb, ba, Y, kv, sgw = x.T
ct, cp = curve[:, 0], curve[:, 1]
seg = pd.read_csv(R4 / 's_dynamics/vol_segments.csv')
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
st = sp.t_us.values * 1e-6; S = sp.s_hat.values
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
def pnl(a, b):
    ia = np.searchsorted(ct, a); ib = np.searchsorted(ct, b)
    ia = min(max(ia, 0), len(cp) - 1); ib = min(max(ib, 0), len(cp) - 1)
    return cp[ib] - cp[ia]
def at(arr, tt, tq):
    return arr[min(max(np.searchsorted(tt, tq, 'right') - 1, 0), len(arr) - 1)]
rows = []
for _, r in seg.iterrows():
    a, b = r.start, r.end
    if np.any((on > a - 25) & (on < a + 5)):
        shocky = True
    else:
        shocky = False
    # detection: first turb==1 after a-2 that stays on for 0.5 s
    m = (t >= a - 2) & (t <= a + 10)
    tt, tu = t[m], turb[m]
    det = np.nan
    for k in range(len(tt)):
        if tu[k] and tt[min(k + 50, len(tt) - 1)] - tt[k] <= 0.6 and tu[k:k + 50].mean() > 0.9:
            det = tt[k]; break
    # end detection: first turb==0 after b-2 staying off 0.5 s
    m2 = (t >= b - 2) & (t <= b + 20)
    tt2, tu2 = t[m2], turb[m2]
    dete = np.nan
    for k in range(len(tt2)):
        if not tu2[k] and tu2[k:k + 50].mean() < 0.1:
            dete = tt2[k]; break
    p0 = at(pos, t, a); S0 = at(S, st, a); S5 = at(S, st, a + 5); S2 = at(S, st, a + 2)
    rows.append(dict(start=a, end=b, dur=b - a, shocky=shocky, det_lag=det - a, end_lag=dete - b,
                     pnl_m3_0=pnl(a - 3, a), pnl_0_3=pnl(a, a + 3), pnl_3_10=pnl(a + 3, a + 10), pnl_seg=pnl(a, b),
                     pnl_end_0_5=pnl(b, b + 5), pnl_end_5_20=pnl(b + 5, b + 20),
                     pos0=p0, dS2=S2 - S0, dS5=S5 - S0, align5=np.sign(p0) * (S5 - S0), pos_end=at(pos, t, b)))
d = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(d.round(2).to_string())
ns = d[~d.shocky]
print('\nnon-shocky starts n=%d: det lag median %.2f mean %.2f; end lag median %.2f' % (len(ns), ns.det_lag.median(), ns.det_lag.mean(), ns.end_lag.median()))
for c in ('pnl_m3_0', 'pnl_0_3', 'pnl_3_10', 'pnl_end_0_5', 'pnl_end_5_20'):
    print(f'  {c:12s} sum {d[c].sum()/1e3:+7.1f}k  mean {d[c].mean():+7.0f}  (non-shocky sum {ns[c].sum()/1e3:+7.1f}k)')
print('  position at start: mean |pos0| %.0f, align5 mean %+.2f (lots x ticks /lot), share aligned %.2f' % (
    d.pos0.abs().mean(), d.align5.mean(), (d.align5 > 0).mean()))
print('  seg pnl per s: %.1f' % (d.pnl_seg.sum() / d.dur.sum()))
# turb time vs truth
vol_true = np.zeros(len(t), bool)
for _, r in seg.iterrows():
    vol_true |= (t >= r.start) & (t < r.end)
print('\nturb vs truth (10 ms samples): P(turb|vol) %.3f  P(turb|calm) %.3f  turb share %.3f  vol share %.3f' % (
    turb[vol_true].mean(), turb[~vol_true].mean(), turb.mean(), vol_true.mean()))
# PnL by (truth, turb) cell
dp = np.diff(cp); tc = ct[:-1]
k = np.clip(np.searchsorted(t, tc, 'right') - 1, 0, len(t) - 1)
sh = np.zeros(len(tc), bool)
for o in on:
    sh |= (tc >= o - 1) & (tc < o + 20)
for vt in (False, True):
    for tb in (0.0, 1.0):
        m = (vol_true[k] == vt) & (turb[k] == tb) & ~sh
        print(f'  truth_vol={vt!s:5s} turb={int(tb)}: time {m.sum()} s, pnl {dp[m].sum()/1e3:+7.1f}k, per s {dp[m].sum()/max(m.sum(),1):+6.1f}')
