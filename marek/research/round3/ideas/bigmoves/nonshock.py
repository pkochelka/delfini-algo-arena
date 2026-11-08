"""Non-shock stale value after each filled market taker (checked at taker + 5 ms, vs S at that time).
Level book from orders/trades deltas. Output per check: side, stale lots/value by edge band, opposite-side depth.
uv run --with pyarrow python nonshock.py NAME [SIMLOG]"""
import sys, numpy as np, pandas as pd, pickle
from common import *
name = sys.argv[1]; simlog = sys.argv[2] if len(sys.argv) > 2 else None
tp, o, tr = load(name)
St, Sv = s_path(name)
sh = shocks(name, (St, Sv))
ons = np.array([s['onset'] for s in sh])
mk = o[~o.ioc & (o.resting_quantity > 0)]
P0 = int(min(o.price.min(), 9000)); NP = int(o.price.max()) - P0 + 2
sidecode = (mk.side.values == 'A').astype(np.int64)
row2 = pd.Series(np.arange(len(o)), index=o.row.values)
# deltas
d_t = [mk.t_us.values]; d_s = [sidecode]; d_p = [mk.price.values]; d_q = [mk.resting_quantity.values]
mrow = row2.reindex(tr.maker_row.values).values
fs = (o.side.values[mrow] == 'A').astype(np.int64)
d_t.append(tr.t_us.values); d_s.append(fs); d_p.append(tr.price.values); d_q.append(-tr.qty.values)
c = mk[mk.end == 'cancelled']
# remaining at cancel = resting - filled_after_insert(=filled - (volume-resting))
rem = c.resting_quantity.values - (c.filled.values - (c.volume.values - c.resting_quantity.values))
d_t.append(c.end_t_us.values.astype(np.int64)); d_s.append((c.side.values == 'A').astype(np.int64)); d_p.append(c.price.values); d_q.append(-rem)
T = np.concatenate(d_t); SD = np.concatenate(d_s); PX = np.concatenate(d_p) - P0; Q = np.concatenate(d_q)
order = np.argsort(T, kind='stable'); T, SD, PX, Q = T[order], SD[order], PX[order], Q[order]
# checks: each filled market taker outside shock windows (onset-1 .. onset+20)
tk = o[o.ioc & (o.filled > 0)]
tt = tk.t.values
k = np.searchsorted(ons, tt + 1.0) - 1
insh = (k >= 0) & (tt < ons[np.clip(k, 0, None)] + 20.0)
tk = tk[~insh]
chk_t = (tk.t_us.values + 5000)
chk_side = np.where(tk.side.values == 'B', 1, -1)  # buy taker -> S up -> stale asks
Sat = Sv[np.clip(np.searchsorted(St, chk_t * 1e-6, 'right') - 1, 0, None)]
vol = pd.read_parquet(R3 / 's_dynamics/s_path.parquet' if name == 'r3' else R2 / 's_dynamics/s_path.parquet', columns=['vol']).vol.values
vat = vol[np.clip(np.searchsorted(St, chk_t * 1e-6, 'right') - 1, 0, None)]
book = np.zeros((2, NP), np.int64); cons = np.zeros((2, NP), np.int64)
THR = [2.0, 3.0, 4.0]; QS = [100, 300, 950, 10**7]
# infinite-room sniper per threshold would interact; run one consuming sniper per (thr) with Q = inf, and record
# what it took in rank order so caps can be applied per check (lots best edge first).
res = {thr: [] for thr in THR}
books = {thr: (np.zeros((2, NP), np.int64), np.zeros((2, NP), np.int64)) for thr in THR}
j = 0
ci = np.argsort(chk_t, kind='stable')
for n in ci:
    tc = chk_t[n]
    while j < len(T) and T[j] <= tc:
        for thr in THR:
            raw, cs = books[thr]
            raw[SD[j], PX[j]] += Q[j]
            if Q[j] < 0 and cs[SD[j], PX[j]] > raw[SD[j], PX[j]]:
                cs[SD[j], PX[j]] = max(raw[SD[j], PX[j]], 0)
        j += 1
    S = Sat[n] - P0; d = chk_side[n]
    for thr in THR:
        raw, cs = books[thr]
        if d > 0:
            hi = int(np.floor(S - thr)); ps = np.arange(max(hi - 40, 0), hi + 1)
            q = raw[1, ps] - cs[1, ps]; e = S - ps; sd_ = 1
            opp = raw[0] - cs[0]; dep = [opp[int(np.floor(S - u)):].sum() for u in (1, 2, 4, 8)]
        else:
            lo = int(np.ceil(S + thr)); ps = np.arange(lo, min(lo + 41, NP))
            q = raw[0, ps] - cs[0, ps]; e = ps - S; sd_ = 0
            opp = raw[1] - cs[1]; dep = [opp[:int(np.ceil(S + u)) + 1].sum() for u in (1, 2, 4, 8)]
        m = q > 0
        if not m.any():
            continue
        q, e, ps = q[m], e[m], ps[m]
        cs[sd_, ps] += q
        ix = np.argsort(-e); q, e = q[ix], e[ix]
        res[thr].append((n, q.sum(), (q * e).sum(), *[(np.minimum(q, np.clip(Q_ - (np.cumsum(q) - q), 0, None)) * e).sum() for Q_ in QS[:3]], *dep))
import collections
for thr in THR:
    R = pd.DataFrame(res[thr], columns=['n', 'L', 'V', 'V100', 'V300', 'V950', 'dep1', 'dep2', 'dep4', 'dep8'])
    R['vol'] = vat[R.n.values]; R['t'] = chk_t[R.n.values] * 1e-6
    R.to_pickle(pathlib.Path(__file__).parent / f'nonshock_{name}_thr{thr:.0f}.pkl')
    for g, Y in R.groupby('vol'):
        big = Y[Y.L > 100]
        print(f"thr {thr} {'vol ' if g else 'calm'} n={len(Y)} L={Y.L.sum():.0f} V={Y.V.sum():.0f} V100={Y.V100.sum():.0f} V300={Y.V300.sum():.0f} V950={Y.V950.sum():.0f} | n(L>100)={len(big)} V-V100 there={big.V.sum()-big.V100.sum():.0f}  dep1/2/4 med at L>100: {big.dep1.median():.0f}/{big.dep2.median():.0f}/{big.dep4.median():.0f}")
raise SystemExit
cols = ['L1', 'V1', 'L2', 'V2', 'L3', 'V3', 'Lb', 'Vb_net2', 'Vb_net4', 'dep1', 'dep2', 'dep4']
X = pd.DataFrame(out, columns=cols); X['t'] = chk_t * 1e-6; X['d'] = chk_side; X['vol'] = vat; X['qtk'] = tk.filled.values
X.to_pickle(pathlib.Path(__file__).parent / f'nonshock_{name}.pkl')
for g, Y in X.groupby('vol'):
    print('vol' if g else 'calm', len(Y), Y[cols].sum().round().to_dict())
