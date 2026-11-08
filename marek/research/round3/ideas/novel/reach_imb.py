"""'Reachable imbalance' predictor: impact in the generator is c*filled^e of each IOC taker, and the book lags S.
So the next buy taker's impact is set by the ask depth within its limit round(S+k) and the next sell taker's by the bid
depth within round(S-k) (k = 1 calm, 2.55 volatile). RI = E[filled_buy^e] - E[filled_sell^e] over the taker size law.
Regress future exact-S moves on RI (null replay book, 50 ms grid, shock windows masked), per regime.
usage: uv run --with pyarrow python reach_imb.py NULL_LOGDIR
"""
import sys, csv, numpy as np, pandas as pd
nd = sys.argv[1]
S = pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet', columns=['t_us', 's_hat', 'vol'])
st = S.t_us.values; sh = S.s_hat.values; vol = S.vol.values
dl = pd.read_csv(f'{nd}/book_deltas.csv')
P0 = 9000; NP = 2000
bid = np.zeros(NP, np.int64); ask = np.zeros(NP, np.int64)
ts = dl.timestamp_us.values; sd = (dl.side.values == 'B'); pr = dl.price.values - P0; qd = dl.qty_diff.values
STEP = 50_000
grid = np.arange(STEP, st[-1], STEP)
# taker size law: rounded lognormal(4.094, 1.397); E[min(q, D)^e] tabulated on D
rng = np.random.default_rng(0); qs = np.maximum(1, np.round(np.exp(rng.normal(4.094, 1.397, 200_000))))
Dtab = np.arange(0, 5001); e = 0.61
qsort = np.sort(qs); cdf_vals = qsort ** e; csum = np.concatenate([[0], np.cumsum(cdf_vals)])
idx = np.searchsorted(qsort, Dtab, side='right')
Ef = (csum[idx] + (len(qs) - idx) * Dtab.astype(float) ** e) / len(qs)   # E[min(q,D)^e]
out = np.zeros((len(grid), 4))
j = 0; n = len(ts)
for gi, g in enumerate(grid):
    while j < n and ts[j] < g:
        if sd[j]: bid[pr[j]] += qd[j]
        else: ask[pr[j]] += qd[j]
        j += 1
    i = np.searchsorted(st, g, side='right') - 1
    s = sh[i]; v = vol[i]; k = 2.55 if v else 1.0
    la = int(round(s + k)) - P0; lb = int(round(s - k)) - P0
    ad = ask[:la + 1].sum(); bd = bid[lb:].sum()
    out[gi] = (min(ad, 5000), min(bd, 5000), s, v)
ad, bd, s, v = out.T
RI = Ef[ad.astype(int)] - Ef[bd.astype(int)]
on = np.array([float(r['onset']) for r in csv.DictReader(open('marek/research/round3/flow/shock_events.csv'))]) * 1e6
bad = np.zeros(len(grid), bool)
for o in on: bad |= (grid >= o - 1e6) & (grid < o + 20e6)
print('RI sd calm %.2f vol %.2f' % (RI[~bad & (v == 0)].std(), RI[~bad & (v == 1)].std()))
for H in (1, 2, 4, 10, 20, 100):
    fut = np.roll(s, -H) - s
    m = ~bad & (np.arange(len(grid)) < len(grid) - H)
    res = []
    for nm, mm, c in (('calm', m & (v == 0), 0.0171), ('vol', m & (v == 1), 0.062)):
        b = np.polyfit(RI[mm], fut[mm], 1)[0]; r = np.corrcoef(RI[mm], fut[mm])[0, 1]
        res.append(f'{nm}: beta {b:.4f} (c={c}) corr {r:+.3f}')
    print(f'H={H*STEP/1e3:.0f}ms', ' | '.join(res))
np.save('/private/tmp/claude-501/-Users-marek-Documents-delfini-algo-arena/c0256e15-f719-43d5-b913-67767d347aa1/scratchpad/ri.npy', np.c_[grid, RI, s, v, bad])
