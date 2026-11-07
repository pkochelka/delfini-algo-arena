"""Distribution of nearby maker prices relative to the taker's limit price (signed so + = toward taker side), calm regime, 5 ms before."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); hi = np.load(C / 'hmm_taker_gamma2.npy')[:, 1] > 0.5
ev = pd.read_csv(D / 'shock_events.csv')
ins = r[r.op == 0]; mk = ins[~ins.ioc]; tk = ins[ins.ioc]
mt = mk.t.to_numpy(); mpx = mk.px.to_numpy(); ms = mk.side.to_numpy()
rows = []
for t, p, s, v in zip(tk.t.to_numpy(), tk.px.to_numpy(), tk.side.to_numpy(), tk.vol.to_numpy()):
    a = np.searchsorted(mt, t - 0.005); b = np.searchsorted(mt, t)
    if b == a: continue
    sg = 1 if s == 0 else -1
    P = mpx[a:b]; S = ms[a:b]
    # same-direction maker quotes: for buy taker, makers' bids are 'below S', asks 'above S'
    bids = P[S == 0]; asks = P[S == 1]
    rows.append(dict(t=t, sg=sg, vol=v, hi=hi[int(t)],
                     # in taker-signed coordinates: x = sg*(maker_px - p); rule predicts bids_rel <= -1 and asks_rel >= -1 (buy)
                     near=(sg * (bids - p)).max() if sg == 1 and len(bids) else ((sg * (asks - p)).max() if sg == -1 and len(asks) else np.nan),
                     far=(sg * (asks - p)).min() if sg == 1 and len(asks) else ((sg * (bids - p)).min() if sg == -1 and len(bids) else np.nan)))
R = pd.DataFrame(rows)
sh = np.zeros(len(R), bool)
for o in ev.onset: sh |= ((R.t > o - 0.5) & (R.t < o + 15)).to_numpy()
R['shock'] = sh
for hh in [False, True]:
    X = R[(R.hi == hh) & ~R.shock]
    print(f'== volatile={hh}: n={len(X)}')
    print(' same-side-as-S-below maker (bid for buy taker), max of sg*(px - p); rule: <= -1'); print(X.near.value_counts(normalize=True).sort_index().round(4).to_string())
    print(' other-side maker (ask for buy taker), min of sg*(px - p); rule: >= -1'); print(X.far.value_counts(normalize=True).sort_index().round(4).to_string())
