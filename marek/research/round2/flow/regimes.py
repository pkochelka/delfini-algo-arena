"""Segments of the 2-state taker HMM; per-regime stats (taker rate per side, maker rate, sizes, |dmid|)."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
z = np.load(C / 'bins1s.npz'); g = np.load(C / 'hmm_taker_gamma2.npy'); mask = z['mask']
hi = g[:, 1] > 0.5
# fill masked bins by neighbouring state (posterior already handles missing)
segs = []; s = 0
for i in range(1, len(hi) + 1):
    if i == len(hi) or hi[i] != hi[s]:
        segs.append((s, i, bool(hi[s]))); s = i
S = pd.DataFrame(segs, columns=['start', 'end', 'high']); S['dur'] = S.end - S.start
print(S[S.high].to_string()); print('high durations', S[S.high].dur.describe().round(1).to_dict())
print('low durations', S[~S.high].dur.describe().round(1).to_dict())
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0]
b = ins.t.astype(int).clip(0, 5999).to_numpy(); ins = ins.assign(hi=hi[b], ok=mask[b])
I = ins[ins.ok]
for h in [False, True]:
    x = I[I.hi == h]; secs = (hi & mask).sum() if h else (~hi & mask).sum()
    tk = x[x.ioc]; mk = x[~x.ioc]
    print(f'regime high={h}: secs {secs}, taker buy {((tk.side==0).sum()/secs):.3f}/s sell {((tk.side==1).sum()/secs):.3f}/s, maker {len(mk)/secs:.2f}/s,'
          f' taker logsize mean {np.log(tk.vol).mean():.3f} sd {np.log(tk.vol).std():.3f}, maker logsize mean {np.log(mk.vol).mean():.3f} sd {np.log(mk.vol).std():.3f},'
          f' |dmid| per s {np.abs(z["dm"][(hi==h)&mask]).mean():.3f}, pathvar per s {z["pv"][(hi==h)&mask].mean():.3f}')
S.to_csv(D / 'taker_regimes.csv', index=False)
