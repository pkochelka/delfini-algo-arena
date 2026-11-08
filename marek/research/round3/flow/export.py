"""Exports: rates_60s.csv (per-minute rates by stream + volatile share), ladder_events.csv (one row per multi-rung ladder)."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True); mem = np.load(C / 'ladder_member.npy')
L = pd.read_pickle(C / 'ladders.pkl'); R = pd.read_csv(D / 'regimes.csv')
ti = ins.t.to_numpy(); ioc = ins.ioc.to_numpy(); sd = ins.side.to_numpy()
def c60(x): return np.bincount((x // 60).astype(int), minlength=100)[:100] / 60
lab = np.zeros(6000)
for _, s in R[R.regime == 'volatile'].iterrows(): lab[int(s.start):int(np.ceil(s.end))] = 1
out = pd.DataFrame({'t0': np.arange(100) * 60, 'vol_share': lab.reshape(100, 60).mean(1),
                    'taker_buy': c60(ti[ioc & (sd == 0)]), 'taker_sell': c60(ti[ioc & (sd == 1)]),
                    'maker_base': c60(ti[~ioc & (mem[:, 0] < 0)]), 'ladder_inserts': c60(ti[~ioc & (mem[:, 0] >= 0)]),
                    'ladder_starts': c60(L.t0.to_numpy())})
out.round(3).to_csv(D / 'rates_60s.csv', index=False)
print(out.corr().round(2).to_string())
E = L.assign(start_depth=np.where(L.side == 0, L.bid_before - L.p0, L.p0 - L.ask_before),
             steps=L.steps.apply(lambda x: ' '.join(map(str, x))), sizes=L.sizes.apply(lambda x: ' '.join(map(str, x))))
E['regime'] = np.where(E.vol, 'volatile', 'calm')
E[['t0', 'side', 'n', 'p0', 'start_depth', 'steps', 'sizes', 'regime']].round(6).to_csv(D / 'ladder_events.csv', index=False)
print(len(E), 'ladders exported')
