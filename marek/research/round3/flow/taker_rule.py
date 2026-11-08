"""Taker limit price vs S: x = sg*(p - S_before). Regular band round(S + sg*k) and aggressive tail, by regime.
Compare with round 2 (k=1 calm, 2.55 vol; tail 2-5 % / 4-8 %)."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
S = pd.read_parquet(D.parent / 's_dynamics/s_path.parquet')
idx = np.flatnonzero(ins.ioc.to_numpy()); sg = np.where(ins.side.to_numpy()[idx] == 0, 1, -1)
s_b = S.s_hat.to_numpy()[idx - 1]; vol = S.vol.to_numpy()[idx]; shock = S.shock.to_numpy()[idx]; sd = S.s_sd.to_numpy()[idx - 1]
p = ins.px.to_numpy()[idx]; x = sg * (p - s_b)
for v in [False, True]:
    m = (vol == v) & ~shock & (sd < 0.3)
    k = 1.0 if not v else 2.55
    reg_band = np.abs(x[m] - k) <= 0.5 + 2 * sd[m]
    print(f'vol={v} n={m.sum()}: x quantiles {np.round(np.quantile(x[m], [.01, .05, .25, .5, .75, .95, .99]), 2)}; in band |x-{k}|<=0.5(+2sd): {reg_band.mean():.3f}')
    e = np.arange(-1, 12.01, 0.25); h = np.histogram(x[m], e)[0]
    print('  hist', ' '.join(f'{a:.2f}:{b}' for a, b in zip(e[:-1], h) if b))
    # fit k: maximize fraction within band
    ks = np.arange(0.5, 3.5, 0.05); fr = [np.mean(np.abs(x[m] - kk) <= 0.5) for kk in ks]; print('  best k', ks[np.argmax(fr)].round(2), 'frac', max(fr).round(3))
    tail = x[m][x[m] > k + 0.5 + 0.3] - (k + 0.5)
    print(f'  tail share {len(tail) / m.sum():.3f}; tail excess mean {tail.mean():.2f} median {np.median(tail):.2f}; KS exp p={stats.kstest(tail - tail.min(), "expon", (0, tail.mean() - tail.min())).pvalue:.3f}')
    print('  tail excess quantiles', np.round(np.quantile(tail, [.1, .25, .5, .75, .9, .99]), 2))
