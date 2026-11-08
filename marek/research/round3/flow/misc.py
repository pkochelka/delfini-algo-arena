"""Within-regime dispersion (1/10/30 s) of takers and base makers; per-1000 s block rates; periodogram peaks;
shock -> regime-switch coupling; ids/timestamps sanity."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True); mem = np.load(C / 'ladder_member.npy')
R = pd.read_csv(D / 'regimes.csv'); ev = pd.read_csv(D / 'shock_events.csv'); V = R[R.regime == 'volatile']
ti = ins.t.to_numpy(); ioc = ins.ioc.to_numpy()
oid = ins.oid.to_numpy(); print('oid consecutive', (np.diff(r.oid[r.op == 0].to_numpy()) == 1).all(), 'distinct ts', (np.diff(r.t.to_numpy()) > 0).all())
lab = np.full(6000, -1)
for _, s in R.iterrows(): lab[int(np.ceil(s.start + 2)):int(s.end - 2)] = 1 if s.regime == 'volatile' else 0
for o in ev.onset: lab[int(o - 2):int(o + 20)] = -1
streams = {'takers': ti[ioc], 'base makers': ti[~ioc & (mem[:, 0] < 0)], 'ladder inserts': ti[~ioc & (mem[:, 0] >= 0)]}
for nm, x in streams.items():
    c = np.bincount(x.astype(int), minlength=6000)[:6000]
    out = []
    for reg in [0, 1]:
        for w in [1, 10, 30]:
            cw = c[:6000 // w * w].reshape(-1, w).sum(1); ok = (lab[:6000 // w * w].reshape(-1, w) == reg).all(1)
            out.append(f'{"vol" if reg else "calm"} w{w}: DI {cw[ok].var() / cw[ok].mean():.2f}')
    print(f'{nm:15s}', ' | '.join(out))
    blk = [c[b * 1000:(b + 1) * 1000][lab[b * 1000:(b + 1) * 1000] == 0].mean() for b in range(6)]
    print('   calm rate per 1000 s block', np.round(blk, 2))
    cc = c.astype(float).copy(); cc[lab < 0] = np.nan
    for reg in [0, 1]: m = lab == reg; cc[m] -= c[m].mean()
    cc = np.nan_to_num(cc); P = np.abs(np.fft.rfft(cc)) ** 2; P = P[1:]
    top = np.argsort(P)[-3:][::-1]; print('   periodogram top peaks: period(s)', np.round(6000 / (top + 1), 1), 'power / median', np.round(P[top] / np.median(P), 1))
# shock -> regime switch coupling
tj = ev.t_jump.to_numpy()
for lo, hi in [(0, 30), (30, 60)]:
    n_in = sum(((V.start > t + lo) & (V.start < t + hi)).sum() for t in tj)
    calm_frac = np.mean([not ((V.start <= t) & (V.end > t)).any() for t in tj])
    exp = len(tj) * calm_frac * (hi - lo) / 155.0
    print(f'vol entries within ({lo},{hi}) s after a jump: {n_in} (expected ≈ {exp:.1f} if independent)')
print('first onset', ev.onset.iloc[0], '; last onset', ev.onset.iloc[-1], '; 6000 - last', 6000 - ev.onset.iloc[-1])
