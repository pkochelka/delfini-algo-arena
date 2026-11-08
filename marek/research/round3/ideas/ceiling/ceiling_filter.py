"""Ceiling angle, step 5: online filter F (r3_best) vs exact S, per regime (event-weighted and time-weighted)."""
import sys, pathlib
import numpy as np, pandas as pd
H = pathlib.Path(__file__).parent
sys.path.insert(0, str(H.parents[2] / 'strategy'))
import eval3
ROOT = H.parents[5]
on, reg = eval3.load_ctx('data/round3.csv')
z = np.load(ROOT / 'data/.cache/round3_S_oracle.npz'); TI, SA, V = z['t_us'], z['S'], z['vol']
ev = np.load(sys.argv[1])
t = ev['t_us']; F = ev['F']; sd = ev['sd']; pv = ev['pv']; turb = ev['turb']
j = np.maximum(np.searchsorted(TI, t, side='right') - 1, 0)
S = SA[j]; vol = V[j]
err = F - S
dt = np.diff(np.append(t, t[-1])) * 1e-6
lab = np.array([eval3.label(x * 1e-6, on, reg) for x in t[::20]]); lab = np.repeat(lab, 20)[:len(t)]
for r in ['calm', 'volatile', 'shock']:
    m = lab == r
    w = dt[m]; e = err[m]
    ae = np.abs(e)
    print(f'{r:9s} time-wtd rmse {np.sqrt((w*e*e).sum()/w.sum()):.3f}  mean|e| {(w*ae).sum()/w.sum():.3f}  '
          f'P(|e|>0.25) {(w*(ae>0.25)).sum()/w.sum():.3f}  P(|e|>0.5) {(w*(ae>0.5)).sum()/w.sum():.3f}  P(|e|>1) {(w*(ae>1)).sum()/w.sum():.3f}  '
          f'median sd {np.median(sd[m]):.3f}  ratio rmse/sd {np.sqrt((w*e*e).sum()/w.sum())/np.sqrt((w*sd[m]**2).sum()/w.sum()):.2f}')
m = lab != 'shock'
print('regime detection (non-shock): P(turb | vol) %.3f  P(turb | calm) %.3f  P(pv>0.5|vol) %.3f  P(pv>0.5|calm) %.3f' % (
    (dt*(turb>0.5)*(vol==1)*m).sum()/(dt*(vol==1)*m).sum(), (dt*(turb>0.5)*(vol==0)*m).sum()/(dt*(vol==0)*m).sum(),
    (dt*(pv>0.5)*(vol==1)*m).sum()/(dt*(vol==1)*m).sum(), (dt*(pv>0.5)*(vol==0)*m).sum()/(dt*(vol==0)*m).sum()))
# shock windows: error right after the jump
sh = []
for o in on:
    a, b = int((o + 2.9) * 1e6), int((o + 3.5) * 1e6)
    k = (t >= a) & (t < b)
    if k.any():
        sh.append((o, np.abs(err[k]).max(), np.abs(err[k][-1])))
print('shock jump window max|e| median %.2f, |e| at +3.5 s median %.2f' % (np.median([x[1] for x in sh]), np.median([x[2] for x in sh])))
