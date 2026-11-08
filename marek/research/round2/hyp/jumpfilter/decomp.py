"""Volatile error decomposition of FV on synth: samples classified by time since last residual jump / last fill-noise event.
uv run python decomp.py STRAT SEED [k=v ...]"""
import sys, ast, json, numpy as np
sys.path.insert(0, 'marek/research/round2/hyp/filter')
import off
from pathlib import Path
from analysis.cache import load_tape
strat, seed = sys.argv[1], int(sys.argv[2])
over = {k: ast.literal_eval(v) for k, v in (a.split('=', 1) for a in sys.argv[3:])}
S, St, tr = off.run(strat, seed, over)
t, F, sd, pv, exc = S.T; e = F - St
SP = np.load(f'marek/research/round2/hyp/filter/S_seed{seed}.npy')
tp = load_tape(f'data/synth/r2_v2_seed{seed}.csv', verbose=False)
o = tp.orders; oo = o[o.ioc & (o.filled > 0)]
ft = np.sort(tp.events.t_us.to_numpy()[oo.row.to_numpy()] / 1e6)
fq = tp.events.t_us.to_numpy()  # unused
dS = np.diff(SP[:, 1]); tS = SP[1:, 0]
# an S change at a fill time (within 1 us) = impact; else residual jump / diffusion / shock
j = np.searchsorted(ft, tS); near = np.zeros(len(tS), bool)
for k in (j - 1, j):
    kk = np.clip(k, 0, len(ft) - 1); near |= np.abs(ft[kk] - tS) < 2e-6
vol = np.zeros(len(t), bool); volS = np.zeros(len(tS), bool)
for a, b, v in tr['segs']:
    if v: vol |= (t >= a) & (t < b); volS |= (tS >= a) & (tS < b)
pre = np.zeros(len(t), bool); since = np.full(len(t), 1e9)
for on in tr['onsets']:
    pre |= (t >= on) & (t < on + 3); m = t >= on + 3; since[m] = np.minimum(since[m], t[m] - on - 3)
vv = vol & ~pre & (since > 5)
jm = ~near & volS & (np.abs(dS) < 20)
print('vol S changes: impact', (near & volS).sum(), 'non-fill', jm.sum(), 'non-fill |dS| mean', np.abs(dS[jm]).mean(),
      'rate/s', jm.sum() / sum(b - a for a, b, v in tr['segs'] if v))
imp = near & volS
print('impact |dS| quantiles', np.quantile(np.abs(dS[imp]), [.5, .9, .99]))
def last(evt):
    i = np.searchsorted(evt, t, 'right') - 1
    return np.where(i >= 0, t - evt[np.clip(i, 0, None)], 1e9)
sj = last(tS[jm]); sj_big = last(tS[jm & (np.abs(dS) > 1)])
big_imp = imp & (np.abs(dS) > 0.6); sbi = last(tS[big_imp])
tot = (e[vv] ** 2).sum()
def rep(name, m):
    m = m & vv
    x = e[m]
    print(f'{name:34s} frac_t={m.sum()/vv.sum():.3f} frac_se={(x**2).sum()/tot:.3f} rmse={np.sqrt((x**2).mean()):.3f} '
          f'sd_med={np.median(sd[m]):.3f} z_rms={np.sqrt(((x/np.maximum(sd[m],1e-3))**2).mean()):.2f}')
rep('all vol', vv)
rep('<1s after resid jump |dS|>1', sj_big < 1)
rep('<1s after any resid jump', sj < 1)
rep('<0.5s after big impact (>0.6)', (sbi < .5) & (sj >= 1))
rep('rest', (sj >= 1) & (sbi >= .5))
print('vol rmse', np.sqrt((e[vv]**2).mean()), 'vol z', np.sqrt(((e[vv]/np.maximum(sd[vv],1e-3))**2).mean()))
ent = np.full(len(t), 1e9)
for a, b, v in tr['segs']:
    if v:
        m = (t >= a) & (t < b); ent[m] = t[m] - a
for a, b in [(0, 1), (1, 3), (3, 10), (10, 1e8)]:
    rep(f'since vol entry {a}-{b}', (ent >= a) & (ent < b))
for a, b in [(0, .5), (.5, .9), (.9, 1.01)]:
    rep(f'pv {a}-{b}', (pv >= a) & (pv < b))
x = np.abs(e[vv]); o_ = np.argsort(-x); c = np.cumsum(x[o_] ** 2) / tot
print('top 4% samples carry', c[int(.04 * len(x))], ' |e|>1 frac', (x > 1).mean(), 'share', (x[x > 1] ** 2).sum() / tot)
