import sys, json, numpy as np
from analysis.cache import load_tape
seed = int(sys.argv[1])
SP = np.load(f'marek/research/round2/hyp/filter/S_seed{seed}.npy'); tr = json.load(open(f'data/synth/r2_v2_seed{seed}.truth.json'))
tp = load_tape(f'data/synth/r2_v2_seed{seed}.csv', verbose=False)
o = tp.orders; oo = o[o.ioc & (o.filled > 0)]
ev = tp.events
tt = ev.t_us.to_numpy()[oo.row.to_numpy()] / 1e6; sg = np.where(ev.side.to_numpy()[oo.row.to_numpy()] == 'B', 1, -1); q = oo.filled.to_numpy()
i = np.searchsorted(SP[:, 0], tt, 'right') - 1  # S row at/after fill
ok = (i >= 1) & (np.abs(SP[np.clip(i,0,None), 0] - tt) < 2e-6)
d = SP[i, 1] - SP[i - 1, 1]
vol = np.zeros(len(tt), bool)
for a, b, v in tr['segs']:
    if v: vol |= (tt >= a) & (tt < b)
for name, m, c, e in [('vol', vol, .0741, .606), ('calm', ~vol, .0142, .625)]:
    m = m & ok
    r = d[m] - sg[m] * c * q[m] ** e
    print(name, m.sum(), 'resid sd', r.std(), 'mean', r.mean(), 'kurt', ((r - r.mean())**4).mean() / r.var()**2, 'q med', np.median(q[m]))
    for lo, hi in [(1, 10), (10, 100), (100, 1e9)]:
        mm = (q[m] >= lo) & (q[m] < hi); print('   q', lo, hi, mm.sum(), 'resid sd', r[mm].std(), 'mean*sign', (r[mm] * sg[m][mm]).mean())
print('unmatched fills', (~ok).sum())
