"""Diagnostics of a smoothed S path: posterior widths, taker offset and maker gap by regime, rule violations."""
import numpy as np, pandas as pd, sys
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'v2'
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; k = d['kind']; p = d['price']; sg = d['sign']
vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
for nm, msk in [('calm', ~vol & ~sh), ('vol', vol & ~sh), ('shock', sh)]:
    print('==', nm, 'n', msk.sum(), 'post sd pct 10/50/90/99:', np.percentile(r['sd'][msk], [10, 50, 90, 99]).round(3),
          '99%% CI width median %.2f' % np.median(r['hi'][msk] - r['lo'][msk]))
    tk = msk & (k >= 4) & (k <= 5)
    x = sg[tk] * (p[tk] - m[tk])
    h = np.histogram(x, bins=[-50, -1, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 7, 10, 20, 200])[0]
    print('  taker sign*(p-S) bins [-50,-1,0,.5,1,1.5,2,2.5,3,3.5,4,5,7,10,20,200]:', h)
    mk = msk & (k <= 3)
    gap = sg[mk] * (m[mk] - p[mk])
    print('  maker gap pct .01/.1/1/50/99:', np.percentile(gap, [0.01, 0.1, 1, 50, 99]).round(3), 'gap<0.05: %d (%.2e)' % ((gap < 0.05).sum(), (gap < 0.05).mean()),
          ' hard viol (S outside CI wrt rule):', int(((sg[mk] > 0) & (r['hi'][mk] < p[mk] + 0.05) | (sg[mk] < 0) & (r['lo'][mk] > p[mk] - 0.05)).sum()))
