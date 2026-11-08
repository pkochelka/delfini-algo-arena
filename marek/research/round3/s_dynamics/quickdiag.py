import sys, numpy as np, prep
tag = sys.argv[1]
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
k = d['kind']; p = d['price']; sg = d['sign']; m = r['mean']; vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
print('ll', r['ll'].sum(), 'sd med', np.median(r['sd']))
for nm, msk in (('calm', ~vol & ~sh), ('vol', vol & ~sh)):
    for kk in (0, 2):
        s = msk & ((k == kk) | (k == kk + 1))
        g = sg[s] * (m[s] - p[s])
        print(nm, 'kind', kk, s.sum(), 'gap q', np.percentile(g, [0.1, 1, 5, 25, 50, 75, 95]).round(3), 'll/ev %.3f' % r['ll'][s].mean())
    s = msk & (k >= 4)
    x = sg[s] * (p[s] - m[s])
    print(nm, 'taker', s.sum(), 'x q', np.percentile(x, [1, 5, 25, 50, 75, 95, 99]).round(2), 'll/ev %.3f' % r['ll'][s].mean())
    print('   taker x hist', np.histogram(x, bins=np.arange(-1, 6.01, 0.25))[0])
