import numpy as np, sys
SP = np.load('marek/research/round2/hyp/filter/S_seed1.npy')
def S(t): return SP[np.searchsorted(SP[:, 0], t, 'right') - 1, 1]
def cat(r):
    t, sd, px, q, ioc, snipe, pv, sg, burst = r
    if snipe == 1: return 'snipe'
    reg = 'shock' if sg == 1 else ('vol' if pv == 1 else ('burst' if burst == 1 else 'calm'))
    return ('take_' if ioc == 1 else 'pass_') + reg
for name in sys.argv[1:]:
    A = np.load(f'marek/research/round2/hyp/filter/fills_{name}_s1.npy')
    print(name, 'fills', len(A), 'last t', A[-1, 0].round(0), 'qty', A[:, 3].sum())
    e0 = (S(A[:, 0]) - A[:, 2]) * A[:, 1] * A[:, 3]
    e10 = (S(A[:, 0] + 10) - A[:, 2]) * A[:, 1] * A[:, 3]
    cs = np.array([cat(r) for r in A])
    for c in sorted(set(cs)):
        m = cs == c
        print(f'  {c:12s} qty={A[m,3].sum():8.0f} edge0={e0[m].sum():9.0f} mark10={e10[m].sum():9.0f} per_unit0={e0[m].sum()/A[m,3].sum():.3f}')
    print(f'  TOTAL edge0={e0.sum():.0f} mark10={e10.sum():.0f}')
