import numpy as np, sys
for n in sys.argv[1:]:
    d = np.load(f'base_{n}.npz'); fl, pt = d['fills'], d['path']
    t, sg, q, px, kind, turb, F, sd, dtk, qtk, age, pos, opx, ah = fl.T
    j = np.minimum(np.searchsorted(pt[:,0], t + 0.1), len(pt)-1); m1 = sg*(pt[j,1]-px)
    e = sg*(F-px)
    print('=====', n)
    m0 = (kind==0)&(turb==0)
    for lo,hi in [(-1,0),(0,1),(1,20),(20,50),(50,100),(100,200),(200,400),(400,800),(800,1e9)]:
        m = m0&(ah>=lo)&(ah<hi)
        if m.sum(): print(f'  ahead[{lo},{hi}) n{m.sum():6} lots {q[m].sum():7.0f} e {(e*q)[m].sum()/q[m].sum():+.3f} mk.1 {(m1*q)[m].sum()/q[m].sum():+.3f} AS {((e-m1)*q)[m].sum()/q[m].sum():+.3f} tot.1 {(m1*q)[m].sum():+.0f}')
    print(' by e and ahead<50 / >=50')
    for lo,hi in [(0,.3),(.3,.5),(.5,.8),(.8,1.2),(1.2,99)]:
        out=[]
        for a0,a1 in [(-1,50),(50,1e9)]:
            m = m0&(e>=lo)&(e<hi)&(ah>=a0)&(ah<a1)
            out.append(f'lots {q[m].sum():7.0f} mk.1 {(m1*q)[m].sum()/max(1,q[m].sum()):+.3f}')
        print(f'  e[{lo},{hi})', ' | '.join(out))
