import numpy as np, sys
for n in sys.argv[1:]:
    d = np.load(f'base_{n}.npz'); fl, pt = d['fills'], d['path']
    t, sg, q, px, kind, turb, F, sd, dtk, qtk, age, pos, opx = fl.T
    def fut(H):
        j = np.minimum(np.searchsorted(pt[:,0], t + H), len(pt)-1); return pt[j,1]
    m1 = sg*(fut(0.1)-px); m10 = sg*(fut(5)-px)
    e = sg*(F-px)
    print('=====', n)
    m0 = (kind==0)&(turb==0)
    print('passive calm by edge vs F at fill (F before this taker impact)')
    for lo,hi in [(-9,0),(0,.1),(.1,.3),(.3,.5),(.5,.8),(.8,1.2),(1.2,99)]:
        m = m0&(e>=lo)&(e<hi)
        if m.sum(): print(f'  e[{lo},{hi}) n{m.sum():6} lots {q[m].sum():7.0f} mk.1 {(m1*q)[m].sum()/q[m].sum():+.3f} mk5 {(m10*q)[m].sum()/q[m].sum():+.3f} tot.1 {(m1*q)[m].sum():+.0f}')
    print('passive calm by time since prev market taker')
    for lo,hi in [(0,.005),(.005,.02),(.02,.05),(.05,.2),(.2,1),(1,99)]:
        m = m0&(dtk>=lo)&(dtk<hi)
        if m.sum(): print(f'  dtk[{lo},{hi}) n{m.sum():6} lots {q[m].sum():7.0f} e {(e*q)[m].sum()/q[m].sum():+.3f} mk.1 {(m1*q)[m].sum()/q[m].sum():+.3f} mk5 {(m10*q)[m].sum()/q[m].sum():+.3f} tot.1 {(m1*q)[m].sum():+.0f}')
    print('take calm by dtk')
    mt=(kind==1)&(turb==0)
    for lo,hi in [(0,.005),(.005,.02),(.02,.05),(.05,.2),(.2,1),(1,99)]:
        m = mt&(dtk>=lo)&(dtk<hi)
        if m.sum(): print(f'  dtk[{lo},{hi}) n{m.sum():6} lots {q[m].sum():7.0f} e {(e*q)[m].sum()/q[m].sum():+.3f} mk.1 {(m1*q)[m].sum()/q[m].sum():+.3f} mk5 {(m10*q)[m].sum()/q[m].sum():+.3f} tot.1 {(m1*q)[m].sum():+.0f}')
