import numpy as np, sys
for n in sys.argv[1:]:
    d = np.load(f'base_{n}.npz'); fl, pt = d['fills'], d['path']
    t, sg, q, px, kind, turb, F, sd, dtk, qtk, age, pos, opx = fl.T
    def fut(H):
        j = np.minimum(np.searchsorted(pt[:,0], t + H), len(pt)-1); return pt[j,1]
    print('=====', n, 'fills', len(fl), 'lots', q.sum())
    mk = {H: sg*(fut(H)-px)*q for H in (0.1, 1, 10, 60)}
    names = {0:'pass',1:'take',2:'ss_pass',3:'snipe'}
    for k in (0,1,3):
        for tb in (0,1):
            m = (kind==k)&(turb==tb)
            if m.sum()==0: continue
            print(f'{names[k]:6} turb{tb} lots {q[m].sum():7.0f}', ' '.join(f'mk{H}: {mk[H][m].sum()/q[m].sum():+.3f}/{mk[H][m].sum():+9.0f}' for H in mk))
    # passive calm by time since last taker
    m0 = (kind==0)&(turb==0)
    print('passive calm by edge at fill (px vs F at fill)')
    e = sg*(F-px)
    # by own fill size bucket
    for lo,hi in [(0,5),(5,20),(20,50),(50,101)]:
        m = m0&(q>=lo)&(q<hi)
        if m.sum(): print(f'  q[{lo},{hi}) n{m.sum():6} lots {q[m].sum():7.0f} mk1 {mk[1][m].sum()/q[m].sum():+.3f} mk10 {mk[10][m].sum()/q[m].sum():+.3f} tot10 {mk[10][m].sum():+.0f}')
    print('passive calm by order age')
    for lo,hi in [(0,.02),(.02,.1),(.1,.5),(.5,2),(2,1e9)]:
        m = m0&(age>=lo)&(age<hi)
        if m.sum(): print(f'  age[{lo},{hi}) n{m.sum():6} lots {q[m].sum():7.0f} mk1 {mk[1][m].sum()/q[m].sum():+.3f} mk10 {mk[10][m].sum()/q[m].sum():+.3f} tot10 {mk[10][m].sum():+.0f}')
    print('passive calm by |pos| before')
    pb = pos - sg*q
    for lo,hi in [(0,20),(20,50),(50,80),(80,101)]:
        m = m0&(np.abs(pb)>=lo)&(np.abs(pb)<hi)
        if m.sum(): print(f'  |pos|[{lo},{hi}) lots {q[m].sum():7.0f} mk10 {mk[10][m].sum()/q[m].sum():+.3f} tot10 {mk[10][m].sum():+.0f}')
    print('take calm by edge at decision (approx F-px at fill)')
    m1 = (kind==1)&(turb==0)
    for lo,hi in [(-9,0),(0,.3),(.3,.6),(.6,1),(1,2),(2,99)]:
        m = m1&(e>=lo)&(e<hi)
        if m.sum(): print(f'  e[{lo},{hi}) n{m.sum():6} lots {q[m].sum():7.0f} mk0.1 {mk[.1][m].sum()/q[m].sum():+.3f} mk10 {mk[10][m].sum()/q[m].sum():+.3f} tot10 {mk[10][m].sum():+.0f}')
    # pnl curve worst seconds
    v = pt[:,4]; tt = pt[:,0]
    sec = np.floor(tt).astype(int); last = np.r_[np.nonzero(np.diff(sec))[0], len(sec)-1]
    vs = v[last]; ds = np.diff(vs); ss = sec[last][1:]
    o = np.argsort(ds)[:15]
    print('worst seconds', [(int(ss[i]), round(ds[i])) for i in o])
    print('sum of worst 30 s', ds[np.argsort(ds)[:30]].sum(), 'total', vs[-1])
