import sys, numpy as np
for fn in sys.argv[1:]:
    z = np.load(fn); fl, pt, tl = z['fills'], z['path'], z['tlog']
    t, F, tu, pos, mid = pt.T
    print('==', fn, 'pnl', float(z['pnl']))
    for r in (0, 1):
        m = tu == r; a = np.abs(pos[m])
        print(f' regime {r}: share {m.mean():.3f} |pos| med {np.median(a):.0f} q25 {np.quantile(a,.25):.0f} q75 {np.quantile(a,.75):.0f} at>=95 {np.mean(a>=95):.2f}')
    # markouts using path F
    def Fat(x): return np.interp(x, t, F)
    ft, sg, q, px, k, ftu, fF, fpos = fl.T
    for H in (1, 10, 60):
        mk = sg * q * (Fat(ft + H) - px)
        out = []
        for kk in (0, 1, 2):
            for r in (0, 1):
                m = (k == kk) & (ftu == r)
                out.append(f'k{int(kk)}r{r}: lots {int(q[m].sum())} mk {mk[m].sum()/1e3:.0f}k ({mk[m].sum()/max(q[m].sum(),1):.2f}/lot)')
        print(f' H{H}: ' + ' | '.join(out))
    # increasing vs reducing fills
    pre = fpos - 0  # pos is recorded before update? fills logged before super -> pos before fill
    inc = np.sign(sg) == np.sign(pre) ; inc |= pre == 0
    mk = sg * q * (Fat(ft + 10) - px)
    for kk in (0, 1):
        for lab, m in (('inc', inc), ('red', ~inc)):
            mm = m & (k == kk)
            print(f'  kind {int(kk)} {lab}: lots {int(q[mm].sum())} mk10 {mk[mm].sum()/1e3:.0f}k {mk[mm].sum()/max(q[mm].sum(),1):.2f}/lot')
    # take truncation: tlog (t, side, L, room, edge, turb, pos); dedupe opportunities: first log after 0.2s gap per side
    if len(tl):
        tt, sd_, L, room, edge, tturb, tpos = tl.T
        keep = np.ones(len(tt), bool); last = {1: -9, -1: -9}
        for i in range(len(tt)):
            s = sd_[i]; keep[i] = tt[i] - last[s] > 0.2; last[s] = tt[i]
        L, room, edge, tturb, tpos, tt2 = L[keep], room[keep], edge[keep], tturb[keep], tpos[keep], tt[keep]
        got = np.minimum(L, np.maximum(room, 0)); flat = np.minimum(L, 100)
        print(f' takes opp {keep.sum()}: L {L.sum():.0f} got {got.sum():.0f} if-flat {flat.sum():.0f} ; edge-weighted got {(got*edge).sum()/1e3:.1f}k flat {(flat*edge).sum()/1e3:.1f}k full {(L*edge).sum()/1e3:.1f}k; turb share {tturb.mean():.2f}')
