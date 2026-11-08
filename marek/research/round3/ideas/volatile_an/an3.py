exec(open('an2.py').read())
# fills by F-edge at send (F - fillprice for buys)
send = {s[1]: s for s in L['send']}
rows = []
for f in fl:
    s = send.get(f[1])
    if s is None or not s[5] or reg(f[0]) != 'vol': continue
    rows.append((f[0], f[2], f[3], f[4], s[6], s[7], s[0]))
a = np.array(rows, float)
t, sd_, p, qq, F, pos0, ts = a.T
eF = sd_ * (F - p)
bins = [-9, 0.5, 1, 1.5, 2, 3, 4, 6, 99]
print('vol IOC fills by F-edge of the fill price: qty, markout vs S at h')
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (eF >= lo) & (eF < hi)
    if not m.any(): continue
    out = ['[%g,%g)' % (lo, hi), int(qq[m].sum())]
    for h in (0, 1, 5, 15, 60):
        out.append('%.2f' % np.average(sd_[m] * (S(t[m] + h) - p[m]), weights=qq[m]))
    print(*out)
# first-sight episodes (pinned vs room), by F-edge at touch
print('episodes by F-edge of touch: n, pinned share, avail qty mean, touch markout at 1,5,15,60')
eT = sd * (Fv - px)
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (eT >= lo) & (eT < hi)
    if not m.any(): continue
    out = ['[%g,%g)' % (lo, hi), m.sum(), '%.2f' % pinned[m].mean(), int(q[m].mean())]
    for h in (1, 5, 15, 60):
        out.append('%.2f/%.2f' % (np.mean(sd[m & pinned] * (S(t0[m & pinned] + h) - px[m & pinned])) if (m&pinned).any() else 0,
                                  np.mean(sd[m & ~pinned] * (S(t0[m & ~pinned] + h) - px[m & ~pinned])) if (m&~pinned).any() else 0))
    print(*out)
