exec(open('an1.py').read().split('samp = np')[0])
send = {s[1]: s for s in L['send']}
sm = np.array([(x[0], x[6], x[7]) for x in L['samp']])
st, sbb, sba = sm.T
rows = []
for f in L['fill']:
    s = send.get(f[1])
    if s is None or not s[5] or reg(f[0]) != 'vol': continue
    rows.append((f[0], f[2], f[3], f[4], s[6]))
t, sg, p, q, F = np.array(rows, float).T
def touch(x, buy):
    i = np.clip(np.searchsorted(st, x, side='right') - 1, 0, len(st) - 1)
    return np.where(buy, sbb[i], sba[i])
print('vol IOC fills: exit-touch (bid for buys, ask for sells) minus fill price, qty-weighted, per lot; and S-edge')
for d in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
    ex = sg * (touch(t + d, sg > 0) - p)
    es = sg * (S(t + d + 0.02) - p)
    print('  +%.2fs: exit %+.2f  S-edge %+.2f  share exit>0 %.2f' % (d, np.average(ex, weights=q), np.average(es, weights=q), np.average(ex > 0, weights=q)))
