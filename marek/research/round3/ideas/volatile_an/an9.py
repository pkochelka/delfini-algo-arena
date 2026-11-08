exec(open('an1.py').read().split('samp = np')[0])
send = {s[1]: s for s in L['send']}
sf = np.load('sfilt.npz')
def SF(t): return sf['f'][np.clip(np.searchsorted(sf['t'], t, side='right') - 1, 0, len(sf['t']) - 1)]
rows = []
for f in L['fill']:
    s = send.get(f[1])
    if s is None or not s[5] or reg(f[0]) != 'vol': continue
    rows.append((f[0], f[2], f[3], f[4], s[6], s[0]))
t, sg, p, q, F, ts = np.array(rows, float).T
for lag in (0.0, 0.02, 0.05, 0.2):
    eS = sg * (S(ts + lag) - p)
    print('lag %.2f: S-edge<0 qty %d of %d (%.2f), sum edge on them %.0f; sum all %.0f' % (lag, q[eS < 0].sum(), q.sum(), q[eS < 0].sum() / q.sum(), (eS * q)[eS < 0].sum(), (eS * q).sum()))
eF = sg * (F - p); eS = sg * (S(ts + 0.05) - p)
err = F - S(ts + 0.05)
print('F-S at vol sends: sd %.2f; |err|>1 share of qty %.3f' % (err.std(), q[abs(err) > 1].sum() / q.sum()))
eSF = sg * (SF(ts) - p)
print('fills that s_filt (causal) would reject (edge<0.3): qty %d, their S-edge sum %.0f' % (q[eSF < 0.3].sum(), (eS * q)[eSF < 0.3].sum()))
