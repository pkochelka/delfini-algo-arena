import pickle, numpy as np, csv, sys, bisect
L = pickle.load(open(sys.argv[1], 'rb'))
R3 = '/Users/marek/Documents/delfini-algo-arena/marek/research/round3/'
sp = np.load('spath.npz'); St, Ss = sp['t'], sp['s']
segs = [(float(r['start']), float(r['end'])) for r in csv.DictReader(open(R3 + 's_dynamics/vol_segments.csv'))]
on = [float(r['onset']) for r in csv.DictReader(open(R3 + 'flow/shock_events.csv'))]
def reg(t):
    i = bisect.bisect_right(on, t + 1.0) - 1
    if i >= 0 and on[i] - 1.0 <= t < on[i] + 20.0: return 'shock'
    for a, b in segs:
        if a <= t < b: return 'vol'
    return 'calm'
def S(t):
    i = np.searchsorted(St, t, side='right') - 1
    return Ss[np.clip(i, 0, len(Ss) - 1)]
samp = np.array([(x[0], x[1], x[2], x[3]) for x in L['samp']])
rg = np.array([reg(t) for t in samp[:, 0]])
for g in ('calm', 'vol', 'shock'):
    m = rg == g
    p = samp[m, 1]
    print(g, 'samples', m.sum(), 'time s', m.sum() * 0.02, '|pos|>=100: %.2f  >=90: %.2f  mean|pos| %.1f  turb %.2f' % ((abs(p) >= 100).mean(), (abs(p) >= 90).mean(), abs(p).mean(), samp[m, 2].mean()))
# fills
fl = L['fill']
send = {s[1]: s for s in L['send']}
print('fills', len(fl))
ft = np.array([f[0] for f in fl]); fs = np.array([f[2] for f in fl]); fp = np.array([f[3] for f in fl], float); fq = np.array([f[4] for f in fl])
ioc = np.array([send[f[1]][5] if f[1] in send else True for f in fl])
fr = np.array([reg(t) for t in ft])
for g in ('calm', 'vol', 'shock'):
    for kind in (True, False):
        m = (fr == g) & (ioc == kind)
        if not m.any(): continue
        out = [g, 'IOC' if kind else 'LMT', 'n', m.sum(), 'qty', fq[m].sum()]
        for h in (0, 0.2, 1, 5, 15, 30, 60, 120):
            e = fs[m] * (S(ft[m] + h) - fp[m])
            out += ['%g:%.2f' % (h, (e * fq[m]).sum() / fq[m].sum())]
        print(*out)
