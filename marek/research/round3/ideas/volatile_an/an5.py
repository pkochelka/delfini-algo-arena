import numpy as np, csv
R3 = '/Users/marek/Documents/delfini-algo-arena/marek/research/round3/'
sp = np.load('spath.npz'); St, Ss = sp['t'], sp['s']
segs = [(float(r['start']), float(r['end'])) for r in csv.DictReader(open(R3 + 's_dynamics/vol_segments.csv'))]
on = [float(r['onset']) for r in csv.DictReader(open(R3 + 'flow/shock_events.csv'))]
g = np.arange(0, 6000, 0.05)
S = Ss[np.clip(np.searchsorted(St, g, side='right') - 1, 0, len(Ss) - 1)]
vol = np.zeros(len(g), bool)
for a, b in segs: vol |= (g >= a + 2) & (g < b - 2)
for o in on: vol &= ~((g >= o - 1) & (g < o + 25))
# transient stock proxy: Y = sum of past dS * exp(-age/T), T=13.6 (permanent share ignored: scale-free regression)
dS = np.diff(S, prepend=S[0])
Y = np.zeros(len(g)); a = np.exp(-0.05 / 13.6)
for i in range(1, len(g)): Y[i] = Y[i - 1] * a + dS[i]
for hp in (0.5, 2, 10):
    k = int(hp / 0.05)
    past = S - np.roll(S, k)
    for hf in (0.5, 1, 2, 5, 15):
        m = int(hf / 0.05)
        fut = np.roll(S, -m) - S
        ok = vol & (np.arange(len(g)) > k) & (np.arange(len(g)) < len(g) - m)
        # require the whole future window inside vol
        ok &= np.roll(vol, -m)
        b = np.polyfit(past[ok], fut[ok], 1)[0]
        c = np.corrcoef(past[ok], fut[ok])[0, 1]
        print('past %4.1fs -> fut %4.1fs  beta %+.3f corr %+.3f' % (hp, hf, b, c))
for hf in (1, 2, 5, 15):
    m = int(hf / 0.05); fut = np.roll(S, -m) - S
    ok = vol & np.roll(vol, -m)
    b = np.polyfit(Y[ok], fut[ok], 1)[0]; c = np.corrcoef(Y[ok], fut[ok])[0, 1]
    print('Y(T=13.6) -> fut %4.1fs beta %+.3f corr %+.3f  sd(Y) %.2f' % (hf, b, c, Y[ok].std()))
