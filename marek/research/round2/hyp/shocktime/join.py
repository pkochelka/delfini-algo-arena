import pickle, csv, sys, bisect, pathlib
import numpy as np
D = pathlib.Path(__file__).parent; ROOT = D.parents[4]
X = pickle.load(open(D / sys.argv[1], 'rb'))
ons = [float(r['onset']) for r in csv.DictReader(open(ROOT/'marek/research/round2/flow/shock_events.csv'))]
st = [l.split() for l in open(D/'stale.txt').read().splitlines()[1:]]
e = np.array(X['eq']); te = e[:,0]; eqv = e[:,1]+e[:,2]*e[:,3]
def at(t): return eqv[min(bisect.bisect_left(te,t), len(te)-1)]
tot=0
print(f"{'onset':>8} {'J':>6} {'L0':>5} {'V0':>6} {'V20':>6} | {'trig_ms':>7} {'dir':>3} {'ssQ':>4} {'ssEdge':>6} {'pnl':>7} {'pen':>6}")
for o, s in zip(ons, st):
    T = o+3.0
    tr = [x for x in X['trig'] if T-0.5 <= x[0] <= T+5]
    a,b = o+2, o+10
    pen = sum(ms*10 for (t0,ms) in X['breaches'] if a<=t0<b)
    fl = [(q,p) for (t,q,p,tag,lim) in X['fills'] if a<=t<b and tag=='ss']
    Q = sum(q for q,_ in fl); lvl = (float(s[1]) + 0)  # J relative to pre-mid
    avg = sum(q*p for q,p in fl)/Q if Q else 0
    d = at(b)-at(a)-pen; tot += d
    tm = f"{(tr[0][0]-T)*1e3:7.1f}" if tr else '   none'
    dr = tr[0][1] if tr else 0
    print(f"{o:8.2f} {s[1]:>6} {s[3]:>5} {s[4]:>6} {s[12]:>6} | {tm} {dr:3d} {Q:4d} {'':>6} {d:7.0f} {pen:6.0f}")
print('total', round(tot))
