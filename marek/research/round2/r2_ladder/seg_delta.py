"""Paired per-volatile-segment PnL deltas of runs vs a reference run (HMM segments, +5 s tail)."""
import csv, sys, math
R = 'marek/research/round2/flow/'
segs = [(float(r['start']), float(r['end'])) for r in csv.DictReader(open(R + 'taker_regimes.csv')) if r['high'] == 'True']
def load(n):
    pn = [float(r['pnl']) for r in csv.DictReader(open(f'marek/logs/r2/round2/{n}/pnl.csv'))]
    return lambda t: pn[min(int(t), len(pn) - 1)]
ref = load(sys.argv[1])
for n in sys.argv[2:]:
    f = load(n)
    d = [(f(b + 5) - f(a)) - (ref(b + 5) - ref(a)) for a, b in segs]
    m = sum(d) / len(d); sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    print(f'{n:<16} sum={sum(d):>8.0f} t={m / sd * math.sqrt(len(d)):5.2f} wins={sum(x > 0 for x in d)}/{len(d)} ' + ' '.join(f'{x:.0f}' for x in d))
