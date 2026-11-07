"""Per-shock PnL by phase for given runs: pre (-1..+2.5 s), jump (+2.5..+5), burst (+5..+20)."""
import csv, sys
R = 'marek/research/round2/flow/'
ev = list(csv.DictReader(open(R + 'shock_events.csv')))
moves = {}
try:
    for r in csv.DictReader(open(R + 'shock_event_moves.csv')):
        moves[round(float(r[list(r)[0]]), 1)] = r
except Exception:
    pass
for name in sys.argv[1:]:
    rows = list(csv.DictReader(open(f'marek/logs/r2/round2/{name}/pnl.csv')))
    pn = [float(r['pnl']) for r in rows]
    f = lambda t: pn[min(int(round(t)), len(pn) - 1)]
    tot = [0, 0, 0]
    per = []
    for e in ev:
        o = float(e['onset'])
        a, b, c = f(o + 2.5) - f(o - 1), f(o + 5) - f(o + 2.5), f(o + 20) - f(o + 5)
        tot[0] += a; tot[1] += b; tot[2] += c
        per.append(round(a + b + c))
    print(f'{name:<10} pre={tot[0]:>8.0f} jump={tot[1]:>8.0f} burst={tot[2]:>8.0f}  per-shock: {per}')
