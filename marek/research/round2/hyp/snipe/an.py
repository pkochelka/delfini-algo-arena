"""Per-shock attribution: an.py TAPE BASE_TAG X_TAG  (reads <tag>_<tape>.pkl)."""
import sys, pickle, json, bisect, pathlib
import numpy as np
D = pathlib.Path(__file__).parent
ROOT = D.parents[4]
tape, bt, xt = sys.argv[1], sys.argv[2], sys.argv[3]
B = pickle.load(open(D / f'{bt}_{tape}.pkl', 'rb'))
X = pickle.load(open(D / f'{xt}_{tape}.pkl', 'rb'))
if tape == 'r2':
    import csv
    ons = [float(r['onset']) for r in csv.DictReader(open(ROOT / 'marek/research/round2/flow/shock_events.csv'))]
    jumps = None
else:
    tr = json.load(open(ROOT / f'data/synth/r2_v2_seed{tape[1:]}.truth.json'))
    ons = tr['onsets']
    jumps = tr['jumps']

def eqf(R):
    e = np.array(R['eq'])
    return e[:, 0], e[:, 1] + e[:, 2] * e[:, 3], e[:, 2]

tb, eb, pb = eqf(B)
tx, ex, px = eqf(X)
def at(t, ts, es):
    i = min(bisect.bisect_left(ts, t), len(ts) - 1)
    return es[i]
fx = X['fills']
br = X['breaches']
tot = 0
print(f"{'onset':>8} {'jump':>6} {'dEq':>8} {'pen':>7} {'net':>8} | {'ssQ':>5} {'ssPx-F':>6} {'xuQ':>5} {'xuPx-F':>6} {'brms':>6} {'maxpos':>6} {'bpos':>5}")
for k, o in enumerate(ons):
    if isinstance(o, (list, tuple)):
        o = o[0]
    a, b = o + 2.0, o + 10.0
    d = (at(b, tx, ex) - at(a, tx, ex)) - (at(b, tb, eb) - at(a, tb, eb))
    pen = sum(ms * 10 for (t0, ms) in br if a <= t0 < b)
    F_end = at(o + 6, tx, np.array(X['eq'])[:, 4])
    ss = [(q, p) for (t, q, p, tag, lim) in fx if a <= t < b and tag == 'ss']
    xu = [(q, p) for (t, q, p, tag, lim) in fx if a <= t < b and tag == 'xu']
    def summ(L):
        Q = sum(q for q, _ in L)
        if not Q:
            return 0, 0.0
        return Q, sum(q * p for q, p in L) / Q - F_end
    sq, sp = summ(ss)
    xq, xp = summ(xu)
    i0, i1 = bisect.bisect_left(tx, a), bisect.bisect_left(tx, b)
    mp = int(np.abs(px[i0:i1]).max()) if i1 > i0 else 0
    i0, i1 = bisect.bisect_left(tb, a), bisect.bisect_left(tb, b)
    bp = int(np.abs(pb[i0:i1]).max()) if i1 > i0 else 0
    brms = sum(ms for (t0, ms) in br if a <= t0 < b)
    j = ''
    if jumps:
        j = round(jumps[k], 1)
    tot += d - pen
    print(f"{o:8.1f} {str(j)[:6]:>6} {d:8.0f} {pen:7.0f} {d - pen:8.0f} | {sq:5d} {sp:6.1f} {xq:5d} {xp:6.1f} {brms:6.0f} {mp:6d} {bp:5d}")
print('sum shock windows', round(tot), 'total delta', X['res']['pnl'] - B['res']['pnl'], 'pen', X['res'].get('breach_penalty'))
