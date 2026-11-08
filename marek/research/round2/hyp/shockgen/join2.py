"""Per-shock table: uv run python join2.py TAPE PKL [--q]"""
import pickle, json, sys, bisect, pathlib
import numpy as np
D = pathlib.Path(__file__).parent
tape, pk = sys.argv[1], sys.argv[2]; quiet = '--q' in sys.argv
X = pickle.load(open(D / pk, 'rb')); S = json.load(open(D / f'stale_{tape}.json'))
e = np.array(X['eq']); te = e[:, 0]; eqv = e[:, 1] + e[:, 2] * e[:, 3]
def at(t): return eqv[min(bisect.bisect_left(te, t), len(te) - 1)]
def pos_at(t): return e[max(bisect.bisect_left(te, t) - 1, 0), 2]
cat = dict(none=0, wrong=0, late=0, ok=0); agg = dict(V5=0, cap=0, xu=0, pnl=0, pen=0, n=0)
catv = dict(none=0, wrong=0, late=0, ok=0); cnt = dict(none=0, wrong=0, late=0, ok=0)
rows = []
if not quiet:
    print(f"{'onset':>8} {'J':>6} {'L5':>5} {'V5':>6} {'V10':>6} {'V20':>6} | {'trig':>6} {'ok':>2} {'pos0':>5} {'ssQ':>5} {'capt':>6} {'xuQ':>5} {'xuC':>6} {'pnl':>7} {'pen':>6} cls")
for s in S:
    o = s['onset']; T = o + 3.0; d = s['d']; lvl = s['lvl']
    tr = [x for x in X['trig'] if T - 0.5 <= x[0] <= T + 5]
    a, b = o + 2, o + 10
    pen = sum(ms * 10 for (t0, ms) in X['breaches'] if a <= t0 < b)
    ss = [(q, p) for (t, q, p, tag, lim) in X['fills'] if T - 0.5 <= t < T + 5 and tag == 'ss']
    xu = [(q, p) for (t, q, p, tag, lim) in X['fills'] if T - 0.5 <= t < T + 5 and tag == 'xu']
    Q = sum(q for q, _ in ss); capt = sum(q * (lvl - p) for q, p in ss)
    xQ = sum(q for q, _ in xu); xC = sum(q * (lvl - p) for q, p in xu)
    pnl = at(b) - at(a) - pen
    L5, V5 = s['LV']['5']; V10 = s['LV']['10'][1]; V20 = s['LV']['20'][1]
    tm = (tr[0][0] - T) * 1e3 if tr else None
    dr = tr[0][1] if tr else 0
    p0 = pos_at(tr[0][0]) if tr else 0
    cls = 'none' if not tr else ('wrong' if dr != d else ('late' if tm > 20 else 'ok'))
    big = abs(s['J']) >= 100
    if not big:
        cnt[cls] += 1; catv[cls] += V5; agg['V5'] += V5; agg['cap'] += capt; agg['xu'] += xC; agg['pnl'] += pnl; agg['pen'] += pen; agg['n'] += 1
    rows.append(dict(onset=o, J=s['J'], L5=L5, V5=V5, tm=tm, cls=cls, Q=Q, capt=capt, pnl=pnl, pen=pen, p0=p0))
    if not quiet:
        print(f"{o:8.2f} {s['J']:6.1f} {L5:5d} {V5:6.0f} {V10:6.0f} {V20:6.0f} | {tm if tm is not None else -1:6.1f} {int(dr==d):2d} {p0:5.0f} {Q:5d} {capt:6.0f} {xQ:5d} {xC:6.0f} {pnl:7.0f} {pen:6.0f} {cls}{' BIG' if big else ''}")
print(tape, 'ex-big n=%d V5=%.0f captured(edge vs new level)=%.0f xu_exit=%.0f pen=%.0f episode_pnl=%.0f' % (agg['n'], agg['V5'], agg['cap'], agg['xu'], agg['pen'], agg['pnl']))
print(tape, 'V5 by class:', {k: (cnt[k], round(catv[k])) for k in cnt})
json.dump(rows, open(D / (pk + '.rows.json'), 'w'))
