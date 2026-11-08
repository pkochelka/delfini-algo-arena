"""Per-shock stale book on r2 from the tape: stale lots/value at T+x ms (T = onset + 3.00)."""
import csv, pathlib, bisect
ROOT = pathlib.Path(__file__).resolve().parents[5]
ons = [float(r['onset']) for r in csv.DictReader(open(ROOT/'marek/research/round2/flow/shock_events.csv'))]
rows = []
with open(ROOT/'data/round2.csv') as fh:
    next(fh)
    for line in fh:
        a = line.rstrip('\n').split(',')
        rows.append((float(a[0]), a[1], int(a[2]), a[3], int(a[4]) if a[4] else 0, int(a[5]) if a[5] else 0, a[6]))
times = [r[0] for r in rows]
book = {}  # oid -> [side, qty, price]
def apply(r):
    t, op, oid, side, vol, px, life = r
    if op == 'Cancel':
        book.pop(oid, None); return
    if life == 'G':
        book[oid] = [side, vol, px]; return
    # FAK taker
    rem = vol
    if side == 'B':
        lv = sorted((o for o in book.items() if o[1][0] == 'A' and o[1][2] <= px), key=lambda x: (x[1][2], x[0]))
    else:
        lv = sorted((o for o in book.items() if o[1][0] == 'B' and o[1][2] >= px), key=lambda x: (-x[1][2], x[0]))
    for k, o in lv:
        if rem <= 0: break
        q = min(rem, o[1]); o[1] -= q; rem -= q
        if o[1] == 0: del book[k]
i = 0
print(f"{'onset':>8} {'J':>5} {'dir':>3} " + ' '.join(f"{'L'+str(x):>5} {'V'+str(x):>6}" for x in (0,5,10,15,20)) + "  firstIns_ms firstTak_ms")
for o in ons:
    T = o + 3.0
    # mid before (bests at T-0.05) and level after: median maker inserts in [T+0.3, T+1.0]
    while i < len(rows) and rows[i][0] < T - 0.0005:
        apply(rows[i]); i += 1
    bids = [v for v in book.values() if v[0]=='B']; asks=[v for v in book.values() if v[0]=='A']
    bb = max(v[2] for v in bids); ba = min(v[2] for v in asks)
    j0 = bisect.bisect_left(times, T+0.3); j1 = bisect.bisect_left(times, T+1.0)
    ins = [ (r[5]) for r in rows[j0:j1] if r[1]=='Insert' and r[6]=='G']
    ins.sort(); lvl = ins[len(ins)//2] if ins else (bb+ba)/2
    J = lvl - (bb+ba)/2; d = 1 if J > 0 else -1
    # first insert / first taker after T
    k = bisect.bisect_left(times, T - 0.0005)
    fi = next((rows[m][0]-T for m in range(k, len(rows)) if rows[m][1]=='Insert' and rows[m][6]=='G'), None)
    ft = next((rows[m][0]-T for m in range(k, len(rows)) if rows[m][6]=='F'), None)
    out = []
    ii = i
    snap = {kk: list(v) for kk, v in book.items()}
    for x in (0, 5, 10, 15, 20):
        while ii < len(rows) and rows[ii][0] < T + x*1e-3:
            apply(rows[ii]); ii += 1
        if d > 0:
            st = [(v[1], lvl - v[2]) for v in book.values() if v[0]=='A' and v[2] < lvl]
        else:
            st = [(v[1], v[2] - lvl) for v in book.values() if v[0]=='B' and v[2] > lvl]
        L = sum(q for q, _ in st); V = sum(q*e for q, e in st)
        out.append((L, V))
    book = snap
    print(f"{o:8.2f} {J:5.1f} {d:3d} " + ' '.join(f"{L:5d} {V:6.0f}" for L, V in out) + f"  {fi*1e3 if fi is not None else -1:8.1f} {ft*1e3 if ft is not None else -1:8.1f}")
