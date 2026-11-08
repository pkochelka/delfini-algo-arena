"""Per-shock stale book for any tape. uv run python stale2.py TAPE -> stale_TAPE.json
T = onset + 3.00; level: synth = pre-mid + truth J; r2 = median maker insert in [T+0.3,T+1.0]."""
import csv, json, pathlib, bisect, sys
ROOT = pathlib.Path(__file__).resolve().parents[5]
D = pathlib.Path(__file__).parent
tape = sys.argv[1]
if tape == 'r2':
    ons = [float(r['onset']) for r in csv.DictReader(open(ROOT/'marek/research/round2/flow/shock_events.csv'))]
    Js = [None]*len(ons); path = ROOT/'data/round2.csv'
else:
    tr = json.load(open(ROOT/f'data/synth/r2_v2_seed{tape[1:]}.truth.json'))
    ons, Js = tr['onsets'], tr['jumps']; path = ROOT/f'data/synth/r2_v2_seed{tape[1:]}.csv'
rows = []
with open(path) as fh:
    next(fh)
    for line in fh:
        a = line.rstrip('\n').split(',')
        rows.append((float(a[0]), a[1], int(a[2]), a[3], int(a[4]) if a[4] else 0, int(a[5]) if a[5] else 0, a[6]))
times = [r[0] for r in rows]
book = {}
def apply(r):
    t, op, oid, side, vol, px, life = r
    if op == 'Cancel':
        book.pop(oid, None); return
    if life == 'G':
        book[oid] = [side, vol, px]; return
    rem = vol
    if side == 'B':
        lv = sorted((o for o in book.items() if o[1][0] == 'A' and o[1][2] <= px), key=lambda x: (x[1][2], x[0]))
    else:
        lv = sorted((o for o in book.items() if o[1][0] == 'B' and o[1][2] >= px), key=lambda x: (-x[1][2], x[0]))
    for k, o in lv:
        if rem <= 0: break
        q = min(rem, o[1]); o[1] -= q; rem -= q
        if o[1] == 0: del book[k]
i = 0; out = []
for o, Jt in zip(ons, Js):
    T = o + 3.0
    while i < len(rows) and rows[i][0] < T - 0.0005:
        apply(rows[i]); i += 1
    bids = [v[2] for v in book.values() if v[0]=='B']; asks=[v[2] for v in book.values() if v[0]=='A']
    if not bids or not asks:
        continue
    bb, ba = max(bids), min(asks); mid = (bb+ba)/2
    if Jt is None:
        j0 = bisect.bisect_left(times, T+0.3); j1 = bisect.bisect_left(times, T+1.0)
        ins = sorted(r[5] for r in rows[j0:j1] if r[1]=='Insert' and r[6]=='G')
        lvl = ins[len(ins)//2] if ins else mid
    else:
        lvl = mid + Jt
    J = lvl - mid; d = 1 if J > 0 else -1
    k = bisect.bisect_left(times, T - 0.0005)
    ft = next((rows[m][0]-T for m in range(k, min(k+5000, len(rows))) if rows[m][6]=='F'), None)
    snap = {kk: list(v) for kk, v in book.items()}
    ii = i; LV = {}
    for x in (0, 5, 10, 20):
        while ii < len(rows) and rows[ii][0] < T + x*1e-3:
            apply(rows[ii]); ii += 1
        if d > 0:
            st = [(v[1], lvl - v[2]) for v in book.values() if v[0]=='A' and v[2] < lvl]
        else:
            st = [(v[1], v[2] - lvl) for v in book.values() if v[0]=='B' and v[2] > lvl]
        LV[x] = (sum(q for q, _ in st), sum(q*e for q, e in st))
    book = snap
    out.append(dict(onset=o, J=J, lvl=lvl, d=d, bb=bb, ba=ba, ft=ft, LV=LV))
json.dump(out, open(D/f'stale_{tape}.json', 'w'))
print(tape, len(out), 'V0 total', round(sum(r['LV'][0][1] for r in out)), 'V0 ex big', round(sum(r['LV'][0][1] for r in out if abs(r['J'])<100)))
