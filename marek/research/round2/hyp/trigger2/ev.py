"""Dump visible cancels / invisible inserts / clips near each shock + controls. python ev.py TAPE -> ev_TAPE.json"""
import json, csv, sys, pathlib, bisect, random
ROOT = pathlib.Path('/Users/marek/Documents/delfini-algo-arena'); D = pathlib.Path(__file__).parent
tape = sys.argv[1]
if tape == 'r2':
    ons = [float(r['onset']) for r in csv.DictReader(open(ROOT/'marek/research/round2/flow/shock_events.csv'))]
    S = json.load(open(ROOT/'marek/research/round2/hyp/shockgen/stale_r2.json')); Js = [s['J'] for s in S]; path = ROOT/'data/round2.csv'
else:
    tr = json.load(open(ROOT/f'data/synth/r2_v2_seed{tape[1:]}.truth.json')); ons, Js = tr['onsets'], tr['jumps']
    path = ROOT/f'data/synth/r2_v2_seed{tape[1:]}.csv'
W = []
for i, o in enumerate(ons):
    W.append([o+3.0-0.25, o+3.15, 'shock', i, o+3.0]); W.append([o+2.3, o+2.7, 'ctl', i, o+2.5])
random.seed(1)
for _ in range(300):
    t0 = random.uniform(100, 5900)
    if all(abs(t0-o) > 15 and abs(t0-o-3) > 3 for o in ons): W.append([t0, t0+0.4, 'rnd', -1, t0+0.25])
W.sort(); starts = [w[0] for w in W]
for w in W: w.append([])
book = {}; bids = {}; asks = {}
DEP = 20
with open(path) as fh:
    next(fh)
    for line in fh:
        a = line.rstrip('\n').split(','); t = float(a[0]); op = a[1]
        k = bisect.bisect_right(starts, t) - 1
        hits = [w for w in W[max(0, k-3):k+1] if w[0] <= t <= w[1]]
        bb = max(bids) if bids else None; ba = min(asks) if asks else None
        if hits and bb is not None and ba is not None:
            ev = None
            if op == 'Cancel':
                o = book.get(a[2])
                if o:
                    vis = (o[1] >= bb - DEP) if o[0] == 'B' else (o[1] <= ba + DEP)
                    ev = ('cb' if o[0] == 'B' else 'ca') if vis else 'ci'
            elif a[6] == 'G':
                s, px = a[3], int(a[5])
                if s == 'B':
                    ev = 'clipU' if (px == ba-1 and ba-bb >= 3) else ('ii' if px < bb - DEP else 'iv')
                else:
                    ev = 'clipD' if (px == bb+1 and ba-bb >= 3) else ('ii' if px > ba + DEP else 'iv')
            else:
                ev = 'tB' if a[3] == 'B' else 'tA'
            for w in hits: w[5].append((round(t - w[4], 6), ev))
        if op == 'Cancel':
            o = book.pop(a[2], None)
            if o:
                lv = bids if o[0]=='B' else asks; lv[o[1]] -= o[2]
                if lv[o[1]] <= 0: del lv[o[1]]
        elif a[6] == 'G':
            s, q, px = a[3], int(a[4]), int(a[5]); book[a[2]] = [s, px, q]
            lv = bids if s=='B' else asks; lv[px] = lv.get(px, 0) + q
        else:
            s, q, px = a[3], int(a[4]), int(a[5]); opp = 'A' if s=='B' else 'B'
            cand = sorted((o for o in book.items() if o[1][0]==opp and (o[1][1] <= px if s=='B' else o[1][1] >= px)),
                          key=lambda x: (x[1][1] if s=='B' else -x[1][1], int(x[0])))
            lv = asks if s=='B' else bids
            for oid, o in cand:
                if q <= 0: break
                f = min(q, o[2]); o[2] -= f; q -= f; lv[o[1]] -= f
                if lv[o[1]] <= 0: del lv[o[1]]
                if o[2] == 0: del book[oid]
out = [dict(kind=w[2], i=w[3], J=(Js[w[3]] if w[2] != 'rnd' else None), ev=w[5]) for w in W]
json.dump(out, open(D/f'ev_{tape}.json', 'w'))
print(tape, len(out))
