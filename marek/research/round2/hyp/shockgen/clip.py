"""First clipped insert (bid at ask-1 / ask at bid+1, spread>=3) after T=onset+3 (truth) and in a control window."""
import json, csv, sys, pathlib, bisect, random
ROOT = pathlib.Path('/Users/marek/Documents/delfini-algo-arena')
tape = sys.argv[1]
if tape == 'r2':
    ons = [float(r['onset']) for r in csv.DictReader(open(ROOT/'marek/research/round2/flow/shock_events.csv'))]
    S = json.load(open(ROOT/'marek/research/round2/hyp/shockgen/stale_r2.json')); Js = [s['J'] for s in S]; path = ROOT/'data/round2.csv'
else:
    tr = json.load(open(ROOT/f'data/synth/r2_v2_seed{tape[1:]}.truth.json')); ons, Js = tr['onsets'], tr['jumps']
    path = ROOT/f'data/synth/r2_v2_seed{tape[1:]}.csv'
# windows: (start, end, kind, idx)
W = []
for i, o in enumerate(ons):
    W.append((o+3.0-0.01, o+3.10, 'shock', i)); W.append((o+2.5, o+2.6, 'ctl', i))
random.seed(1)
for _ in range(60):
    t0 = random.uniform(100, 5900)
    if all(abs(t0-o) > 15 for o in ons): W.append((t0, t0+0.11, 'calm', -1))
W.sort(); starts = [w[0] for w in W]
book = {}; bids = {}; asks = {}
def best():
    return (max(bids) if bids else None, min(asks) if asks else None)
res = {}
with open(path) as fh:
    next(fh)
    for line in fh:
        a = line.rstrip('\n').split(','); t = float(a[0]); op = a[1]
        k = bisect.bisect_right(starts, t) - 1
        hits = [w for w in W[max(0,k-3):k+1] if w[0] <= t <= w[1] and w not in res]
        if op == 'Insert' and a[6] == 'G' and hits:
            bb, ba = best(); px = int(a[5])
            if bb is not None and ba is not None and ba - bb >= 3:
                d = 1 if (a[3] == 'B' and px == ba-1) else (-1 if (a[3] == 'A' and px == bb+1) else 0)
                if d:
                    for w in hits: res[w] = (t - (w[0]+0.01 if w[2]=='shock' else w[0]), d)
        # apply
        if op == 'Cancel':
            o = book.pop(a[2], None)
            if o:
                lv = bids if o[0]=='B' else asks; lv[o[1]] -= o[2]
                if lv[o[1]] <= 0: del lv[o[1]]
        elif a[6] == 'G':
            s, q, px = a[3], int(a[4]), int(a[5]); book[a[2]] = [s, px, q]
            lv = bids if s=='B' else asks; lv[px] = lv.get(px, 0) + q
        else:
            s, q, px = a[3], int(a[4]), int(a[5])
            # taker: consume opposite side resting orders by price-time (approx by oid order)
            opp = 'A' if s=='B' else 'B'
            cand = sorted((o for o in book.items() if o[1][0]==opp and (o[1][1] <= px if s=='B' else o[1][1] >= px)),
                          key=lambda x: (x[1][1] if s=='B' else -x[1][1], x[0]))
            lv = asks if s=='B' else bids
            for oid, o in cand:
                if q <= 0: break
                f = min(q, o[2]); o[2] -= f; q -= f; lv[o[1]] -= f
                if lv[o[1]] <= 0: del lv[o[1]]
                if o[2] == 0: del book[oid]
out = []
for w in W:
    r = res.get(w)
    if w[2] == 'shock':
        J = Js[w[3]]
        if abs(J) >= 100: continue
        out.append(('shock', J, r))
    else:
        out.append((w[2], None, r))
sh = [x for x in out if x[0]=='shock']
hit = [x for x in sh if x[2]]
ok = [x for x in hit if (x[2][1] > 0) == (x[1] > 0)]
dl = sorted(x[2][0]*1e3 for x in hit)
med = lambda v: v[len(v)//2] if v else None
print(tape, 'shocks', len(sh), 'hit', len(hit), 'dir ok', len(ok), 'delay ms med', None if not dl else round(med(dl),1),
      'p25', None if not dl else round(dl[len(dl)//4],1), 'p75', None if not dl else round(dl[3*len(dl)//4],1),
      '| hit<=15ms', sum(1 for x in hit if x[2][0] <= 0.015), 'ok<=15', sum(1 for x in ok if x[2][0] <= 0.015),
      '| ctl FP', sum(1 for x in out if x[0]=='ctl' and x[2]), '/', sum(1 for x in out if x[0]=='ctl'),
      'calm FP', sum(1 for x in out if x[0]=='calm' and x[2]), '/', sum(1 for x in out if x[0]=='calm'))
wrong = [(round(x[1],1), round(x[2][0]*1e3,1)) for x in hit if x not in ok]
print('  wrong (J, ms):', wrong)
json.dump([(x[1], x[2]) for x in sh], open(f'/Users/marek/Documents/delfini-algo-arena/marek/research/round2/hyp/shockgen/clip_{tape}.json','w'))
print('  hits early before T (ms<0):', sum(1 for x in hit if x[2][0] < 0))
