import pickle, csv, sys, numpy as np
d = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else 'jumps/cache/shock_r4_volsig.pkl', 'rb'))
ev = list(csv.DictReader(open('flow/shock_events.csv')))
tot_val = 0
for i, e in enumerate(ev):
    L = d['log'][i]; tj = float(e['t_jump']); J = float(e['jump']); o = float(e['onset']); up = J > 0; sg = 1 if up else -1
    fi = next((k for k, r in enumerate(L) if r['nsn'] > L[0]['nsn'] and r['t'] > o), None)
    k0 = (fi - 1) if fi else max(k for k, r in enumerate(L) if r['t'] < tj)
    r0 = L[k0]; Snew = r0['F'] + J
    side = r0['asks'] if up else r0['bids']
    stale = [(p, q) for p, q, w in side if w == 'market']
    good = sum(q for p, q in stale if (Snew - p) * sg >= 5)
    val = sum(q * ((Snew - p) * sg - 3) for p, q, w in side if w == 'market' and (Snew - p) * sg >= 5)
    tot_val += min(val, 1e9)
    tf = (L[fi]['t'] - tj) * 1e3 if fi else None
    post = [r for r in L if r['t'] >= tj]
    pmax = max((r['pos'] * sg for r in post[:400]), default=0)
    p_pre = L[k0]['pos'] * sg
    print(f"#{i:2d} J={J:7.1f} {e['regime_at_jump'][:4]} fire {str(None if tf is None else round(tf)):>5}ms dir {L[fi]['ss_dir'] if fi else 0:2d} "
          f"edge {L[fi]['edge'] if fi else 0:5.1f} aligned pos pre {p_pre:5d} max {pmax:5d} | stale lots >=5 thru {good:5d}, value {val/1e3:6.1f}k")
print('total stale value', tot_val / 1e3)
