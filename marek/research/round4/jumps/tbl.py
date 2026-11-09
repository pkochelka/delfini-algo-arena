"""Compact table: tbl.py TAG [TAG..] -> set | total d | calm d | vol d | cv d | shock d | shock ex biggest | median shock d (vs last base4 row)."""
import json, sys, statistics
rows = [json.loads(l) for l in open(__file__.replace('tbl.py', 'results.jsonl'))]
tape = 'round3' if '--r3' in sys.argv else 'round4'
rows = [r for r in rows if 'pnl' in r and tape in r['tape']]
bt = 'r3base' if tape == 'round3' else 'base4'
b = [r for r in rows if r['tag'] == bt][-1]
B = b['parts']
print(f"{'set':95s}   total    calm     vol      cv   shock  s-big  smed")
for r in rows:
    if r['tag'] in sys.argv[1:]:
        P = r['parts']; ds = [x - y for x, y in zip(r['per_shock'], b['per_shock'])]
        big = max(ds, key=abs)
        f = lambda x: f'{x/1e3:+7.1f}'
        print(f"{json.dumps(r['set']).replace(' ',''):95s} {r['pnl']/1e3:7.1f} {f(P['calm']-B['calm'])} {f(P['volatile']-B['volatile'])} "
              f"{f(P['calm']+P['volatile']-B['calm']-B['volatile'])} {f(P['shock']-B['shock'])} {f(P['shock']-B['shock']-big)} {statistics.median(ds)/1e3:+5.2f}")
