import json, sys
rows = [json.loads(l) for l in open('/Users/marek/Documents/delfini-algo-arena/marek/research/round2/hyp/results.jsonl')]
rows = [r for r in rows if r['tag'].startswith('sh_')]
base = next(r for r in rows if r['tag'] == (sys.argv[1] if len(sys.argv) > 1 else 'sh_base8'))
T = list(base['pnl'])
print('tag'.ljust(26), ' '.join(t.rjust(7) for t in T), 'synΔ'.rjust(7), 'brch')
for r in rows:
    d = [(r['pnl'].get(t) or 0) - base['pnl'][t] if t in r['pnl'] else None for t in T]
    syn = [x for t, x in zip(T, d) if x is not None and t != 'r2']
    print(r['tag'][:26].ljust(26), ' '.join((f'{x/1000:+7.1f}' if x is not None else '      .') for x in d),
          f'{sum(syn)/max(1,len(syn))/1000:+7.1f}', r['breach'] or '', r['set'] if r['tag'] != base['tag'] else r['pnl'])
