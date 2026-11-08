import json,sys
for l in open('marek/research/round2/hyp/results.jsonl'):
    r=json.loads(l)
    if r['tag'].startswith('drought'):
        v=r.get('vs_base') or {}
        print(r['tag'],r['set'],r['pnl'].get('r2'),{k:(x['d'],x.get('train'),x.get('hold')) for k,x in v.get('per_tape',{}).items()},v.get('win_t'))
