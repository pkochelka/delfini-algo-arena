#!/bin/bash
cd /Users/marek/Documents/delfini-algo-arena
tag=$1; shift
uv run python marek/research/round2/hyp/eval.py strategies/marek/round2/hyp/trigger2.py --tapes ${TAPES:-r2,s1,s2,s3} --set "$@" --tag $tag --base shockgen_C >/dev/null 2>&1
grep "\"tag\": \"$tag\"" marek/research/round2/hyp/results.jsonl | tail -1 | python3 -c "
import json,sys
r=json.loads(sys.stdin.read()); p=r['pnl']; v=r['vs_base']
print(r['tag'], r['set'], ' '.join(f'{k}={v/1e3:.1f}' for k,v in p.items()), 'ex5398=%.1f'%(r['r2_ex5398']/1e3))
print('  deltas', {k:round(x['d']) for k,x in v['per_tape'].items()}, 'ex5398 d', v['per_tape']['r2'].get('ex5398'), 'win_t', v['win_t'])"
