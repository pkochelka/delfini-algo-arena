#!/bin/bash
cd /Users/marek/Documents/delfini-algo-arena
tag=$1; shift
uv run python marek/research/round2/hyp/eval.py strategies/marek/round2/hyp/shockgen.py --tapes ${TAPES:-r2,s1,s2,s3} --set "$@" --tag $tag --base ${BASE:-BASE_final3} >/dev/null 2>&1
tail -1 marek/research/round2/hyp/results.jsonl | python3 -c "
import json,sys
r=json.loads(sys.stdin.read()); p=r['pnl']; v=r['vs_base']
print(r['tag'], r['set'], ' '.join(f'{k}={v/1e3:.1f}' for k,v in p.items()), 'ex5398=%.1f'%(r['r2_ex5398']/1e3), 'sh5398=%.1f'%(r['r2_sh5398']/1e3))
print('  deltas', {k:round(x['d']) for k,x in v['per_tape'].items()}, 'ex5398 d', v['per_tape']['r2'].get('ex5398'), 'win_t', v['win_t'])"
