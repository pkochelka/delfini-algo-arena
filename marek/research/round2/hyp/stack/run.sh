#!/bin/bash
# usage: run.sh tag "k=v k=v" [tapes]
cd /Users/marek/Documents/delfini-algo-arena
T=${3:-r2,s1,s2,s3}
uv run python marek/research/round2/hyp/eval.py strategies/marek/round2/hyp/stack.py --set $2 --tag stack_$1 --base BASE_snipe_best --tapes $T 2>&1 | tail -1 | python3 -c "
import json,sys
r=json.loads(sys.stdin.read()); v=r['vs_base']['per_tape']
print(r['tag'], r['set'], 'r2=%d'%r['pnl']['r2'], ' '.join('%s:%+d'%(k,x['d']) for k,x in v.items()), 'tr=%+d ho=%+d'%(v['r2']['train'],v['r2']['hold']), 't=%.2f'%r['vs_base']['win_t'])
" >> marek/research/round2/hyp/stack/summary.txt
