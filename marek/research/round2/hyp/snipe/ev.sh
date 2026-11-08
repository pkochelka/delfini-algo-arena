#!/bin/bash
# ev.sh TAG [key=val ...]  -> all 8 tapes
cd /Users/marek/Documents/delfini-algo-arena
tag=$1; shift
uv run python marek/research/round2/hyp/eval.py ${STRAT:-strategies/marek/round2/hyp/snipe.py} --tapes ${TAPES:-r2,s1,s2,s3,s4,s5,s6,s7} --set "$@" --tag $tag 2>&1 | tail -1 | python3 -c "
import json,sys
r=json.loads(sys.stdin.read()); p=r['pnl']
print(r['tag'], r['set'], ' '.join(f'{k}={v/1e3:.1f}' for k,v in p.items()), 'synth=%.1f'%(r['synth_mean']/1e3), 'pen', {k:round(v) for k,v in r['breach'].items()})"
