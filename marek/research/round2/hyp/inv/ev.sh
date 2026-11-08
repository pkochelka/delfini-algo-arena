#!/bin/bash
# usage: ev.sh TAG TAPES key=val ...
cd /Users/marek/Documents/delfini-algo-arena
tag=$1; tapes=$2; shift 2
uv run python marek/research/round2/hyp/eval.py ${STRAT:-strategies/marek/round2/hyp/inv.py} --tapes $tapes --tag inv_$tag --set "$@" 2>&1 | tail -1 | python3 -c "
import json,sys
r=json.loads(sys.stDIN.read()) if False else json.loads(sys.stdin.read())
b={'r2':591131,'s1':646964,'s2':657119,'s3':653801,'s4':527090}
print(r['tag'], r['set'], {k:(int(v-b.get(k,0)) if v is not None and k in b else v) for k,v in r['pnl'].items()}, 'dsum', int(sum(v-b[k] for k,v in r['pnl'].items() if k in b)), 'breach',r['breach'])
"
