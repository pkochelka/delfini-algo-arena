#!/bin/bash
# usage: ev.sh tag tapes [k=v ...]
cd /Users/marek/Documents/delfini-algo-arena
tag=$1; tapes=$2; shift 2
if [ $# -gt 0 ]; then S="--set $*"; else S=""; fi
uv run python marek/research/round2/hyp/eval.py strategies/marek/round2/hyp/fast.py --tapes $tapes --tag fast_$tag $S 2>&1 | tail -1 | python3 -c "
import json,sys; r=json.loads(sys.stdin.read()); print(r['tag'], r['set'], {k:round(v/1e3,1) for k,v in r['pnl'].items()}, 'syn', round(r['synth_mean']/1e3,1), 'breach', r['breach'])"
