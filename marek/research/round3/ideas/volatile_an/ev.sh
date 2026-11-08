#!/bin/bash
# usage: ev.sh TAG FILE [k=v ...]   (round3 + synth seed1)
cd /Users/marek/Documents/delfini-algo-arena
tag=$1; f=$2; shift 2
uv run python marek/research/round3/strategy/eval3.py $f --tape ${TAPES:-data/round3.csv,data/synth/r3_v1_seed1.csv} --tag $tag --set "$@" 2>&1 | python3 -c "
import sys,json
for l in sys.stdin:
    try: r=json.loads(l); print(r['tag'], r['tape'][-16:], r.get('pnl'), r.get('breach'), r.get('parts'), r.get('err','')[-300:])
    except Exception: print(l[:300])
"
