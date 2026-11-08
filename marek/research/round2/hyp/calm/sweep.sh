#!/bin/bash
# usage: sweep.sh listfile  (each line: tag|set args)
cd /Users/marek/Documents/delfini-algo-arena
while IFS='|' read -r tag sets; do
  [ -z "$tag" ] && continue
  uv run python marek/research/round2/hyp/eval.py strategies/marek/round2/hyp/calm.py --set $sets --tapes ${TAPES:-r2,s1,s2} --tag calm_$tag 2>&1 | tail -1 | python3 -c "import json,sys; r=json.loads(sys.stdin.read()); print(r['tag'], r['pnl'], round(r['synth_mean']), r['breach'])"
done < "$1"
