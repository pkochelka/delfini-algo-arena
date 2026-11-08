#!/bin/bash
# usage: sweep.sh "cfg1" "cfg2" ...   (each cfg = space separated key=val)
cd /Users/marek/Documents/delfini-algo-arena
for cfg in "$@"; do
  uv run python marek/research/round2/hyp/eval.py strategies/marek/round2/hyp/mech.py --tapes ${TAPES:-r2,s1,s2,s3,s4} --tag "mech $cfg" --set $cfg | tail -1
done
