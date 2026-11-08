#!/bin/bash
# usage: batch.sh TAPES TAG1 "k=v k=v" TAG2 "k=v" ...   runs all concurrently
cd /Users/marek/Documents/delfini-algo-arena
T=$1; shift
while [ $# -gt 0 ]; do
  tag=$1; sets=$2; shift 2
  uv run python marek/research/round2/hyp/eval.py strategies/marek/round2/hyp/shock.py --tapes $T --set $sets --tag sh_$tag > /dev/null &
done
wait
python3 marek/research/round2/hyp/shock/cmp.py
