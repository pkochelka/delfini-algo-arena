#!/bin/bash
# usage: queue.sh TAPES OUTFILE "set1" "set2" ...   (2 at a time)
cd /Users/marek/Documents/delfini-algo-arena
T=$1; O=$2; shift 2
run() { r=$(uv run python marek/research/round2/hyp/eval.py strategies/marek/round2/hyp/filter.py --set $1 --tapes $T --tag "filter $1" | grep -o '"pnl": {[^}]*}, "synth_mean": [0-9.]*'); echo "$1 :: $r" >> $O; }
export -f run; export T O
printf '%s\n' "$@" | xargs -P 2 -I{} bash -c 'run "{}"'
echo DONE >> $O
