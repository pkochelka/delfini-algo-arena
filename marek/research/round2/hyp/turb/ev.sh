#!/bin/bash
# usage: ev.sh outname tapes key=val ...
cd /Users/marek/Documents/delfini-algo-arena
out=$1; tapes=$2; shift 2
uv run python marek/research/round2/hyp/eval.py strategies/marek/round2/hyp/turb.py --tapes $tapes --tag turb_$out --set "$@" > marek/research/round2/hyp/turb/$out.txt 2>&1
