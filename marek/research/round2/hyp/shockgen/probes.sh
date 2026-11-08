#!/bin/bash
# probes.sh PREFIX key=val ...  -> PREFIX_<tape>.pkl for all 8 tapes, 4 at a time
cd /Users/marek/Documents/delfini-algo-arena; D=marek/research/round2/hyp/shockgen; pre=$1; shift
for grp in "r2 s1 s2 s3" "s4 s5 s6 s7"; do
  for t in $grp; do uv run python marek/research/round2/hyp/snipe/probe.py strategies/marek/round2/hyp/shockgen.py $t $D/${pre}_$t.pkl "$@" > $D/${pre}_$t.log 2>&1 & done; wait
done
echo done
