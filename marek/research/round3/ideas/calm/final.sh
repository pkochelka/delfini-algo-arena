#!/bin/bash
# finalist calm_c2 on the remaining synth v1 seeds and the held-out v2 seeds, two processes at a time
cd /Users/marek/Documents/delfini-algo-arena
F=strategies/marek/round3/ideas/calm_a.py
S="take=0.10 tr_take=0.0 tr_take_sd=1.0 theta_keep=0.0"
for T in data/synth/r3_v1_seed2.csv,data/synth/r3_v1_seed3.csv data/synth/r3_v1_seed4.csv,data/synth/r3_v1_seed5.csv \
         data/synth/r3_v1_seed6.csv,data/synth/r3_v2_seed11.csv data/synth/r3_v2_seed12.csv,data/synth/r3_v2_seed13.csv \
         data/synth/r3_v2_seed14.csv; do
  uv run python marek/research/round3/strategy/eval3.py $F --tape $T --tag calm_c2 --set $S | cut -c1-200
done
