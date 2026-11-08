#!/bin/bash
# calm screen 6: combinations of the small positive knobs (calm_a.py; round3 + synth seed1)
cd /Users/marek/Documents/delfini-algo-arena
T=data/round3.csv,data/synth/r3_v1_seed1.csv
F=strategies/marek/round3/ideas/calm_a.py
run() { tag=$1; shift; uv run python marek/research/round3/strategy/eval3.py $F --tape $T --tag $tag --set "$@" | cut -c1-260; }
run calm_c1 take=0.10 tr_take=0.0 tr_take_sd=1.0
run calm_c2 take=0.10 tr_take=0.0 tr_take_sd=1.0 theta_keep=0.0
run calm_trtm1 tr_take=-0.1 tr_take_sd=1.0
run calm_c3 take=0.10 tr_take=0.0 tr_take_sd=1.0 theta_keep=0.0 inv_skew=0.1
run calm_kp0trt0 theta_keep=0.0 tr_take=0.0 tr_take_sd=1.0
