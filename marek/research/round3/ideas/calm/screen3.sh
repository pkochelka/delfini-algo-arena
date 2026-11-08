#!/bin/bash
# calm screen 3: IOC thresholds and reduce-only takes (calm_a.py; round3 + synth seed1)
cd /Users/marek/Documents/delfini-algo-arena
T=data/round3.csv,data/synth/r3_v1_seed1.csv
F=strategies/marek/round3/ideas/calm_a.py
run() { tag=$1; shift; uv run python marek/research/round3/strategy/eval3.py $F --tape $T --tag $tag --set "$@" | cut -c1-260; }
run calm_oc oc_on=1
run calm_oc_tk10sd1 oc_on=1 take=0.10 take_sd=1.0
run calm_tk05 take=0.05
run calm_tk00 take=0.001
run calm_tk10sd1 take=0.10 take_sd=1.0
run calm_tk05sd1 take=0.05 take_sd=1.0
run calm_tk10sd1kp0 take=0.10 take_sd=1.0 theta_keep=0.0
run calm_trt0 tr_take=0.0 tr_take_sd=1.0
run calm_trtm1 tr_take=-0.1 tr_take_sd=1.0
run calm_trt0_50 tr_take=0.0 tr_take_sd=1.0 tr_to=50
