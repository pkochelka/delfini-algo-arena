#!/bin/bash
# calm screen 1: existing PARAMS on r3_best (round3 + synth seed1, 2 processes)
cd /Users/marek/Documents/delfini-algo-arena
T=data/round3.csv,data/synth/r3_v1_seed1.csv
F=strategies/marek/round3/r3_best.py
run() { tag=$1; shift; uv run python marek/research/round3/strategy/eval3.py $F --tape $T --tag $tag --set "$@" | cut -c1-260; }
run calm_pj50 pj_q=50
run calm_pj100 pj_q=100
run calm_kd0 keep_depth=0
run calm_kd2 keep_depth=2
run calm_tk10 take=0.10
run calm_tk20 take=0.20
run calm_tsd1 take_sd=1.0
run calm_tsd3 take_sd=3.0
run calm_kp0 theta_keep=0.0
run calm_kp2 theta_keep=0.2
run calm_ro15 reach_off=1.5
run calm_ro05 reach_off=0.5
run calm_th20 theta=0.20
