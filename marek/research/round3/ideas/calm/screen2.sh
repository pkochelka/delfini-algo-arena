#!/bin/bash
# calm screen 2 (priority queue): calm_a.py mechanisms (round3 + synth seed1)
cd /Users/marek/Documents/delfini-algo-arena
T=data/round3.csv,data/synth/r3_v1_seed1.csv
F=strategies/marek/round3/ideas/calm_a.py
run() { tag=$1; shift; uv run python marek/research/round3/strategy/eval3.py $F --tape $T --tag $tag --set "$@" | cut -c1-260; }
run calm_oc oc_on=1
run calm_oc_tk10sd1 oc_on=1 take=0.10 take_sd=1.0
run calm_tq0 tq_x=0
run calm_tq0d tq_x=0 tq_deep=1
run calm_tq50d tq_x=50 tq_deep=1
run calm_ys0_1 ys_lim=0 ys_y0=1.0
run calm_isk10 inv_skew=0.10
run calm_tk10sd1 take=0.10 take_sd=1.0
run calm_trt0 tr_take=0.0 tr_take_sd=1.0
run calm_tk05sd1 take=0.05 take_sd=1.0
run calm_qk400 tq_x=1000 qk_max=400
run calm_ys50_05 ys_lim=50 ys_y0=0.5
run calm_tk10sd1kp0 take=0.10 take_sd=1.0 theta_keep=0.0
run calm_trtm1 tr_take=-0.1 tr_take_sd=1.0
