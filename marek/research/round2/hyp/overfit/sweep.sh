#!/bin/bash
cd /Users/marek/Documents/delfini-algo-arena
run(){ tag=$1; shift; uv run python marek/research/round2/hyp/eval.py strategies/marek/round2/hyp/overfit.py --set "$@" --tag overfit_$tag --base BASE_final3 2>&1 | tail -1 >> marek/research/round2/hyp/overfit/sweep.log; }
run cap0 xs_cap=0
run cap400 xs_cap=400
run edge3 xs_edge=3.0
run edge6 xs_edge=6.0
run dr0 dr_on=0
run ps0 ps_on=0
run ulim2 xu_lim=2.0
run ts004 xu_ts2=0.04
run ts02 xu_ts2=0.2
run shots1 ss_shots=1
run reval0 xs_reval=0
run lim2_6 xu_lim2=6.0
run sslim5 ss_lim=5
run edgeNone xs_edge=None
