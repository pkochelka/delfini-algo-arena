#!/bin/bash
cd /Users/marek/Documents/delfini-algo-arena
A=marek/research/round3/strategy/ab.sh; F=strategies/marek/round3/r3_lad.py
C=("e10_pt20 passive_t=1 theta_t=2.0 theta_keep_t=1.8" "e10_pt25 passive_t=1 theta_t=2.5 theta_keep_t=2.2" "e10_pv03 pv_thr=0.3" "e10_pv07 pv_thr=0.7" "e10_taket03 take_t=0.3" "e10_taket005 take_t=0.05" "e10_th025 theta=0.25" "e10_th035 theta=0.35" "e10_xs600 xs_cap=600" "e10_xs1200 xs_cap=1200" "e10_ss0 ss_on=0" "e10_sstr3 ss_trade=1 ss_trade_lv=3" "e10_sstr2 ss_trade=1 ss_trade_lv=2")
$A $F "${C[@]:0:9}"
$A $F "${C[@]:9}"
TAPE=synth $A $F "${C[@]}"
