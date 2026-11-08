#!/bin/bash
cd /Users/marek/Documents/delfini-algo-arena
A=marek/research/round3/strategy/ab.sh; F=strategies/marek/round3/r3_lad.py
$A $F "e10" "e10_eta008 eta=0.08" "e10_eta015 eta=0.15" "e10_tr tr_on=1 tr_sr=1" "e10_th_t15 theta_t=1.5 theta_keep_t=1.3" "e10_th_t21 theta_t=2.1 theta_keep_t=1.9" "e10_reach_t3 reach_t=3.0" "e10_sslim10 ss_lim=10" "e10_takesdt1 take_sd_t=1.0"
TAPE=synth $A $F "e10" "e10_eta008 eta=0.08" "e10_eta015 eta=0.15" "e10_tr tr_on=1 tr_sr=1" "e10_th_t15 theta_t=1.5 theta_keep_t=1.3" "e10_th_t21 theta_t=2.1 theta_keep_t=1.9" "e10_reach_t3 reach_t=3.0" "e10_sslim10 ss_lim=10" "e10_takesdt1 take_sd_t=1.0" "e10_lad0 lad_on=0"
