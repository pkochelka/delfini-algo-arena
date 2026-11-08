#!/bin/bash
cd "$(dirname "$0")"
B="vol_sep=1 passive_v=0 take_sd_t=1.0 red_v=1"
./ev.sh rv10 r2,s1 $B theta_v=1.0 theta_keep_v=0.7 & ./ev.sh rv15 r2,s1 $B theta_v=1.5 theta_keep_v=1.2; wait
./ev.sh rv20 r2,s1 $B theta_v=2.0 theta_keep_v=1.7 & ./ev.sh rv07 r2,s1 $B theta_v=0.7 theta_keep_v=0.4; wait
