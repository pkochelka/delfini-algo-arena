#!/bin/bash
cd "$(dirname "$0")"
B="vol_sep=1 passive_v=0 take_sd_t=1.0"
./ev.sh be02 r2,s1 $B burst_exc=0.2 & ./ev.sh be002 r2,s1 $B burst_exc=0.02; wait
./ev.sh pv03 r2,s1 $B pv_thr=0.3 & ./ev.sh pv07 r2,s1 $B pv_thr=0.7; wait
