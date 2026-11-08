#!/bin/bash
cd "$(dirname "$0")"
B="vol_sep=1 passive_v=0 take_sd_t=1.0 unwind_turb=1 unwind_only_t=1"
./ev.sh uw10 r2,s1 $B unwind_u=1.0 unwind_pos=80 unwind_to=50 & ./ev.sh uw15 r2,s1 $B unwind_u=1.5 unwind_pos=80 unwind_to=50; wait
B="vol_sep=1 passive_v=0 take_sd_t=1.0"
./ev.sh vsd25 r2,s1 $B take_sd_v=2.5 & ./ev.sh vt03 r2,s1 $B take_v=0.3; wait
