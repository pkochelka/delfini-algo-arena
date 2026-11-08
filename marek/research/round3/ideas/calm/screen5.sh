#!/bin/bash
cd /Users/marek/Documents/delfini-algo-arena
T=data/round3.csv,data/synth/r3_v1_seed1.csv
F=strategies/marek/round3/ideas/calm_a.py
run() { tag=$1; shift; uv run python marek/research/round3/strategy/eval3.py $F --tape $T --tag $tag --set "$@" | cut -c1-260; }
run calm_oj0b oj_on=1 oj_q=0
run calm_tq0d tq_x=0 tq_deep=1
run calm_oj100b oj_on=1 oj_q=100
# exact-S oracle in calm (research only; round 3 only, the S path is round 3's)
CALM_SRC=/Users/marek/Documents/delfini-algo-arena/strategies/marek/round3/r3_best.py CALM_SHAT=/private/tmp/claude-501/-Users-marek-Documents-delfini-algo-arena/c0256e15-f719-43d5-b913-67767d347aa1/scratchpad/calm/s_hat.npy uv run python marek/research/round3/strategy/eval3.py strategies/marek/round3/ideas/calm_oracle.py --tape data/round3.csv --tag calm_oracle | cut -c1-260
