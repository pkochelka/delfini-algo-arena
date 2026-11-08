#!/bin/bash
# ab.sh FILE "tag k=v k=v" "tag2 k=v" ...   -> runs each config of FILE (eval3.py).
# TAPE=data/round3.csv (default; configs run in parallel) or TAPE=synth (6 seeds per config, configs run 1-2 at a time)
F=$1; shift
TAPE=${TAPE:-data/round3.csv}
n=0
for cfg in "$@"; do
  set -- $cfg; tag=$1; shift
  if [ $# -gt 0 ]; then uv run python marek/research/round3/strategy/eval3.py "$F" --tape "$TAPE" --tag "$tag" --set "$@" &
  else uv run python marek/research/round3/strategy/eval3.py "$F" --tape "$TAPE" --tag "$tag" & fi
  n=$((n+1))
  if [ "$TAPE" = synth ] && [ $((n % 2)) -eq 0 ]; then wait; fi
done
wait
