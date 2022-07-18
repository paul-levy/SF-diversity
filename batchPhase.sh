#!/bin/bash

source activate lcv-python

# 1st arg - cell #
# 2nd arg - dispersion (0 - single gratings; 1 - mixture)
# 3rd arg - exp dir (e.g. V1/ or LGN/)
# 4th arg - plot phase/response by condition?
# 5th arg - make summary plots of rvc fits, phase advance fits?
# 6th arg - optional: direction (default is -1) 

EXP_DIR=$1
FULL=${2:-0}

if [ "$EXP_DIR" = "V1/" ] || [ "$EXP_DIR" = "LGN/" ]; then
  for run in {1..27}
  do
    python3.6 phase_plotting.py $run 0 $EXP_DIR $FULL 1 1 &
  done
  wait
  for run in {28..55}
  do
    python3.6 phase_plotting.py $run 0 $EXP_DIR $FULL 1 1 &
  done
  wait
  for run in {56..81}
  do
    python3.6 phase_plotting.py $run 0 $EXP_DIR $FULL 1 1 &
  done
fi
# leave a blank line at the end
