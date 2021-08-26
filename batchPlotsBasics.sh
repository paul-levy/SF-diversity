#!/bin/bash

source activate lcv-python

### GUIDE (as of 19.11.05)
# V1/ - use dataList_glx.npy, was 35 cells -- now 56 (as of m681)
# V1/ - model recovery (dataList_glx_mr; mr_fitList...), 10 cells
# V1_orig/ - model recovery (dataList_mr; mr_fitList...), 10 cells
# V1_orig/ - standard, 59 cells
# altExp   - standard, 8 cells
# LGN/ - standard, 77 cells

### expDir/ compareSfMix forceSimple

EXP_DIR=$1

if [ "$EXP_DIR" = "V1_BB/" ]; then
  for run in {1..25}
  do
    python plot_basics.py $run V1_BB/ 0 0 &
    python plot_basics.py $run V1_BB/ 0 1 &
  done
  wait
  for run in {26..47}
  do
    python plot_basics.py $run V1_BB/ 0 0 &
    python plot_basics.py $run V1_BB/ 0 1 &
  done
  wait
fi

if [ "$EXP_DIR" = "V1/" ]; then
  for run in {1..45}
  do 
    python3.6 plot_basics.py $run $EXP_DIR 0 0 &
    python3.6 plot_basics.py $run $EXP_DIR 0 1 &
  done
  wait
  for run in {46..81}
  do 
    python3.6 plot_basics.py $run $EXP_DIR 0 0 &
    python3.6 plot_basics.py $run $EXP_DIR 0 1 &
  done
  wait
fi

if [ "$EXP_DIR" = "LGN/" ]; then
  for run in {1..25}
  do 
    python3.6 plot_basics.py $run $EXP_DIR 0 0 &
    python3.6 plot_basics.py $run $EXP_DIR 0 1 &
  done
  wait
  for run in {26..50}
  do 
    python3.6 plot_basics.py $run $EXP_DIR 0 0 &
    python3.6 plot_basics.py $run $EXP_DIR 0 1 &
  done
  wait
  for run in {50..81}
  do 
    python3.6 plot_basics.py $run $EXP_DIR 0 0 &
    python3.6 plot_basics.py $run $EXP_DIR 0 1 &
  done
  wait
fi

if [ "$EXP_DIR" = "V1_orig/" ]; then
  for run in {1..20}
  do 
    python3.6 plot_basics.py $run $EXP_DIR 0 0 &
    python3.6 plot_basics.py $run $EXP_DIR 0 1 &
  done
  wait
  for run in {21..40}
  do 
    python3.6 plot_basics.py $run $EXP_DIR 0 0 &
    python3.6 plot_basics.py $run $EXP_DIR 0 1 &
  done
  wait
  for run in {41..59}
  do 
    python3.6 plot_basics.py $run $EXP_DIR 0 0 &
    python3.6 plot_basics.py $run $EXP_DIR 0 1 &
  done
  wait
fi

if [ "$EXP_DIR" = "altExp/" ]; then
  for run in {1..8}
  do 
    python3.6 plot_basics.py $run $EXP_DIR 0 0 &
    python3.6 plot_basics.py $run $EXP_DIR 0 1 &
  done
  wait
fi

