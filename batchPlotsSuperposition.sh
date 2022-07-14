#!/bin/bash

source activate pytorch-lcv

### GUIDE (as of 19.11.05)
# V1/ - use dataList_glx.npy, was 35 cells -- now 56 (as of m681)
# V1/ - model recovery (dataList_glx_mr; mr_fitList...), 10 cells
# V1_orig/ - model recovery (dataList_mr; mr_fitList...), 10 cells
# V1_orig/ - standard, 59 cells
# altExp   - standard, 8 cells
# LGN/ - standard, 77 cells

## optional arguments after expDir/
## - use_mod_resp (default is 0; put 1 to plot superposition with model responses; in 2, it uses the pytorch model fits)
## - fitType (default is 2 [weighted]; use 1 for flat; etc)
## - excType (1 for gauss deriv [default]; 2 for flex. gauss)
## - useHPCfit (1 to use it; 0 for not)
## - conType (what lgnConType for the model fit?)
## - lgnFrontEnd (which lgnFrontEnd for the model fit?)

EXP_DIR=$1

if [ "$EXP_DIR" = "altExp/" ]; then
  for run in {1..8}
  do
    python3.6 plot_superposition.py $run altExp/ & # with data
  done
fi

if [ "$EXP_DIR" = "LGN/" ]; then
  for run in {1..25}
  do
    python3.6 plot_superposition.py $run LGN/ & # with data
  done
  wait
  for run in {26..50}
  do
    python3.6 plot_superposition.py $run LGN/ & # with data
  done
  wait
  for run in {51..81}
  do
    python3.6 plot_superposition.py $run LGN/ & # with data
  done
fi

if [ "$EXP_DIR" = "V1/" ]; then
  for run in {1..25}
  do
    python3.6 plot_superposition.py $run V1/ & # with data
  done
  wait
  for run in {26..50}
  do
    python3.6 plot_superposition.py $run V1/ & # with data
  done
  wait
  for run in {51..81}
  do
    python3.6 plot_superposition.py $run V1/ & # with data
  done
fi

# leave a blank line at the end
