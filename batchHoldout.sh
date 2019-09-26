#!/bin/bash

### README
# Have you set the dataList name (to read)?
# Have you set the descriptive fit name (to read)?
# Have you set the fitList name (to save)?
# Did you specify if model recovery or not?
### Go to model_responses.py first

# params are: cellNum, expDir, lossType, fitType, initFromCurr, trackSteps
# initFromCurr - (0; don't...1; do;...-1; initialize from other fitType (i.e. if flat, initialize from weighted))
# see model_responses.py for additional details

### GUIDE (as of 19.05.04)
# V1/ - use dataList_glx.npy, 17 cells
# V1/ - model recovery (dataList_glx_mr; mr_fitList...), 10 cells
# V1_orig/ - model recovery (dataList_mr; mr_fitList...), 10 cells
# V1_orig/ - standard, 59 cells
# altExp   - standard, 8 cells
###

source activate lcv-python

for run in {1..8}; do
  python holdoutFits.py $run altExp/ 4 1 0 0 &
  python holdoutFits.py $run altExp/ 4 2 0 0 &

  python holdoutFits.py $run altExp/ 4 1 1 0 &
  python holdoutFits.py $run altExp/ 4 2 1 0 &

done
