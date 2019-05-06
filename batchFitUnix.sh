#!/bin/bash

### README
# Have you set the dataList name (to read)?
# Have you set the descriptive fit name (to read)?
# Have you set the fitList name (to save)?
# Did you specify if model recovery or not?
### Go to model_responses.py first

# params are: cellNum, expDir, lossType, fitType, initFromCurr, trackSteps
# see model_responses.py for additional details

### GUIDE (as of 19.05.04)
# V1/ - use dataList_glx.npy, 17 cells
# V1/ - model recovery (dataList_glx_mr; mr_fitList...), 10 cells
# V1-_orig - model recovery (dataList_mr; mr_fitList...), 10 cells
# V1-_orig - standard, 59 cells
# altExp   - standard, 8 cells
###

source activate lcv-python

#for run in {1..17}; do
#  python model_responses.py $run V1/ 4 1 0 0 &
#  python model_responses.py $run V1/ 4 2 0 0 &
#done

# for model recovery
for run in {1..10}; do
  python model_responses.py $run V1/ 4 1 0 0 &
  python model_responses.py $run V1/ 4 2 0 0 &
done
