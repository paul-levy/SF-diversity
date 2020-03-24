#!/bin/bash

### README
# Have you set the dataList name (to read)?
# Have you set the descriptive fit name (to read)?
# Have you set the fitList name (to save)?
# Did you specify if model recovery or not?
### Go to model_responses.py first

########
# params are: cellNum, expDir, lossType, fitType, initFromCurr, trackSteps, [kMult], [rvcMod]
########
#   cellNum, expDir - obvious
#   lossType - which loss function
#      1 - (sqrt(mod) - sqrt(data)).^2
#      2 - poiss
#      3 - modPoiss
#      4 - chiSq
#   fitType - which type of normalization?
#      1 - no normalization [flat]
#      2 - gaussian-weighted normalization responses [most common]
#      3 - gaussian-weighted c50/norm "constant"
#      4 - gaussian-weighted (flexible/two-halved) normalization responses
#   initFromCurr - 
#      0 - don't...
#      1 - do
#      -1 - initialize from other fitType (e.g. if flat, initialize from weighted))
#   trackSteps - [1/0] save the steps in the optimization procedure
#   kMult - if using the chiSq loss function, what is the multiplier (see model_responses/helper_fcns)
#   rvcMod - which model of RVC are we using?
#      0 - movshon (see hf for details)
#      1 - naka-rushton [default]
#      2 - peirce

#### see model_responses.py for additional details

### GUIDE (as of 19.11.05)
# V1/ - use dataList_glx.npy, was 35 cells -- now 56 (as of m681)
# V1/ - model recovery (dataList_glx_mr; mr_fitList...), 10 cells
# V1_orig/ - model recovery (dataList_mr; mr_fitList...), 10 cells
# V1_orig/ - standard, 59 cells
# altExp   - standard, 8 cells
# LGN/ - standard, 77 cells
###

source activate lcv-python

#################
#################

for run in {1..77}; do
  python model_responses.py $run V1/ 4 1 -1 0 0.1 1 &
  python model_responses.py $run V1/ 4 2 -1 0 0.1 1 &

  #python model_responses.py $run V1_orig/ 4 1 0 0 0.01 &
  #python model_responses.py $run V1_orig/ 4 2 0 0 0.01 &

  #python model_responses.py $run LGN/ 4 1 0 0 0.1 0 &
  #python model_responses.py $run LGN/ 4 2 0 0 0.1 0 &

done

# for model recovery
#for run in {11..15}; do
#  python model_responses.py $run V1/ 4 1 0 0 &
#  python model_responses.py $run V1/ 4 2 0 0 &
#done
