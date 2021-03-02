#!/bin/bash

### README
# Have you set the dataList name (to read)?
# Have you set the descriptive fit name (to read)?
# Have you set the fitList name (to save)?
# Did you specify if model recovery or not?
### Go to model_responses.py first

########
# params are: cellNum, expDir, lossType, fitType, lgnFrontOn, initFromCurr, trackSteps, [kMult], [rvcMod]
########
#   cellNum, expDir - obvious
#   excType - which excitatory filter?
#      1 - deriv. order of gaussian (the first/usual/most common; default)
#      2 - flex. gauss (i.e. two-halved gaussian)
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
#   lgnFrontOn - will you have the lgnFrontEnd [1] (m::p f_c is 1::3), [2] (2::3), [99] (joint fit one LGN front end for all cells in the directory) or none [<=0]
#   initFromCurr - 
#      0 - don't...
#      1 - do
#      -1 - initialize from other fitType (e.g. if flat, initialize from weighted))
#   trackSteps - [1/0] save the steps in the optimization procedure
#   kMult - if using the chiSq loss function, what is the multiplier (see model_responses/helper_fcns)
#   newMethod - default is "old" method; 1 is new method (actually compute, work with PSTH rather than averaged)
#   vecCorrected - make vector math adjustments for F1 responses? Yes [1], No [0; default]
#   fixRespExp - (default is None, meaning a free resp. exp; otherwise, specify a value; use value <= 0 to make it None)
#   lgnConType - 1 is default; for 2 or 3, we average the RVCs of M & P and apply that function for M & P SF responses
#              --- for (2), the value is fixed at 0.5; for (3), it's simply equal to the optimized-for mWeight/1-mWeight
#   toPar - (Default is 0/False; 1 will be True, i.e. do parallelization)

#### see model_responses.py for additional details

### GUIDE (as of 19.11.05)
# V1/ - use dataList_glx.npy, was 35 cells -- now 56 (as of m681)
# V1/ - model recovery (dataList_glx_mr; mr_fitList...), 10 cells
# V1_orig/ - model recovery (dataList_mr; mr_fitList...), 10 cells
# V1_orig/ - standard, 59 cells
# altExp   - standard, 8 cells
# LGN/ - standard, 77 cells
###

### TEMP GUIDE (as of 20.05.07)
# V1/ - 8 cells
# V1_orig/ - 3 cells
# altExp   - 3 cells
# V1_BB/ - 41 cells (if dl=dataList_210222; else, 20 cells)
###


source activate pytorch-lcv

#################
#################
# Used with newest version of calling the model fits (adjusted on  21.02.07)
EXP_DIR=$1
EXC_TYPE=$2
LOSS=$3

for run in {1..41}; do
  # py ---------fun---------------------#----dir---e-l-f---i-t--k----vec-------
  #-----------------------------------------------------LGN--------nm--fixExp--
  #python3.6 model_responses_pytorch.py $run V1_BB/ 2 1 2 0 0 1 0.10 1 1 -1 &
  #python3.6 model_responses_pytorch.py $run V1_BB/ 2 1 1 1 0 1 0.10 1 1 -1 & # lgn (type a)
  #python3.6 model_responses_pytorch.py $run V1_BB/ 2 1 1 2 0 1 0.10 1 1 -1 # lgn type B

  ###########
  #### New model fitting procedure/comparisons, per T.M.; started on 21.02.06
  #### - Rather than comparing V1-only, wght against LGN-stage, flat, we'll do 2x2 matrix:
  #### -- LGN stage will ALWAYS be present, but we toggle:
  #### ---- flat (x) vs weighted gain control (y)
  #### ---- lgn stage with separate RVC for M&P (1) vs averaged RVC applied to both M&P (2)
  #### Last run per loss type:
  #### ---  sqrt: 21.02.09 20:43 (V1/, V1_BB; full matrix)
  #### --- poiss: 21.02.09 20:52 (V1/, V1_BB; full matrix)
  ###########
  # --- Current loss: Sqrt loss
  # py ---------fun---------------------#---dir-e-l-f---i-t--k----vec--con-----  
  #--------------------------------------------------LGN--------nm--fixExp-----
  # - (x1)
  python3.6 model_responses_pytorch.py $run $EXP_DIR $EXC_TYPE $LOSS 1 1 0 1 0.10 1 1 -1 1 &
  # - (y1)
  python3.6 model_responses_pytorch.py $run $EXP_DIR $EXC_TYPE $LOSS 2 1 0 1 0.10 1 1 -1 1 &
  # - (x2)
  python3.6 model_responses_pytorch.py $run $EXP_DIR $EXC_TYPE $LOSS 1 1 0 1 0.10 1 1 -1 2 &
  # - (y2)
  python3.6 model_responses_pytorch.py $run $EXP_DIR $EXC_TYPE $LOSS 2 1 0 1 0.10 1 1 -1 2 &
    ################
  #### END current procedure (started 21.02)
  ################


  # --- LGN flat, V1 wght, init, Poiss loss
  #python3.6 model_responses_pytorch.py $run V1/ 2 2 2 0 0 1 0.10 1 1 -1 & ### V1
  #python3.6 model_responses_pytorch.py $run V1_orig/ 2 2 2 0 0 1 0.10 1 1 -1 & ### V1
  #python3.6 model_responses_pytorch.py $run altExp/ 2 2 2 0 0 1 0.10 1 1 -1 & ### V1
  #python3.6 model_responses_pytorch.py $run V1_BB/ 2 2 2 0 0 1 0.10 1 1 -1 & ### V1
  #python3.6 model_responses_pytorch.py $run V1_BB/ 2 2 1 1 0 1 0.10 1 1 -1 &  # LGN
  #python3.6 model_responses_pytorch.py $run V1/ 2 2 1 1 0 1 0.10 1 1 -1 &  # LGN
  #python3.6 model_responses_pytorch.py $run V1_orig/ 2 2 1 1 0 1 0.10 1 1 -1 &  # LGN
  #python3.6 model_responses_pytorch.py $run altExp/ 2 2 1 1 0 1 0.10 1 1 -1 &  # LGN

  # py ---------fun---------------------#----dir---e-l-f---i-t--k----vec-------  
  #-----------------------------------------------------LGN--------nm--fixExp--

  # --- LGN flat, V1 wght, init, sqrt loss
  #python3.6 model_responses_pytorch.py $run V1_orig/ 2 1 2 0 0 1 0.10 1 1 -1 & ### V1
  #python3.6 model_responses_pytorch.py $run altExp/ 2 1 2 0 0 1 0.10 1 1 -1 & ### V1
  #python3.6 model_responses_pytorch.py $run V1/ 2 1 2 0 0 1 0.10 1 1 -1 & ### V1
  #python3.6 model_responses_pytorch.py $run V1_BB/ 2 1 2 0 0 1 0.10 1 1 -1 & ### V1
  #python3.6 model_responses_pytorch.py $run V1_BB/ 2 1 1 1 0 1 0.10 1 1 -1 & # LGN
  #python3.6 model_responses_pytorch.py $run V1/ 2 1 1 1 0 1 0.10 1 1 -1 & # LGN
  #python3.6 model_responses_pytorch.py $run V1_orig/ 2 1 1 1 0 1 0.10 1 1 -1 & # LGN
  #python3.6 model_responses_pytorch.py $run altExp/ 2 1 1 1 0 1 0.10 1 1 -1 & # LGN
  # py ---------fun---------------------#----dir---e-l-f---i-t--k----vec-------  
  #-----------------------------------------------------LGN--------nm--fixExp--

  # --- LGN flat, V1 wght, init, modPoiss loss
  #python3.6 model_responses_pytorch.py $run V1_orig/ 2 3 2 0 0 1 0.10 1 1 -1 & ### V1
  #python3.6 model_responses_pytorch.py $run altExp/ 2 3 2 0 0 1 0.10 1 1 -1 & ### V1
  #python3.6 model_responses_pytorch.py $run V1/ 2 3 2 0 0 1 0.10 1 1 -1 & ### V1
  #python3.6 model_responses_pytorch.py $run V1_BB/ 2 3 2 0 0 1 0.10 1 1 -1 & ### V1
  #python3.6 model_responses_pytorch.py $run V1_BB/ 2 3 1 1 0 1 0.10 1 1 -1 & # LGN
  #python3.6 model_responses_pytorch.py $run V1/ 2 3 1 1 0 1 0.10 1 1 -1 & # LGN
  #python3.6 model_responses_pytorch.py $run V1_orig/ 2 3 1 1 0 1 0.10 1 1 -1 & # LGN
  #python3.6 model_responses_pytorch.py $run altExp/ 2 3 1 1 0 1 0.10 1 1 -1 & # LGN
  # py ---------fun---------------------#----dir---e-l-f---i-t--k----vec-------  
  #-----------------------------------------------------LGN--------nm--fixExp--

done
