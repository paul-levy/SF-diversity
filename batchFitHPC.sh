#!/bin/bash
	
#SBATCH --time=8:00:00
#SBATCH --mem=8GB

#SBATCH --job-name=modresp

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=MR_%A_%a.out
#SBATCH --error=MR_%A_%a.err

# params are: cellNum, expDir, lossType, fitType, initFromCurr
# see model_responses.py for additional details

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

### TEMP GUIDE (as of 20.05.07)
# V1/ - 8 cells
# V1_orig/ - 3 cells
# altExp   - 3 cells
###

module purge
module load python3/intel/3.6.3
module load numpy/python3.6/intel/1.14.0
#source activate lcv-python

python model_responses.py $SLURM_ARRAY_TASK_ID V1/ 2 4 1 0 1 0.1 1
python model_responses.py $SLURM_ARRAY_TASK_ID V1/ 2 4 2 0 1 0.1 1

#python model_responses.py $SLURM_ARRAY_TASK_ID V1_orig/ 2 4 1 0 0 0.1 1 
#python model_responses.py $SLURM_ARRAY_TASK_ID V1_orig/ 2 4 2 0 0 0.1 1 

#python model_responses.py $SLURM_ARRAY_TASK_ID altExp/ 2 4 1 0 0 0.1 1 
#python model_responses.py $SLURM_ARRAY_TASK_ID altExp/ 2 4 2 0 0 0.1 1 

