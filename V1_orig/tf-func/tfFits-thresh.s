#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=8GB

#SBATCH --job-name=princeTFv1flat

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=tf_%A_%a.out
#SBATCH --error=tf_%A_%a.err

module purge
source /home/pl1465/SF_diversity/tf2.7/python2.7.12/bin/activate

# params are: cellNum, stopThresh, learning rate, lossType, fitType, subset_frac, initFromCurr
#   lossType:
#     1 - sqrt
#     2 - poisson
#     3 - modPoiss
#   fitType:
#     1 - flat normalization
#     2 - gaussian weighting of norm responses
#     3 - (flexible) gaussian c50 filter
#   subset_frac = 0 means take all data (no subsampling)
python mod_resp_trackNLL.py $SLURM_ARRAY_TASK_ID 1e-6 0.1 3 1 1 0
 
# leave a blank line at the end

