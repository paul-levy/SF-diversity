#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=16GB

#SBATCH --job-name=evalModGauss

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=tfgfMP_%A_%a.out
#SBATCH --error=tfHgMP_%A_%a.err

module purge
source /home/pl1465/SF_diversity/tf2.7/python2.7.12/bin/activate

# params are: cellNum, stopThresh (NOT CURRENTLY USED), learning rate (NOT CURRENTLY USED), lossType, fitType, subset_frac (NOT CURRENTLY USED), initFromCurr
#   lossType:
#     1 - sqrt
#     2 - poisson
#     3 - modPoiss
#   fitType:
#     1 - flat normalization
#     2 - gaussian weighting of norm responses
#     3 - (flexible) gaussian c50 filter
#   subset_frac = 0 means take all data (no subsampling)
python evalModel.py $SLURM_ARRAY_TASK_ID 1e-5 0.1 3 2 1 0
 
# leave a blank line at the end
