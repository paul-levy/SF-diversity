#!/bin/bash

#SBATCH --time=5:00:00
#SBATCH --mem=16GB

#SBATCH --job-name=princeTFHone

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=tfH_%A_%a.out
#SBATCH --error=tfH_%A_%a.err

module purge
source /home/pl1465/SF_diversity/Analysis/tf2.7/python2.7.12/bin/activate

# cellnum errThresh learningRate fitType subsampleFrac paramsFromCurrFits
#   fitType:
#     1 - sqrt
#     2 - poisson
#     3 - modPoiss
#   subset_frac = 0 means take all data (no subsampling)
python mod_resp_trackNLL.py $SLURM_ARRAY_TASK_ID 1e-8 0.005 2 0.9 1
 
# leave a blank line at the end

