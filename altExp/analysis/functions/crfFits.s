#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --mem=4GB

#SBATCH --job-name=princeCRF

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=cF_%A_%a.out
#SBATCH --error=cF_%A_%a.err

module purge
source /home/pl1465/SF_diversity/Analysis/tf2.7/python2.7.12/bin/activate

python crf_fit.py $SLURM_ARRAY_TASK_ID 0 1
python crf_fit.py $SLURM_ARRAY_TASK_ID 1 1
  
# leave a blank line at the end

