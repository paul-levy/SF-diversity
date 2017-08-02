#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --mem=4GB

#SBATCH --job-name=princeDescr

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=df_%A_%a.out
#SBATCH --error=df_%A_%a.err

module purge
module load tensorflow/python2.7/20170218

python descr_fit.py $SLURM_ARRAY_TASK_ID 25 1
  
# leave a blank line at the end

