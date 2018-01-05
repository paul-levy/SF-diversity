#!/bin/bash

#SBATCH --time=1:30:00
#SBATCH --mem=4GB

#SBATCH --job-name=princeDescr

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=df_%A_%a.out
#SBATCH --error=df_%A_%a.err

module purge
source /home/pl1465/SF_diversity/Analysis/tf2.7/python2.7.12/bin/activate

python descr_fit.py $SLURM_ARRAY_TASK_ID 10
  
# leave a blank line at the end

