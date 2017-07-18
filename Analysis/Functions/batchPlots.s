#!/bin/bash

#SBATCH --time=00:15:00
#SBATCH --mem=1500M0B

#SBATCH --job-name=sfPlots

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=plt_%A_%a.out
#SBATCH --error=plt_%A_%a.err

module purge
module load tensorflow/python2.7/20170218

python plotting.py $SLURM_ARRAY_TASK_ID
 
# leave a blank line at the end

