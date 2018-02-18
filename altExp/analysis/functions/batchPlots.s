#!/bin/bash

#SBATCH --time=00:45:00
#SBATCH --mem=1500M0B

#SBATCH --job-name=sfPlots

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=plt_%A_%a.out
#SBATCH --error=plt_%A_%a.err

module purge
source /home/pl1465/SF_diversity/Analysis/tf2.7/python2.7.12/bin/activate

python plotting.py $SLURM_ARRAY_TASK_ID
 
# leave a blank line at the end

