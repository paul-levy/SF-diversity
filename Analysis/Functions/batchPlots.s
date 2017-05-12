#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --time=00:05:00
#SBATCH --mem=1500MB

#SBATCH --job-name=princeDescr

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=df_%A_%a.out
#SBATCH --error=df_%A_%a.err

module purge
module load tensorflow/python2.7/20170218

python plotting.py $SLURM_ARRAY_TASK_ID
 
# leave a blank line at the end

