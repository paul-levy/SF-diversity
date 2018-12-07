#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --mem=4GB

#SBATCH --job-name=norm_resp

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=nr_%A_%a.out
#SBATCH --error=nr_%A_%a.err

module purge
source /home/pl1465/SF_diversity/tf2.7/python2.7.12/bin/activate

python create_norm_resp.py $SLURM_ARRAY_TASK_ID
 
# leave a blank line at the end

