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
module load seaborn/0.7.1

# second param is fit_type:
	# 1 - least squares
	# 2 - square root
	# 3 - poisson
	# 4 - modulated poission
python plotting.py $SLURM_ARRAY_TASK_ID 4
 
# leave a blank line at the end

