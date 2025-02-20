#!/bin/bash

#SBATCH --time=00:45:00
#SBATCH --mem=1500M0B

#SBATCH --job-name=sfPlots

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=sand_%A_%a.out
#SBATCH --error=sand_%A_%a.err

module purge
source /home/pl1465/SF_diversity/tf2.7/python2.7.12/bin/activate
module load seaborn/0.7.1

# second param is fit_type:
	# 1 - least squares
	# 2 - square root
	# 3 - poisson
	# 4 - modulated poission
# third param is 1 (gaussian) or 0 (helper_functions asymmetry) calculation for normalization
# if third param is 1:
#   4/5 params are (in log coordinates) mean and std of gaussian
python norm_sandbox.py $SLURM_ARRAY_TASK_ID 4 1
#python norm_sandbox.py $SLURM_ARRAY_TASK_ID 4 1 0 0.1
 
# leave a blank line at the end

