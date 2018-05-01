#!/bin/bash

#SBATCH --time=00:15:00
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
	# 1 - square root
	# 2 - poisson
	# 3 - modulated poission
# third param is 1 (gaussian) or 0 (helper_functions asymmetry) calculation for normalization
# if third param is 1:
#   4/5 [optional] params are (in log coordinates) mean and std of gaussian
#   if not given, then they will be chosen from the optimization (if done) or randomly from a distrubition
python plotting.py $SLURM_ARRAY_TASK_ID 1 1
 
# leave a blank line at the end

