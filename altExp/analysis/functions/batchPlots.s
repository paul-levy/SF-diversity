#!/bin/bash

#SBATCH --time=00:25:00
#SBATCH --mem=1500MB

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
# third param is crf_fit_type: (i.e. what loss function for naka-rushton fits)
        # same as above
# fourth param is descr_fit_type: (i.e. what loss function for descriptive gaussian fits)
        # same as above
# fifth param is 0 (no norm sims) or 1 (do normalization simulations)
# sixth param is 1 (gaussian) or 0 (helper_functions asymmetry) calculation for normalization
# if sixth param is 1:
#   7/8 [optional] params are (in log coordinates) mean and std of gaussian
# if sixth param is 2:
#   7/8/9 [optional] params are std of left/right halves, and offset (i.e. bottom/lowest c50)
python plotting.py $SLURM_ARRAY_TASK_ID 3 1 1 0 0
 
# leave a blank line at the end

