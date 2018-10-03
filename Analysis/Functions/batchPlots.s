#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --mem=1500MB

#SBATCH --job-name=sfPlots

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=plt_%A_%a.out
#SBATCH --error=plt_%A_%a.err

module purge
source /home/pl1465/SF_diversity/Analysis/tf2.7/python2.7.12/bin/activate
module load seaborn/0.7.1

# second param is loss_type:
	# 1 - square root
	# 2 - poisson
	# 3 - modulated poission
# third param is fit_type:
	# 1 - flat normalization
	# 2 - gaussian weighted normalization
	# 3 - c50/normalization "constant" filter
# fourth param is log_y: (1 for log y coordinate)
# if third param is 1: standard asymmetric normalization (or just flat...)
# if third param is 2:
#   4/5 [optional] params are (in log coordinates) mean and std of gaussian
#   if not given, then they will be chosen from the optimization (if done) or randomly from a distrubition
# if third param is 3:
#   4/5/6 [optional] params are std of left/right halves, and offset (i.e. bottom/lowest c50), and peak of c50 curve
python plotting.py $SLURM_ARRAY_TASK_ID 3 2 1
 
# leave a blank line at the end

