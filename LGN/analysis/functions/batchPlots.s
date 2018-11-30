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

# for the below: put 0 to avoid plotting/loading that type (except for fit_type)
# second param is loss_type (for full model):
	# 1 - least squares
	# 2 - square root
	# 3 - poisson
	# 4 - modulated poission
# third param is fit_type (for full model):
	# 1 - flat normalization
	# 2 - gaussian weighting of normalization responses
	# 3 - c50 controlled by gaussian
# fourth param is crf_fit_type: (i.e. what loss function for naka-rushton fits)
        # 1 - lsq
	# 2 - square root
	# 3 - poisson
	# 4 - modulated poission
# fifth  param is sf_loss_type: (i.e. what loss function for difference of gaussian fits)
        # 1 - poiss
        # 2 - sqrt
        # 3 - sach
# sixth param is sf_DoG_model: (i.e. which DoG model)
        # 1 - sach
        # 2 - tony
# seventh param is 0 (no norm sims) or 1 (do normalization simulations)
# eighth param is -1 or +1 (phase direction for loading rvc/phAdv fits)
# ninth param is 0 (plot for viewing/general) vs 1 (plot for editing in illustrator)
python plotting.py $SLURM_ARRAY_TASK_ID 4 1 0 3 2 0 1 0 0
 
# leave a blank line at the end

