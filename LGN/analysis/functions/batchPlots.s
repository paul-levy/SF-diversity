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

# second param is plotType:
        # 0 - just f0 (i.e. DC mean)
        # 1 - just f1 (i.e. response amplitude at stimulus frequency, meaning drift rate)
        # 2 - f0 and f1
# for the below: put 0 to avoid plotting/loading that type
# third param is loss_type:
	# 1 - square root
	# 2 - poisson
	# 3 - modulated poission
# fourth param is fit_type:
	# 1 - flat normalization
	# 2 - gaussian weighting of normalization responses
	# 3 - c50 controlled by gaussian
# fifth param is crf_fit_type: (i.e. what loss function for naka-rushton fits)
        # 1 - lsq
	# 2 - square root
	# 3 - poisson
	# 4 - modulated poission
# sixth  param is descr_fit_type: (i.e. what loss function for descriptive gaussian fits)
        # same as above, no mod_poiss
# seventh param is 0 (no norm sims) or 1 (do normalization simulations)
python plotting.py $SLURM_ARRAY_TASK_ID 0 0 0 0 0 0 0
 
# leave a blank line at the end

