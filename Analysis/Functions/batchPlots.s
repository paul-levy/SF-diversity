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
	# 4 - chi squared
# third param is fit_type:
	# 1 - flat normalization
	# 2 - gaussian weighted normalization
	# 3 - c50/normalization "constant" filter
# fourth param is log_y: (1 for log y coordinate)
python plotting.py $SLURM_ARRAY_TASK_ID 4 1 0
 
# leave a blank line at the end

