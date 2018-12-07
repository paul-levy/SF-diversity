#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --mem=1500MB

#SBATCH --job-name=sfPlots

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=plt_%A_%a.out
#SBATCH --error=plt_%A_%a.err

module purge
source /home/pl1465/SF_diversity/tf2.7/python2.7.12/bin/activate
module load seaborn/0.7.1

# second param is loss_type:
	# 1 - square root
	# 2 - poisson
	# 3 - modulated poission
	# 4 - chi squared
# third param is loss function for descriptive fits:
        # lsq (1), sqrt (2), poiss (3)
# fourth param is log_y: (1 for log y coordinate)
# automatically plots flat + weighted fits
python plot_compare.py $SLURM_ARRAY_TASK_ID 4 3 0
 
# leave a blank line at the end

