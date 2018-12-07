#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --mem=4GB

#SBATCH --job-name=princeRVC

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=rvc_%A_%a.out
#SBATCH --error=rvc_%A_%a.err

module purge
source /home/pl1465/SF_diversity/tf2.7/python2.7.12/bin/activate

# CELL_ID || DATASET || FIX/FREE-c50 || fitType || [optional: nIterations; default = 1, i.e. no repeats]
	# note: currently, crf_fit.py calls non-bootstrapped optimization; see crf_fit.py for details
        # DATASET:
                # 1 - altExp
                # 2 - LGN (my data, beginning with m675)
                # 3 - to be continued
	# fitType:
		# 1 - least squares
		# 2 - minimize sq(sqrt(pred) - sqrt(resp))
		# 3 - poisson
		# 4 - modulated poisson (Goris)

python crf_fit.py $SLURM_ARRAY_TASK_ID 2 0 2
python crf_fit.py $SLURM_ARRAY_TASK_ID 2 1 2

# leave a blank line at the end

