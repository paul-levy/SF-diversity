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

# second param is baseline_sub - for separate SF plots by contrast, subtract baseline from resp?
# third param is fit_type:
	# 1 - poisson
	# 2 - square root
	# 3 - sach's loss function (See descr_fit.py)
python plotting.py $SLURM_ARRAY_TASK_ID 3 1 0
 
# leave a blank line at the end

