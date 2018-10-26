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

# second param is loss type (sf tuning):
	# 1 - poisson
	# 2 - square root
	# 3 - sach's loss function (See descr_fit.py)
# third param is DoG model (sf tuning):
	# 1 - sach's
	# 2 - tony's 
# 4th param is load from file (1; Tony's fits) or use params from my fits (0)
python plotting.py $SLURM_ARRAY_TASK_ID 3 2 0
 
# leave a blank line at the end

