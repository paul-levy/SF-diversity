#!/bin/bash

#SBATCH --time=00:25:00
#SBATCH --mem=1500MB

#SBATCH --job-name=descrFits

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=df_%A_%a.out
#SBATCH --error=df_%A_%a.err

module purge
source /home/pl1465/SF_diversity/Analysis/tf2.7/python2.7.12/bin/activate
module load seaborn/0.7.1

# 1st arg - cell #
# 2nd arg - disp (0 or 1)
# 3rd arg - fit phase advance?
# 4th arg - fit rvc?
# 5th arg - fit DoG descriptive sf tuning?
# 6th arg - optional - direction
# 7th arg - optional - regularize the gains in the DoG fits (<0.1 works well...)
python descr_fits.py $SLURM_ARRAY_TASK_ID 0 0 1 0 1 0
python descr_fits.py $SLURM_ARRAY_TASK_ID 1 0 1 0 1 0
 
# leave a blank line at the end

