#!/bin/bash

#SBATCH --time=00:25:00
#SBATCH --mem=1500MB

#SBATCH --job-name=phasePlots

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=phi_%A_%a.out
#SBATCH --error=phi_%A_%a.err

module purge
source /home/pl1465/SF_diversity/Analysis/tf2.7/python2.7.12/bin/activate
module load seaborn/0.7.1

# 1st arg - cell #
# 2nd arg - dispersion (0 - single gratings; 1 - mixture)
# 3rd arg - plot phase/response by condition?
# 4th arg - make summary plots of rvc fits, phase advance fits?
python phase_plotting.py $SLURM_ARRAY_TASK_ID 0 0 1
 
# leave a blank line at the end

