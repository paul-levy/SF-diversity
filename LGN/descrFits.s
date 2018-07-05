#!/bin/bash

#SBATCH --time=0:20:00
#SBATCH --mem=1GB

#SBATCH --job-name=princeDescr

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=df_%A_%a.out
#SBATCH --error=df_%A_%a.err

module purge
source /home/pl1465/SF_diversity/Analysis/tf2.7/python2.7.12/bin/activate

# cellNum, numRepititions, baseline sub yes (1) or no (0), loss_type
# loss_type: 1 - poiss
#            2 - sqrt
#            3 - sach (see descr_fit.py for details)
python descr_fit.py $SLURM_ARRAY_TASK_ID 4 0 3
  
# leave a blank line at the end

