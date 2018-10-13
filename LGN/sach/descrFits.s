#!/bin/bash

#SBATCH --time=6:30:00
#SBATCH --mem=1500MB

#SBATCH --job-name=descrFitsSach

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=df_%A_%a.out
#SBATCH --error=df_%A_%a.err

module purge
source /home/pl1465/SF_diversity/Analysis/tf2.7/python2.7.12/bin/activate

# cellNum, numRepititions [default =4], loss_type, DoG model
# loss_type: 1 - poiss
#            2 - sqrt
#            3 - sach [DEFAULT] (see descr_fit.py for details)
# DoG model: 1 - sach; 2 - Tony fomulation
python descr_fit.py $SLURM_ARRAY_TASK_ID 1000 3 1
  
# leave a blank line at the end
