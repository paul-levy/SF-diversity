#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --time=1:00:00
#SBATCH --mem=4GB

#SBATCH --job-name=princeDescr

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=df_%A_%a.out
#SBATCH --error=df_%A_%a.err

module purge
module load tensorflow/python2.7/20170218

for iii in $(seq 1 59); do
  python descr_fit.py ${iii} 25
done
 
# leave a blank line at the end

