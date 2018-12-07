#!/bin/bash

#SBATCH --time=00:15:00
#SBATCH --mem=1500MB

#SBATCH --job-name=fourier

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=ft_%A_%a.out
#SBATCH --error=ft_%A_%a.err

module purge
source /home/pl1465/SF_diversity/Analysis/tf2.7/python2.7.12/bin/activate
module load seaborn/0.7.1

python fourier.py $SLURM_ARRAY_TASK_ID
