#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=4:00:00
#PBS -l mem=4GB
#PBS -N Fitzzz
#PBS -M pl1465@nyu.edu
#PBS -m abe
#PBS -j oe 
#PBS -t 1-40

module purge
module load python3/intel/3.5.1
module load tensorflow/python3.5.1/20161029

python3.5 model_responses.py $PBSARRAYID 40000 1
 
# leave a blank line at the end

