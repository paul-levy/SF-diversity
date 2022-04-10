#!/bin/bash
	
#SBATCH --time=6:30:00
#SBATCH --mem-per-cpu=1GB

#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1

#SBATCH --job-name=sf_descr

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=./hpc_out/df_%A_%a.out
#SBATCH --error=./hpc_out/df_%A_%a.err

module purge

EXP_DIR=$1
RVC_FIT=$2
DESCR_FIT=$3
BOOT_REPS=$4
JOINT=${5:-0}
LOSS=${6:-2}

singularity exec --nv \
	    --overlay /scratch/pl1465/pyt_220124/pytorch_greene.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; bash batchDescr_sep_HPC.sh $SLURM_ARRAY_TASK_ID $EXP_DIR $RVC_FIT $DESCR_FIT $BOOT_REPS $JOINT 0 $LOSS"

