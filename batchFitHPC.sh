#!/bin/bash
	
#SBATCH --time=6:00:00
#SBATCH --mem=4GB

#SBATCH --ntasks=4 

#SBATCH --job-name=modresp

#SBATCH --mail-user=pl1465@nyu.edu
#SBATCH --mail-type=ALL

#SBATCH --output=./hpc_out/MR_%A_%a.out
#SBATCH --error=./hpc_out/MR_%A_%a.err

module purge
module load anaconda3/2020.02/

source /share/apps/anaconda3/2020.02/etc/profile.d/conda.sh
conda activate /scratch/pl1465/SF_diversity/pytorch/
export PATH=/scratch/pl1465/SF_diversity/pytorch/bin:$PATH

EXP_DIR=$1
EXC_TYPE=$2
LOSS=$3

# - flat gain control, LGN with separate M&P RVC
srun -n1 --nodes=1 python model_responses_pytorch.py $SLURM_ARRAY_TASK_ID $EXP_DIR $EXC_TYPE $LOSS 1 1 0 1 0.10 1 1 -1 1 &
# - weighted gain control, LGN with separate M&P RVC
srun -n1 --nodes=1 python model_responses_pytorch.py $SLURM_ARRAY_TASK_ID $EXP_DIR $EXC_TYPE $LOSS 2 1 0 1 0.10 1 1 -1 1 &
# - flat gain control, LGN with common M&P RVC
srun -n1 --nodes=1 python model_responses_pytorch.py $SLURM_ARRAY_TASK_ID $EXP_DIR $EXC_TYPE $LOSS 1 1 0 1 0.10 1 1 -1 2 &
# - weighted gain control, LGN with common M&P RVC
srun -n1 --nodes=1 python model_responses_pytorch.py $SLURM_ARRAY_TASK_ID $EXP_DIR $EXC_TYPE $LOSS 2 1 0 1 0.10 1 1 -1 2 &

wait
