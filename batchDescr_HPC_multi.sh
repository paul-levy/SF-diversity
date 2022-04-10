#!/bin/bash

JOINT=$1
BOOT=${2:-0}
LONG=${3:-0}
LOSS=${4:-2}

if [ $LONG -eq 0 ]; then
    PYCALL="batchDescr_HPC.sh"
    echo 'haha'
elif [ $LONG -eq 1 ]; then
    PYCALL="batchDescr_HPC_longer.sh"
    echo 'long'
elif [ $LONG -eq 2 ]; then
    PYCALL="batchDescr_HPC_muchlonger.sh"
    echo 'longer'
fi

sbatch --array=1-81 $PYCALL LGN/ 0 1 $BOOT $JOINT $LOSS
sbatch --array=1-81 $PYCALL V1/ 0 1 $BOOT $JOINT $LOSS
sbatch --array=1-59 batchDescr_HPC.sh V1_orig/ 0 1 $BOOT $JOINT $LOSS
sbatch --array=1-47 $PYCALL V1_BB/ 0 1 $BOOT $JOINT $LOSS
sbatch --array=1-8 $PYCALL altExp/ 0 1 $BOOT $JOINT $LOSS
