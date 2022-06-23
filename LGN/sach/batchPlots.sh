#!/bin/bash

source activate lcv-python

# second param is loss type (sf tuning):
	# 1 - poisson
	# 2 - square root
	# 3 - sach's loss function (See descr_fit.py)
# third param is DoG model (sf tuning):
        # 0 - flex gauss
	# 1 - sach's
	# 2 - tony's 
# 4th param is RVC model
        # 0 - Tony
        # 1 - NR
        # 2 - Pierce NR

DOG_MOD=${1:-1}
DOG_LOSS=${2:-3}
RVC_MOD=${3:-0}
JOINT=${4:-0}
HPC=${5:-0}
PHADJ=${6:-0}
SEM=${7:-0}
ZFREQ=${8:-1}

for run in {1..34}
do
  python3.6 plotting.py $run $DOG_LOSS $DOG_MOD $RVC_MOD $JOINT $HPC $PHADJ $SEM $ZFREQ 0 &
done

 
# leave a blank line at the end

