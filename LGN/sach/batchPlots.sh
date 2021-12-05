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
# 5th param is load from file (1; Tony's fits) or use params from my fits (0) --- use 0

DOG_MOD=${1:-1}
DOG_LOSS=${2:-3}
RVC_MOD=${3:-0}
JOINT=${4:-0}

for run in {1..34}
do
  python3.6 plotting.py $run $DOG_LOSS $DOG_MOD $RVC_MOD $JOINT 0 & 
done

 
# leave a blank line at the end

