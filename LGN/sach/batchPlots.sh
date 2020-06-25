#!/bin/bash

source activate lcv-python

# second param is loss type (sf tuning):
	# 1 - poisson
	# 2 - square root
	# 3 - sach's loss function (See descr_fit.py)
# third param is DoG model (sf tuning):
	# 1 - sach's
	# 2 - tony's 
# 4th param is load from file (1; Tony's fits) or use params from my fits (0) --- use 0

for run in {1..34}
do
  python plotting.py $run 3 1 0 & 
  #python plotting.py $run 3 1 0 & 
done

 
# leave a blank line at the end

