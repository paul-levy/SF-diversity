#!/bin/bash

# second param is loss_type:
	# 1 - square root
	# 2 - poisson
	# 3 - modulated poission
	# 4 - chi squared
# third param is expInd
	# 1 - V1_orig
	# 2 - altExp (V1)
	# 3 - LGN
	# 4 - ...
# fourth param is log_y: (1 for log y coordinate)

for run in {1..34}
do
  /e/2.3/p3/wangzhuo/anaconda3/bin/python3 plot_compare.py $run 4 3 0 &
done

# leave a blank line at the end

