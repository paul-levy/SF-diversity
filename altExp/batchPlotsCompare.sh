#!/bin/bash

# second param is loss_type:
	# 1 - square root
	# 2 - poisson
	# 3 - modulated poission
	# 4 - chi squared
# third param is log_y: (1 for log y coordinate)

for run in {1..8}
do
  /e/2.3/p3/wangzhuo/anaconda3/bin/python3 plot_compare.py $run 4 0 &
done

# leave a blank line at the end

