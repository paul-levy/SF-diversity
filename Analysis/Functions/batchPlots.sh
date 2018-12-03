#!/bin/bash

# second param is loss_type:
	# 1 - square root
	# 2 - poisson
	# 3 - modulated poission
	# 4 - chi squared
# third param is fit_type:
	# 1 - flat normalization
	# 2 - gaussian weighted normalization
	# 3 - c50/normalization "constant" filter
# fourth param is log_y: (1 for log y coordinate)

for run in {1..59}
do
  /e/2.3/p3/wangzhuo/anaconda3/bin/python3 plotting.py $run 4 1 0 &
  /e/2.3/p3/wangzhuo/anaconda3/bin/python3 plotting.py $run 4 2 0 &
done

# leave a blank line at the end

