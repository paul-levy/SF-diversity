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
# fourth param is crf_fit_type: (i.e. what loss function for naka-rushton fits)
        # 1 - lsq
	# 2 - square root
	# 3 - poisson
	# 4 - modulated poission
# fifth  param is descr_fit_type: (i.e. what loss function for descriptive gaussian fits)
        # same as above, no mod_poiss
# sixth param is 0 (no norm sims) or 1 (do normalization simulations)

for run in {1..8}
do
  /e/2.3/p3/wangzhuo/anaconda3/bin/python3 plotting_simple.py $run 4 1 3 3 0 &
  /e/2.3/p3/wangzhuo/anaconda3/bin/python3 plotting_simple.py $run 4 2 3 3 0 &
done

# leave a blank line at the end

