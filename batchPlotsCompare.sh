#!/bin/bash

# second param is loss_type:
	# 1 - square root
	# 2 - poisson
	# 3 - modulated poission
	# 4 - chi squared
# third param is expDir (e.g. V1/ or LGN/)
# fourth param is log_y: (1 for log y coordinate)

source activate lcv-python

for run in {1..34}
do
  python plot_compare.py $run 4 LGN/ 0 &
done

# leave a blank line at the end

