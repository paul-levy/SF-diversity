#!/bin/bash

# second param is loss_type:
	# 1 - square root
	# 2 - poisson
	# 3 - modulated poission
	# 4 - chi squared
# third param is expDir (e.g. V1/ or LGN/)
# fourth param is std/sem as variance measure: (1 sem (default))

source activate lcv-python

for run in {1..59}
do
  python plot_compare.py $run 4 V1_orig/ 1 &
done

# leave a blank line at the end

