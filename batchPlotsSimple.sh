#!/bin/bash

# second param is loss_type:
	# 1 - square root
	# 2 - poisson
	# 3 - modulated poission
	# 4 - chi squared
# third param is fit (normalization) type
        # 1 - flat
        # 2 - gaussian
# fourth param is expDir (e.g. V1/ or LGN/)
# fifth param is - are we plotting model recovery (1) or no (0)?
# sixth param is std/sem as variance measure: (1 sem (default))

source activate lcv-python

for run in {1..10}
do
  python plot_simple.py $run 4 1 V1_orig/ 1 1 &
  python plot_simple.py $run 4 2 V1_orig/ 1 1 &
done

# leave a blank line at the end
