#!/bin/bash

# first param is cellNum
# second param is loss_type:
	# 1 - square root
	# 2 - poisson
	# 3 - modulated poission
	# 4 - chi squared
# third param is expDir (e.g. V1/ or LGN/)
# fourth param is fit type (i.e. weighted or flat)
# fifth param is duration of output video (in seconds, e.g. 5)
# sixth param is # steps of optimization b/t adjacent frames in the video, e.g. 5 or 50 (typically, there are O(500, 1000) opt steps)

source activate lcv-python

for run in {1..5}
do
  python plot_animate.py $run 4 V1/ 1 10 25 &
  python plot_animate.py $run 4 V1/ 2 10 25 &
done

# leave a blank line at the end

