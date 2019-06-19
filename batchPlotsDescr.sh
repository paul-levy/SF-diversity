#!/bin/bash

### README
# have you set the dataList name?
# have you set the fitList base name?
# have you set the descrFits base name?
### see plot_simple.py for changes/details

# second param is expDir (e.g. V1/ or LGN/)
# third param is descrMod:
	# 0 - flex
	# 1 - sach
	# 2 - tony
# fourth param is loss type
        # 1 - lsq
        # 2 - sqrt
        # 3 - poiss
        # 4 - sach
# fifth param is f0/f1 (i.e. if 1, load rvcFits)
# sixth param is std/sem as variance measure: (1 sem (default))

source activate lcv-python

for run in {1..59}
do
  python plot_descr.py $run V1_orig/ 0 3 0 1 &
  python plot_descr.py $run V1_orig/ 1 3 0 1 &
done

# leave a blank line at the end
