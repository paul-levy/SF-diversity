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

### TODO: Fix here and in helper_fcns (and thus in all plotting...) to access correct rvcFits
###       ALSO must make rvcFits (or get_spikes - make the choice) access F0/F1 depending on 

for run in {1..16}
do
  python plot_descr.py $run V1/ 0 2 1 1 &
  python plot_descr.py $run V1/ 0 3 1 1 &
done

# leave a blank line at the end
