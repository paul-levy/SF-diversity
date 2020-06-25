#!/bin/bash

### README
# have you set the dataList name?
# have you set the fitList base name?
# have you set the descrFits base name?
### see plot_simple.py for changes/details

# second param is expDir (e.g. V1/ or LGN/)
# third param is descrMod: (descr)
	# 0 - flex
	# 1 - sach
	# 2 - tony
# fourth param is loss type (descr)
        # 1 - lsq
        # 2 - sqrt
        # 3 - poiss
        # 4 - sach
# fifth param is joint (descr) - 0/1
# sixth param is f0/f1 (i.e. if 1, load rvcFits)
# seventh param is which rvcModel to get/plot (0: movshon; 1: naka-rushton; 2: peirce)
# eigth param is std/sem as variance measure: (1 sem (default))
# ninth param is force log Y for byDisp/allCon and CRF/allSfs: (0/no (default))

source activate lcv-python

### TODO: Fix here and in helper_fcns (and thus in all plotting...) to access correct rvcFits
###       ALSO must make rvcFits (or get_spikes - make the choice) access F0/F1 depending on 

### GUIDE (as of 19.11.05)
# V1/ - use dataList_glx.npy, was 35 cells -- now 56 (as of m681)
# V1/ - model recovery (dataList_glx_mr; mr_fitList...), 10 cells
# V1_orig/ - model recovery (dataList_mr; mr_fitList...), 10 cells
# V1_orig/ - standard, 59 cells
# altExp   - standard, 8 cells
# LGN/ - standard, 77 cells

for run in {1..77}
do  
  python plot_descr.py $run LGN/ 0 2 0 1 0 1 1 &
  #python plot_descr.py $run LGN/ 1 4 0 1 0 1 1 &
  #python plot_descr.py $run V1/ 0 2 0 1 1 1 1 &
  #python plot_descr.py $run altExp/ 0 2 0 0 1 1 1 &
  #python plot_descr.py $run V1_orig/ 0 2 0 0 1 1 1 &
  #python plot_descr.py $run altExp/ 0 2 0 1 1 1 1 &
  #python plot_descr.py $run LGN/ 1 1 0 1 0 1 1 &
done


#python plot_descr.py 3 LGN/ 1 4 0 1 0 1 1
#python plot_descr.py 8 LGN/ 1 4 0 1 0 1 1 
#python plot_descr.py 19 LGN/ 1 4 0 1 0 1 1 
#python plot_descr.py 24 LGN/ 1 4 0 1 0 1 1 
#python plot_descr.py 25 LGN/ 1 4 0 1 0 1 1 
#python plot_descr.py 28 LGN/ 1 4 0 1 0 1 1 
#python plot_descr.py 31 LGN/ 1 4 0 1 0 1 1 
#python plot_descr.py 33 LGN/ 1 4 0 1 0 1 1 
#python plot_descr.py 41 LGN/ 1 4 0 1 0 1 1 
#python plot_descr.py 45 LGN/ 1 4 0 1 0 1 1 
#python plot_descr.py 53 LGN/ 1 4 0 1 0 1 1 
#python plot_descr.py 54 LGN/ 1 4 0 1 0 1 1 
#python plot_descr.py 70 LGN/ 1 4 0 1 0 1 1 

# leave a blank line at the end
