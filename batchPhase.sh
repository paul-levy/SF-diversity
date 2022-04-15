#!/bin/bash

source activate lcv-python

# 1st arg - cell #
# 2nd arg - dispersion (0 - single gratings; 1 - mixture)
# 3rd arg - exp dir (e.g. V1/ or LGN/)
# 4th arg - plot phase/response by condition?
# 5th arg - make summary plots of rvc fits, phase advance fits?
# 6th arg - optional: direction (default is -1) 

FULL=${1:-0}

for run in {1..27}
do
  # LGN 
  python3.6 phase_plotting.py $run 0 LGN/ $FULL 1 1 &
done
wait
for run in {28..55}
do
  # LGN 
  python3.6 phase_plotting.py $run 0 LGN/ $FULL 1 1 &
done
wait
for run in {56..81}
do
  # LGN 
  python3.6 phase_plotting.py $run 0 LGN/ $FULL 1 1 &
done
wait

# leave a blank line at the end
