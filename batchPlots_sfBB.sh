#!/bin/bash

### README
### WARN: have you set the fitList base name?

# NOTE: assumes normType = 1 (flat) and normType = 2 (gauss) are present
# -- will choose normType = 1 for LGN, =2 for non-LGN front-end

# 2nd param is excType: 1 (gaussian deriv); 2 (flex. gauss)
# 3rd param is loss_type:
	# 1 - square root
	# 2 - poisson
	# 3 - modulated poission
	# 4 - chi squared
# 4 param is expDir (e.g. altExp/ or LGN/)
# 5 param is lgnFrontEnd (choose LGN type; will be comparing against non-LGN type)
# 6 param is diffPlot (i.e. plot everything relative to flat model prediction)
# 7 param is interpModel (i.e. interpolate model?)
# 8th param is kMult (0.01, 0.05, 0.10, usually...)
# 9th param is whether (1) or not (0) to do vector correction F1...
# 10th param is whether to include the onset transient correction for F1 responses (use onsetDur in mS to use (e.g. 100); 0 to do without)
# 11th param is respExpFixed (-1 for not fixed, then specific value for a fit with fixed respExp [e.g. 1 or 2])
# 12th param is std/sem as variance measure: (1 sem (default))

source activate pytorch-lcv

LOSS=2

for run in {1..20}
do
  ######
  ## New version
  ######
  # --------------------------e-l--dir--nrm-lgn-dif--kmul--onsr--sem-----
  # --------------------------------------con--inp----cor-rExp-------
  # modA: flat, fixed RVC, lgn A; modB: wght, fixed RVC, lgnA
  python3.6 plot_sfBB.py $run 2 $LOSS V1_BB/ 12 22 11 0 0 0.05 1 0 -1 1 & # no diff, not interpolated
  # modA: flat, fixed RVC, lgn A; modB: wght, standard RVC, lgnA
  python3.6 plot_sfBB.py $run 2 $LOSS V1_BB/ 12 21 11 0 0 0.05 1 0 -1 1 & # no diff, not interpolated
  # modA: flat, standard RVC, lgn A; modB: wght, standard RVC, lgnA
  python3.6 plot_sfBB.py $run 2 $LOSS V1_BB/ 12 11 11 0 0 0.05 1 0 -1 1 & # no diff, not interpolated
  # pytorch mod; modA: wght, fixed RVC, lgn A; modB: wght, standard RVC, lgnA
  python3.6 plot_sfBB.py $run 2 $LOSS V1_BB/ 22 21 11 0 0 0.05 1 0 -1 1 & # no diff, not interpolated

  ######
  ## End of new version
  ######


  #python3.6 plot_sfBB.py $run 2 1 V1_BB/ 1 0 0 0.05 1 0 -1 1 & # no diff, not interpolated
  #python3.6 plot_sfBB.py $run 2 2 V1_BB/ 1 0 0 0.05 1 0 -1 1 & # no diff, not interpolated
  #python3.6 plot_sfBB.py $run 2 3 V1_BB/ 1 0 0 0.05 1 0 -1 1 & # no diff, not interpolated
  # DATA ONLY, WITH PHASE CORRECTION  
  #python3.6 plot_sfBB.py $run -1 2 V1_BB/ 1 0 0 0.05 1 80 -1 1 & # no diff, not interpolated
  # --------------------------e-l--dir----dif--kmul--rExp------
  # ------------------------------------lgn-inp----corr-sem-------
done

# leave a blank line at the end

