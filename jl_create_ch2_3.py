import numpy as np
import os, sys
import helper_fcns as hf
import scipy.stats as ss
from scipy.stats.mstats import gmean
from scipy.stats import ks_2samp, kstest, linregress
import itertools, pdb

area = sys.argv[1];
jointType = int(sys.argv[2]);

brief=True; # save boot iter metrics, too?

if area == 'V1':

  ########################
  #### V1
  ########################

  expDirs = ['V1_orig/', 'altExp/', 'V1/', 'V1_BB/']
  expNames = ['dataList.npy', 'dataList.npy', 'dataList_221011.npy', 'dataList_210721.npy']

  nExpts = len(expDirs);

  # these are usually same for all expts...we'll "tile" below
  useHPCfit = 1;

  ### model specifications
  vecCorrected = 0
  pytorch_mod = 1; #
  excType, lossType = 1, 1
  fixRespExp = 2; # i
  scheduler = False;
  normA, normB = 2, 1; # for weighted, 5 is wghtGain (2 is "old" weighted model)
  lgnA, lgnB = 0, 0; # (1) means have an LGN, original filters
  conA, conB = 1, 1; # (1) separate M&P; (2) is equal M+P; (4) is all parvo
  loc_str = 'HPC' if useHPCfit else '';
  ### organize the mod specs....
  modSpecs = dict();
  wght_specs = dict([('excType', excType),
                     ('lossType', lossType),
                     ('normType', normA),
                     ('lgnOn', lgnA),
                     ('lgnType', conA),
                     ('fixRespExp', fixRespExp),
                     ('scheduler', scheduler),
                   ])
  flat_specs = dict([('excType', excType),
                     ('lossType', lossType),
                     ('normType', normB),
                     ('lgnOn', lgnB),
                     ('lgnType', conB),
                     ('fixRespExp', fixRespExp),
                     ('scheduler', scheduler),
                   ])
  modSpecs['wght'] = wght_specs;
  modSpecs['flat'] = flat_specs;
  ### organize the mod specs....
  fitBase = 'fitList%s_pyt_nr221119d%s%s' % (loc_str, '_noRE' if fixRespExp is not None else '', '_noSched' if scheduler==False else '');
  fitNamesWght = [hf.fitList_name(fitBase, normA, lossType, lgnA, conA, vecCorrected, fixRespExp=None, excType=excType)]
  fitNamesFlat = [hf.fitList_name(fitBase, normB, lossType, lgnB, conB, vecCorrected, fixRespExp=None, excType=excType)]
  cv_fitNamesWght = [hf.fitList_name(fitBase, normA, lossType, lgnA, conA, vecCorrected, fixRespExp=None, CV=True, excType=excType)]
  cv_fitNamesFlat = [hf.fitList_name(fitBase, normB, lossType, lgnB, conB, vecCorrected, fixRespExp=None, CV=True, excType=excType)]
  ####
  # descrFits - loss type determined by comparison (choose best; see modCompare.ipynb::Descriptive Fits)
  ####
  dogMod = 3; # 1 (sach), 2 (Tony), 3 (d-DoG-S), 4 (N/A), 5 (d-DoG-S Hawk)
  # dir. order is 'V1_orig/', 'altExp/', 'V1/', 'V1_BB/'
  if jointType == 10:
    #dogNames = ['descrFitsHPC_220811vEs_sqrt_ddogs_JTflankShiftCopyCtrRaSlope.npy', 'descrFitsHPC_220811vEs_phAdj_sqrt_ddogs_JTflankShiftCopyCtrRaSlope.npy', 'descrFitsHPC_220811vEs_phAdj_sqrt_ddogs_JTflankShiftCopyCtrRaSlope.npy']
    # TEMPORARY UNTIL WE RUN V1/joint10 on the V1-only dataList [currently running on 22.11.27]
    dogNames = ['descrFitsHPC_221126vEs_sqrt_ddogs_JTflankShiftCopyCtrRaSlope.npy', 'descrFitsHPC_221126vEs_sqrt_ddogs_JTflankShiftCopyCtrRaSlope.npy', 'descrFitsHPC_221126vEs_phAdj_sqrt_ddogs_JTflankShiftCopyCtrRaSlope.npy', 'descrFitsHPC_220826vEs_phAdj_sqrt_ddogs_JTflankShiftCopyCtrRaSlope.npy']
  # THE BELOW ARE NOT UPDATED AS OF 22.11.27
  if jointType == 0:
    dogNames = ['descrFitsHPC_220811vEs_sqrt_ddogs.npy', 'descrFitsHPC_220811vEs_phAdj_sqrt_ddogs.npy', 'descrFitsHPC_220811vEs_phAdj_sqrt_ddogs.npy'];
  if jointType == 7:
    dogNames = ['descrFitsHPC_220609_sqrt_ddogs_JTflankSurrShapeCtrRaSlope.npy'];
  if jointType == 9:
    dogNames = ['descrFitsHPC_220801vEs_phAdj_sqrt_ddogs_JTflankFixedCopyCtrRaSlope.npy', 'descrFitsHPC_220801vEs_phAdj_sqrt_ddogs_JTflankFixedCopyCtrRaSlope.npy', 'descrFitsHPC_220801vEs_sqrt_ddogs_JTflankFixedCopyCtrRaSlope.npy']
    #dogNames = ['descrFitsHPC_220721vEs_phAdj_sqrt_ddogs_JTflankFixedCopyCtrRaSlope.npy', 'descrFitsHPC_220721vEs_phAdj_sqrt_ddogs_JTflankFixedCopyCtrRaSlope.npy', 'descrFitsHPC_220720vEs_sqrt_ddogs_JTflankFixedCopyCtrRaSlope.npy']
    #dogNames = ['descrFitsHPC_220721_phAdj_sqrt_ddogs_JTflankFixedCopyCtrRaSlope.npy', 'descrFitsHPC_220721_phAdj_sqrt_ddogs_JTflankFixedCopyCtrRaSlope.npy', 'descrFitsHPC_220720_sqrt_ddogs_JTflankFixedCopyCtrRaSlope.npy']
    #dogNames = ['descrFitsHPC_220720vEs_phAdj_sqrt_ddogs_JTflankFixedCopyCtrRaSlope.npy']
    #dogNames = ['descrFitsHPC_220707vEs_sqrt_ddogs_JTflankFixedCopyCtrRaSlope.npy']

  descrMod = 0; # which model for the diff. of gauss fits (0/1/2: flex/sach/tony)
  descrNames = ['descrFits_210304_sqrt_flex.npy', 'descrFits_210914_sqrt_flex.npy', 'descrFits_210914_sqrt_flex.npy', 'descrFits_210916_sqrt_flex.npy'];
  #descrNames = ['descrFits_190503_sqrt_flex.npy', 'descrFits_190503_sqrt_flex.npy', 'descrFits_191023_sqrt_flex.npy'];

  rvcNames = ['rvcFitsHPC_221126_f0_NR.npy', 'rvcFitsHPC_221126_f0_NR.npy', 'rvcFitsHPC_221126_NR_pos.npy', 'rvcFitsHPC_221126_vecF1_NR.npy']
  #rvcNames = ['rvcFitsHPC_220609_vecF1_NR.npy']
  #rvcNames = ['rvcFitsHPC_220609_f0_NR.npy', 'rvcFitsHPC_220609_vecF1_NR.npy']
  #rvcNames = ['rvcFitsHPC_220609_f0_NR.npy', 'rvcFitsHPC_220609_vecF1_NR.npy', 'V1_BB/structures/rvcFitsHPC_220609_vecF1_NR.npy']
  rvcMods = [1];
  #rvcNames = ['rvcFits_210914_f0_NR.npy', 'rvcFits_210914_f0_NR.npy', 'rvcFits_210914_vecF1_NR.npy', 'rvcFits_210916_vecF1_NR.npy'];
  #rvcNames = ['rvcFits_191023_f0_NR.npy', 'rvcFits_191023_f0_NR.npy', 'rvcFits_191023_NR_pos.npy'];
  #rvcMods = [1,1,1,1]; # 0-mov; 1-Nakarushton; 2-Peirce
  # rvcNames   = ['rvcFits_f0.npy'];
  # pack to easily tile
  expt = [expDirs, expNames, fitNamesWght, fitNamesFlat, descrNames, dogNames, rvcNames, rvcMods, cv_fitNamesWght, cv_fitNamesFlat];
  for exp_i in range(len(expt)):
      if len(expt[exp_i]) == 1:
          expt[exp_i] = expt[exp_i] * nExpts;
  # now unpack for use
  expDirs, expNames, fitNamesWght, fitNamesFlat, descrNames, dogNames, rvcNames, rvcMods, cv_fitNamesWght, cv_fitNamesFlat = expt;

  base_dir = os.getcwd() + '/';

  ###
  # what do we want to track for each cell?
  jointList = []; # we'll pack dictionaries in a list...

  #### these are now defaults in hf.jl_create - but here, nonetheless, for reference!

  # any parameters we need for analysis below?
  varExplThresh = -np.Inf; # i.e. only include if the fit explains >X (e.g. 75)% variance
  dog_varExplThresh = -np.Inf; # i.e. only include if the fit explains >X (e.g. 75)% variance
  #varExplThresh = 70; # i.e. only include if the fit explains >X (e.g. 75)% variance
  #dog_varExplThresh = 70; # i.e. only include if the fit explains >X (e.g. 75)% variance

  sf_range = [0.1, 10]; # allowed values of 'mu' for fits - see descr_fit.py for details

  conDig = 1; # i.e. round to nearest tenth (1 decimal place)
  rawInd = 0; # for accessing ratios/differences that we pass into diffsAtThirdCon

  muLoc = 2; # prefSF is in location '2' of parameter arrays

  # NOTE: the real code for creating the jointList has been moved to helper_fcns!
  # WARNING: This takes [~/<]10 minutes (as of 20.04.14)
  # jointList_V1full = hf.jl_create(base_dir, [expDirs[-1]], [expNames[-1]], [fitNamesWght[-1]], [fitNamesFlat[-1]], [descrNames[-1]], [dogNames[-1]], [rvcNames[-1]], [rvcMods[-1]])

  jointList = hf.jl_create(base_dir, expDirs, expNames, fitNamesWght, fitNamesFlat, descrNames, dogNames, rvcNames, rvcMods, varExplThresh=varExplThresh, dog_varExplThresh=dog_varExplThresh, descrMod=descrMod, dogMod=dogMod, jointType=jointType, reducedSave=True, briefVersion=brief,  cv_fitNamesWght=cv_fitNamesWght, cv_fitNamesFlat=cv_fitNamesFlat, toPar=True, modSpecs=modSpecs)

  from datetime import datetime
  suffix = datetime.today().strftime('%y%m%d')
  #suffix = '220926'

  varExplThresh_str = varExplThresh if varExplThresh > 0 else 0;
  dog_varExplThresh_str = dog_varExplThresh if dog_varExplThresh > 0 else 0;
  np.save(base_dir + 'jointList_V1_%svE_vT%02d_dvT%02d_m%dj%d' % (suffix, varExplThresh_str, dog_varExplThresh_str, dogMod, jointType), jointList)

########################
#### END V1
########################

if area == 'LGN':

  ########################
  #### LGN
  ########################

  # expDirs (and expNames must always be of the right length, i.e. specify for each expt dir 
  ### LGN version
  #expDirs = ['LGN/']
  #expNames = ['dataList.npy']
  expDirs = ['LGN/', 'LGN/sach/'];
  expNames = ['dataList_220222.npy', 'sachData.npy']
  #expDirs = ['LGN/sach/'];
  #expNames = ['sachData.npy']

  nExpts = len(expDirs);

  # these are usually same for all expts...we'll "tile" below
  # fitBase = 'fitList_190502cA';
  fitBase = 'fitList_191023c';
  #fitBase = 'fitList_210304';
  fitNamesWght = ['%s_wght_chiSq.npy' % fitBase];
  fitNamesFlat = ['%s_flat_chiSq.npy' % fitBase];
  ####
  # descrFits - loss type determined by comparison (choose best; see modCompare.ipynb::Descriptive Fits)
  ####
  #dogNames = ['descrFits_s210304_sach_sach.npy'];
  #descrNames = ['descrFits_s210304_sqrt_flex.npy'];
  dogMod = 1; # 1 (sach) or 2 (Tony)

  if jointType == 7: # [0/1/2/3 --> NONE//fix gs,rs//fix rs//fix rc,rs]
    dogNames = ['descrFitsHPC_220810vEs_phAdj_sqrt_sach_JTsurrShapeCtrRaSlope.npy', 'descrFitsHPC_s220810vEs_phAdj_sqrt_sach_JTsurrShapeCtrRaSlope.npy'];
    #dogNames = ['descrFitsHPC_220702vE_phAdj_sqrt_sach_JTsurrShapeCtrRaSlope.npy', 'descrFitsHPC_s220730vE_phAdj_sqrt_sach_JTsurrShapeCtrRaSlope.npy'];
    #dogNames = ['descrFitsHPC_220610_phAdj_sqrt_sach_JTsurrShapeCtrRaSlope.npy', 'descrFitsHPC_s220610_phAdj_sqrt_sach_JTsurrShapeCtrRaSlope.npy'];
  if jointType == 2: # [0/1/2/3 --> NONE//fix gs,rs//fix rs//fix rc,rs]
    dogNames = ['descrFitsHPC_220810vEs_phAdj_sqrt_sach_JTsurrShape.npy', 'descrFitsHPC_s220810vEs_phAdj_sqrt_sach_JTsurrShape.npy'];
  if jointType == 0: # [0/1/2/3 --> NONE//fix gs,rs//fix rs//fix rc,rs]
    dogNames = ['descrFitsHPC_220810vEs_phAdj_sqrt_sach.npy', 'descrFitsHPC_s220810vEs_phAdj_sqrt_sach.npy'];

  descrMod = 0; # which model for the diff. of gauss fits (0/1/2: flex/sach/tony)
  descrNames = ['descrFits_211005_sqrt_flex.npy', 'descrFits_s211006_sach_flex.npy'];

  rvcMods = [1,0]; # 0-mov (blank); 1-Nakarushton (NR); 2-Peirce (peirce)
  rvcNames = ['rvcFitsHPC_220926_NR_pos.npy', 'rvcFitsHPC_220531_pos.npy'];
  #rvcNames = ['rvcFitsHPC_220531_pos.npy', 'rvcFitsHPC_220531_pos.npy'];
  #rvcNames = ['rvcFitsHPC_220506_vecF1.npy', 'rvcFitsHPC_220508_vecF1.npy'];
  #rvcNames = ['rvcFits_210914_pos.npy', 'rvcFitsHPC_220219_pos.npy'];
  #rvcMods = [0]; # 0-mov (blank); 1-Nakarushton (NR); 2-Peirce (peirce)
  #rvcNames = ['rvcFits_210304.npy']; 
  #rvcNames = ['rvcFits_191023_pos.npy']; 
  #rvcMods = [0]; # 0-mov (blank); 1-Nakarushton (NR); 2-Peirce (peirce)

  # pack to easily tile
  expt = [expDirs, expNames, fitNamesWght, fitNamesFlat, descrNames, dogNames, rvcNames];
  for exp_i in range(len(expt)):
      if len(expt[exp_i]) == 1:
          expt[exp_i] = expt[exp_i] * nExpts;
  # now unpack for use
  expDirs, expNames, fitNamesWght, fitNamesFlat, descrNames, dogNames, rvcNames = expt;

  base_dir = os.getcwd() + '/';

  #####
  # what do we want to track for each cell?
  jointList = []; # we'll pack dictionaries in a list...

  #### these are now defaults in hf.jl_create - but here, nonetheless, for reference!

  # any parameters we need for analysis below?
  #varExplThresh = 65; # i.e. only include if the fit explains >X (e.g. 75)% variance
  #dog_varExplThresh = 65; # i.e. only include if the fit explains >X (e.g. 75)% variance
  varExplThresh = -np.Inf#60; # i.e. only include if the fit explains >X (e.g. 75)% variance
  dog_varExplThresh = -np.Inf#60; # i.e. only include if the fit explains >X (e.g. 75)% variance

  sf_range = [0.01, 10]; # allowed values of 'mu' for fits - see descr_fit.py for details

  # NOTE: the real code for creating the jointList has been moved to helper_fcns!
  # WARNING: This takes ~10 minutes (as of 09.06.19)
  jointList = hf.jl_create(base_dir, expDirs, expNames, fitNamesWght, fitNamesFlat, descrNames, dogNames, rvcNames, rvcMods, varExplThresh=varExplThresh, dog_varExplThresh=dog_varExplThresh, descrMod=descrMod, dogMod=dogMod, jointType=jointType, reducedSave=True, briefVersion=brief)

  from datetime import datetime
  suffix = datetime.today().strftime('%y%m%d')
  #suffix = '220926';

  varExplThresh_str = varExplThresh if varExplThresh > 0 else 0;
  dog_varExplThresh_str = dog_varExplThresh if dog_varExplThresh > 0 else 0;
  jt_str = '_jt%d' % jointType
  np.save(base_dir + 'jointList_LGN_%s_vT%02d_dvT%02d%s' % (suffix, varExplThresh_str, dog_varExplThresh_str, jt_str), jointList)
