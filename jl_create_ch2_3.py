import numpy as np
import os, sys
import helper_fcns as hf
import scipy.stats as ss
from scipy.stats.mstats import gmean
from scipy.stats import ks_2samp, kstest, linregress
import itertools, pdb

area = sys.argv[1];
if len(sys.argv)>2:
  jointType = int(sys.argv[2]);
else:
  jointType = 10 if area=='V1' else 7; # default to 10 and 7
if len(sys.argv)>3:
  normIn = int(sys.argv[3]);
  normA,normB = int(np.floor(normIn/10)), int(np.mod(normIn, 10))
else:
  normA = None; normB = None;

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
  excType = [1]
  lossType = [1]
  fixRespExp = [2]; 
  hasCV = [1];
  scheduler = [0];
  # these values - specifying the model - will be used with itertools to get all model combinations
  norms = [1, 2]; # flat, weighted
  lgnFrontEnds= [0,1,4]; # off, normal, shift independent
  lgnCons = [1];
  # --- now, generate all combinations!
  loc_str = 'HPC' if useHPCfit else '';

  ##### get the name --- and tile any value which appears only once (should be N model times)
  # name
  fL_name = 'fitList%s_pyt_nr230107' % (loc_str); # we'll add noRE, noSched after
  # expand the model subtypes into the full list of model fits
  expCodes = np.vstack(list(itertools.product(*(norms, lgnFrontEnds, lgnCons)))); # will be [nMod x nFeatures]
  normTypes = expCodes[:,0];
  lgnFrontEnds = expCodes[:,1];
  lgnConTypes = expCodes[:,2];
  nMods = len(normTypes);
  # tile
  expt = [excType, lossType, fixRespExp, hasCV, scheduler];
  for exp_i in range(len(expt)):
      if len(expt[exp_i]) == 1:
          expt[exp_i] = expt[exp_i] * nMods;
  # now unpack for use
  excTypes, lossTypes, fixRespExp, hasCV, scheduler = expt

  ########
  ### organize the mod specs....
  ########
  modSpecs = dict();
  modSpecs['nMods'] = nMods
  modSpecs['excType'] = excTypes
  modSpecs['lossType'] = lossTypes
  modSpecs['normType'] = normTypes
  modSpecs['lgnFrontEnd'] = lgnFrontEnds
  modSpecs['lgnConType'] = lgnConTypes
  modSpecs['scheduler'] = scheduler
  modSpecs['hasCV'] = hasCV
  modSpecs['fixRespExp'] = fixRespExp

  ####
  # descrFits - loss type determined by comparison (choose best; see modCompare.ipynb::Descriptive Fits)
  ####
  dogMod = 3; # 1 (sach), 2 (Tony), 3 (d-DoG-S), 4 (N/A), 5 (d-DoG-S Hawk)
  # dir. order is 'V1_orig/', 'altExp/', 'V1/', 'V1_BB/'
  if jointType == 10:
    # TEMPORARY UNTIL WE RUN V1/joint10 on the V1-only dataList [currently running on 22.11.27]
    dogNames = ['descrFitsHPC_221126vEs_sqrt_ddogs_JTflankShiftCopyCtrRaSlope.npy', 'descrFitsHPC_221126vEs_sqrt_ddogs_JTflankShiftCopyCtrRaSlope.npy', 'descrFitsHPC_221126vEs_phAdj_sqrt_ddogs_JTflankShiftCopyCtrRaSlope.npy', 'descrFitsHPC_221126vEs_phAdj_sqrt_ddogs_JTflankShiftCopyCtrRaSlope.npy']
  # THE BELOW ARE NOT UPDATED AS OF 22.11.27
  if jointType == 0:
    dogNames = ['descrFitsHPC_220811vEs_sqrt_ddogs.npy', 'descrFitsHPC_220811vEs_phAdj_sqrt_ddogs.npy', 'descrFitsHPC_220811vEs_phAdj_sqrt_ddogs.npy'];
  if jointType == 7:
    dogNames = ['descrFitsHPC_220609_sqrt_ddogs_JTflankSurrShapeCtrRaSlope.npy'];
  if jointType == 9:
    dogNames = ['descrFitsHPC_220801vEs_phAdj_sqrt_ddogs_JTflankFixedCopyCtrRaSlope.npy', 'descrFitsHPC_220801vEs_phAdj_sqrt_ddogs_JTflankFixedCopyCtrRaSlope.npy', 'descrFitsHPC_220801vEs_sqrt_ddogs_JTflankFixedCopyCtrRaSlope.npy']

  descrMod = 0; # which model for the diff. of gauss fits (0/1/2: flex/sach/tony)
  descrNames = ['descrFits_210304_sqrt_flex.npy', 'descrFits_210914_sqrt_flex.npy', 'descrFits_210914_sqrt_flex.npy', 'descrFits_210916_sqrt_flex.npy'];

  rvcNames = ['rvcFitsHPC_221126_f0_NR.npy', 'rvcFitsHPC_221126_f0_NR.npy', 'rvcFitsHPC_221126_NR_pos.npy', 'rvcFitsHPC_221126_vecF1_NR.npy']
  rvcMods = [1];
  # pack to easily tile
  expt = [expDirs, expNames, descrNames, dogNames, rvcNames, rvcMods];
  for exp_i in range(len(expt)):
      if len(expt[exp_i]) == 1:
          expt[exp_i] = expt[exp_i] * nExpts;
  # now unpack for use
  expDirs, expNames, descrNames, dogNames, rvcNames, rvcMods = expt;

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

  jointList = hf.jl_create(base_dir, expDirs, expNames, None, None, descrNames, dogNames, rvcNames, rvcMods, varExplThresh=varExplThresh, dog_varExplThresh=dog_varExplThresh, descrMod=descrMod, dogMod=dogMod, jointType=jointType, reducedSave=True, briefVersion=brief,  toPar=True, modSpecs=modSpecs, flexModels=True, flBase_name=fL_name)

  from datetime import datetime
  suffix = datetime.today().strftime('%y%m%d')
  #suffix = '220926'

  varExplThresh_str = varExplThresh if varExplThresh > 0 else 0;
  dog_varExplThresh_str = dog_varExplThresh if dog_varExplThresh > 0 else 0;
  np.save(base_dir + 'jointList_wMods_V1_%svE_vT%02d_dvT%02d_m%dj%d' % (suffix, varExplThresh_str, dog_varExplThresh_str, dogMod, jointType), jointList)
  # -- old version of the naming scheme
  #np.save(base_dir + 'jointList_wMods_freeRE_V1_%svE_vT%02d_dvT%02d_m%dj%d_mA%dmB%d' % (suffix, varExplThresh_str, dog_varExplThresh_str, dogMod, jointType, normA, normB), jointList)

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

  jointList = hf.jl_create(base_dir, expDirs, expNames, fitNamesWght, fitNamesFlat, descrNames, dogNames, rvcNames, rvcMods, varExplThresh=varExplThresh, dog_varExplThresh=dog_varExplThresh, descrMod=descrMod, dogMod=dogMod, jointType=jointType, reducedSave=True, briefVersion=brief, flexModels=False)

  from datetime import datetime
  suffix = datetime.today().strftime('%y%m%d')
  #suffix = '220926';

  varExplThresh_str = varExplThresh if varExplThresh > 0 else 0;
  dog_varExplThresh_str = dog_varExplThresh if dog_varExplThresh > 0 else 0;
  jt_str = '_jt%d' % jointType
  np.save(base_dir + 'jointList_LGN_%s_vT%02d_dvT%02d%s' % (suffix, varExplThresh_str, dog_varExplThresh_str, jt_str), jointList)
