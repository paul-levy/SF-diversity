import numpy as np
import sys
import helper_fcns as hf
import scipy.optimize as opt
import os
from time import sleep
import warnings
import pdb

# personal mac
#dataPath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/analysis/structures/';
#save_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/analysis/figures/';
# prince cluster
dataPath = '/home/pl1465/SF_diversity/LGN/analysis/structures/';
save_loc = '/home/pl1465/SF_diversity/LGN/analysis/structures/';

expName = 'dataList.npy'
phAdvName = 'phaseAdvanceFits'
rvcName = 'rvcFits'

### 1: Recreate Movshon, Kiorpes, Hawken, Cavanaugh '05 figure 6 analyses
''' These plots show response versus contrast data (with model fit) AND
    phase/amplitude plots with a model fit for response phase as a function of response amplitude
   
    First, given the FFT-derived response amplitude and phase, determine the response phase relative
    to the stimulus by taking into account the stimulus phase. 
    Then, make a simple linear model fit (line + constant offset) of the response phase as a function
    of response amplitude.

    The key non-intuitive step in the analysis is as follows: Rather than simply fitting an RVC curve
    with the FFT-derived response amplitudes, we determine the "true"/expected response phase given
    the measured response amplitude. Then, we project the observed vector onto a line which represents
    the "true"/expected response vector (i.e. the response vector with the expected phase, given the amplitude).
    Thus, the "fixed" response amplitude is = measuredAmp * cos(phiM - phiE)
    (phiM/E are the measured and expected response phases, respectively)

    This value will always be <= measuredAmp, which is what we want - if the response phase deviated
    from what it should be, then there was noise, and that noise shouldn't contribute to our response.
  
    Then, with these set of response amplitudes, fit an RVC (same model as in the '05 paper)
'''

def phase_advance_fit(cell_num, data_loc = dataPath, phAdvName=phAdvName, to_save = 1, disp=0, dir=-1):
  ''' Given the FFT-derived response amplitude and phase, determine the response phase relative
      to the stimulus by taking into account the stimulus phase. 
      Then, make a simple linear model fit (line + constant offset) of the response phase as a function
      of response amplitude.
      SAVES loss/optimized parameters/and phase advance (if default "to_save" value is kept)
      RETURNS phAdv_model, all_opts
  '''

  dataList = hf.np_smart_load(data_loc + 'dataList.npy');
  cellStruct = hf.np_smart_load(data_loc + dataList['unitName'][cell_num-1] + '_sfm.npy');
  if dir == 1:
    phAdvName = phAdvName + '_pos.npy';
  if dir == -1:
    phAdvName = phAdvName + '_neg.npy';

  # first, get the set of stimulus values:
  _, stimVals, valConByDisp, _, _ = hf.tabulate_responses(cellStruct);
  allCons = stimVals[1];
  allSfs = stimVals[2];

  # for all con/sf values for this dispersion, compute the mean amplitude/phase per condition
  allAmp, allPhi, allTf = hf.get_all_fft(cellStruct, disp, dir=dir); 
     
  # now, compute the phase advance
  conInds = valConByDisp[disp];
  conVals = allCons[conInds];
  nConds = len(allAmp); # this is how many conditions are present for this dispersion
  # recall that nConds = nCons * nSfs
  allCons = [conVals] * nConds; # repeates list and nests
  phAdv_model, all_opts, all_phAdv, all_loss = hf.phase_advance(allAmp, allPhi, conVals, allTf);

  if os.path.isfile(data_loc + phAdvName):
      phFits = hf.np_smart_load(data_loc + phAdvName);
  else:
      phFits = dict();

  # update stuff - load again in case some other run has saved/made changes
  if os.path.isfile(data_loc + phAdvName):
      print('reloading phAdvFits...');
      phFits = hf.np_smart_load(data_loc + phAdvName);
  if cell_num-1 not in phFits:
    phFits[cell_num-1] = dict();
  phFits[cell_num-1]['loss'] = all_loss;
  phFits[cell_num-1]['params'] = all_opts;
  phFits[cell_num-1]['phAdv'] = all_phAdv;

  if to_save:
    np.save(data_loc + phAdvName, phFits);
    print('saving phase advance fit for cell ' + str(cell_num));

  return phAdv_model, all_opts;

def rvc_adjusted_fit(cell_num, data_loc = dataPath, rvcName=rvcName, to_save=1, disp=0, dir=-1):
  ''' Piggy-backing off of phase_advance_fit above, get prepare to project the responses onto the proper phase to get the correct amplitude
      Then, with the corrected response amplitudes, fit the RVC model
  '''

  dataList = hf.np_smart_load(data_loc + 'dataList.npy');
  cellStruct = hf.np_smart_load(data_loc + dataList['unitName'][cell_num-1] + '_sfm.npy');
  if dir == 1:
    rvcName = rvcName + '_pos.npy';
  if dir == -1:
    rvcName = rvcName + '_neg.npy';

  # first, get the set of stimulus values:
  _, stimVals, valConByDisp, _, _ = hf.tabulate_responses(cellStruct);
  allCons = stimVals[1];
  allSfs = stimVals[2];
  valCons = allCons[valConByDisp[disp]];

  # calling phase_advance fit, use the phAdv_model and optimized paramters to compute the true response amplitude
  # given the measured/observed amplitude and phase of the response
  phAdv_model, all_opts = phase_advance_fit(cell_num, dir=dir, to_save = 0); # don't save
  allAmp, allPhi, _ = hf.get_all_fft(cellStruct, disp, dir=dir);
  # get just the mean amp/phi and put into convenient lists
  allAmpMeans = [[x[0] for x in sf] for sf in allAmp]; # mean is in the first element; do that for each [mean, std] pair in each list (split by sf)
  allPhiMeans = [[x[0] for x in sf] for sf in allPhi]; # mean is in the first element; do that for each [mean, var] pair in each list (split by sf)

  # use the original measure of varaibility if/when using weighted loss function in hf.rvc_fit
  allAmpStd = [[x[1] for x in sf] for sf in allAmp]; # std is in the first element; do that for each [mean, std] pair in each list (split by sf)

  adjMeans = hf.project_resp(allAmpMeans, allPhiMeans, phAdv_model, all_opts);
  consRepeat = [valCons] * len(adjMeans);
  rvc_model, all_opts, all_conGains, all_loss = hf.rvc_fit(adjMeans, consRepeat, allAmpStd);

  if os.path.isfile(data_loc + rvcName):
      rvcFits = hf.np_smart_load(data_loc + rvcName);
  else:
      rvcFits = dict();

  # update stuff - load again in case some other run has saved/made changes
  if os.path.isfile(data_loc + rvcName):
      print('reloading rvcFits...');
      rvcFits = hf.np_smart_load(data_loc + rvcName);
  if cell_num-1 not in rvcFits:
    rvcFits[cell_num-1] = dict();
  rvcFits[cell_num-1]['loss'] = all_loss;
  rvcFits[cell_num-1]['params'] = all_opts;
  rvcFits[cell_num-1]['conGain'] = all_conGains;
  rvcFits[cell_num-1]['adjMeans'] = adjMeans;
  rvcFits[cell_num-1]['stds'] = allAmpStd;

  if to_save:
    np.save(data_loc + rvcName, rvcFits);
    print('saving rvc fit for cell ' + str(cell_num));

  return rvc_model, all_opts, all_conGains, adjMeans;

if __name__ == '__main__':

    if len(sys.argv) < 2:
      print('uhoh...you need at least one argument(s) here');
      exit();

    cell_num = int(sys.argv[1]);
    if len(sys.argv) > 2:
      dir = int(sys.argv[2]);
    else:
      dir = None;
    print('Running cell %d' % cell_num);

    # then, put what to run here...
    phase_advance_fit(cell_num);
    rvc_adjusted_fit(cell_num);
    if dir:
      phase_advance_fit(cell_num, dir=dir);
      rvc_adjusted_fit(cell_num, dir=dir);
