import numpy as np
import helper_fcns

import pdb

### Similar to helper_fcns, but meant specifically for the sfBB_* series of experiments

def get_baseOnly_resp(expInfo):
  ''' returns the distribution of responses, mean/s.e.m. and unique sfXcon for each base stimulus in the sfBB_* series
  '''

  byTrial = expInfo['trial'];

  baseOnlyTr = np.logical_and(byTrial['baseOn'], ~byTrial['maskOn']) # baseON and maskOFF
  baseSf_all, baseCon_all = byTrial['sf'][1, baseOnlyTr], byTrial['con'][1, baseOnlyTr]

  sf_con_pairs = np.stack((baseSf_all, baseCon_all), axis=1)
  unique_pairs = np.unique(sf_con_pairs, axis=0);

  baseResp_dc = []; baseResp_f1 = [];
  baseSummary_dc = np.zeros((len(unique_pairs), 2)); baseSummary_f1 = np.zeros((len(unique_pairs), 2));

  for ii, up in enumerate(unique_pairs):

      # we have the unique pairs, now cycle through and do the same thing here we did with the other base stimulus....
      baseSf_curr, baseCon_curr = up;
      baseOnly_curr = np.logical_and(baseOnlyTr, np.logical_and(byTrial['sf'][1,:]==baseSf_curr,
                                                               byTrial['con'][1,:]==baseCon_curr))
      baseDC, baseF1 = expInfo['spikeCounts'][baseOnly_curr], expInfo['f1_base'][baseOnly_curr];

      baseResp_dc.append(baseDC); baseResp_f1.append(baseF1);

      baseSummary_dc[ii, :] = [np.mean(baseDC), np.std(baseDC)/len(baseDC)];
      baseSummary_f1[ii, :] = [np.mean(baseF1), np.std(baseF1)/len(baseF1)];

  return [baseResp_dc, baseResp_f1], [baseSummary_dc, baseSummary_f1], unique_pairs;


def get_mask_resp(expInfo, withBase=0, maskF1 = 1):
  ''' return the DC, F1 matrices [mean, s.e.m.] for responses to the mask only in the sfBB_* series 
      For programs (e.g. sfBB_varSF) with multiple base conditions, the order returned here is guaranteed
      to be the same as the unique base conditions given in get_baseOnly_resp
  '''

  conDig = 3; # round contrast to nearest thousandth (i.e. 0.001)
  byTrial = expInfo['trial'];

  maskCon, maskSf = expInfo['maskCon'], expInfo['maskSF'];
  # if we want with the base, we'll have to consider how many unique base conditions there are

  respsDC = []; respsF1 = [];

  if withBase == 0:
    # first, the logical which gives mask-only trials
    baseMatch = np.logical_and(byTrial['maskOn'], ~byTrial['baseOn']);
    nBase = 1;
  elif withBase == 1:
    # first, the logical which gives mask+base trials
    baseMatch = np.logical_and(byTrial['maskOn'], byTrial['baseOn']);
    _, _, baseConds = get_baseOnly_resp(expInfo);
    nBase = len(baseConds);

  maskResp_dc = []; maskResp_f1 = [];

  #pdb.set_trace();

  for up in np.arange(nBase):

    # make a 3d matrix of base+mask responses - SF x CON x [mean, SEM]
    maskCon, maskSf = np.unique(np.round(expInfo['maskCon'], conDig)), expInfo['maskSF'];
    respMatrixDC = np.nan * np.zeros((len(maskCon), len(maskSf), 2));
    respMatrixF1 = np.nan * np.zeros((len(maskCon), len(maskSf), 2));

    if withBase == 1: # then subset based on the particular base condition
      # we have the unique pairs, now cycle through and do the same thing here we did with the other base stimulus....
      baseSf_curr, baseCon_curr = baseConds[up];
      currTr = np.logical_and(baseMatch, np.logical_and(byTrial['sf'][1,:]==baseSf_curr, 
                                                        byTrial['con'][1,:]==baseCon_curr));
    else:
      currTr = baseMatch;


    for mcI, mC in enumerate(maskCon):
        conOk = (np.round(byTrial['con'][0,:], conDig) == mC)
        for msI, mS in enumerate(maskSf):
            sfOk = (byTrial['sf'][0,:] == mS)
            trialsOk = np.logical_and(currTr, np.logical_and(conOk, sfOk));

            currDC = expInfo['spikeCounts'][trialsOk];
            if maskF1 == 1:
              currF1 = expInfo['f1_mask'][trialsOk];
            else:
              currF1 = expInfo['f1_base'][trialsOk];
            dcMean, f1Mean = np.mean(currDC), np.mean(currF1)
            respMatrixDC[mcI, msI, :] = [dcMean, np.std(currDC)/len(currDC)]
            respMatrixF1[mcI, msI, :] = [f1Mean, np.std(currF1)/len(currF1)]

    maskResp_dc.append(respMatrixDC); maskResp_f1.append(respMatrixF1);

  if nBase == 1:
    return maskResp_dc[0], maskResp_f1[0];
  else:
    return maskResp_dc, maskResp_f1;

