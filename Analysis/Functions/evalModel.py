import numpy as np
import helper_fcns as hf
import model_responses as mod_resp
import scipy.optimize as opt
from scipy.stats import norm, mode, poisson, nbinom
import sys
import os

import pdb

def getConstraints(fitType):
        # 00 = preferred spatial frequency   (cycles per degree) || [>0.05]
        # 01 = derivative order in space || [>0.1]
        # 02 = normalization constant (log10 basis) || unconstrained
        # 03 = response exponent || >1
        # 04 = response scalar || >1e-3
        # 05 = early additive noise || [0, 1]; was [0.001, 1] - see commented out line below
        # 06 = late additive noise || >0.01
        # 07 = variance of response gain || >1e-3
        # if fitType == 2
        # 08 = mean of normalization weights gaussian || [>-2]
        # 09 = std of ... || >1e-3 or >5e-1
        # if fitType == 3
        # 08 = the offset of the c50 tuning curve which is bounded between [v_sigOffset, 1] || [0, 0.75]
        # 09 = standard deviation of the gaussian to the left of the peak || >0.1
        # 10 = "" to the right "" || >0.1
        # 11 = peak (i.e. sf location) of c50 tuning curve 

    zero = (0.05, None);
    one = (0.1, None);
    two = (None, None);
    three = (1, None);
    four = (1e-3, None);
    five = (0, 1); # why? if this is always positive, then we don't need to set awkward threshold (See ratio = in GiveBof)
    six = (0.01, None); # if always positive, then no hard thresholding to ensure rate (strictly) > 0
    seven = (1e-3, None);
    if fitType == 1:
      eight = (0, 0); # flat normalization (i.e. no tilt)
      return (zero,one,two,three,four,five,six,seven,eight);
    if fitType == 2:
      eight = (-2, None);
      nine = (5e-1, None);
      return (zero,one,two,three,four,five,six,seven,eight,nine);
    elif fitType == 3:
      eight = (0, 0.75);
      nine = (1e-1, None);
      ten = (1e-1, None);
      eleven = (0.05, None);
      return (zero,one,two,three,four,five,six,seven,eight,nine,ten,eleven);
    else: # mistake!
      return [];

def setModel(cellNum, stopThresh, lr, lossType = 1, fitType = 1, subset_frac = 1, initFromCurr = 1, holdOutCondition = None):
    # Given just a cell number, will fit the Robbe-inspired V1 model to the data
    #
    # stopThresh is the value (in NLL) at which we stop the fitting (i.e. if the difference in NLL between two full steps is < stopThresh, stop the fitting
    #
    # LR is learning rate
    #
    # lossType
    #   1 - loss := square(sqrt(resp) - sqrt(pred))
    #   2 - loss := poissonProb(spikes | modelRate)
    #   3 - loss := modPoiss model (a la Goris, 2014)
    #
    # fitType - what is the model formulation?
    #   1 := flat normalization
    #   2 := gaussian-weighted normalization responses
    #   3 := gaussian-weighted c50/norm "constant"
    #
    # holdOutCondition - [d, c, sf] or None
    #   which condition should we hold out from the dataset
 
    ########
    # Load cell
    ########
    #loc_data = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/Analysis/Structures/'; # personal mac
    loc_data = '/home/pl1465/SF_diversity/Analysis/Structures/'; # Prince cluster 

    # fitType
    if fitType == 1:
      fL_suffix1 = '_flat';
    elif fitType == 2:
      fL_suffix1 = '_wght';
    elif fitType == 3:
      fL_suffix1 = '_c50';
    # lossType
    if lossType == 1:
      fL_suffix2 = '_sqrt.npy';
    elif lossType == 2:
      fL_suffix2 = '_poiss.npy';
    elif lossType == 3:
      fL_suffix2 = '_modPoiss.npy';

    dataList = hf.np_smart_load(str(loc_data + 'dataList.npy'));
    dataNames = dataList['unitName'];

    print('loading data structure...');
    S = hf.np_smart_load(str(loc_data + dataNames[cellNum-1] + '_sfm.npy')); # why -1? 0 indexing...
    print('...finished loading');
    trial_inf = S['sfm']['exp']['trial'];
    prefOrEst = mode(trial_inf['ori'][1]).mode;
    trialsToCheck = trial_inf['con'][0] == 0.01;
    prefSfEst = mode(trial_inf['sf'][0][trialsToCheck==True]).mode;
    
    ########

    # 00 = preferred spatial frequency   (cycles per degree)
    # 01 = derivative order in space
    # 02 = normalization constant        (log10 basis)
    # 03 = response exponent
    # 04 = response scalar
    # 05 = early additive noise
    # 06 = late additive noise
    # 07 = variance of response gain - only used if lossType = 3
    # if fitType == 2
    # 08 = mean of (log)gaussian for normalization weights
    # 09 = std of (log)gaussian for normalization weights
    # if fitType == 3
    # 08 = the offset of the c50 tuning curve which is bounded between [v_sigOffset, 1] || [0, 1]
    # 09 = standard deviation of the gaussian to the left of the peak || >0.1
    # 10 = "" to the right "" || >0.1
    # 11 = peak of offset curve
    
    curr_params = [];
    initFromCurr = 0; # override initFromCurr so that we just go with default parameters

    if np.any(np.isnan(curr_params)): # if there are nans, we need to ignore...
      curr_params = [];
      initFromCurr = 0;

    pref_sf = float(prefSfEst) if initFromCurr==0 else curr_params[0];
    dOrdSp = np.random.uniform(1, 3) if initFromCurr==0 else curr_params[1];
    normConst = -0.8 if initFromCurr==0 else curr_params[2]; # why -0.8? Talked with Tony, he suggests starting with lower sigma rather than higher/non-saturating one
    #normConst = np.random.uniform(-1, 0) if initFromCurr==0 else curr_params[2];
    respExp = np.random.uniform(1, 3) if initFromCurr==0 else curr_params[3];
    respScalar = np.random.uniform(10, 1000) if initFromCurr==0 else curr_params[4];
    noiseEarly = np.random.uniform(0.001, 0.1) if initFromCurr==0 else curr_params[5];
    noiseLate = np.random.uniform(0.1, 1) if initFromCurr==0 else curr_params[6];
    varGain = np.random.uniform(0.1, 1) if initFromCurr==0 else curr_params[7];
    if fitType == 1:
      inhAsym = 0; 
    if fitType == 2:
      normMean = np.random.uniform(-1, 1) if initFromCurr==0 else curr_params[8];
      normStd = np.random.uniform(0.1, 1) if initFromCurr==0 else curr_params[9];
    if fitType == 3:
      sigOffset = np.random.uniform(0, 0.05) if initFromCurr==0 else curr_params[8];
      stdLeft = np.random.uniform(1, 5) if initFromCurr==0 else curr_params[9];
      stdRight = np.random.uniform(1, 5) if initFromCurr==0 else curr_params[10];
      sigPeak = float(prefSfEst) if initFromCurr==0 else curr_params[11];

    print('Initial parameters:\n\tsf: ' + str(pref_sf)  + '\n\td.ord: ' + str(dOrdSp) + '\n\tnormConst: ' + str(normConst));
    print('\n\trespExp ' + str(respExp) + '\n\trespScalar ' + str(respScalar));
    
    #########
    # Now get all the data we need
    #########    
    # stimulus information
    
    # vstack to turn into array (not array of arrays!)
    stimOr = np.vstack(trial_inf['ori']);

    #purge of NaNs...
    mask = np.isnan(np.sum(stimOr, 0)); # sum over all stim components...if there are any nans in that trial, we know
    objWeight = np.ones((stimOr.shape[1]));    

    # and get rid of orientation tuning curve trials
    oriBlockIDs = np.hstack((np.arange(131, 155+1, 2), np.arange(132, 136+1, 2))); # +1 to include endpoint like Matlab

    oriInds = np.empty((0,));
    for iB in oriBlockIDs:
        indCond = np.where(trial_inf['blockID'] == iB);
        if len(indCond[0]) > 0:
            oriInds = np.append(oriInds, indCond);

    # get rid of CRF trials, too? Not yet...
    conBlockIDs = np.arange(138, 156+1, 2);
    conInds = np.empty((0,));
    for iB in conBlockIDs:
       indCond = np.where(trial_inf['blockID'] == iB);
       if len(indCond[0]) > 0:
           conInds = np.append(conInds, indCond);

    objWeight[conInds.astype(np.int64)] = 1; # for now, yes it's a "magic number"    

    mask[oriInds.astype(np.int64)] = True; # as in, don't include those trials either!
    # hold out a condition if we have specified, and adjust the mask accordingly
    if holdOutCondition is not None:
      # dispInd: [1, 5]...conInd: [1, 2]...sfInd: [1, 11]
      # first, get all of the conditions... - blockIDs by condition known from Robbe code
      dispInd = holdOutCondition[0];
      conInd = holdOutCondition[1];
      sfInd = holdOutCondition[2];

      StimBlockIDs  = np.arange(((dispInd-1)*(13*2)+1)+(conInd-1), ((dispInd)*(13*2)-5)+(conInd-1)+1, 2); # +1 to include the last block ID
      currBlockID = StimBlockIDs[sfInd-1];
      holdOutTr = np.where(trial_inf['blockID'] == currBlockID)[0];
      mask[holdOutTr.astype(np.int64)] = True; # as in, don't include those trials either!
      
    # Set up model here - get the parameters and parameter bounds
    if fitType == 1:
      param_list = (pref_sf, dOrdSp, normConst, respExp, respScalar, noiseEarly, noiseLate, varGain, inhAsym);
    elif fitType == 2:
      param_list = (pref_sf, dOrdSp, normConst, respExp, respScalar, noiseEarly, noiseLate, varGain, normMean, normStd);
    elif fitType == 3:
      param_list = (pref_sf, dOrdSp, normConst, respExp, respScalar, noiseEarly, noiseLate, varGain, sigOffset, stdLeft, stdRight, sigPeak);
    all_bounds = getConstraints(fitType);
   
    # now set up the optimization
    obj = lambda params: mod_resp.SFMGiveBof(params, structureSFM=S, normType=fitType, lossType=lossType, maskIn=~mask)[0];
    tomin = opt.minimize(obj, param_list, bounds=all_bounds);

    opt_params = tomin['x'];
    NLL = tomin['fun'];

    if holdOutCondition is not None:
      holdoutNLL, _, = mod_resp.SFMGiveBof(opt_params, structureSFM=S, normType=fitType, lossType=lossType, trialSubset=holdOutTr);
    else:
      holdoutNLL = [];

    return NLL, opt_params, holdoutNLL;

if __name__ == '__main__':

    if len(sys.argv) < 8:
      print('uhoh...you need seven arguments here'); # and one is the script itself...
      print('See evalModel.py or evalModel.s for guidance');
      exit();

    cellNum = int(sys.argv[1]);
    lossType = int(sys.argv[4]);
    fitType = int(sys.argv[5]);

    print('Running cell ' + str(cellNum) + ' with NLL step threshold of ' + sys.argv[2] + ' with learning rate ' + sys.argv[3]);
    print('Additionally, each iteration will have ' + sys.argv[6] + ' of the data (subsample fraction)');

    # fitType
    if fitType == 1:
      fL_suffix1 = '_flat';
    elif fitType == 2:
      fL_suffix1 = '_wght';
    elif fitType == 3:
      fL_suffix1 = '_c50';
    # lossType
    if lossType == 1:
      fL_suffix2 = '_sqrt.npy';
    elif lossType == 2:
      fL_suffix2 = '_poiss.npy';
    elif lossType == 3:
      fL_suffix2 = '_modPoiss.npy';

    nDisps = 5;
    nCons = 2;
    nSfs = 11;

    NLLs = np.nan * np.empty((nDisps, nCons, nSfs));
    holdoutNLLs = np.nan * np.empty((nDisps, nCons, nSfs));
    allParams = np.array(np.nan * np.empty((nDisps, nCons, nSfs)), dtype='O');

    save_name = 'hQ_%d_%s%s' % (cellNum, fL_suffix1, fL_suffix2);

    save_loc = '/home/pl1465/SF_diversity/Analysis/Structures/'; # Prince cluster    
    #save_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/Analysis/Structures/'

    holdoutRes = dict();
   
    for d in range(1): # only care about single gratings for now
      for c in reversed(range(nCons)):
        for s in range(nSfs):

          holdOutCondition = [d+1, c+1, s+1];
          print('holding out %s' % holdOutCondition);

          NLL, params, holdoutNLL = setModel(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]), holdOutCondition=holdOutCondition);

          NLLs[d, c, s] = NLL;
          allParams[d, c, s] = params;
          holdoutNLLs[d, c, s] = holdoutNLL;

        # update and save after each disp x con combination
        holdoutRes['NLL'] = NLLs;
        holdoutRes['params'] = allParams;
        holdoutRes['holdoutNLL'] = holdoutNLLs;

        np.save(save_loc + save_name, holdoutRes);

