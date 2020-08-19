import math, cmath, numpy, os
import helper_fcns as hf
from scipy.stats import norm, mode, lognorm, nbinom, poisson
from numpy.matlib import repmat
import scipy.optimize as opt
import multiprocessing as mp
import datetime
import sys
import warnings

import pdb

fft = numpy.fft

try:
  cellNum      = int(sys.argv[1]);
except:
  cellNum = numpy.nan;
try:
  dataListName = hf.get_datalist(sys.argv[2]); # argv[2] is expDir
except:
  dataListName = None;

modRecov = 0;

#rvcBaseName = 'rvcFits_191023'; # a base set of RVC fits used for initializing c50 in opt...(full name, except .npy
rvcBaseName = 'rvcFits_200507'; # a base set of RVC fits used for initializing c50 in opt...(full name, except .npy)

# now, get descrFit name (ask if modRecov, too)
if modRecov == 1:
  try:
    fitType      = int(sys.argv[5]);
    descrFitName = 'mr%s_descrFits_190503_poiss_flex.npy' % hf.fitType_suffix(fitType);
  except:
    warnings.warn('Could not load descrFits in model_responses');
  # now try RVC base
  try:
    fitType      = int(sys.argv[5]);
    rvcBase      = 'mr%s_%s.npy' % (hf.fitType_suffix(fitType), rvcBaseName);
  except:
    warnings.warn('Could not load base RVC in model_responses');
    rvcBase = None;
else:
  try:
    descrFitName = 'descrFits_200507_sqrt_flex.npy' # dataset 200507
    #descrFitName = 'descrFits_191023_poiss_flex.npy' # full dataset 
  except:
    warnings.warn('Could not load descrFits in model_responses');
  # now try RVC base
  try:
    rvcBase      = '%s.npy' % rvcBaseName;
  except:
    warnings.warn('Could not load base RVC in model_responses');
    rvcBase = None;

# create global variables for use in saving each step of the iteration
params_glob = [];
loss_glob = [];
resp_glob = [];

###
# TODO?: get_normPrms   - get the parameters for the normalization pool, depending on normType
# oriFilt               - used only in plotting to create 2D image of receptive field/filter

# SFMSimpleResp_trial   - SFMSimpleResp, but per trial
# SFMSimpleResp_par     - uses ..._trial to fully replicate SFMSimpleResp
# SFMSimpleResp  - compute the response of the cell's linear filter to a given (set of) stimuli

# SFMNormResp_trial    - as for SFMSimpleResp_trial, but for normalization pool response
# SFMNormResp_par    - as for SFMSimpleResp_par
# SFMNormResp    - compute the response of the normalization pool to stimulus; not called in optimization, but called in simulation

# GetNormResp    - wrapper for SFMNormResp
# SimpleNormResp - a simplified normalization calculation per discussions with JAM
# SFMGiveBof     - gathers simple and normalization responses to get full model response; optimization is here!
# SFMSimulateNew - as in SFMGiveBof, but without optimization
# SFMSimulate    - as in SFMGiveBof, but without optimization; can pass in arbitrary stimuli here
# setModel       - wrapper+ for optimization and saving optimization results

# orientation filter used in plotting (only, I think?)
###

def oriFilt(imSizeDeg, pixSizeDeg, prefSf, prefOri, dOrder, aRatio):
    
    # returns a spatial filter which is dth-order derivate of 2-D Gaussian

    # imSizeDeg  = image size in degrees
    # pixSizeDeg = pixel size in degrees
    # prefSf     = peak spatial frequency in cycles/deg
    # prefOri    = peak orientation in radians
    # dOrder     = derivative order (integer)
    # aRatio     = aspect ratio of the Gaussian 
    #
    # oriTuning = (cos(oris-prefOri).^2 .* exp((aRatio^2-1)*cos(oris-prefOri).^2)).^(dOrder/2);
    # sfTuning  = sfs.^dOrder .* exp(- sfs.^2 ./ (2*sx^2));


    pixPerDeg = 1/pixSizeDeg;
    npts2     = round(0.5*imSizeDeg*pixPerDeg);
    psfPixels = 2*npts2*prefSf/pixPerDeg;                                      # convert peak sf from cycles/degree to pixels
    sx        = psfPixels/max(math.sqrt(dOrder), 0.01);                             # MAGIC
    sy        = sx/aRatio;
    
    [X, Y] = numpy.mgrid[-npts2:npts2, -npts2:npts2];
    rX    = numpy.cos(prefOri) * X + numpy.sin(prefOri) * Y;
    rY    = -numpy.sin(prefOri) * X + numpy.cos(prefOri) * Y;

    ffilt = numpy.exp(-(pow(rX, 2) / (2 * pow(sx, 2)) + pow(rY, 2) / (2 * pow(sy, 2)))) * pow(-1j*rX, dOrder);
    
    filt = fft.fftshift(fft.ifft2(fft.ifftshift(ffilt)));
    return filt.real;

def SFMSimpleResp_trial(trNum, channel, trialInf, stimParams = [], expInd = 1, excType = 1, lgnFrontParams=None):
  ''' Will be used for parallelizing SFMSimpleResp - compute the response for one frame
  '''
  p = trNum; # just for ease of typing

  # Overhead - saving things, preparing for calculation
  # - Load the data structures, etc
  if stimParams: # i.e. if we actually have non-empty stimParams
    make_own_stim = 1;
  else:
    make_own_stim = 0;
  z             = trialInf;
  nSf           = 1;
  nStimComp     = hf.get_exp_params(expInd).nStimComp;
  nFrames       = hf.num_frames(expInd);
  # - load the filter parameters
  # -- get spatial coordinates
  xCo = 0;                                                              # in visual degrees, centered on stimulus center
  yCo = 0;                                                              # in visual degrees, centered on stimulus center
  prefSf = channel['pref']['sf'];                              # in cycles per degree
  prefTf = round(numpy.nanmean(trialInf['tf'][0]));     # in cycles per second
  # -- Get derivative order in space and time
  dOrdTi = channel['dord']['ti'];
  if excType == 1:
    dOrdSp = channel['dord']['sp'];
  # -- (or, if needed get sigmaLow/High)
  if excType == 2:
    sigLow = channel['sigLow'];
    sigHigh = channel['sigHigh'];
  # unpack the LGN weights, but we'll only use if lgnFrontEnd>0
  mWeight = channel['mWeight'];
  pWeight = 1-mWeight;

  # preset the simpleResp as zeros
  simpleResp = numpy.zeros((nFrames, )); # nFrames, [blank] since just one trial!
 
  # Set stim parameters
  if make_own_stim == 1:

    all_stim = hf.makeStimulus(stimParams['stimFamily'], stimParams['conLevel'], \
                                                            stimParams['sf_c'], stimParams['template'], expInd=expInd);

    stimOr = all_stim['Ori'];
    stimTf = all_stim['Tf'];
    stimCo = all_stim['Con'];
    stimPh = all_stim['Ph'];
    stimSf = all_stim['Sf'];
  else:
    stimOr = numpy.array([z['ori'][x][p] * numpy.pi/180 for x in range(nStimComp)]);
    stimPh = numpy.array([z['ph'][x][p] * numpy.pi/180 for x in range(nStimComp)]);
    stimTf = numpy.array([z['tf'][x][p] for x in range(nStimComp)]);
    stimCo = numpy.array([z['con'][x][p] for x in range(nStimComp)]);
    stimSf = numpy.array([z['sf'][x][p] for x in range(nStimComp)]);

  if numpy.count_nonzero(numpy.isnan(stimOr)): # then this is a blank stimulus, no computation to be done
    return simpleResp;

  # I. Orientation, spatial frequency and temporal frequency
  # Compute orientation tuning - removed 17.18.7

  ### NEW: 20.07.16: LGN filtering stage
  ### Assumptions: No interaction between SF/con -- which we know is not true...
  # - first, SF tuning: model 2 (Tony)
  try: # if these fields are not in lgnFrontParams, then we weren't trying to have an lgnFrontParams
    DoGmodel = lgnFrontParams['dogModel'];
    dog_m = lgnFrontParams['dog_m'];
    dog_p = lgnFrontParams['dog_p'];
    rvcMod = lgnFrontParams['rvcModel'];
    params_m = lgnFrontParams['rvc_m'];
    params_p = lgnFrontParams['rvc_p'];

    resps_m = hf.get_descrResp(dog_m, stimSf, DoGmodel, minThresh=0.1)
    resps_p = hf.get_descrResp(dog_p, stimSf, DoGmodel, minThresh=0.1)
    # -- make sure we normalize by the true max response:
    sfTest = numpy.geomspace(0.1, 10, 1000);
    max_m = numpy.max(hf.get_descrResp(dog_m, sfTest, DoGmodel, minThresh=0.1));
    max_p = numpy.max(hf.get_descrResp(dog_p, sfTest, DoGmodel, minThresh=0.1));
    # -- then here's our selectivity per component for the current stimulus
    selSf_m = numpy.divide(resps_m, max_m);
    selSf_p = numpy.divide(resps_p, max_p);
    # - then RVC response: # rvcMod 0 (Movshon)
    rvc_mod = hf.get_rvc_model();
    selCon_m = rvc_mod(*params_m, stimCo)
    selCon_p = rvc_mod(*params_p, stimCo)
    # -- then here's our final responses per component for the current stimulus
    lgnSel = mWeight*(selSf_m*selCon_m) + pWeight*(selSf_p*selCon_p);
  except:
    lgnSel = np.ones_like(stimSf);

  if excType == 1:
    # Compute spatial frequency tuning - Deriv. order Gaussian
    sfRel = stimSf / prefSf;
    s     = pow(stimSf, dOrdSp) * numpy.exp(-dOrdSp/2 * pow(sfRel, 2));
    sMax  = pow(prefSf, dOrdSp) * numpy.exp(-dOrdSp/2);
    sNl   = s/sMax;
    selSf = sNl;
  elif excType == 2:
    # Compute spatial frequency tuning - flexible Gauss
    sfRel = numpy.divide(stimSf, prefSf);
    # - set the sigma appropriately, depending on what the stimulus SF is
    sigma = numpy.multiply(sigLow, [1]*len(sfRel));
    sigma[[x for x in range(len(sfRel)) if sfRel[x] > 1]] = sigHigh;
    # - now, compute the responses (automatically normalized, since max gaussian value is 1...)
    s     = [numpy.exp(-numpy.divide(numpy.square(numpy.log(x)), 2*numpy.square(y))) for x,y in zip(sfRel, sigma)];
    selSf = s; 

  # Compute temporal frequency tuning - removed 19.05.13

  # II. Phase, space and time
  omegaX = stimSf * numpy.cos(stimOr); # the stimulus in frequency space
  omegaY = stimSf * numpy.sin(stimOr);
  omegaT = stimTf;

  P = numpy.empty((nFrames, 3)); # nFrames for number of frames, two for x and y coordinate, one for time
  P[:,0] = 2*numpy.pi*xCo*numpy.ones(nFrames,);  # P is the matrix that contains the relative location of each filter in space-time (expressed in radians)
  P[:,1] = 2*numpy.pi*yCo*numpy.ones(nFrames,); # P(:,0) and p(:,1) describe location of the filters in space

  # Pre-allocate some variables
  if nSf == 1:
    respSimple = numpy.zeros(nFrames,);
  else:
    respSimple = numpy.zeros(nFrames, nSf);

  for iF in range(nSf):
    if isinstance(xCo, int):
      factor = 1;
    else:
      factor = len(xCo);

    linR1 = numpy.zeros((nFrames*factor, nStimComp)); # pre-allocation
    linR2 = numpy.zeros((nFrames*factor, nStimComp));
    linR3 = numpy.zeros((nFrames*factor, nStimComp));
    linR4 = numpy.zeros((nFrames*factor, nStimComp));

    computeSum = 0; # important constant: if stimulus contrast or filter sensitivity equals zero there is no point in computing the response

    framesDiv = numpy.arange(nFrames)/float(nFrames); # used in stimPos calc, but same regardless of loop
    for c in range(nStimComp): # there are up to nine stimulus components
      selSi = selSf[c] * lgnSel[c]; # lgnSel will be 1 if we're not using the front end

      if selSi != 0 and stimCo[c] != 0:
        computeSum = 1;

        # Use the effective number of frames displayed/stimulus duration
        stimPos = framesDiv + stimPh[c] / (2*numpy.pi*stimTf[c]); # nFrames + the appropriate phase-offset
        P3Temp  = numpy.full_like(P[:, 1], stimPos);
        P[:,2]  = 2*numpy.pi*P3Temp; # P(:,2) describes relative location of the filters in time.

        omegas = numpy.vstack((omegaX[c], omegaY[c], omegaT[c])); # make this a 3 x len(omegaX) array
        rComplex = selSi*stimCo[c]*numpy.exp(1j*numpy.dot(P, omegas));

        linR1[:,c] = rComplex.real.reshape(linR1[:,c].shape);  # four filters placed in quadrature
        linR2[:,c] = -1*rComplex.real.reshape(linR2[:,c].shape);
        linR3[:,c] = rComplex.imag.reshape(linR3[:,c].shape);
        linR4[:,c] = -1*rComplex.imag.reshape(linR4[:,c].shape);

      if computeSum == 1:
        respSimple1 = numpy.maximum(0, linR1.sum(1)); # superposition and half-wave rectification,...
        respSimple2 = numpy.maximum(0, linR2.sum(1));
        respSimple3 = numpy.maximum(0, linR3.sum(1));
        respSimple4 = numpy.maximum(0, linR4.sum(1));

        # if channel is tuned, it is phase selective...
        # NOTE: 19.05.14 - made response always complex...(wow)! See git for previous version
        if nSf == 1:
          respComplex = pow(respSimple1, 2) + pow(respSimple2, 2) \
              + pow(respSimple3, 2) + pow(respSimple4, 2); 
          respSimple = numpy.sqrt(numpy.divide(respComplex, 4)); # div by 4 to avg across all filters
        else:
          respComplex = pow(respSimple1, 2) + pow(respSimple2, 2) \
              + pow(respSimple3, 2) + pow(respSimple4, 2);
          respSimple[iF, :] = numpy.sqrt(numpy.divide(respComplex, 4)); # div by 4 to avg across all filters

  # Store response in desired format
  return respSimple;

##########

def SFMSimpleResp_par(S, channel, stimParams = [], expInd = 1, trialInf = None, excType = 1, lgnFrontEnd = 0):
    # returns object (class?) with simpleResp and other things

    # SFMSimpleResp       Computes response of simple cell for sfmix experiment

    # SFMSimpleResp(varargin) returns a complex cell response for the
    # mixture stimuli used in sfMix. The cell's receptive field is the n-th
    # derivative of a 2-D Gaussian that need not be circularly symmetric.

    # 1/23/17 - Edits: Added stimParamsm, make_own_stim so that I can set what
    # stimuli I want when simulating from model

    make_own_stim = 0;
    if stimParams: # i.e. if we actually have non-empty stimParams
        make_own_stim = 1;
        if not 'template' in stimParams:
            stimParams['template'] = S;
        if not 'repeats' in stimParams:
            stimParams['repeats'] = 10; # why 10? To match experimental #repetitions

    # Load the data structure
    T = S['sfm'];
    if trialInf is None: # otherwise, we've passed in explicit trial information!
      trialInf = T['exp']['trial'];

    # Get preferred stimulus values
    prefSf = channel['pref']['sf'];                              # in cycles per degree
    # CHECK LINE BELOW
    prefTf = round(numpy.nanmean(trialInf['tf'][0]));     # in cycles per second

    # Get directional selectivity - removed 7/18/17

    # Get derivative order in space and time
    dOrdTi = channel['dord']['ti'];
    if excType == 1:
      dOrdSp = channel['dord']['sp'];
    # (or, if needed get sigmaLow/High)
    if excType == 2:
      sigLow = channel['sigLow'];
      sigHigh = channel['sigHigh'];
    # unpack the LGN weights, but we'll only use if lgnFrontEnd>0
    mWeight = channel['mWeight'];

    # Get aspect ratio in space - removed 7/18/17

    # Get spatial coordinates
    xCo = 0;                                                              # in visual degrees, centered on stimulus center
    yCo = 0;                                                              # in visual degrees, centered on stimulus center

    # Store some results in M
    M = dict();
    pref = dict();
    dord = dict();
    pref.setdefault('sf', prefSf);
    pref.setdefault('tf', prefTf);
    pref.setdefault('xCo', xCo);
    pref.setdefault('yCo', yCo);
    if excType == 1:
      dord.setdefault('sp', dOrdSp);
    dord.setdefault('ti', dOrdTi);
    
    M.setdefault('pref', pref);
    M.setdefault('dord', dord);
    if excType == 2:
      M.setdefault('sig', (sigLow, sigHigh));
    M.setdefault('mWeight', mWeight);

    # Pre-allocate memory
    if make_own_stim == 1:
        nTrials = stimParams['repeats']; # to keep consistent with number of repetitions used for each stim. condition
    else: # CHECK THIS GUY BELOW
        try:
          nTrials = len(trialInf['num']);
        except:
          nTrials = len(trialInf['con'][0]); 
          # if we've defined our own stimuli, then we won't have "num"; just get number of trials from stim components
    trialInfSlim = dict();
    trialInfSlim['ori'] = trialInf['ori'];
    trialInfSlim['ph'] = trialInf['ph'];
    trialInfSlim['tf'] = trialInf['tf'];
    trialInfSlim['con'] = trialInf['con'];
    trialInfSlim['sf'] = trialInf['sf'];
    trialInfSlim['num'] = trialInf['num'];

    DoGmodel = 2;
    if lgnFrontEnd == 1:
      dog_m = [1, 3, 0.3, 0.4]; # k, f_c, k_s, j_s
      dog_p = [1, 9, 0.5, 0.4];
    elif lgnFrontEnd == 2:
      dog_m = [1, 6, 0.3, 0.4]; # k, f_c, k_s, j_s
      dog_p = [1, 9, 0.5, 0.4];
    if lgnFrontEnd > 0:
      # specify rvc parameters
      rvcMod = 0;
      params_m = [0, 12.5, 0.05];
      params_p = [0, 17.5, 0.50];
      # save everything for use later
      M['dogModel'] = 2;
      M['dog_m'] = dog_m;
      M['dog_p'] = dog_p;
      M['rvcModel'] = rvcMod;
      M['rvc_m'] = params_m;
      M['rvc_p'] = params_p;

    # Compute simple cell response for all trials
    # HERE we multiprocess!
    from functools import partial
    fn_perTi = partial(SFMSimpleResp_trial, channel=channel, trialInf=trialInfSlim, expInd=expInd, excType=excType, stimParams=[], lgnFrontParams=lgnFrontParams);
    nCpu = mp.cpu_count();
    with mp.Pool(processes = nCpu) as pool:
      simpleAsList = pool.map(fn_perTi, range(nTrials));
    #simpleAsList = [fn_perTi(ti) for ti in range(nTrials)];
    M['simpleResp'] = numpy.transpose(numpy.vstack(simpleAsList));

    return M;

##########


# SFMSimpleResp - Used in Robbe V1 model - excitatory, linear filter response
def SFMSimpleResp(S, channel, stimParams = [], expInd = 1, trialInf = None, excType = 1, lgnFrontEnd = 0):
    # returns object (class?) with simpleResp and other things

    # SFMSimpleResp       Computes response of simple cell for sfmix experiment

    # SFMSimpleResp(varargin) returns a complex cell response for the
    # mixture stimuli used in sfMix. The cell's receptive field is the n-th
    # derivative of a 2-D Gaussian that need not be circularly symmetric.

    # 1/23/17 - Edits: Added stimParamsm, make_own_stim so that I can set what
    # stimuli I want when simulating from model

    make_own_stim = 0;
    if stimParams: # i.e. if we actually have non-empty stimParams
        make_own_stim = 1;
        if not 'template' in stimParams:
            stimParams['template'] = S;
        if not 'repeats' in stimParams:
            stimParams['repeats'] = 10; # why 10? To match experimental #repetitions

    # Load the data structure
    T = S['sfm'];
    if trialInf is None: # otherwise, we've passed in explicit trial information!
      trialInf = T['exp']['trial'];

    # Get preferred stimulus values
    prefSf = channel['pref']['sf'];                              # in cycles per degree
    # CHECK LINE BELOW
    prefTf = round(numpy.nanmean(trialInf['tf'][0]));     # in cycles per second

    # Get directional selectivity - removed 7/18/17

    # Get derivative order in space and time
    dOrdTi = channel['dord']['ti'];
    if excType == 1:
      dOrdSp = channel['dord']['sp'];
    # (or, if needed get sigmaLow/High)
    if excType == 2:
      sigLow = channel['sigLow'];
      sigHigh = channel['sigHigh'];
    # unpack the LGN weights, but we'll only use if lgnFrontEnd > 0
    mWeight = channel['mWeight'];
    pWeight = 1-mWeight;

    # Get aspect ratio in space - removed 7/18/17

    # Get spatial coordinates
    xCo = 0;                                                              # in visual degrees, centered on stimulus center
    yCo = 0;                                                              # in visual degrees, centered on stimulus center

    # Store some results in M
    M = dict();
    pref = dict();
    dord = dict();
    pref.setdefault('sf', prefSf);
    pref.setdefault('tf', prefTf);
    pref.setdefault('xCo', xCo);
    pref.setdefault('yCo', yCo);
    if excType == 1:
      dord.setdefault('sp', dOrdSp);
    dord.setdefault('ti', dOrdTi);
    
    M.setdefault('pref', pref);
    M.setdefault('dord', dord);
    if excType == 2:
      M.setdefault('sig', (sigLow, sigHigh));

    # Pre-allocate memory
    z             = trialInf;
    nSf           = 1;
    nStimComp     = hf.get_exp_params(expInd).nStimComp;
    nFrames       = hf.num_frames(expInd);
    if make_own_stim == 1:
        nTrials = stimParams['repeats']; # to keep consistent with number of repetitions used for each stim. condition
    else: # CHECK THIS GUY BELOW
        try:
          nTrials = len(z['num']);
        except:
          nTrials = len(z['con'][0]); 
          # if we've defined our own stimuli, then we won't have "num"; just get number of trials from stim components
    
    # set it zero
    M['simpleResp'] = numpy.zeros((nFrames, nTrials));

    DoGmodel = 2;
    if lgnFrontEnd == 1:
      dog_m = [1, 3, 0.3, 0.4]; # k, f_c, k_s, j_s
      dog_p = [1, 9, 0.5, 0.4];
    elif lgnFrontEnd == 2:
      dog_m = [1, 6, 0.3, 0.4]; # k, f_c, k_s, j_s
      dog_p = [1, 9, 0.5, 0.4];
    if lgnFrontEnd > 0:
      # specify rvc parameters
      rvcMod = 0;
      params_m = [0, 12.5, 0.05];
      params_p = [0, 17.5, 0.50];
      # save everything for use later
      M['dogModel'] = 2;
      M['dog_m'] = dog_m;
      M['dog_p'] = dog_p;
      M['rvcModel'] = rvcMod;
      M['rvc_m'] = params_m;
      M['rvc_p'] = params_p;

    # Compute simple cell response for all trials
    for p in range(nTrials): 
    
        # Set stim parameters
        if make_own_stim == 1:

            all_stim = hf.makeStimulus(stimParams['stimFamily'], stimParams['conLevel'], \
                                                                    stimParams['sf_c'], stimParams['template'], expInd=expInd);

            stimOr = all_stim['Ori'];
            stimTf = all_stim['Tf'];
            stimCo = all_stim['Con'];
            stimPh = all_stim['Ph'];
            stimSf = all_stim['Sf'];
        else:
            stimOr = numpy.empty((nStimComp,));
            stimTf = numpy.empty((nStimComp,));
            stimCo = numpy.empty((nStimComp,));
            stimPh = numpy.empty((nStimComp,));
            stimSf = numpy.empty((nStimComp,));
            
            for iC in range(nStimComp):
                stimOr[iC] = z['ori'][iC][p] * numpy.pi/180; # in radians
                stimTf[iC] = z['tf'][iC][p];          # in cycles per second
                stimCo[iC] = z['con'][iC][p];         # in Michelson contrast
                stimPh[iC] = z['ph'][iC][p] * numpy.pi/180;  # in radians
                stimSf[iC] = z['sf'][iC][p];          # in cycles per degree
                
        if numpy.count_nonzero(numpy.isnan(stimOr)): # then this is a blank stimulus, no computation to be done
            continue;
                
        # I. Orientation, spatial frequency and temporal frequency
        # Compute orientation tuning - removed 17.18.7

        ### NEW: 20.07.16: LGN filtering stage
        ### Assumptions: No interaction between SF/con -- which we know is not true...
        # - first, SF tuning: model 2 (Tony)
        if lgnFrontEnd > 0:
          resps_m = hf.get_descrResp(dog_m, stimSf, DoGmodel, minThresh=0.1)
          resps_p = hf.get_descrResp(dog_p, stimSf, DoGmodel, minThresh=0.1)
          # -- make sure we normalize by the true max response:
          sfTest = numpy.geomspace(0.1, 10, 1000);
          max_m = numpy.max(hf.get_descrResp(dog_m, sfTest, DoGmodel, minThresh=0.1));
          max_p = numpy.max(hf.get_descrResp(dog_p, sfTest, DoGmodel, minThresh=0.1));
          # -- then here's our selectivity per component for the current stimulus
          selSf_m = numpy.divide(resps_m, max_m);
          selSf_p = numpy.divide(resps_p, max_p);
          # - then RVC response: # rvcMod 0 (Movshon)
          rvc_mod = hf.get_rvc_model();
          selCon_m = rvc_mod(*params_m, stimCo)
          selCon_p = rvc_mod(*params_p, stimCo)
          # -- then here's our final responses per component for the current stimulus
          lgnSel = mWeight*(selSf_m*selCon_m) + pWeight*(selSf_p*selCon_p);

        if excType == 1:
          # Compute spatial frequency tuning - Deriv. order Gaussian
          sfRel = stimSf / prefSf;
          s     = pow(stimSf, dOrdSp) * numpy.exp(-dOrdSp/2 * pow(sfRel, 2));
          sMax  = pow(prefSf, dOrdSp) * numpy.exp(-dOrdSp/2);
          sNl   = s/sMax;
          selSf = sNl;
        elif excType == 2:
          # Compute spatial frequency tuning - flexible Gauss
          sfRel = numpy.divide(stimSf, prefSf);
          # - set the sigma appropriately, depending on what the stimulus SF is
          sigma = numpy.multiply(sigLow, [1]*len(sfRel));
          sigma[[x for x in range(len(sfRel)) if sfRel[x] > 1]] = sigHigh;
          # - now, compute the responses (automatically normalized, since max gaussian value is 1...)
          s     = [numpy.exp(-numpy.divide(numpy.square(numpy.log(x)), 2*numpy.square(y))) for x,y in zip(sfRel, sigma)];
          selSf = s; 

        # Compute temporal frequency tuning - removed 19.05.13

        # II. Phase, space and time
        omegaX = stimSf * numpy.cos(stimOr); # the stimulus in frequency space
        omegaY = stimSf * numpy.sin(stimOr);
        omegaT = stimTf;

        P = numpy.empty((nFrames, 3)); # nFrames for number of frames, two for x and y coordinate, one for time
        P[:,0] = 2*numpy.pi*xCo*numpy.ones(nFrames,);  # P is the matrix that contains the relative location of each filter in space-time (expressed in radians)
        P[:,1] = 2*numpy.pi*yCo*numpy.ones(nFrames,); # P(:,0) and p(:,1) describe location of the filters in space

        # Pre-allocate some variables
        if nSf == 1:
            respSimple = numpy.zeros(nFrames,);
        else:
            respSimple = numpy.zeros(nFrames, nSf);

        for iF in range(nSf):
            if isinstance(xCo, int):
                factor = 1;
            else:
                factor = len(xCo);

            linR1 = numpy.zeros((nFrames*factor, nStimComp)); # pre-allocation
            linR2 = numpy.zeros((nFrames*factor, nStimComp));
            linR3 = numpy.zeros((nFrames*factor, nStimComp));
            linR4 = numpy.zeros((nFrames*factor, nStimComp));
            
            computeSum = 0; # important constant: if stimulus contrast or filter sensitivity equals zero there is no point in computing the response

            for c in range(nStimComp): # there are up to nine stimulus components
                #selSi = selSf[c]; # filter sensitivity for the sinusoid in the frequency domain
                # NEW: 20.07.16 -- why divide by 2 for the LGN stage? well, if selectivity is at peak for M and P, sum will be 2 (both are already normalized) // could also just sum...
                if lgnFrontEnd > 0:
                  selSi = selSf[c] * lgnSel[c]; # filter sensitivity for the sinusoid in the frequency domain
                else:
                  selSi = selSf[c]
                if selSi != 0 and stimCo[c] != 0:
                    computeSum = 1;
                                   
                    # Use the effective number of frames displayed/stimulus duration
                    stimPos = numpy.asarray(range(nFrames))/float(nFrames) + \
                                            stimPh[c] / (2*numpy.pi*stimTf[c]); # nFrames + the appropriate phase-offset
                    P3Temp  = numpy.full_like(P[:, 1], stimPos);
                    #P3Temp  = repmat(stimPos, 1, len(xCo));
                    P[:,2]  = 2*numpy.pi*P3Temp; # P(:,2) describes relative location of the filters in time.

                    omegas = numpy.vstack((omegaX[c], omegaY[c], omegaT[c])); # make this a 3 x len(omegaX) array
                    rComplex = selSi*stimCo[c]*numpy.exp(1j*numpy.dot(P, omegas));

                    linR1[:,c] = rComplex.real.reshape(linR1[:,c].shape);  # four filters placed in quadrature
                    linR2[:,c] = -1*rComplex.real.reshape(linR2[:,c].shape);
                    linR3[:,c] = rComplex.imag.reshape(linR3[:,c].shape);
                    linR4[:,c] = -1*rComplex.imag.reshape(linR4[:,c].shape);

                if computeSum == 1:
                    respSimple1 = numpy.maximum(0, linR1.sum(1)); # superposition and half-wave rectification,...
                    respSimple2 = numpy.maximum(0, linR2.sum(1));
                    respSimple3 = numpy.maximum(0, linR3.sum(1));
                    respSimple4 = numpy.maximum(0, linR4.sum(1));

                    # if channel is tuned, it is phase selective...
                    # NOTE: 19.05.14 - made response always complex...(wow)! See git for previous version
                    if nSf == 1:
                          respComplex = pow(respSimple1, 2) + pow(respSimple2, 2) \
                              + pow(respSimple3, 2) + pow(respSimple4, 2); 
                          respSimple = numpy.sqrt(numpy.divide(respComplex, 4)); # div by 4 to avg across all filters
                    else:
                          respComplex = pow(respSimple1, 2) + pow(respSimple2, 2) \
                              + pow(respSimple3, 2) + pow(respSimple4, 2);
                          respSimple[iF, :] = numpy.sqrt(numpy.divide(respComplex, 4)); # div by 4 to avg across all filters
                        
        # Store response in desired format
        M['simpleResp'][:,p] = respSimple;
        
    return M;
        
def SFMNormResp(unitName, loadPath, normPool, stimParams = [], expInd = 1, overwrite = 0):

# returns M which contains normPool response, trial_used, filter preferences

# SFNNormResp       Computes normalization response for sfMix experiment

# SFMNormResp(unitName, varargin) returns a simulated V1-normalization
# response to the mixture stimuli used in sfMix (version 'expInd'). The normalization pool
# consists of spatially distributed filters, tuned for orientation, spatial 
# frequency, and temporal frequency. The tuning functions decribe responses
# of spatial filters obtained by taking a d-th order derivative of a 
# 2-D Gaussian. The pool includes both filters that are broadly and 
# narrowly tuned for spatial frequency. The filters are chosen such that 
# their summed responses approximately tile the spatial frequency domain
# between 0.3 c/deg and 10 c/deg.

# 1/23/17 - Edited, like SFMSimpleResp, to allow for making own stimuli

# 1/25/17 - Allowed 'S' to be passed in by checking if unitName is a string
#       or not (if ischar...)
#   Discovery that rComplex = line is QUITE SLOW. Need to speed up...
#  So, decompose that operation into [static] + [dynamic]
#   where static is the part of computation that doesn't change with frame
#   and dynamic does, meaning only the latter needs to computed repeatedly!

# 1/30/17 - Allow for passing in of specific trials from which to draw phase/tf

    make_own_stim = 0;
    if stimParams: # i.e. if we actually have non-empty stimParams
        make_own_stim = 1;

    # If unitName is char/string, load the data structure
    # Otherwise, we assume that we already are passing loaded data structure
    if isinstance(unitName, str):
        S = hf.np_smart_load(loadPath + unitName + '_sfm.npy')
    else:
        S = unitName;

    if overwrite == 0:
      if 'mod' in S['sfm']:
        if 'normalization' in S['sfm']['mod']:
          print('Overwrite flag is set to 0, and %s already has a computed normalization response' % (loadPath+unitName+'_sfm.npy'));
          return S['sfm']['mod']['normalization'];

    if make_own_stim:
        if not 'template' in stimParams:
            stimParams['template'] = S;
        if not 'repeats' in stimParams:
            stimParams['repeats'] = 10; # why 10? To match experimental #repetitions
    
    T = S['sfm']; # we assume first sfm if there exists more than one
        
    # Get filter properties in spatial frequency domain
    gain = numpy.empty((len(normPool['n'])));
    for iB in range(len(normPool['n'])):
        prefSf_new = numpy.logspace(numpy.log10(.1), numpy.log10(30), normPool['nUnits'][iB]);
        if iB == 0:
            prefSf = prefSf_new;
        else:
            prefSf = [prefSf, prefSf_new];
        gain[iB]   = normPool['gain'][iB];
       
    # Get filter properties in direction of motion and temporal frequency domain
    # for prefOr: whatever mode is for stimulus orientation of any component, that's prefOr
    # for prefTf: whatever mean is for stim. TF of any component, that's prefTf
    prefOr = (numpy.pi/180)*mode(T['exp']['trial']['ori'][0]).mode;   # in radians
    prefTf = round(numpy.nanmean(T['exp']['trial']['tf'][0]));       # in cycles per second
    
    # Compute spatial coordinates filter centers (all in phase, 4 filters per period)
    
    stimSi = T['exp']['size']; # in visual degrees
    stimSc = 1.75;                     # in cycles per degree, this is approximately center frequency of stimulus distribution
    nCycle = stimSi*stimSc;
    radius = math.sqrt(pow(math.ceil(4*nCycle), 2)/math.pi);
    vec    = numpy.arange(-math.ceil(radius), math.ceil(radius)+1, 1);
    # hideous python code...fix this when you are learned
    yTemp  = .25/stimSc*repmat(vec, len(vec), 1).transpose();
    xTemp  = .25/stimSc*repmat(vec, len(vec), 1);
    ind    = numpy.sign(stimSi/2 - numpy.sqrt(pow(xTemp,2) + pow(yTemp,2)));
    mask  = numpy.ma.masked_greater(ind, 0).mask; # find the ind > 0; returns object, .mask to get bool array
    xCo = xTemp[mask]; # in visual degrees, centered on stimulus center
    yCo = yTemp[mask]; # in visual degrees, centered on stimulus center
    
    # Store some results in M
    M          = dict();
    pref       = dict();
    pref.setdefault('or', prefOr);
    pref.setdefault('sf', prefSf);
    pref.setdefault('tf', prefTf);
    pref.setdefault('xCo', xCo);
    pref.setdefault('yCo', yCo);
    M.setdefault('pref', pref);
                               
    # Pre-allocate memory
    z          = T['exp']['trial'];
    nSf        = 0;
    nStimComp  = hf.get_exp_params(expInd).nStimComp;
    nFrames    = hf.num_frames(expInd);
    
    if not isinstance(prefSf, int):
        for iS in range(len(prefSf)):
            nSf = nSf + len(prefSf[iS]);
    else:
        nSf = 1;
    
    if make_own_stim == 1:
        nTrials = stimParams['repeats']; # keep consistent with 10 repeats per stim. condition
    else:
        nTrials  = len(z['num']);

    trial_used = numpy.zeros(nTrials);
        
    M['normResp'] = numpy.zeros((nTrials, nSf, nFrames));
    
    # Compute normalization response for all trials
    if not make_own_stim:
        print('Computing normalization response for ' + unitName + ' ...');

    for p in range(nTrials):
       
        # Set stim parameters
        if make_own_stim == 1:
            # So... If we want to pass in specific trials, check that those
            # trials exist
            # if there are enough for the trial 'p' we are at now, then
            # grab that one; otherwise get the first
            if 'trial_used' in stimParams:
                if stimParams['trial_used'] >= p:
                    stimParams['template']['trial_used'] = stimParams['trial_used'][p];
                else:
                    stimParams['template']['trial_used'] = stimParams['trial_used'][0];
            
            all_stim = hf.makeStimulus(stimParams['stimFamily'], stimParams['conLevel'], \
                                    stimParams['sf_c'], stimParams['template'], expInd=expInd);
            
            stimOr = all_stim['Ori'];
            stimTf = all_stim['Tf'];
            stimCo = all_stim['Con'];
            stimPh = all_stim['Ph'];
            stimSf = all_stim['Sf'];
            trial_used[p] = all_stim['trial_used'];
            
        else:
            stimOr = numpy.empty((nStimComp,));
            stimTf = numpy.empty((nStimComp,));
            stimCo = numpy.empty((nStimComp,));
            stimPh = numpy.empty((nStimComp,));
            stimSf = numpy.empty((nStimComp,));
            for iC in range(nStimComp):
                stimOr[iC] = z['ori'][iC][p] * math.pi/180; # in radians
                stimTf[iC] = z['tf'][iC][p];          # in cycles per second
                stimCo[iC] = z['con'][iC][p];         # in Michelson contrast
                stimPh[iC] = z['ph'][iC][p] * math.pi/180;  # in radians
                stimSf[iC] = z['sf'][iC][p];          # in cycles per degree

        if numpy.count_nonzero(numpy.isnan(stimOr)): # then this is a blank stimulus, no computation to be done
            continue;
                
        # I. Orientation, spatial frequency and temporal frequency
        # matrix size: nComponents x nFilt (i.e., number of stimulus components by number of orientation filters)
          
        # Compute SF tuning
        for iB in range(len(normPool['n'])):
            sfRel = repmat(stimSf, len(prefSf[iB]), 1).transpose() / repmat(prefSf[iB], nStimComp, 1);
            s     = pow(repmat(stimSf, len(prefSf[iB]), 1).transpose(), normPool['n'][iB]) \
                        * numpy.exp(-normPool['n'][iB]/2 * pow(sfRel, 2));
            sMax  = pow(repmat(prefSf[iB], nStimComp, 1), normPool['n'][iB]) * numpy.exp(-normPool['n'][iB]/2);
            if iB == 0:
                selSf = gain[iB] * s / sMax;
            else:
                selSf = [selSf, gain[iB] * s/sMax];
                
        # Orientation
        selOr = numpy.ones(nStimComp); 
        # all stimulus components of the spatial frequency mixtures were shown at the cell's preferred direction of motion
        
        # Compute temporal frequency tuning - removed TF tuning 19.13.05
        selTf = numpy.ones(nStimComp); 

        # II. Phase, space and time
        omegaX = stimSf * numpy.cos(stimOr); # the stimulus in frequency space
        omegaY = stimSf * numpy.sin(stimOr);
        omegaT = stimTf;

        P = numpy.empty((nFrames*len(xCo), 3)); # nFrames for number of frames, two for x and y coordinate, one for time
        P[:,0] = 2*math.pi*repmat(xCo, 1, nFrames); # P is the matrix that contains the relative location of each filter in space-time (expressed in radians)
        P[:,1] = 2*math.pi*repmat(yCo, 1, nFrames); # P(:,1) and p(:,2) describe location of the filters in space
               
        # Pre-allocate some variables
        respComplex = numpy.zeros((nSf, len(xCo), nFrames));
        
        selSfVec = numpy.zeros((nStimComp, nSf));
        where = 0;
        for iB in range(len(selSf)):
            selSfVec[:, where:where+normPool['nUnits'][iB]] = selSf[iB];
            where = where + normPool['nUnits'][iB];
        
        # Modularize computation - Compute the things that are same for all filters (iF)
        for c in range(nStimComp):  # there are up to nine stimulus components

            if stimCo[c] != 0: #if (selSi ~= 0 && stimCo(c) ~= 0)

                # Use the effective number of frames displayed/stimulus duration
                stimPos = numpy.asarray(range(nFrames))/float(nFrames) + \
                                        stimPh[c] / (2*math.pi*stimTf[c]); # nFrames + the appropriate phase-offset
                P3Temp  = repmat(stimPos, 1, len(xCo));
                P[:,2]  = 2*math.pi*P3Temp; # P(:,2) describes relative location of the filters in time.
            
                omegas = numpy.vstack((omegaX[c], omegaY[c], omegaT[c])); # make this a 3 x len(omegaX) array
                
                rComplex_curr = stimCo[c]*numpy.exp(1j*numpy.dot(P, omegas));
                if c == 0:
                    rComplex_static = rComplex_curr;
                else:
                    rComplex_static = numpy.append(rComplex_static, rComplex_curr, 1);
        
        for iF in range(nSf):
            linR1 = numpy.zeros((nFrames*len(xCo), nStimComp)); # pre-allocation
            linR2 = numpy.zeros((nFrames*len(xCo), nStimComp));
            linR3 = numpy.zeros((nFrames*len(xCo), nStimComp));
            linR4 = numpy.zeros((nFrames*len(xCo), nStimComp));
            computeSum = 0;  # important: if stim contrast or filter sensitivity = zero, no point in computing  response

            # Modularize - Now do the things that are filter-dependent
            for c in range(nStimComp): # there are up to nine stimulus components
                selSi = selOr[c]*selSfVec[c,iF]*selTf[c];    # filter sensitivity for the sinusoid in the frequency domain

                if selSi > 1 or stimCo[c] > 1:
                    pdb.set_trace();

                if selSi != 0 and stimCo[c] != 0:
                    #print('HERE!');
                    #print('SelSi: {0} | stimCo[c]: {1}'.format(selSi, stimCo[c]));
                    computeSum = 1;
                    # now were mostly repeating a simple multiply rather
                    # than exp...
                    rComplex = selSi * rComplex_static[:, c];
        
                    linR1[:,c] = rComplex.real.reshape(linR1[:,c].shape);  # four filters placed in quadrature
                    linR2[:,c] = -1*rComplex.real.reshape(linR2[:,c].shape);
                    linR3[:,c] = rComplex.imag.reshape(linR3[:,c].shape);
                    linR4[:,c] = -1*rComplex.imag.reshape(linR4[:,c].shape);
            
            if computeSum == 1:
                respSimple1 = pow(numpy.maximum(0, linR1.sum(1)), 2); # superposition and half-wave rectification,...
                respSimple2 = pow(numpy.maximum(0, linR2.sum(1)), 2);
                respSimple3 = pow(numpy.maximum(0, linR3.sum(1)), 2);
                respSimple4 = pow(numpy.maximum(0, linR4.sum(1)), 2);
                
                respComplex[iF,:,:] = numpy.reshape(respSimple1 + respSimple2 + respSimple3 + respSimple4, [len(xCo), nFrames]);
        
                if numpy.count_nonzero(numpy.isnan(respComplex[iF,:,:])) > 0:
                    pdb.set_trace();
                
        # integration over space (compute average response across space, normalize by number of spatial frequency channels)

        respInt = respComplex.mean(1) / len(normPool['n']);

        # square root to bring everything in linear contrast scale again
        M['normResp'][p,:,:] = respInt;   

    M.setdefault('trial_used', trial_used);
 
    # if you make/use your own stimuli, just return the output, M;
    # otherwise, save the responses
   
    # THIS NOT GUARANTEED :)
    if not make_own_stim: 
        print('Saving, it seems. In ' + str(loadPath));
        # Save the simulated normalization response in the units structure
        if 'mod' not in S['sfm']:
          S['sfm']['mod'] = dict();
        S['sfm']['mod']['normalization'] = M;
        numpy.save(loadPath + unitName + '_sfm.npy', S)
        
    return M;

def GetNormResp(iU, loadPath, stimParams = [], expDir=[], expInd=None, overwrite=0, dataListName=dataListName):
    ''' GETNORMRESP    Runs the code that computes the response of the
     normalization pool for the recordings in the SfDiv project.
     Returns 'M', result from SFMNormResp
    '''

    # Robbe Goris, 10-30-2015

    M = dict();
    
    # Set characteristics normalization pool
    # The pool includes broad and narrow filters

    # The exponents of the filters used to approximately tile the spatial frequency domain
    n = numpy.array([.75, 1.5]);
    # The number of cells in the broad/narrow pool
    nUnits = numpy.array([12, 15]);
    # The gain of the linear filters in the broad/narrow pool
    gain = numpy.array([.57, .614]);

    normPool = {'n': n, 'nUnits': nUnits, 'gain': gain};

    curr_dir = expDir + 'structures/';
    loadPath = loadPath + curr_dir; # the loadPath passed in is the base path, then we add the directory for the specific experiment
    
    if isinstance(iU, int): # if we've passd in an index to the datalist
        dataList = hf.np_smart_load(loadPath + dataListName);
        unitName = str(dataList['unitName'][iU-1]);
        if expInd is None:
          expInd = hf.exp_name_to_ind(dataList['expType'][iU-1]);
        M = SFMNormResp(unitName, loadPath, normPool, expInd=expInd, overwrite=overwrite);
    else:
        unitName = iU;
        M = SFMNormResp(unitName, [], normPool, stimParams, expInd=expInd, overwrite=overwrite);

    return M;

def SimpleNormResp(S, expInd, gs_mean=None, gs_std=None, normType=2, trialArtificial=None, lgnFrontParams=None):
  ''' A simplified version of the normalization response, in effect, without filters
    The contrast of each stimulus component will be squared and weighted with the same 
    weighting function typically applied (whether that be flat or tuned)
    This replace Linh, which is (nFrames x nTrials) matrix of responses
  '''
  np = numpy;

  if trialArtificial is not None:
    trialInf = trialArtificial;
  else:
    trialInf = S['sfm']['exp']['trial'];
  cons = np.vstack([comp for comp in trialInf['con']]);
  consSq = np.square(cons);
  # cons (and wghts) will be (nComps x nTrials) 
  wghts = hf.genNormWeightsSimple(S, gs_mean, gs_std, normType, trialInf, lgnFrontParams);

  # now put it all together
  resp = np.multiply(wghts, consSq);
  respPerTr = np.sqrt(resp.sum(0)); # i.e. sum over components, then sqrt
  nFrames = hf.num_frames(expInd);
  respByFr = np.array(nFrames * [respPerTr]); # broadcast - response will be same for every frame

  return respByFr;

def SFMGiveBof(params, structureSFM, normType=1, lossType=1, trialSubset=None, maskOri=True, maskIn=None, expInd=1, rvcFits=None, trackSteps=False, overwriteSpikes=None, kMult = 0.10, cellNum=cellNum, excType=1, compute_varExpl=0, lgnFrontEnd=0):
    '''
    Computes the negative log likelihood for the LN-LN model
       Optional arguments: //note: true means include in mask, false means exclude
       trialSubset - pass in the trials you want to evaluate (ignores all other masks)
       maskOri     - in the optimization, we don't include the orientation tuning curve - skip that in the evaluation of loss, too
       maskIn      - pass in a mask (overwrite maskOri and trialSubset, i.e. highest presedence) 
       expInd      - which experiment number?
       rvcFits     - if included, then we'll get adjusted spike counts instead of "raw" saved ones
       trackSteps  - track the NLL within an optimization run
       ovrwSpikes  - overwrite
       kMult       - used for chiSq loss func.
       excType    - which excitatory filter? (1 is the usual, deriv. order Gaussian; 2 is flex. gauss)
     
    Returns NLL ###, respModel
       Note: to keep in sync with the organization/gathering of measured spiking responses, we return
         respModel as spikes/trial (and not a rate of spks/s)
         - (and, of course), we are fitting to the raw spike counts, not the rates
    '''

    # 00 = preferred spatial frequency   (cycles per degree)
    # if excType == 1:
      # 01 = derivative order in space
    # elif excType == 2:
      # 01 = sigma for SF lower than sfPref
      # -2 = sigma for SF higher than sfPref (i.e. the last parameter)
    # 02 = normalization constant        (log10 basis)
    # 03 = response exponent
    # 04 = response scalar
    # 05 = early additive noise
    # 06 = late additive noise
    # 07 = variance of response gain    
    # if fitType == 1 (flat normalization)
    #   08 = asymmetry ("historically", bounded [-0.35, 0.35], currently just "flat")
    # if fitType == 2 (gaussian-weighted normalization responses)
    #   08 = mean of normalization weights gaussian
    #   09 = std of ...
    # if fitType == 3 (gaussian-weighted c50/norm "constant")
    #   08 = offset of c50 tuning filter (filter bounded between [sigOffset, 1]
    #   09/10 = standard deviations to the left and right of the peak of the c50 filter
    #   11 = peak (in sf cpd) of c50 filter
    # if fitType == 4 (gaussian-weighted (flexible/two-halved) normalization responses)
    #   08 = mean of normalization weights gaussian
    #   09/10 = std (left/right) of ...
    # USED ONLY IF lgnFrontEnd > 0
    # -1 = mWeight (with pWeight = 1-mWeight)

    T = structureSFM['sfm'];

    # Get parameter values
    # Excitatory channel
    pref = {'sf': params[0]};
    mWeight = params[-1];
    if excType == 1:
      dord = {'sp': params[1], 'ti': 0.25}; # deriv order in temporal domain = 0.25 ensures broad tuning for temporal frequency
      excChannel = {'pref': pref, 'dord': dord, 'mWeight': mWeight};
    elif excType == 2:
      sigLow = params[1]; sigHigh = params[-1-numpy.sign(lgnFrontEnd)]; # if lgnFrontEnd > 0, then it's the 2nd last param; otherwise, it's the last one
      dord = {'ti': 0.25}; # deriv order in temporal domain = 0.25 ensures broad tuning for temporal frequency
      excChannel = {'pref': pref, 'dord': dord, 'sigLow': sigLow, 'sigHigh': sigHigh, 'mWeight': mWeight};

    # Inhibitory channel
    # nothing in this current iteration - 7/7/17

    # Other (nonlinear) model components
    sigma    = pow(10, params[2]); # normalization constant
    respExp  = params[3]; # response exponent
    scale    = params[4]; # response scalar

    # Noise parameters
    noiseEarly = params[5];   # early additive noise
    noiseLate  = params[6];  # late additive noise
    varGain    = params[7];  # multiplicative noise

    ### Normalization parameters
    normParams = hf.getNormParams(params, normType);
    if normType == 1:
      inhAsym = normParams;
      gs_mean = None; gs_std = None; # replacing the "else" in commented out 'if normType == 2 or normType == 4' below
    elif normType == 2:
      gs_mean = normParams[0];
      gs_std  = normParams[1];
    elif normType == 3:
      # sigma calculation
      offset_sigma = normParams[0];  # c50 filter will range between [v_sigOffset, 1]
      stdLeft      = normParams[1];  # std of the gaussian to the left of the peak
      stdRight     = normParams[2]; # '' to the right '' 
      sfPeak       = normParams[3]; # where is the gaussian peak?
    elif normType == 4: # two-halved Gaussian...
      gs_mean = normParams[0];
      gs_std = normParams[1];
    else:
      inhAsym = normParams;

    # Evaluate prior on response exponent -- corresponds loosely to the measurements in Priebe et al. (2004)
    #priorExp = lognorm.pdf(respExp, 0.3, 0, numpy.exp(1.15)); # matlab: lognpdf(respExp, 1.15, 0.3);
    #NLLExp   = 0; #-numpy.log(priorExp);

    if normType == 3:
      filter = hf.setSigmaFilter(sfPeak, stdLeft, stdRight);
      scale_sigma = -(1-offset_sigma);
      evalSfs = structureSFM['sfm']['exp']['trial']['sf'][0]; # the center SF of all stimuli
      sigmaFilt = hf.evalSigmaFilter(filter, scale_sigma, offset_sigma, evalSfs);
    else:
      sigmaFilt = numpy.square(sigma); # i.e. square the normalization constant

    ''' unused
    # Compute weights for suppressive signals
    nInhChan = T['mod']['normalization']['pref']['sf'];
    nTrials = len(T['exp']['trial']['num']);
    inhWeight = [];
    nFrames = hf.num_frames(expInd);

    if normType == 2 or normType == 4:
      inhWeightMat = hf.genNormWeights(structureSFM, nInhChan, gs_mean, gs_std, nTrials, expInd, normType);
    else: # normType == 1 or anything else,
      gs_mean = None; gs_std = None;
      for iP in range(len(nInhChan)):
          inhWeight = numpy.append(inhWeight, 1 + inhAsym*(numpy.log(T['mod']['normalization']['pref']['sf'][iP]) \
                                              - numpy.mean(numpy.log(T['mod']['normalization']['pref']['sf'][iP]))));
      # assumption by creation (made by Robbe) - only two normalization pools
      inhWeightT1 = numpy.reshape(inhWeight, (1, len(inhWeight)));
      inhWeightT2 = repmat(inhWeightT1, nTrials, 1);
      inhWeightT3 = numpy.reshape(inhWeightT2, (nTrials, len(inhWeight), 1));
      inhWeightMat  = numpy.tile(inhWeightT3, (1,1,nFrames));
    '''
                              
    # Evaluate sfmix experiment
    for iR in range(1): #range(len(structureSFM['sfm'])): # why 1 for now? We don't have S.sfm as array (just one)
        T = structureSFM['sfm']; # [iR]

        # the lgn params (lgnPass) will be specified in SFMSimpleResp[_par] if lgnFrontEnd > 0
        E = SFMSimpleResp(structureSFM, excChannel, expInd=expInd, excType=excType, lgnFrontEnd=lgnFrontEnd);
        #E = SFMSimpleResp_par(structureSFM, excChannel, expInd=expInd, excType=excType, lgnFrontEnd=lgnFrontEnd);

        #timing/debugging parallelization
        # Get simple cell response for excitatory channel
        #E = '''SFMSimpleResp(structureSFM, excChannel, expInd=expInd, excType=excType);'''
        #Epar = '''SFMSimpleResp_par(structureSFM, excChannel, expInd=expInd, excType=excType);'''

        #import timeit
        #etime = timeit.timeit(stmt=E, globals={'structureSFM': structureSFM, 'excChannel': excChannel, 'expInd': expInd, 'excType': excType, 'SFMSimpleResp': SFMSimpleResp}, number=15);
        #ePartime = timeit.timeit(stmt=Epar, globals={'structureSFM': structureSFM, 'excChannel': excChannel, 'expInd': expInd, 'excType': excType, 'SFMSimpleResp_par': SFMSimpleResp_par}, number=15);

        # Extract simple cell response (half-rectified linear filtering)
        Lexc = E['simpleResp']; # [nFrames x nTrials]

        # Get inhibitory response (pooled responses of complex cells tuned to wide range of spatial frequencies, square root to bring everything in linear contrast scale again)
        #Linh = numpy.sqrt((inhWeightMat*T['mod']['normalization']['normResp']).sum(1)).transpose();
        if lgnFrontEnd > 0:
          lgnPass = E;
        else:
          lgnPass = None;
        Linh = SimpleNormResp(structureSFM, expInd, gs_mean, gs_std, normType, lgnFrontParams=lgnPass); # [nFrames x nTrials]
 
        # Compute full model response (the normalization signal is the same as the subtractive suppressive signal)
        numerator     = noiseEarly + Lexc;
        denominator   = pow(sigmaFilt + pow(Linh, 2), 0.5); # square Linh added 7/24 - was mistakenly not fixed earlier
        #ratio         = pow(numerator/denominator, respExp);
        # NOTE^^^: TODO - turned off the 0 thresholding to see what happens and to better fit LGN responses, which - when adjusted - can be negative        
        ratio         = pow(numpy.maximum(0, numerator/denominator), respExp);
        meanRate      = ratio.mean(0);
        respModel     = noiseLate + scale*meanRate; # respModel[iR]
        rateModel     = respModel / T['exp']['trial']['duration'];
        # and get the spike count
        if overwriteSpikes is not None:
          spikeCount = hf.get_spikes(T['exp']['trial'], rvcFits=rvcFits, expInd=expInd, overwriteSpikes=overwriteSpikes);
        else:
          f1f0_rat = hf.compute_f1f0(T['exp']['trial'], cellNum, expInd, loc_data=None)[0];
          # TODO: should add line forcing F1 if LGN experiment...
          # -- rvcMod = - 1 only because rvcFits already contains the loaded fits (tells func call to use rvcName as fits)
          spikeRate = hf.get_adjusted_spikerate(T['exp']['trial'], cellNum, expInd, dataPath=None, rvcName=rvcFits, rvcMod=-1, baseline_sub=False);
          # NOTE: this is now a spikerate, not spike count, so let's convert (by multiplying rate by stimDir)
          spikeCount = numpy.multiply(spikeRate, hf.get_exp_params(expInd).stimDur);
          # now, we're recasting as an int, AND (necessary for simple cells), rounding to the nearest integer
          # -- yes, int32 cannot represent NaNs, but all of these values are masked out, anyway, so the artifact is not harmful
          spikeCount = numpy.rint(spikeCount).astype(numpy.int32);

        ### Masking the data - which trials will we include
        # now get the "right" subset of the data for evaluating loss (e.x. by default, orientation tuning trials are not included)
        if maskOri and expInd == 1: # as of 11.18.26, we can only hold out/evaluate subset on original V1 data 
          # start with all trials...
          mask = numpy.ones_like(spikeCount, dtype=bool); # i.e. true
          # and get rid of orientation tuning curve trials
          oriBlockIDs = numpy.hstack((numpy.arange(131, 155+1, 2), numpy.arange(132, 136+1, 2))); # +1 to include endpoint like Matlab

          oriInds = numpy.empty((0,));
          for iB in oriBlockIDs:
              indCond = numpy.where(T['exp']['trial']['blockID'] == iB);
              if len(indCond[0]) > 0:
                  oriInds = numpy.append(oriInds, indCond);
          mask[oriInds.astype(numpy.int64)] = False;
        else: # just go with all trials
          # start with all trials...
          mask = numpy.ones_like(spikeCount, dtype=bool); # i.e. true
        # BUT, if we pass in trialSubset, then use this as our mask (i.e. overwrite the above mask)
        if trialSubset is not None: # i.e. if we passed in some trials to specifically include, then include ONLY those (i.e. skip the rest)
          # start by including NO trials
          mask = numpy.zeros_like(spikeCount, dtype=bool); # i.e. true
          mask[trialSubset.astype(numpy.int64)] = True;

        if maskIn is not None:
          mask = maskIn; # overwrite the mask with the one we've passed in!

        # organize responses so that we can package them for evaluating varExpl...
        _, _, expByCond, expAll = hf.organize_resp(spikeCount, structureSFM['sfm']['exp']['trial'], expInd, mask);
        _, _, modByCond, modAll = hf.organize_resp(respModel, structureSFM['sfm']['exp']['trial'], expInd, mask);
        # - and now compute varExpl - first for SF tuning curves, then for RVCs...
        nDisp, nSf, nCon = expByCond.shape;
        vE_SF = numpy.nan * numpy.zeros((nDisp, nCon));
        vE_con = numpy.nan * numpy.zeros((nDisp, nSf));
        
        if compute_varExpl == 1:
          for dI in numpy.arange(nDisp):
            for sI in numpy.arange(nSf):
               vE_con[dI, sI] = hf.var_explained(hf.nan_rm(expByCond[dI, sI, :]), hf.nan_rm(modByCond[dI, sI, :]), None);
            for cI in numpy.arange(nCon):
               vE_SF[dI, cI] = hf.var_explained(hf.nan_rm(expByCond[dI, :, cI]), hf.nan_rm(modByCond[dI, :, cI]), None);
          
        if lossType == 1:
          # alternative loss function: just (sqrt(modResp) - sqrt(neurResp))^2
          # sqrt - now handles negative responses by first taking abs, sqrt, then re-apply the sign 
          lsq = numpy.square(numpy.sign(respModel[mask])*numpy.sqrt(numpy.abs(respModel[mask])) - numpy.sign(spikeCount[mask])*numpy.sqrt(numpy.abs(spikeCount[mask])));
          #lsq = numpy.square(numpy.add(numpy.sqrt(respModel[mask]), -numpy.sqrt(spikeCount[mask])));
          NLL = numpy.mean(lsq);
          nll_notSum = numpy.square(numpy.add(numpy.sign(respModel)*numpy.sqrt(numpy.abs(respModel)), -numpy.sign(spikeCount)*numpy.sqrt(numpy.abs(spikeCount))));
          #varExpl_split = [hf.var_explained(dr, mr, None) for dr, mr in zip(exp_responses[0], mod_responses[0])];
        elif lossType == 2:
          poiss_llh = numpy.log(poisson.pmf(spikeCount[mask], respModel[mask]));
          nll_notSum = poiss_llh;
          NLL = numpy.mean(-poiss_llh);
          nll_notSum = -numpy.log(poisson.pmf(spikeCount, respModel));
          varExpl_split = [];
        elif lossType == 3:
          # Get predicted spike count distributions
          mu  = numpy.maximum(.01, respModel[mask]); # The predicted mean spike count; respModel[iR]
          var = mu + (varGain*pow(mu,2));                        # The corresponding variance of the spike count
          r   = pow(mu,2)/(var - mu);                           # The parameters r and p of the negative binomial distribution
          p   = r/(r + mu);
          llh = nbinom.pmf(spikeCount[mask], r, p); # Likelihood for each pass under doubly stochastic model
          NLL = numpy.mean(-numpy.log(llh)); # The negative log-likelihood of the whole data-set; [iR]
          nll_notSum = numpy.nan; # FIX/TODO
          varExpl_split = [];
        elif lossType == 4: #chi squared
          exp_responses = [expByCond.flatten(), numpy.nanvar(expAll, axis=3).flatten()];
          mod_responses = [modByCond.flatten(), numpy.nanvar(modAll, axis=3).flatten()];
          NLL, nll_notSum = hf.chiSq(exp_responses, mod_responses, kMult = kMult);
  
    if trackSteps == True:
      global params_glob, loss_glob, resp_glob;
      params_glob.append(params);
      loss_glob.append(NLL);
      resp_glob.append(respModel);

    return NLL, respModel, nll_notSum, vE_SF, vE_con; # add varExpl stuff here...

def SFMsimulateNew(params, structureSFM, disp, con, sf_c, normType=1, expInd=1, nRepeats=None, excType=1):
  ''' New version of SFMsimulate...19.05.13 create date
      See helper_fcns/makeStimulusRef for details on input parameters
  '''
  T = structureSFM['sfm'];
  # now we have stimuluated trials!
  trialSim = hf.makeStimulusRef(T['exp']['trial'], disp, con, sf_c, expInd, nRepeats);

  trialInf = trialSim;
  stimDur = hf.get_exp_params(expInd).stimDur;

  ### Get parameter values
  # Excitatory channel
  pref = {'sf': params[0]};
  if excType == 1:
    dord = {'sp': params[1], 'ti': 0.25}; # deriv order in temporal domain = 0.25 ensures broad tuning for temporal frequency
    excChannel = {'pref': pref, 'dord': dord};
  elif excType == 2:
    sigLow = params[1]; sigHigh = params[-1-numpy.sign(lgnFrontEnd)]; # if lgnFrontEnd>0, then it's the 2nd last param; otherwise, it's the last one (i.e. -1 - [sign(0)=0] = -1)
    dord = {'ti': 0.25}; # deriv order in temporal domain = 0.25 ensures broad tuning for temporal frequency
    excChannel = {'pref': pref, 'dord': dord, 'sigLow': sigLow, 'sigHigh': sigHigh};

  # Other (nonlinear) model components
  sigma    = pow(10, params[2]); # normalization constant
  respExp  = params[3]; # response exponent
  scale    = params[4]; # response scalar

  # Noise parameters
  noiseEarly = params[5];   # early additive noise
  noiseLate  = params[6];  # late additive noise
  varGain    = params[7];  # multiplicative noise

  ### Normalization parameters
  normParams = hf.getNormParams(params, normType);
  if normType == 1:
    inhAsym = normParams;
  elif normType == 2:
    gs_mean = normParams[0];
    gs_std  = normParams[1];
  elif normType == 3:
    # sigma calculation
    offset_sigma = normParams[0];  # c50 filter will range between [v_sigOffset, 1]
    stdLeft      = normParams[1];  # std of the gaussian to the left of the peak
    stdRight     = normParams[2]; # '' to the right '' 
    sfPeak       = normParams[3]; # where is the gaussian peak?
  elif normType == 4:
    gs_mean, gs_std = normParams[0], normParams[1]
  else:
    inhAsym = normParams;

  ########################
  #### the following is not used, since we use the simple normalization calculation...
  #### (other than defining gs_mean/gs_std as None if flat normalization...
  # Compute weights for suppressive signals
  if normType == 3:
    filter = hf.setSigmaFilter(sfPeak, stdLeft, stdRight);
    scale_sigma = -(1-offset_sigma);
    evalSfs = trialInf['sf'][0]; # the center SF of all stimuli
    sigmaFilt = hf.evalSigmaFilter(filter, scale_sigma, offset_sigma, evalSfs);
  else:
    sigmaFilt = numpy.square(sigma); # i.e. normalization constant squared

  if normType == 2 or normType == 4:
    inhWeightMat = [];
    #inhWeightMat = hf.genNormWeights(structureSFM, nInhChan, gs_mean, gs_std, nTrials, expInd, normType);
  else: # normType == 1 or anything else, we just go with 
    gs_mean = None; gs_std = None;
    '''
    for iP in range(len(nInhChan)):
        inhWeight = numpy.append(inhWeight, 1 + inhAsym*(numpy.log(T['mod']['normalization']['pref']['sf'][iP]) \
                                            - numpy.mean(numpy.log(T['mod']['normalization']['pref']['sf'][iP]))));
    # assumption (made by Robbe) - only two normalization pools
    inhWeightT1 = numpy.reshape(inhWeight, (1, len(inhWeight)));
    inhWeightT2 = repmat(inhWeightT1, nTrials, 1);
    inhWeightT3 = numpy.reshape(inhWeightT2, (nTrials, len(inhWeight), 1));
    inhWeightMat  = numpy.tile(inhWeightT3, (1,1,nFrames));
    '''
  ########################

  # Evaluate sfmix experiment
  T = structureSFM['sfm']; # [iR]

  # Get simple cell response for excitatory channel
  E = SFMSimpleResp(structureSFM, excChannel, stimParams=[], expInd=expInd, trialInf=trialInf, excType=excType);

  # Extract simple cell response (half-rectified linear filtering)
  Lexc = E['simpleResp'];

  # Get inhibitory response (pooled responses of complex cells tuned to wide range of spatial frequencies, square root to bring everything in linear contrast scale again)
  # NOTE (19.05.13): GetNormResp is ignored, since we use the simple normalization response, now
  Linh = SimpleNormResp(structureSFM, expInd, gs_mean, gs_std, normType, trialInf);
  '''
  normResp = GetNormResp(structureSFM, [], stimParams, expInd=expInd);
  if unweighted == 1:
    return [], [], Lexc, normResp['normResp'], [];
  #Linh = numpy.sqrt((inhWeightMat*normResp['normResp']).sum(1)).transpose();
  '''
  # Compute full model response (the normalization signal is the same as the subtractive suppressive signal)
  numerator     = noiseEarly + Lexc;
  # taking square root of denominator (after summing squares...) to bring in line with computation in Carandini, Heeger, Movshon, '97
  denominator   = pow(sigmaFilt + pow(Linh, 2), 0.5); # squaring Linh - edit 7/17
  ratio         = pow(numerator/denominator, respExp);
  meanRate      = ratio.mean(0);
  respModel     = noiseLate + scale*meanRate; # respModel[iR]
  rateModel     = respModel / stimDur;

  return rateModel, Linh, Lexc, denominator;

def SFMsimulate(params, structureSFM, stimFamily, con, sf_c, unweighted = 0, normType=1, expInd=1, excType=1, lgnFrontEnd=0):
    # Currently, will get slightly different stimuli for excitatory and inhibitory/normalization pools
    # But differences are just in phase/TF, but for TF, drawn from same distribution, anyway...
    # 4/27/18: if unweighted = 1, then do the calculation/return normResp with weights applied; otherwise, just return the unweighted filter responses

    #print('simulate!');
    
    T = structureSFM['sfm'];

    # Get parameter values
    # Excitatory channel
    pref = {'sf': params[0]};
    if lgnFrontEnd > 0:
      mWeight = params[-1];
    if excType == 1:
      dord = {'sp': params[1], 'ti': 0.25}; # deriv order in temporal domain = 0.25 ensures broad tuning for temporal frequency
      excChannel = {'pref': pref, 'dord': dord, 'mWeight': mWeight};
    elif excType == 2:
      sigLow = params[1]; sigHigh = params[-1-numpy.sign(lgnFrontEnd)]; # if lgnFrontEnd>0, then it's the 2nd last param; otherwise, it's the last one
      dord = {'ti': 0.25}; # deriv order in temporal domain = 0.25 ensures broad tuning for temporal frequency
      excChannel = {'pref': pref, 'dord': dord, 'sigLow': sigLow, 'sigHigh': sigHigh, 'mWeight': mWeight};

    # Other (nonlinear) model components
    sigma    = pow(10, params[2]); # normalization constant
    respExp  = params[3]; # response exponent
    scale    = params[4]; # response scalar

    # Noise parameters
    noiseEarly = params[5];   # early additive noise
    noiseLate  = params[6];  # late additive noise
    varGain    = params[7];  # multiplicative noise

    ### Normalization parameters
    normParams = hf.getNormParams(params, normType);
    if normType == 1:
      inhAsym = normParams;
    elif normType == 2:
      gs_mean = normParams[0];
      gs_std  = normParams[1];
    elif normType == 3:
      # sigma calculation
      offset_sigma = normParams[0];  # c50 filter will range between [v_sigOffset, 1]
      stdLeft      = normParams[1];  # std of the gaussian to the left of the peak
      stdRight     = normParams[2]; # '' to the right '' 
      sfPeak       = normParams[3]; # where is the gaussian peak?
    elif normType == 4:
      gs_mean, gs_std = normParams[0], normParams[1]
    else:
      inhAsym = normParams;

    # Get stimulus structure ready...
    stimParams = dict();
    stimParams['stimFamily'] = stimFamily;
    stimParams['conLevel'] = con;
    stimParams['sf_c'] = sf_c;
    stimParams['repeats'] = 1; # defaults to 10 anyway, in makeStimulus.py
    
    # Compute weights for suppressive signals
    nInhChan = T['mod']['normalization']['pref']['sf'];
    nTrials = stimParams['repeats'];
    inhWeight = [];
    nFrames = hf.num_frames(expInd); # always
    stimDur = hf.get_exp_params(expInd).stimDur;

    if normType == 3:
      filter = hf.setSigmaFilter(sfPeak, stdLeft, stdRight);
      scale_sigma = -(1-offset_sigma);
      evalSfs = structureSFM['sfm']['exp']['trial']['sf'][0]; # the center SF of all stimuli
      sigmaFilt = hf.evalSigmaFilter(filter, scale_sigma, offset_sigma, evalSfs);
    else:
      sigmaFilt = numpy.square(sigma); # i.e. normalization constant squared

    if normType == 2 or normType == 4:
      inhWeightMat = hf.genNormWeights(structureSFM, nInhChan, gs_mean, gs_std, nTrials, expInd, normType);
    else: # normType == 1 or anything else, we just go with 
      gs_mean = None; gs_std = None;
      for iP in range(len(nInhChan)):
          inhWeight = numpy.append(inhWeight, 1 + inhAsym*(numpy.log(T['mod']['normalization']['pref']['sf'][iP]) \
                                              - numpy.mean(numpy.log(T['mod']['normalization']['pref']['sf'][iP]))));
      # assumption (made by Robbe) - only two normalization pools
      inhWeightT1 = numpy.reshape(inhWeight, (1, len(inhWeight)));
      inhWeightT2 = repmat(inhWeightT1, nTrials, 1);
      inhWeightT3 = numpy.reshape(inhWeightT2, (nTrials, len(inhWeight), 1));
      inhWeightMat  = numpy.tile(inhWeightT3, (1,1,nFrames));
                              
    # Evaluate sfmix experiment
    T = structureSFM['sfm']; # [iR]
    
    # Get simple cell response for excitatory channel
    E = SFMSimpleResp(structureSFM, excChannel, stimParams, expInd=expInd, excType=excType, lgnFrontEnd=lgnFrontEnd);

    # Extract simple cell response (half-rectified linear filtering)
    Lexc = E['simpleResp'];

    # Get inhibitory response (pooled responses of complex cells tuned to wide range of spatial frequencies, square root to bring everything in linear contrast scale again)
    # NOTE (19.05.13): GetNormResp is ignored, since we use the simple normalization response, now
    normResp = GetNormResp(structureSFM, [], stimParams, expInd=expInd);
    if unweighted == 1:
      return [], [], Lexc, normResp['normResp'], [];
    #Linh = numpy.sqrt((inhWeightMat*normResp['normResp']).sum(1)).transpose();
    Linh = SimpleNormResp(structureSFM, expInd, gs_mean, gs_std, normType);

    # Compute full model response (the normalization signal is the same as the subtractive suppressive signal)
    numerator     = noiseEarly + Lexc;
    # taking square root of denominator (after summing squares...) to bring in line with computation in Carandini, Heeger, Movshon, '97
    denominator   = pow(sigmaFilt + pow(Linh, 2), 0.5); # squaring Linh - edit 7/17
    ratio         = pow(numpy.maximum(0, numerator/denominator), respExp);
    meanRate      = ratio.mean(0);
    respModel     = noiseLate + scale*meanRate; # respModel[iR]
    rateModel     = respModel / stimDur;

    return respModel, Linh, Lexc, normResp['normResp'], denominator;

def setModel(cellNum, expDir, lossType = 1, fitType = 1, initFromCurr = 1, fL_name=None, trackSteps=False, holdOutCondition = None, modRecov = None, rvcBase=rvcBaseName, rvcMod=1, dataListName=dataListName, kMult=0.1, excType=1, lgnFrontEnd=0, fixRespExp=None):
    # Given just a cell number, will fit the Robbe-inspired V1 model to the data for a particular experiment (expInd)
    #
    # lossType
    #   1 - loss := square(sqrt(resp) - sqrt(pred))
    #   2 - loss := poissonProb(spikes | modelRate)
    #   3 - loss := modPoiss model (a la Goris, 2014)
    #   4 - loss := chi squared (a la Cavanaugh, 2002)
    #
    # fitType - what is the model formulation?
    #   1 := flat normalization
    #   2 := gaussian-weighted normalization responses
    #   3 := gaussian-weighted c50/norm "constant"
    #   4 := gaussian-weighted (flexible/two-halved) normalization responses
    #
    # excType - 1 (deriv. ord of gauss); 2 (flex. gauss)
    #
    # holdOutCondition - [[d, c, sf]*N] or None
    #   which condition should we hold out from the dataset
    #   note that it is passed in as list of lists  

    ########
    # Load cell
    ########
    loc_base = os.getcwd() + '/'; # ensure there is a "/" after the final directory
    loc_data = loc_base + expDir + 'structures/';

    if 'pl1465' in loc_base:
      loc_str = 'HPC';
    else:
      loc_str = '';
    if fL_name is None: # otherwise, it's already defined...
      if modRecov == 1:
        fL_name = 'mr_fitList%s_190516cA' % loc_str
      else:
        #fL_name = 'fitList%s_200417%s' % (loc_str, hf.chiSq_suffix(kMult));
        #fL_name = 'fitList%s_200418%s' % (loc_str, hf.chiSq_suffix(kMult));
        #fL_name = 'fitList%s_200507%s' % (loc_str, hf.chiSq_suffix(kMult));
        #fL_name = 'fitList%s_200418%s_TNC' % (loc_str, hf.chiSq_suffix(kMult));
        #fL_name = 'fitList%s_190321c' % loc_str
        if excType == 1:
          fL_name = 'fitList%s_200417' % (loc_str);
        elif excType == 2:
          fL_name = 'fitList%s_200507' % (loc_str);
        #fL_name = 'fitList%s_200519%s' % (loc_str, hf.chiSq_suffix(kMult));
        #fL_name = 'fitList%s_200522%s' % (loc_str, hf.chiSq_suffix(kMult));

    if lossType == 4: # chiSq...
      fL_name = '%s%s' % (fL_name, hf.chiSq_suffix(kMult));

    if fixRespExp is not None:
      fL_name = '%s_re%d' % (fL_name, np.round(fixRespExp*10)); # suffix to indicate that the response exponent is fixed...

    if lgnFrontEnd == 1:
      fL_name = '%s_LGN' % fL_name # implicit "a" at the end of LGN...
    if lgnFrontEnd == 2:
      fL_name = '%s_LGNb' % fL_name

    np = numpy;

    fitListName = hf.fitList_name(base=fL_name, fitType=fitType, lossType=lossType);
    # get the name for the stepList name, regardless of whether or not we run this now
    stepListName = str(fitListName.replace('.npy', '_details.npy'));

    print('\nFitList: %s' % fitListName);

    if os.path.isfile(loc_data + fitListName):
      fitList = hf.np_smart_load(str(loc_data + fitListName));
    else:
      fitList = dict();

    dataList = hf.np_smart_load(str(loc_data + dataListName));
    dataNames = dataList['unitName'];

    expInd = hf.exp_name_to_ind(dataList['expType'][cellNum-1]);
    
    print('loading data structure from %s...' % loc_data);
    S = hf.np_smart_load(str(loc_data + dataNames[cellNum-1] + '_sfm.npy')); # why -1? 0 indexing...
    print('...finished loading');
    trial_inf = S['sfm']['exp']['trial'];

    ## Is this a model recovery fit?
    recovSpikes = None;
    if modRecov is not None: # Not very clean, but it will have to do for now
      if modRecov == 1:
        try:
          recovSpikes = hf.get_recovInfo(S, normType)[1];
        except:
          warnings.warn('You are not set up to run model recovery analysis with this norm type!\nSetting recovery spikes to None');
    # get prefSfEst
    try: 
      dfits = hf.np_smart_load(loc_data + descrFitName);
      hiCon = -1;
      prefSfEst = dfits[cellNum-1]['prefSf'][0][hiCon]; # get high contrast, single grating prefSf
      sigLo, sigHi = dfits[cellNum-1]['params'][0, hiCon, 3:5]; # parameter locations for sigmaLow/High
    except:
      if expInd == 1:
        prefOrEst = mode(trial_inf['ori'][1]).mode;
        trialsToCheck = trial_inf['con'][0] == 0.01;
        prefSfEst = mode(trial_inf['sf'][0][trialsToCheck==True]).mode;
      else:
        prefOrEst = 0;
        allSfs    = np.unique(trial_inf['sf'][0]);
        allSfs    = allSfs[~np.isnan(allSfs)]; # remove NaN...
        prefSfEst = np.median(allSfs);

    # load RVC fits, then get normConst estimate
    rvcFits = hf.get_rvc_fits(loc_data, expInd, cellNum, rvcName=rvcBase, rvcMod=rvcMod);
    try: 
      peakSf = prefSfEst; # just borrow from above
      stimVals = hf.tabulate_responses(S, expInd)[1];
      all_sfs = stimVals[2];
      # now, get the index corresponding to that peak SF and get the c50 from the corresponding RVC fit
      prefSfInd = np.argmin(np.abs(all_sfs - peakSf));
      # get the c50, but take log10 (we optimize in that space rather than in contrast)
      c50_est = np.log10(hf.c50_empirical(rvcMod, rvcFits[0]['params'][prefSfInd])[0]);
      normConst = np.minimum(c50_est, np.log10(0.25)); # don't let a c50 value larger than X as the starting point
    except:
      # why -1? Talked with Tony, he suggests starting with lower sigma rather than higher/non-saturating one
      normConst = -2; # i.e. c50 = 0.01 (1% contrast); yes, it's low...

    ########
    # 00 = preferred spatial frequency   (cycles per degree)
    # if excType == 1:
      # 01 = derivative order in space
    # elif excType == 2:
      # 01 = sigma for SF lower than sfPref
      # -1-lgnFrontEnd = sigma for SF higher than sfPref (i.e. the last parameter)
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
    # if fitType == 4
    # 08 = mean of (log)gaussian for normalization weights
    # 09/10 = std of (log)gaussian (to left/right) for normalization weights
    # USED ONLY IF lgnFrontEnd == 1
    # -1 = mWeight (with pWeight = 1-mWeight)

    if cellNum-1 in fitList:
      try:
        curr_params = fitList[cellNum-1]['params']; # load parameters from the fitList! this is what actually gets updated...
        currNLL = fitList[cellNum-1]['NLL']; # exists - either from real fit or as placeholder
      except:
        curr_params = [];
        currNLL = 1e4;
        initFromCurr = 0; # override initFromCurr so that we just go with default parameters
        fitList[cellNum-1] = dict();
        fitList[cellNum-1]['NLL'] = 1e4; # large initial value...
    else: # set up basic fitList structure...
      curr_params = [];
      currNLL = 1e4;
      initFromCurr = 0; # override initFromCurr so that we just go with default parameters
      fitList[cellNum-1] = dict();
      fitList[cellNum-1]['NLL'] = 1e4; # large initial value...
    # get the list of NLL per run
    try:
      nll_history = fitList[cellNum-1]['nll_history'];
    except:
      nll_history = np.array([]);

    if numpy.any(numpy.isnan(curr_params)): # if there are nans, we need to ignore...
      curr_params = [];
      initFromCurr = 0;

    pref_sf = float(prefSfEst) if initFromCurr==0 else curr_params[0];
    if excType == 1:
      dOrdSp = np.random.uniform(1, 3) if initFromCurr==0 else curr_params[1];
    elif excType == 2:
      sigLow = np.random.uniform(1, 4) if initFromCurr==0 else curr_params[1];
      sigHigh = np.random.uniform(0.1, 2) if initFromCurr==0 else curr_params[-1-numpy.sign(lgnFrontEnd)]; # if lgnFrontEnd == 0, then it's the last param; otherwise it's the 2nd to last param
    normConst = normConst if initFromCurr==0 else curr_params[2];
    #respExp = 1 if initFromCurr==0 else curr_params[3];
    respExp = np.random.uniform(1.5, 2.5) if initFromCurr==0 else curr_params[3];
    respScalar = np.random.uniform(10, 200) if initFromCurr==0 else curr_params[4];
    noiseEarly = np.random.uniform(0.001, 0.01) if initFromCurr==0 else curr_params[5]; # 02.27.19 - (dec. up. bound to 0.01 from 0.1)
    noiseLate = np.random.uniform(0.1, 1) if initFromCurr==0 else curr_params[6];
    varGain = np.random.uniform(0.1, 1) if initFromCurr==0 else curr_params[7];
    if lgnFrontEnd > 0:
      # Now, the LGN weighting 
      mWeight = np.random.uniform(0.25, 0.75) if initFromCurr==0 else curr_params[-1];

    if fitType == 1:
      inhAsym = 0; 
    if fitType == 2:
      normMean = np.log10(pref_sf) if initFromCurr==0 else curr_params[8]; # start as matched to excFilter
      normStd = 1.5 if initFromCurr==0 else curr_params[9]; # start at high value (i.e. broad)
    if fitType == 3:
      sigOffset = np.random.uniform(0, 0.05) if initFromCurr==0 else curr_params[8];
      stdLeft = np.random.uniform(1, 5) if initFromCurr==0 else curr_params[9];
      stdRight = np.random.uniform(1, 5) if initFromCurr==0 else curr_params[10];
      sigPeak = float(prefSfEst) if initFromCurr==0 else curr_params[11];
    if fitType == 4:
      normMean = np.log10(pref_sf) if initFromCurr==0 else curr_params[8]; # start as matched to excFilter
      normStdL = 1.5 if initFromCurr==0 else curr_params[9]; # start at high value (i.e. broad)
      normStdR = 1.5 if initFromCurr==0 else curr_params[10]; # start at high value (i.e. broad)
   
    ### Now, if we want to initialize the core paramers with the other fit type...
    if initFromCurr == -1 and (fitType==1 or fitType==2 or fitType == 4): # then initialize from the opposite case...
      if fitType==1:
        altType = 2; # TODO? Decide whether alt for flat is wght or flex?
      elif fitType==2 or fitType == 4:
        altType = 1;
      altFL = hf.fitList_name(base=fL_name, fitType=altType, lossType=lossType);
      try:
        altFits = hf.np_smart_load(loc_data + altFL);
        if cellNum-1 in altFits:
          altParams = altFits[cellNum-1]['params'];
          if excType == 1:
            pref_sf,dOrdSp,normConst,respExp,respScalar,noiseEarly,noiseLate,varGain = altParams[0:8];
          elif excType == 2:
            pref_sf,sigLow,normConst,respExp,respScalar,noiseEarly,noiseLate,varGain = altParams[0:8];
            sigHigh = altParams[-1-np.sign(lgnFrontEnd)]; # if lgnFrontEnd > 0, then it's the 2nd last param; otherwise, it's the last one
          if lgnFrontEnd > 0:
            mWeight = altParams[-1];
          else:
            mWeight = np.nan;
      except:
        warnings.warn('Could not initialize with alternate-fit parameters; defaulting to typical process');

    if excType == 1:
      print('Initial parameters:\n\tsf: ' + str(pref_sf)  + '\n\td.ord: ' + str(dOrdSp) + '\n\tnormConst: ' + str(normConst));
    elif excType == 2:
      print('Initial parameters:\n\tsf: ' + str(pref_sf)  + '\n\tsigLow: ' + str(sigLow) + '\n\tsigHigh: ' + str(sigHigh) + '\n\tnormConst: ' + str(normConst));
    print('\n\trespExp ' + str(respExp) + '\n\trespScalar ' + str(respScalar) + '\n\tmagnoWeight: ' + str(mWeight));
    
    #########
    # Now get all the data we need
    #########    
    # stimulus information
    
    # vstack to turn into array (not array of arrays!)
    stimOr = np.vstack(trial_inf['ori']);

    #purge of NaNs...
    mask = np.isnan(np.sum(stimOr, 0)); # sum over all stim components...if there are any nans in that trial, we know
    objWeight = np.ones((stimOr.shape[1]));    

    if expInd == 1:
      # get rid of orientation tuning curve trials
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
      for cond in holdOutCondition: # i.e. we pass in as array of [disp, con, sf] combinations
        val_trials = hf.get_valid_trials(S, cond[0], cond[1], cond[2], expInd)[0];
        mask[val_trials] = True; # as in, don't include those trials either!
      
    # Set up model here - get the parameters and parameter bounds
    # -- note that we automatically add mWeight to the paramlist, but we'll trim it off if needed (eaiser to do in terms of code logic)
    if fitType == 1:
      if excType == 1:
        param_list = (pref_sf, dOrdSp, normConst, respExp, respScalar, noiseEarly, noiseLate, varGain, inhAsym, mWeight);
      elif excType == 2:
        param_list = (pref_sf, sigLow, normConst, respExp, respScalar, noiseEarly, noiseLate, varGain, inhAsym, sigHigh, mWeight);
    elif fitType == 2:
      if excType == 1:
        param_list = (pref_sf, dOrdSp, normConst, respExp, respScalar, noiseEarly, noiseLate, varGain, normMean, normStd, mWeight);
      elif excType == 2:
        param_list = (pref_sf, sigLow, normConst, respExp, respScalar, noiseEarly, noiseLate, varGain, normMean, normStd, sigHigh, mWeight);
    ### TODO: add excType = [1/2] and mWeight (lgnFrontEnd>0) to fitType == 3/4 sections 
    ### TODO: -- and adjust hf.getConstraints accordingly
    elif fitType == 3:
      param_list = (pref_sf, dOrdSp, normConst, respExp, respScalar, noiseEarly, noiseLate, varGain, sigOffset, stdLeft, stdRight, sigPeak, mWeight);
    elif fitType == 4:
      param_list = (pref_sf, dOrdSp, normConst, respExp, respScalar, noiseEarly, noiseLate, varGain, normMean, normStdL, normStdR, mWeight);
    all_bounds = hf.getConstraints(fitType, excType, fixRespExp);
    if lgnFrontEnd == 0: # then we'll trim off the last constraint, which is mWeight bounds (and the last param, which is mWeight)
      param_list = param_list[0:-1];
      all_bounds = all_bounds[0:-1];
   
    ## NOW: set up the objective function
    obj = lambda params: SFMGiveBof(params, structureSFM=S, normType=fitType, lossType=lossType, maskIn=~mask, expInd=expInd, rvcFits=rvcFits, trackSteps=trackSteps, overwriteSpikes=recovSpikes, kMult=kMult, excType=excType, lgnFrontEnd=lgnFrontEnd)[0];

    print('...now minimizing!'); 
    if 'TNC' in fL_name:
      tomin = opt.minimize(obj, param_list, bounds=all_bounds, method='TNC');
    else:
      tomin = opt.minimize(obj, param_list, bounds=all_bounds);

    opt_params = tomin['x'];
    NLL = tomin['fun'];

    ## we've finished optimization, so reload again to make sure that this  NLl is better than the currently saved one
    ## -- why do we have to do it again here? We may be running multiple fits for the same cells at the same time
    ## --   and we want to make sure that if one of those has updated, we don't overwrite that opt. if it's better
    if os.path.exists(loc_data + fitListName):
      fitList = hf.np_smart_load(str(loc_data + fitListName));
      try: # well, even if fitList loads, we might not have currNLL, so we have to have an exception here
        currNLL = fitList[cellNum-1]['NLL']; # exists - either from real fit or as placeholder
      except:
        pass; # we've already defined the currNLL...

    ### SAVE: Now we save the results, including the results of each step, if specified
    print('...finished. Current NLL (%.2f) vs. previous NLL (%.2f)' % (NLL, currNLL)); 
    # reload fitlist in case changes have been made with the file elsewhere!
    if os.path.exists(loc_data + fitListName):
      fitList = hf.np_smart_load(str(loc_data + fitListName));
    # else, nothing to reload!!!
    # but...if we reloaded fitList and we don't have this key (cell) saved yet, recreate the key entry...
    if cellNum-1 not in fitList:
      fitList[cellNum-1] = dict();
    # now, if the NLL is now the best, update this
    if NLL < currNLL:
      fitList[cellNum-1]['NLL'] = NLL;
      fitList[cellNum-1]['params'] = opt_params;
      # NEW: Also save whether or not fit was success, exit message (18.12.01)
      fitList[cellNum-1]['success'] = tomin['success'];
      fitList[cellNum-1]['message'] = tomin['message'];
      # NEW: Also save *when* this most recent fit was made (19.02.04); and nll_history below
      fitList[cellNum-1]['time'] = datetime.datetime.now();
      # NEW: Also also save entire loss/optimization structure
      fitList[cellNum-1]['opt'] = tomin;
    else:
      print('new NLL not less than currNLL, not saving result, but updating ovreal fit list (i.e. tracking each fit)');
    fitList[cellNum-1]['nll_history'] = np.append(nll_history, NLL);
    numpy.save(loc_data + fitListName, fitList);
    # now the step list, if needed
    if trackSteps and NLL < currNLL:
      if os.path.exists(loc_data + stepListName):
        stepList = hf.np_smart_load(str(loc_data + stepListName));
      else:
        stepList = dict();
      stepList[cellNum-1] = dict();
      stepList[cellNum-1]['params'] = params_glob;
      stepList[cellNum-1]['loss']   = loss_glob;
      stepList[cellNum-1]['resp']   = resp_glob;
      numpy.save(loc_data + stepListName, stepList);

   # TODO: make holdOutTr for these many-holdout conditions
#    if holdOutCondition is not None:
#      holdoutNLL, _, = SFMGiveBof(opt_params, structureSFM=S, normType=fitType, lossType=lossType, #trialSubset=holdOutTr, expInd=expInd, rvcFits=rvcFits);
#    else:
#      holdoutNLL = [];
    holdoutNLL = [];

    return NLL, opt_params, holdoutNLL;

if __name__ == '__main__':

    if len(sys.argv) < 9:
      print('uhoh...you need 9 arguments here'); # and one is the script itself...
      print('See this file (setModel) or batchFitUnix.sh for guidance');
      exit();

    cellNum      = int(sys.argv[1]);
    expDir       = sys.argv[2];
    excType      = int(sys.argv[3]);
    lossType     = int(sys.argv[4]);
    fitType      = int(sys.argv[5]);
    lgnFrontOn   = int(sys.argv[6]);
    initFromCurr = int(sys.argv[7]);
    trackSteps   = int(sys.argv[8]);

    if len(sys.argv) > 9:
      kMult = float(sys.argv[9]);
    else:
      kMult = 0.10; # default (see modCompare.ipynb for details)

    if len(sys.argv) > 10:
      rvcMod = float(sys.argv[10]);
    else:
      rvcMod = 1; # default (naka-rushton)

    if len(sys.argv) > 11:
      fixRespExp = float(sys.argv[11]);
      if fixRespExp <= 0: # this is the code to not fix the respExp
        fixRespExp = None;
    else:
      fixRespExp = None; # default (see modCompare.ipynb for details)

    import time
    start = time.process_time();
    setModel(cellNum, expDir, lossType, fitType, initFromCurr, trackSteps=trackSteps, modRecov=modRecov, kMult=kMult, rvcMod=rvcMod, excType=excType, lgnFrontEnd=lgnFrontOn, fixRespExp=fixRespExp);
    enddd = time.process_time();
    print('Took %d time -- NO par!!!' % (enddd-start));
