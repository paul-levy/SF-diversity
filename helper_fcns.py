import math, numpy, random
from scipy.stats import norm, mode, poisson, nbinom
from scipy.stats.mstats import gmean as geomean
from numpy.matlib import repmat
import scipy.optimize as opt
import os
import importlib as il
from time import sleep
sqrt = math.sqrt
log = math.log
exp = math.exp
import pdb
import warnings

# Functions:

### basics

# np_smart_load - be smart about using numpy load
# bw_lin_to_log
# bw_log_to_lin
# exp_name_to_ind - given the name of an exp (e.g. sfMixLGN), return the expInd
# get_exp_params  - given an index for a particular version of the sfMix experiments, return parameters of that experiment (i.e. #stimulus components)
# get_exp_ind     - given a .npy for a given sfMix recording, determine the experiment index
# fitList_name
# phase_fit_name
# descrFit_name
# angle_xy

### fourier

# make_psth - create a psth for a given spike train
# spike_fft - compute the FFT for a given PSTH, extract the power at a given set of frequencies 

### descriptive fits to sf tuning/basic data analyses

# DiffOfGauss - standard difference of gaussians
# DoGsach - difference of gaussians as implemented in sach's thesis
# dog_prefSf - compute the prefSf for a given DoG model/parameter set
# dog_prefSfMod - fit a simple model of prefSf as f'n of contrast
# dog_charFreq - given a model/parameter set, return the characteristic frequency of the tuning curve
# dog_charFreqMod - smooth characteristic frequency vs. contrast with a functional form/fit
# var_explained - compute the variance explained for a given model fit/set of responses
# chiSq

# deriv_gauss - evaluate a derivative of a gaussian, specifying the derivative order and peak
# get_prefSF - Given a set of parameters for a flexible gaussian fit, return the preferred SF
# compute_SF_BW - returns the log bandwidth for height H given a fit with parameters and height H (e.g. half-height)
# fix_params - Intended for parameters of flexible Gaussian, makes all parameters non-negative
# flexible_Gauss - Descriptive function used to describe/fit SF tuning
# blankResp - return mean/std of blank responses (i.e. baseline firing rate) for sfMixAlt experiment
# get_valid_trials - rutrn list of valid trials given disp/con/sf
# get_valid_sfs - return list indices (into allSfs) of valid sfs for given disp/con
# tabulate_responses - Organizes measured and model responses for sfMixAlt experiment
# organize_adj_responses - wrapper for organize_adj_responses within each experiment subfolder
# organize_resp       -
# get_spikes - get correct # spikes for a given cell (will get corrected spikes if needed)
# get_rvc_fits - return the rvc fits for a given cell (if applicable)
# mod_poiss - computes "r", "p" for modulated poisson model (neg. binomial)
# naka_rushton
# fit_CRF
# random_in_range - random real-valued number between A and B
# nbinpdf_log - was used with sfMix optimization to compute the negative binomial probability (likelihood) for a predicted rate given the measured spike count

# getSuppressiveSFtuning - returns the normalization pool response
# makeStimulus - was used last for sfMix experiment to generate arbitrary stimuli for use with evaluating model
# getNormParams  - given the model params and fit type, return the relevant parameters for normalization
# genNormWeights - used to generate the weighting matrix for weighting normalization pool responses
# setSigmaFilter - create the filter we use for determining c50 with SF
# evalSigmaFilter - evaluate an arbitrary filter at a set of spatial frequencies to determine c50 (semisaturation contrast)
# setNormTypeArr - create the normTypeArr used in SFMGiveBof/Simulate to determine the type of normalization and corresponding parameters
# getConstraints - return the constraints used in model optimization

def np_smart_load(file_path, encoding_str='latin1'):

   if not os.path.isfile(file_path):
     return [];
   loaded = [];
   while(True):
     try:
         loaded = numpy.load(file_path, encoding=encoding_str).item();
         break;
     except IOError: # this happens, I believe, because of parallelization when running on the cluster; cannot properly open file, so let's wait and then try again
         sleep(10); # i.e. wait for 10 seconds
     except EOFError: # this happens, I believe, because of parallelization when running on the cluster; cannot properly open file, so let's wait and then try again
       sleep(10); # i.e. wait for 10 seconds

   return loaded;

def bw_lin_to_log( lin_low, lin_high ):
    # Given the low/high sf in cpd, returns number of octaves separating the
    # two values

    return numpy.log2(lin_high/lin_low);

def bw_log_to_lin(log_bw, pref_sf):
    # given the preferred SF and octave bandwidth, returns the corresponding
    # (linear) bounds in cpd

    less_half = numpy.power(2, numpy.log2(pref_sf) - log_bw/2);
    more_half = numpy.power(2, log_bw/2 + numpy.log2(pref_sf));

    sf_range = [less_half, more_half];
    lin_bw = more_half - less_half;
    
    return lin_bw, sf_range

def exp_name_to_ind(expName):
    ''' static mapping from name of experiment to expInd
    '''
    if expName == 'sfMix':
      expInd = 1;
    elif expName == 'sfMixAlt':
      expInd = 2;
    elif expName == 'sfMixLGN':
      expInd = 3;
    elif expName == 'sfMixInt':
      expInd = 4;
    elif expName == 'sfMixHalfInt':
      expInd = 5;
    return expInd;

def get_exp_params(expInd, forceDir=None):
    '''  returns the following
                (max) nComponents in each stimulus
                # of stimulus families (i.e. how many dispersion levels)
                # of contrasts (in the sfMix core, i.e. not how many for single gratings)
                list of how many components in each level
                directory (relative to /common)
    '''
    class exp_params:
      
      def __init__(self, expInd):
        if expInd == 1: # original V1 experiment - m657, m658, m660
          self.nStimComp = 9;
          self.nFamilies = 5; 
          self.comps     = [1, 3, 5, 7, 9];
          self.nCons     = 2;
          self.nSfs      = 11;
          self.nCells    = 59;
          self.dir       = 'V1_orig/'
          self.stimDur   = 1; # in seconds
          self.fps       = 120; # frame rate (in Hz, i.e. frames per second)
        elif expInd == 2: # V1 alt exp - m670, m671
          self.nStimComp = 7;
          self.nFamilies = 4;
          self.comps     = [1, 3, 5, 7]
          self.nCons     = 4;
          self.nSfs      = 11;
          self.nCells    = 8; # check...
          self.dir       = 'altExp/'
          self.stimDur   = 1; # in seconds
          self.fps       = 120; # frame rate (in Hz, i.e. frames per second)
        elif expInd == 3: # (original) LGN experiment - m675; two recordings from V1 exp. m676 (in V1/)
          self.nStimComp = 5;
          self.nFamilies = 2;
          self.comps     = [1, 5];
          self.nCons     = 4;
          self.nSfs      = 11;
          self.nCells    = 34;
          self.dir       = 'LGN/'
          self.stimDur   = 1; # in seconds
          self.fps       = 120; # frame rate (in Hz, i.e. frames per second)
        elif expInd == 4: # V1 "Int" - same as expInd = 2, but with integer TFs (keeping separate to track # cells)
          self.nStimComp = 7;
          self.nFamilies = 4;
          self.comps     = [1, 3, 5, 7]
          self.nCons     = 4;
          self.nSfs      = 11;
          self.nCells    = 1;
          self.dir       = 'V1/'
          self.stimDur   = 1; # in seconds
          self.fps       = 120; # frame rate (in Hz, i.e. frames per second)
        elif expInd == 5: # V1 "halfInt" - same as expInd = 4, but with stimDur = 2
          self.nStimComp = 7;
          self.nFamilies = 4;
          self.comps     = [1, 3, 5, 7]
          self.nCons     = 4;
          self.nSfs      = 11;
          self.nCells    = 4;
          self.dir       = 'V1/'
          self.stimDur   = 2; # in seconds
          self.fps       = 120; # frame rate (in Hz, i.e. frames per second)

        if forceDir is not None:
          self.dir       = forceDir;

    return exp_params(expInd);

def get_exp_ind(filePath, fileName):
    '''  returns the following:
           index of experiment (see get_exp_params)
           name of experiment (e.g. sfMix, sfMixHalfInt)

         this function relies on the fact that all .npy files (in /structures) have an associated matlab file
         in /recordings with the full experiment name
           EXCEPT: V1_orig files...
    '''
    if 'V1_orig' in filePath:
      return 'sfMix'; # need to handle this case specially, since we don't have /recordings/ for this experiment

    name_root = fileName.split('_')[0];
    orig_files = os.listdir(filePath + '../recordings/');
    for f in orig_files:
      if f.startswith(name_root) and '.xml' in f and 'sfMix' in f: # make sure it's not a *_sfm.mat that is in .../recordings/, and that .xml and "sfMix" are in the file
        # we rely on the following: fileName#run[expType].[exxd/xml]
        expName = f.split('[')[1].split(']')[0];
        return exp_name_to_ind(expName), expName;

    return None, None; # uhoh...

def num_frames(expInd):
  exper = get_exp_params(expInd);
  dur = exper.stimDur;
  fps = exper.fps;
  nFrames    = dur*fps;
  return nFrames;

def fitList_name(base, fitType, lossType):
  ''' use this to get the proper name for the full model fits
  '''
  # first the fit type
  if fitType == 1:
    fitSuf = '_flat';
  elif fitType == 2:
    fitSuf = '_wght';
  elif fitType == 3:
    fitSuf = '_c50';
  # then the loss type
  if lossType == 1:
    lossSuf = '_sqrt.npy';
  elif lossType == 2:
    lossSuf = '_poiss.npy';
  elif lossType == 3:
    lossSuf = '_modPoiss.npy';
  return str(base + fitSuf + lossSuf);

def phase_fit_name(base, dir, byTrial=0):
  ''' Given the base name for a file, append the flag for the phase direction (dir)
      This is used for phase advance fits
  '''
  if byTrial == 1:
    byTr = '_byTr'
  elif byTrial == 0:
    byTr = '';

  if dir == 1:
    base = base + byTr + '_pos.npy';
  if dir == -1:
    base = base + byTr + '_neg.npy';
  return base;

def descrFit_name(lossType, modelName = None):
  ''' if modelName is none, then we assume we're fitting descriptive tuning curves to the data
      otherwise, pass in the fitlist name in that argument, and we fit descriptive curves to the model
      this simply returns the name
  '''
  # load descrFits
  if lossType == 1:
    floss_str = '_lsq';
  elif lossType == 2:
    floss_str = '_sqrt';
  elif lossType == 3:
    floss_str = '_poiss';
  elif lossType == 4:
    floss_str = '_sach';
  descrFitBase = 'descrFits%s' % floss_str;

  if modelName is None:
    descrName = '%s.npy' % descrFitBase;
  else:
    descrName = '%s_%s.npy' % (descrFitBase, modelName);
    
  return descrName;

def angle_xy(x_coord, y_coord):
   ''' return list of angles (in deg) given list of x/y coordinates (i.e. polar coordinates)
   ''' 
   np = numpy;
   def smart_angle(x, y, th): 
     if x>=0 and y>=0: # i.e. quadrant 1
       return th;
     if x<=0 and y>=0: # i.e. quadrant 2
       return 180 - th;
     if x<=0 and y<=0: # i.e. quadrant 3
       return 180 + th;
     if x>=0 and y<=0:
       return 360 - th;

   th = [np.rad2deg(np.arctan(np.abs(y_coord[i]/x_coord[i]))) for i in range(len(x_coord))];
   theta = [smart_angle(x_coord[i], y_coord[i], th[i]) for i in range(len(th))];
   return theta;

### fourier

def make_psth(spikeTimes, binWidth=1e-3, stimDur=1):
    # given an array of arrays of spike times, create the PSTH for a given bin width and stimulus duration
    # i.e. spikeTimes has N arrays, each of which is an array of spike times

    binEdges = numpy.linspace(0, stimDur, 1+stimDur/binWidth);
    
    all = [numpy.histogram(x, bins=binEdges) for x in spikeTimes]; 
    psth = [x[0] for x in all];
    bins = [x[1] for x in all];
    return psth, bins;

def spike_fft(psth, tfs = None):
    ''' given a psth (and optional list of component TFs), compute the fourier transform of the PSTH
        if the component TFs are given, return the FT power at the DC, and at all component TFs
        note: if only one TF is given, also return the power at f2 (i.e. twice f1, the stimulus frequency)
    '''
    np = numpy;

    full_fourier = [np.fft.fft(x) for x in psth];
    spectrum = [np.abs(np.fft.fft(x)) for x in psth];

    if tfs:
      rel_power = [spectrum[i][tfs[i]] for i in range(len(tfs))];
    else:
      rel_power = [];

    return spectrum, rel_power, full_fourier;

### descriptive fits to sf tuning/basic data analyses

def DiffOfGauss(gain, f_c, gain_s, j_s, stim_sf):
  ''' Difference of gaussians 
  gain      - overall gain term
  f_c       - characteristic frequency of the center, i.e. freq at which response is 1/e of maximum
  gain_s    - relative gain of surround (e.g. gain_s of 0.5 says peak surround response is half of peak center response
  j_s       - relative characteristic freq. of surround (i.e. char_surround = f_c * j_s)
  '''
  np = numpy;
  dog = lambda f: np.maximum(0, gain*(np.exp(-np.square(f/f_c)) - gain_s * np.exp(-np.square(f/(f_c*j_s)))));

  norm = np.max(dog(stim_sf));

  dog_norm = lambda f: dog(f) / norm;

  return dog(stim_sf), dog_norm(stim_sf);

def DoGsach(gain_c, r_c, gain_s, r_s, stim_sf):
  ''' Difference of gaussians as described in Sach's thesis
  gain_c    - gain of the center mechanism
  r_c       - radius of the center
  gain_s    - gain of surround mechanism
  r_s       - radius of surround
  '''
  np = numpy;
  dog = lambda f: np.maximum(0, gain_c*np.pi*np.square(r_c)*np.exp(-np.square(f*np.pi*r_c)) - gain_s*np.pi*np.square(r_s)*np.exp(-np.square(f*np.pi*r_s)));

  norm = np.max(dog(stim_sf));
  dog_norm = lambda f: dog(f) / norm;

  return dog(stim_sf), dog_norm(stim_sf);

def dog_prefSf(modParams, dog_model=2, all_sfs=numpy.logspace(-1, 1, 11)):
  ''' Compute the preferred SF given a set of DoG parameters
  '''
  sf_bound = (numpy.min(all_sfs), numpy.max(all_sfs));
  if dog_model == 1:
    obj = lambda sf: -DoGsach(*modParams, stim_sf=sf)[0];
  elif dog_model == 2:
    obj = lambda sf: -DiffOfGauss(*modParams, stim_sf=sf)[0];
  init_sf = numpy.median(all_sfs);
  optz = opt.minimize(obj, init_sf, bounds=(sf_bound, ))
  return optz['x'];

def dog_prefSfMod(descrFit, allCons, disp=0, varThresh=65, dog_model=2):
  ''' Given a descrFit dict for a cell, compute a fit for the prefSf as a function of contrast
      Return ratio of prefSf at highest:lowest contrast, lambda of model, params
  '''
  np = numpy;
  # the model
  psf_model = lambda offset, slope, alpha, con: offset + slope*np.power(con-con[0], alpha);
  # gather the values
  #   only include prefSf values derived from a descrFit whose variance explained is gt the thresh
  validInds = np.where(descrFit['varExpl'][disp, :] > varThresh)[0];
  if len(validInds) == 0: # i.e. no good fits...
    return np.nan, [], [];
  if 'prefSf' in descrFit:
    prefSfs = descrFit['prefSf'][disp, validInds];
  else:
    prefSfs = [];
    for i in validInds:
      psf_curr = dog_prefSf(descrFit['params'][disp, validInds], dog_model);
      prefSfs.append(psf_curr);
  conVals = allCons[validInds];
  weights = descrFit['varExpl'][disp, validInds];
  # set up the optimization
  obj = lambda params: np.sum(np.multiply(weights,
        np.square(psf_model(params[0], params[1], params[2], conVals) - prefSfs)))
  init_offset = prefSfs[0];
  conRange = conVals[-1] - conVals[0];
  init_slope = (prefSfs[-1] - prefSfs[0]) / conRange;
  init_alpha = 0.4; # most tend to be saturation (i.e. contrast exp < 1)
  # run
  optz = opt.minimize(obj, [init_offset, init_slope, init_alpha], bounds=((0, None), (None, None), (0.25, 4)));
  opt_params = optz['x'];
  # ratio:
  extrema = psf_model(*opt_params, con=(conVals[0], conVals[-1]))
  pSfRatio = extrema[-1] / extrema[0]

  return pSfRatio, psf_model, opt_params;

def dog_charFreq(prms, DoGmodel=1):
  if DoGmodel == 1: # sach
      r_c = prms[1];
      f_c = 1/(numpy.pi*r_c)
  elif DoGmodel == 2: # tony
      f_c = prms[1];

  return f_c;

def dog_charFreqMod(descrFit, allCons, varThresh=70, DoGmodel=1, lowConCut = 0.1, disp=0):
  ''' Given a descrFit dict for a cell, compute a fit for the charFreq as a function of contrast
      Return ratio of charFreqat highest:lowest contrast, lambda of model, params, the value of the charFreq at the valid contrasts, the corresponding valid contrast
      Note: valid contrast means a contrast which is greater than the lowConCut and one for which the Sf tuning fit has a variance explained gerat than varThresh
  '''
  np = numpy;
  # the model
  fc_model = lambda offset, slope, alpha, con: offset + slope*np.power(con-con[0], alpha);
  # gather the values
  #   only include prefSf values derived from a descrFit whose variance explained is gt the thresh
  if disp == 0:
    inds = np.asarray([0, 1, 2, 3, 4, 5, 7, 9, 11]);
  elif disp == 1:
    inds = np.asarray([6, 8, 10]);
  validInds = np.where((descrFit['varExpl'][disp, inds] > varThresh) & (allCons > lowConCut))[0];
  conVals = allCons[validInds];

  if len(validInds) == 0: # i.e. no good fits...
    return np.nan, None, None, None, None;
  if 'charFreq' in descrFit:
    charFreqs = descrFit['charFreq'][disp, inds[validInds]];
  else:
    charFreqs = [];
    for i in validInds:
      cf_curr = dog_charFreq(descrFit['params'][disp, i], DoGmodel);
      charFreqs.append(cf_curr);
  weights = descrFit['varExpl'][disp, inds[validInds]];
  # set up the optimization
  obj = lambda params: np.sum(np.multiply(weights,
        np.square(fc_model(params[0], params[1], params[2], conVals) - charFreqs)))
  init_offset = charFreqs[0];
  conRange = conVals[-1] - conVals[0];
  init_slope = (charFreqs[-1] - charFreqs[0]) / conRange;
  init_alpha = 0.4; # most tend to be saturation (i.e. contrast exp < 1)
  # run
  optz = opt.minimize(obj, [init_offset, init_slope, init_alpha], bounds=((0, None), (None, None), (0.25, 4)));
  opt_params = optz['x'];
  # ratio:
  extrema = fc_model(*opt_params, con=(conVals[0], conVals[-1]))
  fcRatio = extrema[-1] / extrema[0]

  return fcRatio, fc_model, opt_params, charFreqs, conVals;


def var_explained(data_resps, modParams, sfVals, dog_model = 2):
  ''' given a set of responses and model parameters, compute the variance explained by the model 
  '''
  np = numpy;
  resp_dist = lambda x, y: np.sum(np.square(x-y))/np.maximum(len(x), len(y))
  var_expl = lambda m, r, rr: 100 * (1 - resp_dist(m, r)/resp_dist(r, rr));

  # organize data responses (adjusted)
  data_mean = np.mean(data_resps) * np.ones_like(data_resps);

  # compute model responses
  if dog_model == 1:
    mod_resps = DoGsach(*modParams, stim_sf=sfVals)[0];
  if dog_model == 2:
    mod_resps = DiffOfGauss(*modParams, stim_sf=sfVals)[0];

  return var_expl(mod_resps, data_resps, data_mean);

def chiSq(data_resps, model_resps, stimDur=1):
  ''' given a set of measured and model responses, compute the chi-squared (see Cavanaugh et al '02a)
      assumes: resps are mean/variance for each stimulus condition (e.g. like a tuning curve)
        with each condition a tuple (or 2-array) with [mean, var]
  '''
  np = numpy;
  rats = np.divide(data_resps[1], data_resps[0]);
  nan_rm = lambda x: x[~np.isnan(x)]
  neg_rm = lambda x: x[x>0]; # particularly for adjusted responses, a few values might be negative; remove these from the rho calculation
  rho = geomean(neg_rm(nan_rm(rats))); # only need neg_rm, but just being explicit
  k   = 0.10 * rho * np.nanmax(data_resps[0]) # default kMult from Cavanaugh is 0.01

  # some conditions might be blank (and therefore NaN) - remove them!
  num = data_resps[0] - model_resps[0];
  valid = ~np.isnan(num);
  data_resp_recenter = data_resps[0][valid] - np.min(data_resps[0][valid]);
  # the numerator is (.)^2, and therefore always >=0; the denominator is now "recentered" so that the values are >=0
  # thus, chi will always be >=0, avoiding a fit which maximizes the numerator to reduce the loss (denom was not strictly >0)   
  chi = np.sum(np.divide(np.square(num[valid]), k + data_resp_recenter*rho/stimDur));

  return chi;

##

def deriv_gauss(params, stimSf = numpy.logspace(numpy.log10(0.1), numpy.log10(10), 101)):

    prefSf = params[0];
    dOrdSp = params[1];

    sfRel = stimSf / prefSf;
    s     = pow(stimSf, dOrdSp) * numpy.exp(-dOrdSp/2 * pow(sfRel, 2));
    sMax  = pow(prefSf, dOrdSp) * numpy.exp(-dOrdSp/2);
    sNl   = s/sMax;
    selSf = sNl;

    return selSf, stimSf;

def get_prefSF(flexGauss_fit):
   ''' Given a set of parameters for a flexible gaussian fit, return the preferred SF
   '''
   return flexGauss_fit[2];

def compute_SF_BW(fit, height, sf_range):

    # 1/16/17 - This was corrected in the lead up to SfN (sometime 11/16). I had been computing
    # octaves not in log2 but rather in log10 - it is field convention to use
    # log2!

    # Height is defined RELATIVE to baseline
    # i.e. baseline = 10, peak = 50, then half height is NOT 25 but 30
    
    bw_log = numpy.nan;
    SF = numpy.empty((2, 1));
    SF[:] = numpy.nan;

    # left-half
    left_full_bw = 2 * (fit[3] * sqrt(2*log(1/height)));
    left_cpd = fit[2] * exp(-(fit[3] * sqrt(2*log(1/height))));

    # right-half
    right_full_bw = 2 * (fit[4] * sqrt(2*log(1/height)));
    right_cpd = fit[2] * math.exp((fit[4] * sqrt(2*math.log(1/height))));

    if left_cpd > sf_range[0] and right_cpd < sf_range[-1]:
        SF = [left_cpd, right_cpd];
        bw_log = log(right_cpd / left_cpd, 2);

    # otherwise we don't have defined BW!
    
    return SF, bw_log

def fix_params(params_in):

    # simply makes all input arguments positive
 
    # R(Sf) = R0 + K_e * EXP(-(SF-mu)^2 / 2*(sig_e)^2) - K_i * EXP(-(SF-mu)^2 / 2*(sig_i)^2)

    return [abs(x) for x in params_in] 

def flexible_Gauss(params, stim_sf, minThresh=0.1):
    # The descriptive model used to fit cell tuning curves - in this way, we
    # can read off preferred SF, octave bandwidth, and response amplitude

    respFloor       = params[0];
    respRelFloor    = params[1];
    sfPref          = params[2];
    sigmaLow        = params[3];
    sigmaHigh       = params[4];

    # Tuning function
    sf0   = [x/sfPref for x in stim_sf];

    sigma = numpy.multiply(sigmaLow, [1]*len(sf0));

    sigma[[x for x in range(len(sf0)) if sf0[x] > 1]] = sigmaHigh;

    # hashtag:uglyPython
    shape = [math.exp(-pow(math.log(x), 2) / (2*pow(y, 2))) for x, y in zip(sf0, sigma)];
                
    return [max(minThresh, respFloor + respRelFloor*x) for x in shape];

def blankResp(cellStruct):
    # as of 01.03.19, should check if this works for all experiment variants
    if 'sfm' in cellStruct:
      tr = cellStruct['sfm']['exp']['trial'];
    else:
      tr = cellStruct;
    blank_tr = tr['spikeCount'][numpy.isnan(tr['con'][0])];
    mu = numpy.mean(blank_tr);
    sig = numpy.std(blank_tr);
    
    return mu, sig, blank_tr;
    
def get_valid_trials(data, disp, con, sf, expInd):
  ''' Given a data and the disp/con/sf indices (i.e. integers into the list of all disps/cons/sfs
      Determine which trials are valid (i.e. have those stimulus criteria)
      RETURN list of valid trials, lists for all dispersion values, all contrast values, all sf values
  '''
  _, stimVals, _, validByStimVal, _ = tabulate_responses(data, expInd);

  # gather the conditions we need so that we can index properly
  valDisp = validByStimVal[0];
  valCon = validByStimVal[1];
  valSf = validByStimVal[2];

  allDisps = stimVals[0];
  allCons = stimVals[1];
  allSfs = stimVals[2];

  val_trials = numpy.where(valDisp[disp] & valCon[con] & valSf[sf]);

  return val_trials, allDisps, allCons, allSfs;

def get_valid_sfs(data, disp, con, expInd):
  ''' Self explanatory, innit? Returns the indices (into allSfs) of valid sfs for the given condition
  '''
  _, stimVals, _, validByStimVal, _ = tabulate_responses(data, expInd);

  # gather the conditions we need so that we can index properly
  valDisp = validByStimVal[0];
  valCon = validByStimVal[1];
  valSf = validByStimVal[2];

  allDisps = stimVals[0];
  allCons = stimVals[1];
  allSfs = stimVals[2];

  val_sfs = [];
  for i in range(len(allSfs)):
    val_trials = numpy.where(valDisp[disp] & valCon[con] & valSf[i]);
    if len(val_trials[0]) > 0:
      val_sfs.append(i);

  return val_sfs;

def tabulate_responses(cellStruct, expInd, modResp = []):
    ''' Given cell structure (and opt model responses), returns the following:
        (i) respMean, respStd, predMean, predStd, organized by condition; pred is linear prediction
        (ii) all_disps, all_cons, all_sfs - i.e. the stimulus conditions of the experiment
        (iii) the valid contrasts for each dispersion level
        (iv) valid_disp, valid_con, valid_sf - which conditions are valid for this particular cell
        (v) modRespOrg - the model responses organized as in (i) - only if modResp argument passed in
        NOTE: We pass in the overall spike counts (modResp; either real or predicted), and compute 
          the spike *rates* (i.e. spikes/s)
    '''
    np = numpy;
    conDig = 3; # round contrast to the thousandth
    exper = get_exp_params(expInd);
    stimDur = exper.stimDur;
    
    if 'sfm' in cellStruct: 
      data = cellStruct['sfm']['exp']['trial'];
    else: # we've passed in sfm.exp.trial already
      data = cellStruct;

    if expInd == 1: # this exp structure only has 'con', 'sf'; perform some simple ops to get everything as in other exp structures
      v1_dir = exper.dir.replace('/', '.');
      v1_hf = il.import_module(v1_dir+'helper_fcns');
      data['total_con'] = np.sum(data['con'], -1);
      data['cent_sf']   = data['sf'][0]; # first component, i.e. center SF, is at position 0
      data['num_comps'] = v1_hf.get_num_comps(data['con'][0]);
      # next, let's prune out the trials from the ori and RVC measurements
      oriBlockIDs = np.hstack((np.arange(131, 155+1, 2), np.arange(132, 136+1, 2))); # +1 to include endpoint like Matlab
      conBlockIDs = np.arange(138, 156+1, 2);
      invalidIDs  = np.hstack((oriBlockIDs, conBlockIDs));
      blockIDs = data['blockID'];
      inval    = np.in1d(blockIDs, invalidIDs)
    else:
      inval    = np.in1d(np.zeros_like(data['total_con']), 1); # i.e. all are valid, since there's only core sfMix trials

    all_cons = np.unique(np.round(data['total_con'][~inval], conDig));
    all_cons = all_cons[~np.isnan(all_cons)];

    all_sfs = np.unique(data['cent_sf'][~inval]);
    all_sfs = all_sfs[~np.isnan(all_sfs)];

    all_disps = np.unique(data['num_comps'][~inval]);
    all_disps = all_disps[all_disps>0]; # ignore zero...

    nCons = len(all_cons);
    nSfs = len(all_sfs);
    nDisps = len(all_disps);
    
    respMean = np.nan * np.empty((nDisps, nSfs, nCons));
    respStd = np.nan * np.empty((nDisps, nSfs, nCons));
    predMean = np.nan * np.empty((nDisps, nSfs, nCons));
    predStd = np.nan * np.empty((nDisps, nSfs, nCons));

    if len(modResp) == 0: # as in, if it isempty
        modRespOrg = [];
        mod = 0;
    else:
        nRepsMax = 20; # assume we'll never have more than 20 reps for any given condition...
        modRespOrg = np.nan * np.empty((nDisps, nSfs, nCons, nRepsMax));
        mod = 1;
        
    val_con_by_disp = [];
    valid_disp = dict();
    valid_con = dict();
    valid_sf = dict();

    for d in range(nDisps):
        val_con_by_disp.append([]);

        valid_disp[d] = data['num_comps'] == all_disps[d];
        for con in range(nCons):

            valid_con[con] = np.round(data['total_con'], conDig) == all_cons[con];

            for sf in range(nSfs):

                valid_sf[sf] = data['cent_sf'] == all_sfs[sf];

                valid_tr = valid_disp[d] & valid_sf[sf] & valid_con[con];
                ## TODO: fix an issue here with floats being treated as integers...(run plot_compare.py for cell from expInd=1)
                if expInd == 1: # also take care of excluding ori/rvc trials
                  valid_tr = valid_tr & ~inval;
                if np.all(np.unique(valid_tr) == False):
                    continue;

                respMean[d, sf, con] = np.mean(data['spikeCount'][valid_tr]/stimDur);
                respStd[d, sf, con] = np.std((data['spikeCount'][valid_tr]/stimDur));
                
                curr_pred = 0;
                curr_var = 0; # variance (std^2) adds
                for n_comp in range(all_disps[d]):
                    # find information for each component, find corresponding trials, get response, sum
                        # Note: unique(.) will only be one value, since all equiv stim have same equiv componentss 
                    curr_con = np.unique(data['con'][n_comp][valid_tr]);
                    val_con = np.round(data['total_con'], conDig) == np.round(curr_con, conDig);
                    curr_sf = np.unique(data['sf'][n_comp][valid_tr]);
                    val_sf = np.round(data['cent_sf'], conDig) == np.round(curr_sf, conDig);
                    
                    val_tr = val_con & val_sf & valid_disp[0] # why valid_disp[0]? we want single grating presentations!

                    if np.all(np.unique(val_tr) == False):
                        #print('empty...');
                        continue;
                    
                    curr_pred = curr_pred + np.mean(data['spikeCount'][val_tr]/stimDur);
                    curr_var = curr_var + np.var(data['spikeCount'][val_tr]/stimDur);
                    
                predMean[d, sf, con] = curr_pred;
                predStd[d, sf, con] = np.sqrt(curr_var);
                
                if mod: # convert spike counts in each trial to spike rate (spks/s)
                    nTrCurr = sum(valid_tr); # how many trials are we getting?
                    modRespOrg[d, sf, con, 0:nTrCurr] = modResp[valid_tr]/stimDur;

            if np.any(~np.isnan(respMean[d, :, con])):
                if ~np.isnan(np.nanmean(respMean[d, :, con])):
                    val_con_by_disp[d].append(con);
                    
    return [respMean, respStd, predMean, predStd], [all_disps, all_cons, all_sfs], val_con_by_disp, [valid_disp, valid_con, valid_sf], modRespOrg;

def organize_adj_responses(data, rvcFits, expInd):
  ''' Used as a wrapper to call the organize_adj_responses function for a given experiment
      We set the organize_adj_responses separately for each experiment since some versions don't have adjusted responses
        and for those that do, some aspects of the calculation may differ
  '''
  dir = get_exp_params(expInd).dir;
  to_import = dir.replace('/', '.') + 'helper_fcns';
  new_hf = il.import_module(to_import);
  if hasattr(new_hf, 'organize_adj_responses'):
    adjResps = new_hf.organize_adj_responses(data, rvcFits)[1]; # 2nd returned argument (pos 1) is responses by trial
  else:
    warnings.warn('this experiment (as given by ind) does not have an organize_adj_responses call!');
    adjResps = None;
  return adjResps;

def organize_resp(spikes, expStructure, expInd):
    ''' organizes the responses by condition given spikes, experiment structure, and expInd
    '''
    # the blockIDs are fixed...
    exper = get_exp_params(expInd);
    nFam = exper.nFamilies;
    nCon = exper.nCons;
    nCond = exper.nSfs;
    nReps = 20; # never more than 20 reps per stim. condition
    
    if expInd == 1: # only the original V1 exp (expInd=1) has separate ori and RVC measurements
      # Analyze the stimulus-driven responses for the orientation tuning curve
      oriBlockIDs = numpy.hstack((numpy.arange(131, 155+1, 2), numpy.arange(132, 136+1, 2))); # +1 to include endpoint like Matlab
      if 'sfm' in expStructure:
        data = expStructure['sfm']['exp']['trial'];
      else:
        data = expStructure;


      rateOr = numpy.empty((0,));
      for iB in oriBlockIDs:
          indCond = numpy.where(data['blockID'] == iB);
          if len(indCond[0]) > 0:
              rateOr = numpy.append(rateOr, numpy.mean(spikes[indCond]));
          else:
              rateOr = numpy.append(rateOr, numpy.nan);

      # Analyze the stimulus-driven responses for the contrast response function
      conBlockIDs = numpy.arange(138, 156+1, 2);
      iC = 0;

      rateCo = numpy.empty((0,));
      for iB in conBlockIDs:
          indCond = numpy.where(data['blockID'] == iB);   
          if len(indCond[0]) > 0:
              rateCo = numpy.append(rateCo, numpy.mean(spikes[indCond]));
          else:
              rateCo = numpy.append(rateCo, numpy.nan);
    else:
      rateOr = None;
      rateCo = None;

    # Analyze the stimulus-driven responses for the spatial frequency mixtures
    if expInd == 1: # have to do separate work, since the original V1 experiment has tricker stimuli (i.e. not just sfMix, but also ori and rvc measurements done separately)
      v1_dir = exper.dir.replace('/', '.');
      v1_hf = il.import_module(v1_dir + 'helper_fcns');
      _, _, rateSfMix, allSfMix = v1_hf.organize_modResp(spikes, data);
    else:
      allSfMix  = tabulate_responses(expStructure, expInd, spikes)[4];
      rateSfMix = numpy.nanmean(allSfMix, -1);

    return rateOr, rateCo, rateSfMix, allSfMix;  

def get_spikes(data, rvcFits = None, expInd = None):
  ''' Given the data (S.sfm.exp.trial), if rvcFits is None, simply return saved spike count;
                                        else return the adjusted spike counts (e.g. LGN, expInd 3)
  '''
  if rvcFits is None:
    spikes = data['spikeCount'];
  else:
    if expInd is None:
      warnings.warn('Should pass in expInd; defaulting to 3');
      expInd = 3; # should be specified, but just in case
    spikes = organize_adj_responses(data, rvcFits, expInd);
  return spikes;

def get_rvc_fits(loc_data, expInd, cellNum, rvcName='rvcFits', direc=1):
  ''' simple function to return the rvc fits needed for adjusting responses
  '''
  if expInd == 3: # for now, only the LGN experiment has the response adjustment
    rvcFits = np_smart_load(str(loc_data + phase_fit_name(rvcName, direc)));
    try:
      rvcFits = rvcFits[cellNum-1];
    except: # if the RVC fits haven't been done...
      warnings.warn('This experiment type (expInd=3) usually has associated RVC fits for resposne adjustment');
      rvcFits = None;
  else:
    rvcFits = None;

  return rvcFits;

def mod_poiss(mu, varGain):
    np = numpy;
    var = mu + (varGain * np.power(mu, 2));                        # The corresponding variance of the spike count
    r   = np.power(mu, 2) / (var - mu);                           # The parameters r and p of the negative binomial distribution
    p   = r/(r + mu)

    return r, p

def naka_rushton(con, params):
    np = numpy;
    base = params[0];
    gain = params[1];
    expon = params[2];
    c50 = params[3];

    return base + gain*np.divide(np.power(con, expon), np.power(con, expon) + np.power(c50, expon));

def fit_CRF(cons, resps, nr_c50, nr_expn, nr_gain, nr_base, v_varGain, fit_type):
	# fit_type (i.e. which loss function):
		# 1 - least squares
		# 2 - square root
		# 3 - poisson
		# 4 - modulated poisson
    np = numpy;

    n_sfs = len(resps);

    # Evaluate the model
    loss_by_sf = np.zeros((n_sfs, 1));
    for sf in range(n_sfs):
        all_params = (nr_c50, nr_expn, nr_gain, nr_base);
        param_ind = [0 if len(i) == 1 else sf for i in all_params];

        nr_args = [nr_base[param_ind[3]], nr_gain[param_ind[2]], nr_expn[param_ind[1]], nr_c50[param_ind[0]]]; 
	# evaluate the model
        pred = naka_rushton(cons[sf], nr_args); # ensure we don't have pred (lambda) = 0 --> log will "blow up"
        
        if fit_type == 4:
	    # Get predicted spike count distributions
          mu  = pred; # The predicted mean spike count; respModel[iR]
          var = mu + (v_varGain * np.power(mu, 2));                        # The corresponding variance of the spike count
          r   = np.power(mu, 2) / (var - mu);                           # The parameters r and p of the negative binomial distribution
          p   = r/(r + mu);
	# no elif/else

        if fit_type == 1 or fit_type == 2:
          # error calculation
          if fit_type == 1:
            loss = lambda resp, pred: np.sum(np.power(resp-pred, 2)); # least-squares, for now...
          if fit_type == 2:
            loss = lambda resp, pred: np.sum(np.square(np.sqrt(resp) - np.sqrt(pred)));

          curr_loss = loss(resps[sf], pred);
          loss_by_sf[sf] = np.sum(curr_loss);

        else:
          # if likelihood calculation
          if fit_type == 3:
            loss = lambda resp, pred: poisson.logpmf(resp, pred);
            curr_loss = loss(resps[sf], pred); # already log
          if fit_type == 4:
            loss = lambda resp, r, p: np.log(nbinom.pmf(resp, r, p)); # Likelihood for each pass under doubly stochastic model
            curr_loss = loss(resps[sf], r, p); # already log
          loss_by_sf[sf] = -np.sum(curr_loss); # negate if LLH

    return np.sum(loss_by_sf);

def random_in_range(lims, size = 1):

    return [random.uniform(lims[0], lims[1]) for i in range(size)]

def nbinpdf_log(x, r, p):
    from scipy.special import loggamma as lgamma

    # We assume that r & p are tf placeholders/variables; x is a constant
    # Negative binomial is:
        # gamma(x+r) * (1-p)^x * p^r / (gamma(x+1) * gamma(r))
    
    # Here we return the log negBinomial:
    noGamma = x * numpy.log(1-p) + (r * numpy.log(p));
    withGamma = lgamma(x + r) - lgamma(x + 1) - lgamma(r);
    
    return numpy.real(noGamma + withGamma);

# 

def getSuppressiveSFtuning(sfs = numpy.logspace(-2, 2, 1000)): 
    # written when still new to python. Probably to matlab-y...
    # Not updated for sfMixAlt - 1/31/18
    # normPool details are fixed, ya?
    # plot model details - exc/suppressive components
    omega = sfs;

    # Compute suppressive SF tuning
    # The exponents of the filters used to approximately tile the spatial frequency domain
    n = numpy.array([.75, 1.5]);
    # The number of cells in the broad/narrow pool
    nUnits = numpy.array([12, 15]);
    # The gain of the linear filters in the broad/narrow pool
    gain = numpy.array([.57, .614]);

    normPool = {'n': n, 'nUnits': nUnits, 'gain': gain};
    # Get filter properties in spatial frequency domain
    gain = numpy.empty((len(normPool.get('n'))));
    for iB in range(len(normPool.get('n'))):
        prefSf_new = numpy.logspace(numpy.log10(.1), numpy.log10(30), normPool.get('nUnits')[iB]);
        if iB == 0:
            prefSf = prefSf_new;
        else:
            prefSf = [prefSf, prefSf_new];
        gain[iB]   = normPool.get('gain')[iB];

    for iB in range(len(normPool.get('n'))):
        sfRel = numpy.matlib.repmat(omega, len(prefSf[iB]), 1).transpose() / prefSf[iB]
        s     = numpy.power(numpy.matlib.repmat(omega, len(prefSf[iB]), 1).transpose(), normPool['n'][iB]) \
                    * numpy.exp(-normPool['n'][iB]/2 * numpy.square(sfRel));
        sMax  = numpy.power(prefSf[iB], normPool['n'][iB]) * numpy.exp(-normPool['n'][iB]/2);
        if iB == 0:
            selSf = gain[iB] * s / sMax;
        else:
            selSf = [selSf, gain[iB] * s/sMax];

    return numpy.hstack((selSf[0], selSf[1]));

def makeStimulus(stimFamily, conLevel, sf_c, template, expInd=1):

# returns [Or, Tf, Co, Ph, Sf, trial_used]

# 1/23/16 - This function is used to make arbitrary stimuli for use with
# the Robbe V1 model. Rather than fitting the model responses at the actual
# experimental stimuli, we instead can simulate from the model at any
# arbitrary spatial frequency in order to determine peak SF
# response/bandwidth/etc

# If argument 'template' is given, then orientation, phase, and tf_center will be
# taken from the template, which will be actual stimuli from that cell

    if expInd>1:
      warnings.warn('The contrasts used here match only the 1st, original V1 experiment structure');
    # Fixed parameters
    exper = get_exp_params(expInd);
    num_gratings = exper.nStimComp;
    num_families = exper.nFamilies;
    comps        = exper.comps;

    spreadVec = numpy.logspace(math.log10(.125), math.log10(1.25), num_families);
    octSeries  = numpy.linspace(1.5, -1.5, num_gratings);

    # set contrast and spatial frequency
    if conLevel == 1:
        total_contrast = 1;
    elif conLevel == 2:
        total_contrast = 1/3;
    elif conLevel>=0 and conLevel<1:
        total_contrast = conLevel;
        conLevel = 1; # just set to 1 (i.e. get "high contrast" block IDs, used if expInd=1; see valid_blockIDs = ...)
    else:
        total_contrast = 1; # default to that
        
    spread     = spreadVec[stimFamily-1];
    profTemp = norm.pdf(octSeries, 0, spread);
    profile    = profTemp/sum(profTemp);

    if stimFamily == 1: # do this for consistency with actual experiment - for stimFamily 1, only one grating is non-zero; round gives us this
        profile = numpy.round(profile);

    Sf = numpy.power(2, octSeries + numpy.log2(sf_c)); # final spatial frequency
    Co = numpy.dot(profile, total_contrast); # final contrast

    ## The others
    
    # get orientation - IN RADIANS
    trial = template.get('sfm').get('exp').get('trial');
    OriVal = mode(trial.get('ori')[0]).mode * numpy.pi/180; # pick arbitrary grating, mode for this is cell's pref Ori for experiment
    Or = numpy.matlib.repmat(OriVal, 1, num_gratings)[0]; # weird tuple [see below], just get the array we care about...
    
    if template.get('trial_used') is not None: #use specified trial
        trial_to_copy = template.get('trial_used');
    else: # get random trial for phase, TF
        # we'll draw from a random trial with the same stimulus family
        if expInd == 1:
          valid_blockIDs = numpy.arange((stimFamily-1)*(13*2)+1+(conLevel-1), ((stimFamily)*(13*2)-5)+(conLevel-1), 2)
        else:
          valid_blockIDs = trial['blockID'][numpy.where(trial['num_comps'] == comps[stimFamily-1])];
        num_blockIDs = len(valid_blockIDs);
        # for phase and TF
        valid_trials = trial.get('blockID') == valid_blockIDs[random.randint(0, num_blockIDs-1)] # pick a random block ID
        valid_trials = numpy.where(valid_trials)[0]; # 0 is to get array...weird "tuple" return type...
        trial_to_copy = valid_trials[random.randint(0, len(valid_trials)-1)]; # pick a random trial from within this

    trial_used = trial_to_copy;
    
    # grab Tf and Phase [IN RADIANS] from each grating for the given trial
    Tf = numpy.asarray([i[trial_to_copy] for i in trial.get('tf')]);
    Ph = numpy.asarray([i[trial_to_copy] * math.pi/180 for i in trial.get('ph')]);
    
    if numpy.any(numpy.isnan(Tf)):
      pdb.set_trace();

    # now, sort by contrast (descending) with ties given to lower SF:
    inds_asc = numpy.argsort(Co); # this sorts ascending
    inds_des = inds_asc[::-1]; # reverse it
    Or = Or[inds_des];
    Tf = Tf[inds_des];
    Co = Co[inds_des];
    Ph = Ph[inds_des];
    Sf = Sf[inds_des];
    
    return {'Ori': Or, 'Tf' : Tf, 'Con': Co, 'Ph': Ph, 'Sf': Sf, 'trial_used': trial_used}

def getNormParams(params, normType):
  ''' pass in param list, normType (1=original "tilt'; 2=gaussian weighting (wght); 3=con-dep sigma)
  '''
  if normType == 1:
    if len(params) > 8:
      inhAsym = params[8];
    else:
      inhAsym = 0; 
    return inhAsym;
  elif normType == 2:
    gs_mean = params[8];
    gs_std  = params[9];
    return gs_mean, gs_std;
  elif normType == 3:
    # sigma calculation
    offset_sigma = params[8];  # c50 filter will range between [v_sigOffset, 1]
    stdLeft      = params[9];  # std of the gaussian to the left of the peak
    stdRight     = params[10]; # '' to the right '' 
    sfPeak       = params[11]; # where is the gaussian peak?
    return offset_sigma, stdLeft, stdRight, sfPeak;
  else:
    inhAsym = 0;
    return inhAsym;

def genNormWeights(cellStruct, nInhChan, gs_mean, gs_std, nTrials, expInd):
  np = numpy;
  # A: do the calculation here - more flexibility
  inhWeight = [];
  nFrames = num_frames(expInd);
  T = cellStruct['sfm'];
  nInhChan = T['mod']['normalization']['pref']['sf'];
        
  for iP in range(len(nInhChan)): # two channels: narrow and broad

    # if asym, put where '0' is
    curr_chan = len(T['mod']['normalization']['pref']['sf'][iP]);
    log_sfs = np.log(T['mod']['normalization']['pref']['sf'][iP]);
    new_weights = norm.pdf(log_sfs, gs_mean, gs_std);
    inhWeight = np.append(inhWeight, new_weights);
    
  inhWeightT1 = np.reshape(inhWeight, (1, len(inhWeight)));
  inhWeightT2 = repmat(inhWeightT1, nTrials, 1);
  inhWeightT3 = np.reshape(inhWeightT2, (nTrials, len(inhWeight), 1));
  inhWeightMat  = np.tile(inhWeightT3, (1,1,nFrames));

  return inhWeightMat;

def setSigmaFilter(sfPref, stdLeft, stdRight, filtType = 1):
  '''
  For now, we are parameterizing the semisaturation contrast filter as a "fleixble" Gaussian
  That is, a gaussian parameterized with a mean, and a standard deviation to the left and right of that peak/mean
  We set the baseline of the filter to 0 and the overall amplitude to 1
  '''
  filter = dict();
  if filtType == 1:
    filter['type'] = 1; # flexible gaussian
    filter['params'] = [0, 1, sfPref, stdLeft, stdRight]; # 0 for baseline, 1 for respAmpAbvBaseline

  return filter;

def evalSigmaFilter(filter, scale, offset, evalSfs):
  '''
  filter is the type of filter to be evaluated (will be dictionary with necessary parameters)
  scale, offset are used to scale and offset the filter shape
  evalSfs - which sfs to evaluate at
  '''

  params = filter['params'];  
  if filter['type'] == 1: # flexibleGauss
    filterShape = numpy.array(flexible_Gauss(params, evalSfs, 0)); # 0 is baseline/minimum value of flexible_Gauss
  elif filter['type'] == 2:
    filterShape = deriv_gauss(params, evalSfs)[0]; # take the first output argument only

  evalC50 = scale*filterShape + offset - scale 
  # scale*filterShape will be between [scale, 0]; then, -scale makes it [0, -scale], where scale <0 ---> -scale>0
  # finally, +offset means evalC50 is [offset, -scale+offset], where -scale+offset will typically = 1
  return evalC50;

def setNormTypeArr(params, normTypeArr = []):
  '''
  Used to create the normTypeArr array which is called in model_responses by SFMGiveBof and SFMsimulate to set
  the parameters/values used to compute the normalization signal for the full model

  Requires the model parameters vector; optionally takes normTypeArr as input

  Returns the normTypeArr
  '''

  # constants
  c50_len = 12; # 12 parameters if we've optimized for the filter which sets c50 in a frequency-dependent way
  gauss_len = 10; # 10 parameters in the model if we've optimized for the gaussian which weights the normalization filters
  asym_len = 9; # 9 parameters in the model if we've used the old asymmetry calculation for norm weights

  inhAsym = 0; # set to 0 as default

  # now do the work
  if normTypeArr:
    norm_type = int(normTypeArr[0]); # typecast to int
    if norm_type == 2: # c50
      if len(params) == c50_len:
        filt_offset = params[8];
        std_l = params[9];
        std_r = params[10];
        filt_peak = params[11];
      else:
        if len(normTypeArr) > 1:
          filt_offset = normTypeArr[1];
        else: 
          filt_offset = random_in_range([0.05, 0.2])[0]; 
        if len(normTypeArr) > 2:
          std_l = normTypeArr[2];
        else:
          std_l = random_in_range([0.5, 5])[0]; 
        if len(normTypeArr) > 3:
          std_r = normTypeArr[3];
        else: 
          std_r = random_in_range([0.5, 5])[0]; 
        if len(normTypeArr) > 4:
          filt_peak = normTypeArr[4];
        else: 
          filt_peak = random_in_range([1, 6])[0]; 
      normTypeArr = [norm_type, filt_offset, std_l, std_r, filt_peak];

    elif norm_type == 1:
      if len(params) == gauss_len: # we've optimized for these parameters
        gs_mean = params[8];
        gs_std = params[9];
      else:
        if len(normTypeArr) > 1:
          gs_mean = normTypeArr[1];
        else:
          gs_mean = random_in_range([-1, 1])[0];
        if len(normTypeArr) > 2:
          gs_std = normTypeArr[2];
        else:
          gs_std = numpy.power(10, random_in_range([-2, 2])[0]); # i.e. 1e-2, 1e2
      normTypeArr = [norm_type, gs_mean, gs_std]; # save in case we drew mean/std randomly
    
    elif norm_type == 0:
      if len(params) == asym_len:
        inhAsym = params[8];
      if len(normTypeArr) > 1: # then we've passed in inhAsym to override existing one, if there is one
        inhAsym = normTypeArr[1];
      normTypeArr = [norm_type, inhAsym];

  else:
    norm_type = 0; # i.e. just run old asymmetry computation
    if len(params) == asym_len:
      inhAsym = params[8];
    if len(normTypeArr) > 1: # then we've passed in inhAsym to override existing one, if there is one
      inhAsym = normTypeArr[1];
    normTypeArr = [norm_type, inhAsym];

  return normTypeArr;

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
