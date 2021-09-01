import math, numpy, random
from scipy.stats import norm, mode, poisson, nbinom, sem
from scipy.stats.mstats import gmean as geomean
from numpy.matlib import repmat
from helper_fcns_sfBB import get_resp_str
import scipy.optimize as opt
import os, sys
import importlib as il
import itertools
from time import sleep
sqrt = math.sqrt
log = math.log
exp = math.exp
from functools import partial
import multiprocessing as mp
import pdb
import warnings

# Functions:

### I. basics

# np_smart_load - be smart about using numpy load
# nan_rm        - remove nan from array
# bw_lin_to_log
# bw_log_to_lin
# sf_highCut      - compute the (high) SF at which the response falls to X (a fraction) of the peak
# sf_com          - model-free calculation of the tuning curve's center-of-mass
# sf_var          - model-free calculation of the variance in the measured responses
# get_datalist    - given the experiment directory, get the data list name
# exp_name_to_ind - given the name of an exp (e.g. sfMixLGN), return the expInd
# get_exp_params  - given an index for a particular version of the sfMix experiments, return parameters of that experiment (i.e. #stimulus components)
# get_exp_ind     - given a .npy for a given sfMix recording, determine the experiment index
# parse_exp_name  - parse the experiment name into m#, unit#, p#, file ext
# num_frames      - compute/return the number of frames per stimulus condition given expInd
# fitType_suffix  - get the string corresponding to a fit (i.e. normalization) type
# lossType_suffix - get the string corresponding to a loss type
# chiSq_suffix    - what suffix (e.g. 'a' or 'c') given the chiSq multiplier value
# fitList_name    - put together the name for the fitlist 
# phase_fit_name  
# descrMod_name   - returns string for descriptive model fit
# descrLoss_name  - returns string for descriptive model loss type
# descrFit_name   - name for descriptive fits
# rvc_mod_suff    - what is the suffix for the given rvcModel (e.g. 'NR', 'peirce')
# rvc_fit_name    - name for rvcFits
# angle_xy
# flatten_list
# switch_inner_outer

### II. fourier, and repsonse-phase adjustment

# make_psth - create a psth for a given spike train
# make_psth_slide - create a sliding psth (not vectorized)
# fit_onset_transient - make a fit to the onset transient of a PSTH 
# manual_fft - "Manual" FFT, including (optional) onset transient
# fft_amplitude - adjust the FFT amplitudes as needed for a real signal (i.e. double non-DC amplitudes)
# spike_fft - compute the FFT for a given PSTH, extract the power at a given set of frequencies 
# compute_f1_byTrial - returns amp, phase for each trial at the stimulus TFs (i.e. full F1, not just amplitude)
# adjust_f1_byTrial  - akin to the same in hf_sfBB, returns vector-corrected amplitudes for each trial in order
# compute_f1f0 - compute the ratio of F1::F0 for the stimulus closest to optimal

### III. phase/more psth

# project_resp - project the measured response onto the true/predicted phase and determine the "true" response amplitude
# project_resp_cond - project the individual responses for a given condition
# first_ph0 - for a given stimulus start phase, compute how much of a cycle (and how much time) before the stimulus gets to the start of a cycle (i.e. ph=0)
# fold_psth - fold a psth for a given number of cycles (given spike times)
# get_true_phase - compute the response phase relative to the stimulus phase given a response phase (rel. to trial time window) and a stimulus phase (rel. to trial start)
# polar_vec_mean - compute the vector mean given a set of amplitude/phase pairs for responses on individual trials
# get_all_fft - extract the amp/phase for a condition or set of conditions
# get_phAdv_model - return the lambda function describing the responsePhase-as-function-of-respAmplitude model
# get_recovInfo - get the model recovery parameters/spikes, if applicable
# phase_advance - compute the phase advance (a la Movshon/Kiorpes/+ 2005)
# tf_to_ind - convert the given temporal frequency into an (integer) index into the fourier spectrum

### IV. descriptive fits to sf tuning/basic data analyses

# get_rvc_model - return the lambda function describing the rvc model
# naka_rushton - naka-rushton form of the response-versus-contrast, with flexibility to evaluate super-saturating RVCs (Peirce 2007)
# rvc_fit - fit response versus contrast with a model used in Movshon/Kiorpes/+ 2005

# DiffOfGauss - standard difference of gaussians
# DoGsach - difference of gaussians as implemented in sach's thesis
# var_explained - compute the variance explained for a given model fit/set of responses
# chiSq      - compute modified chiSq loss value as described in Cavanaugh et al
# get_c50 - get the c50 given the rvcModel and model parameters5
# c50_empirical - compute the effective/emperical c50 by optimization
# descr_prefSf - compute the prefSf for a given DoG model/parameter set
# dog_prefSfMod - fit a simple model of prefSf as f'n of contrast
# dog_charFreq - given a model/parameter set, return the characteristic frequency of the tuning curve
# dog_charFreqMod - smooth characteristic frequency vs. contrast with a functional form/fit
# dog_get_param - get the radius/gain given the parameters and specified DoG model

# dog_loss - compute the DoG loss, given responses and model parameters
# dog_init_params - given the responses, estimate initial parameters for a given DoG model
# dog_fit - used to fit the Diff of Gauss responses -- either separately for each con, or jointly for all cons within a given dispersion

# deriv_gauss - evaluate a derivative of a gaussian, specifying the derivative order and peak
# get_prefSF - Given a set of parameters for a flexible gaussian fit, return the preferred SF
# compute_SF_BW - returns the log bandwidth for height H given a fit with parameters and height H (e.g. half-height)
# fix_params - Intended for parameters of flexible Gaussian, makes all parameters non-negative
# flexible_Gauss - Descriptive function used to describe/fit SF tuning
# get_descrResp - get the SF descriptive response

#######
## V. jointList interlude
#######

# jl_perCell - how to oranize everything, given a cell (called by jl_create, in a loop)
# jl_create - create the jointList
# jl_get_metric_byCon()
# jl_get_metric_highComp()

######
## IV. return to fits/analysis (part ii)
######

# blankResp - return mean/sem of blank responses (i.e. baseline firing rate) for sfMixAlt experiment
# get_valid_trials - return list of valid trials given disp/con/sf
# get_valid_sfs - return list indices (into allSfs) of valid sfs for given disp/con

# get_condition - trial-by-trial f0/f1 for given condition
# get_condition_adj - as above, but adj responses
# get_isolated_response - collect responses (mean/sem/trial) of comps of stimulus when presented in isolation
# get_isolated_responseAdj - as above, but adj responses

# tabulate_responses - Organizes measured and model responses for sfMixAlt experiment
# organize_adj_responses - wrapper for organize_adj_responses within each experiment subfolder
# organize_resp       -
# get_spikes - get correct # spikes for a given cell (will get corrected spikes if needed) trial-by-trial
# get_rvc_fits - return the rvc fits for a given cell (if applicable)
# get_adjusted_spikerate - wrapper for get_spikes which gives us the correct/adjusted (if needed) spike rate (per second)
# mod_poiss - computes "r", "p" for modulated poisson model (neg. binomial)
# fit_CRF
# random_in_range - random real-valued number between A and B
# nbinpdf_log - was used with sfMix optimization to compute the negative binomial probability (likelihood) for a predicted rate given the measured spike count

# getSuppressiveSFtuning - returns the normalization pool response
# makeStimulusRef - new way of making stimuli (19.05.13)
# makeStimulus - was used last for sfMix experiment to generate arbitrary stimuli for use with evaluating model
# getNormParams  - given the model params and fit type, return the relevant parameters for normalization
# genNormWeightsSimple - for the simple version of the normalization model
# genNormWeights - used to generate the weighting matrix for weighting normalization pool responses
# setSigmaFilter - create the filter we use for determining c50 with SF
# evalSigmaFilter - evaluate an arbitrary filter at a set of spatial frequencies to determine c50 (semisaturation contrast)
# setNormTypeArr - create the normTypeArr used in SFMGiveBof/Simulate to determine the type of normalization and corresponding parameters; DEPRECATED?
# getConstraints - return the constraints used in model optimization
# getConstraints_joint - return the constraints used in model optimization

#######
## VI. fitting/analysis for basic characterizations (i.e. NON sfMix* programs, like ori16, etc)
#######
# oriCV - return the orientation circular variance measure (Xing et al, 2004)
# oriTune - return prefOri, oriBW, bestFitParams, struct from optimization call
# tfTune - return prefTf, tfBW (in octaves)
# sizeTune - data and model derived metrics, curves for plotting (disk and annulus tuning)
# rvcTune - returns c50 and conGain value (if it can be computed)

# get_basic_tunings - wrapper for the above, and calls to ExpoAnalysisTools/python/readBasicCharacterization


##################################################################
##################################################################
##################################################################
### I. BASICS
##################################################################
##################################################################
##################################################################


def np_smart_load(file_path, encoding_str='latin1', allow_pickle=True):
   # added allow_pickle=True to handle different numpy version in pytorch env. compared to lcv-python env.

   if not os.path.isfile(file_path):
     return [];
   loaded = [];
   nTry = 10;
   while(nTry > 0):
     try:
         loaded = numpy.load(file_path, encoding=encoding_str, allow_pickle=True).item();
         break;
     except IOError: # this happens, I believe, because of parallelization when running on the cluster; cannot properly open file, so let's wait and then try again
        sleep_time = random_in_range([3, 5])[0];
        sleep(sleep_time); # i.e. wait for 10 seconds
     except EOFError: # this happens, I believe, because of parallelization when running on the cluster; cannot properly open file, so let's wait and then try again
        sleep_time = random_in_range([3, 5])[0];
        sleep(sleep_time); # i.e. wait for 10 seconds
     except: # pickling error???
        sleep_time = random_in_range([3, 5])[0];
        sleep(sleep_time); # i.e. wait for 10 seconds
     nTry -= 1 #don't try indefinitely!

   return loaded;

def nan_rm(x):
   return x[~numpy.isnan(x)];

def bw_lin_to_log( lin_low, lin_high ):
    ''' Given the low/high sf in cpd, returns number of octaves separating the
        two values
    '''

    return numpy.log2(lin_high/lin_low);

def bw_log_to_lin(log_bw, pref_sf):
    ''' given the preferred SF and octave bandwidth, returns the corresponding
        (linear) bandwidth and bounds in cpd
    '''

    less_half = numpy.power(2, numpy.log2(pref_sf) - log_bw/2);
    more_half = numpy.power(2, log_bw/2 + numpy.log2(pref_sf));

    sf_range = [less_half, more_half];
    lin_bw = more_half - less_half;
    
    return lin_bw, sf_range

def sf_highCut(params, sfMod, frac, sfRange=(0.1, 10), baseline_sub=None):
  ''' given a fraction of peak response (e.g. 0.5 or 1/e or...), compute the frequency (above the peak)
      at which the response reaches that response fraction
        note: we can pass in obj_thresh (i.e. if the optimized objective function is greater than this value,
              we count the model as NOT having a defined cut-off)
  '''
  np = numpy;
  peak_sf = descr_prefSf(params, sfMod, sfRange);
  if baseline_sub is None: # if we pass in baseline_sub, we're measuring for "frac" reduction in response modulation above baseline response
    to_sub = np.array(0);
  else:
    to_sub = baseline_sub;
  peak_resp = get_descrResp(params, np.array([peak_sf]), sfMod)[0] - to_sub; # wrapping, since we expect list/array in get_descrResp
  # now, what is the criterion response you need?
  crit_resp = peak_resp * frac;
  crit_obj = lambda fc: np.square((get_descrResp(params, np.array([fc]), sfMod)[0] - to_sub) - crit_resp);
  sfs_to_test = np.geomspace(peak_sf, sfRange[1], 500);
  match_ind = np.argmin([crit_obj(x) for x in sfs_to_test]);
  match_sf = sfs_to_test[match_ind];

  if match_sf == sfRange[1]: # i.e. we don't really have a cut-off in this range
    return np.nan;
  else:
    return match_sf;

def sf_com(resps, sfs):
  ''' model-free calculation of the tuning curve's center-of-mass
      input: resps, sfs (np arrays; sfs in linear cpd)
      output: center of mass of tuning curve (in linear cpd)
  '''
  np = numpy;
  com = lambda resp, sf: np.dot(np.log2(sf), np.array(resp))/np.sum(resp);
  try:
    return np.power(2, com(resps, sfs));
  except:
    return np.nan

def sf_var(resps, sfs, sf_cm):
  ''' model-free calculation of the tuning curve's center-of-mass
      input: resps, sfs (np arrays), and center of mass (sfs, com in linear cpd)
      output: variance measure of tuning curve
  '''
  np = numpy;
  sfVar = lambda cm, resp, sf: np.dot(resp, np.abs(np.log2(sf)-np.log2(cm)))/np.sum(resp);
  try:
    return sfVar(sf_cm, resps, sfs);
  except:
    return np.nan

def get_datalist(expDir, force_full=0):
  if expDir == 'V1_orig/':
    return 'dataList_200507.npy' if force_full==0 else 'dataList.npy'; # limited set of data
    #return 'dataList.npy';
  elif expDir == 'altExp/':
    return 'dataList_200507.npy' if force_full==0 else 'dataList.npy'; # limited set of data
    #return 'dataList.npy';
  elif expDir == 'LGN/':
    return 'dataList_210524.npy'
    #return 'dataList.npy';
  elif expDir == 'V1/':
    return 'dataList_glx_200507.npy' if force_full==0 else 'dataList_210721.npy' #'dataList_glx.npy'; # limited set of data
    #return 'dataList_glx_200507.npy' if force_full==0 else 'dataList_210528.npy' #'dataList_glx.npy'; # limited set of data
    #return 'dataList_glx.npy';
  elif expDir == 'V1_BB/':
    return 'dataList_210721.npy';
    #return 'dataList_210222.npy';

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
    elif expName == 'sfMixLGNhalfInt':
      expInd = 6;
    elif 'sfBB' in expName:
      expInd = -1 # sfBB for now...
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
        ### LGN versions
        elif expInd == 3 or expInd == 6: 
          self.nStimComp = 5;
          self.nFamilies = 2;
          self.comps     = [1, 5];
          self.nCons     = 4;
          self.nSfs      = 11;
          self.nCells    = 34;
          self.dir       = 'LGN/'
          self.fps       = 120; # frame rate (in Hz, i.e. frames per second)
          if expInd == 3: # (original) LGN experiment - m675 and beyond; two recordings from V1 exp. m676 (in V1/)
            self.stimDur   = 1; # in seconds
          elif expInd == 6: # (updated "halfInt") LGN experiment - m680 and beyond
            self.stimDur   = 2; # in seconds
        ### full (V1) versions
        elif expInd == 4 or expInd == 5:
          self.nStimComp = 7;
          self.nFamilies = 4;
          self.comps     = [1, 3, 5, 7]
          self.nCons     = 4;
          self.nSfs      = 11;
          self.nCells    = 1;
          self.dir       = 'V1/'
          self.fps       = 120; # frame rate (in Hz, i.e. frames per second)
          if expInd == 4: # V1 "Int" - same as expInd = 2, but with integer TFs (keeping separate to track # cells)
            self.stimDur   = 1; # in seconds
          elif expInd == 5: # V1 "halfInt" - same as expInd = 4, but with halfinteger TFs
            self.stimDur   = 2; # in seconds
        ### sfBB
        elif expInd == -1: # sfBB for now...
          self.nStimComp = 2;
          self.nFamilies = -1; # does not apply here...
          self.comps     = [2]
          self.nCons     = 7;
          self.nSfs      = 7;
          self.nCells    = 1;
          self.dir       = 'V1_BB/'
          self.fps       = 120; # frame rate (in Hz, i.e. frames per second)
          self.stimDur   = 1; # in seconds

        if forceDir is not None:
          self.dir       = forceDir;

    return exp_params(expInd);

def get_exp_ind(filePath, fileName, overwriteExpName=None):
    '''  returns the following:
           index of experiment (see get_exp_params)
           name of experiment (e.g. sfMix, sfMixHalfInt)

         this function relies on the fact that all .npy files (in /structures) have an associated matlab file
         in /recordings with the full experiment name
           EXCEPT: V1_orig files...
    '''
    if overwriteExpName is not None:
      return exp_name_to_ind(overwriteExpName), overwriteExpName;

    if 'V1_orig' in filePath:
      return 1, 'sfMix'; # need to handle this case specially, since we don't have /recordings/ for this experiment

    if fileName.startswith('mr'): # i.e. model recovery...
      name_root = fileName.split('_')[1];
    else:
      name_root = fileName.split('_')[0];
    orig_files = os.listdir(filePath + '../recordings/');
    for f in orig_files:
      if f.startswith(name_root) and '.xml' in f and 'sfMix' in f: # make sure it's not a *_sfm.mat that is in .../recordings/, and that .xml and "sfMix" are in the file
        # we rely on the following: fileName#run[expType].[exxd/xml]
        expName = f.split('[')[1].split(']')[0];
        return exp_name_to_ind(expName), expName;

    return None, None; # uhoh...

def parse_exp_name(name):
  ''' Assuming the name format is mNNNpZZlY#[NAME].EXT
      - TODO: Decide on and adapt to naming convention for sorting-derived cells
      - return 'mNNN', unit #, pen # [or NONE], run # [or NONE], prog name [or NONE], file extension [or NONE]
  '''
  _, nameOnly = os.path.split(name); # in case there is a file with path, ignore the path
  ### Make in loop? Should be straightforward

  # let's see if we have an extension
  splt = nameOnly.split('.') # we should only have one '.' in the full name, only if there is an extension
  name = splt[0]; # recover the name from that split...
  if len(splt) == 1: # i.e. no file extension to split
     ext = None
  else:
     ext = splt[1]
  # let's see if we have a program name
  splt = name.split('[');
  name = splt[0];
  if len(splt) == 1: # i.e. no program name
    progName = None
  else:
    progName = splt[1].split(']')[0];
  # Get a run number, if applicable
  splt = name.split('#');
  name = splt[0];
  if len(splt) == 1: # i.e. no run #
     runNumber = None
  else:
     runNumber = int(splt[1]);
  # Get unit #
  splt = name.split('r');
  if len(splt) == 1:
     splt = name.split('l'); 
  name = splt[0];
  if len(splt) == 1: # i.e. no unit #
     unitNum = None
  else:
     unitNum = int(splt[1]);
  # Get penetration #, if applicable
  splt = name.split('p');
  name = splt[0];
  if len(splt) == 1: # i.e. no run #
     penNumber = None
  else:
     penNumber = int(splt[1]);
  mStr = name;

  return mStr, unitNum, penNumber, runNumber, progName, ext;  

def num_frames(expInd):
  ''' compute/return the number of frames per stimulus condition given expInd '''
  exper = get_exp_params(expInd);
  dur = exper.stimDur;
  fps = exper.fps;
  nFrames    = dur*fps;
  return nFrames;

def fitType_suffix(fitType):
  ''' Use this to get the string referring to a given normalization type
  '''
  if fitType == 1:
    fitSuf = '_flat';
  elif fitType == 2:
    fitSuf = '_wght';
  elif fitType == 3:
    fitSuf = '_c50';
  elif fitType == 4:
    fitSuf = '_flex';
  elif fitType == 5:
    fitSuf = '_wghtGain';
  return fitSuf;

def lossType_suffix(lossType):
  ''' Use this to get the string referring to a given loss function
  '''
  if lossType == 1:
    lossSuf = '_sqrt.npy';
  elif lossType == 2:
    lossSuf = '_poiss.npy';
  elif lossType == 3:
    lossSuf = '_modPoiss.npy';
  elif lossType == 4:
    lossSuf = '_chiSq.npy';
  return lossSuf;

def lgnType_suffix(lgnType, lgnConType=1):
  ''' Use this to get the string referring to a given loss function
  '''
  if lgnType == 0:
    lgnSuf = '';
  elif lgnType == 1:
    lgnSuf = '_LGN';
  elif lgnType == 2:
    lgnSuf = '_LGNb';
  elif lgnType == 9:
    lgnSuf = '_jLGN';

  if lgnConType == 1: # this means separate RVC for M & P channels
    conSuf = '';
  elif lgnConType == 2: # this means one averaged RVC, with equal weighting for M & P (i.e. fixed)
    conSuf = 'f'; # fixed
  elif lgnConType == 3: # as for 2, but M vs P weighting for RVC is yoked to the mWeight model parameter (optimized)
    conSuf = 'y'; # "yoked"
  elif lgnConType == 4: # parvo-only front-end
    conSuf = 'p';

  return '%s%s' % (lgnSuf, conSuf);

def chiSq_suffix(kMult):
  ''' We need a multiplying constant when using the chiSquared loss (see chiSq within this file)
      I denote this multiplier with a flag; this function returns that flag based on the value
  '''
  if kMult == 0.01:
    return 'a';
  elif kMult == 0.05:
    return 'b';
  elif kMult == 0.10:
    return 'c';
  else: # if it's not one of the default values, then add a special suffix
    asStr = str(kMult);
    afterDec = asStr.split('.')[1];
    # why? don't want (e.g.) Z0.02, just Z02 - we know multiplier values are 0<x<<1
    return 'z%s' % afterDec;

def fitList_name(base, fitType, lossType, lgnType=None, lgnConType=1, vecCorrected=0, CV=0, fixRespExp=None, kMult=0.1):
  ''' use this to get the proper name for the full model fits
      - kMult used iff lossType == 4
  '''
  # first the fit type
  fitSuf = fitType_suffix(fitType);
  # then the loss type
  lossSuf = lossType_suffix(lossType);
  # IF lgnType/lgnConType are given, then we can add that, too
  if lgnType is not None:
    lgnSuf = lgnType_suffix(lgnType, lgnConType);
  else:
    lgnSuf = '';
  vecSuf = '_vecF1' if vecCorrected else '';
  CVsuf = '_CV' if CV else '';
  reSuf = np.round(fixRespExp*10) if fixRespExp is not None else ''; # for fixing response exponent (round to nearest tenth)
  kMult = chiSq_suffix(kMult) if lossType == 4 else '';  

  # order is as follows
  return str(base + kMult + vecSuf + CVsuf + reSuf + lgnSuf + fitSuf + lossSuf);

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
  if dir is None or dir == 0:
    base = base + '.npy';
  return base;

def descrMod_name(DoGmodel):
  ''' returns the string for a given SF descriptive model fit
  '''
  if DoGmodel == 0:
    modStr = 'flex';
  elif DoGmodel == 1:
    modStr = 'sach';
  elif DoGmodel == 2:
    modStr = 'tony';
  return modStr;

def descrLoss_name(lossType):
  ''' returns the string for a given SF descriptive loss type
  '''
  if lossType == 1:
    floss_str = '_lsq';
  elif lossType == 2:
    floss_str = '_sqrt';
  elif lossType == 3:
    floss_str = '_poiss';
  elif lossType == 4:
    floss_str = '_sach';
  else:
    floss_str = '';
  return floss_str

def descrFit_name(lossType, descrBase=None, modelName = None):
  ''' if modelName is none, then we assume we're fitting descriptive tuning curves to the data
      otherwise, pass in the fitlist name in that argument, and we fit descriptive curves to the model
      this simply returns the name
  '''
  # load descrFits
  floss_str = descrLoss_name(lossType);
  if descrBase is None:
    descrBase = 'descrFits';
  descrFitBase = '%s%s' % (descrBase, floss_str);

  if modelName is None:
    descrName = '%s.npy' % descrFitBase;
  else:
    descrName = '%s_%s.npy' % (descrFitBase, modelName);
    
  return descrName;

def rvc_mod_suff(modNum):
   ''' returns the suffix for a given rvcModel number'''
   if modNum == 0:
     suff = '';
   elif modNum == 1:
     suff = '_NR';
   elif modNum == 2:
     suff = '_peirce';
   
   return suff;

def rvc_fit_name(rvcBase, modNum, dir=1, vecF1=None):
   ''' returns the correct suffix for the given RVC model number and direction (pos/neg)
   '''
   vecSuff = '_vecF1' if vecF1==1 else '';
   suff = rvc_mod_suff(modNum);
   base = rvcBase + vecSuff + suff;

   return phase_fit_name(base, dir);

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

def flatten_list(l):
  ''' turn ((a), (b), ...) into (a, b, ...) '''
  flatten = lambda l: [item for sublist in l for item in sublist];
  return flatten(l);

def switch_inner_outer(x, asnp = False):
  ''' switch the inner and outer parts of a list '''
  if asnp: # i.e. cast each as numpy array
    switch_inner_outer = lambda arr: [numpy.array([x[i] for x in arr]) for i in range(len(arr[0]))];
  else:
    switch_inner_outer = lambda arr: [[x[i] for x in arr] for i in range(len(arr[0]))];
  return switch_inner_outer(x);

##################################################################
##################################################################
##################################################################
### II. FOURIER
##################################################################
##################################################################
##################################################################

def make_psth(spikeTimes, binWidth=1e-3, stimDur=1):
    # given an array of arrays of spike times, create the PSTH for a given bin width and stimulus duration
    # i.e. spikeTimes has N arrays, each of which is an array of spike times

    # -- added int(*) to ensure the # of steps is an int (required for newer np versions)
    binEdges = numpy.linspace(0, stimDur, 1+int(stimDur/binWidth));
    
    all = [numpy.histogram(x, bins=binEdges) for x in spikeTimes]; 
    psth = [x[0] for x in all];
    bins = [x[1] for x in all];
    return psth, bins;

def make_psth_slide(spikeTimes, binWidth=25e-3, stimDur=1, binSlide=1e-3, debug=0):
    ''' binWidth in S will be x +/- binWidth s; each bin will be centered binSlide s away (default is 1e-3, i.e. 1ms)
        NOTE: This assumes all spikeTimes will be of same stimDur and we use common slide/width values
        -- Yes, this is vectorized
    '''
    np = numpy;

    spikeTimes = [sorted(x) for x in spikeTimes]; # to use searchsorted, each spikeTimes array must be sorted!
    time_centers = np.linspace(0, stimDur, 1+int(stimDur/binSlide));
    idx1 = [np.array([np.searchsorted(spikeTimesCurr,tc-binWidth,'right') for tc in time_centers]) for spikeTimesCurr in spikeTimes];
    idx2 = [np.array([np.searchsorted(spikeTimesCurr,tc+binWidth,'left') for tc in time_centers]) for spikeTimesCurr in spikeTimes];

    # Using the lower/upper bounds of each bin, determine how many S of each time bin are valid times (i.e. between [0, stimDur])
    binLow = time_centers - binWidth;
    binHigh = time_centers + binWidth;
    full_width = 2*binWidth;
    binLow[binLow>0] = 0
    binHigh = binHigh-stimDur;
    binHigh[binHigh<0] = 0
    div_factor = full_width - (binHigh-binLow)
    counts = [np.divide(idx2Curr-idx1Curr, 1e3*div_factor) for idx2Curr,idx1Curr in zip(idx2,idx1)];

    if debug:
        return counts, time_centers, idx1, idx2, div_factor
    else:
        return counts, time_centers

def fit_onset_transient(psth, bins, onsetWidth=100, stimDur=1, whichMod=0, toNorm=1):
   ''' psth, bins should be from the make_psth_slide call
       onsetWidth is in mS, stimDur is in S
   '''
   np = numpy;

   onsetInd = int((1e-3*onsetWidth/stimDur)*len(psth));
   onset_rate, onset_bins = psth[0:onsetInd], bins[0:onsetInd]
   onset_toFit = onset_rate - np.min(psth);
   # make the fit to the onset transient...
   if whichMod == 0:
      oy = np.polyfit(onset_bins, onset_toFit, deg=10);
      asMod = np.poly1d(oy); # make a callable function given the fit
      full_transient = np.zeros_like(bins[1:]);
      full_transient[0:onsetInd] = asMod(onset_bins);
      if toNorm == 1:
         full_transient = np.divide(full_transient, np.max(full_transient));
   elif whichMod == 1: # make other methods...
      # FIRST: create a filter which forces the onset to start from 0 and end at 0
      # assume: 10% of onset to get to max, 10% to decay to 0 at end
      nBins = len(onset_bins)
      onsFilt = np.ones_like(onset_bins)
      onCut = int(0.1*nBins);
      onsFilt[0:onCut] = np.square(np.linspace(0,1,onCut))
      onsFilt[nBins-onCut:nBins] = np.square(np.linspace(1,0,onCut))

      # what form to fit? "trunc_norm" (i.e. a truncated gaussian)
      from scipy.stats import truncnorm
      a = 0; b = onsetWidth;
      my_truncnorm = lambda amp,loc, scale, x: amp*truncnorm.pdf(x, (a-loc)/scale, (b-loc)/scale, loc=loc, scale=scale);
      loss = lambda prms: np.sum(np.square(my_truncnorm(*prms, 1e3*onset_bins) - onset_toFit))
      init_fit = [0.2,50,10]
      best_fit = opt.minimize(loss, init_fit)
      full_transient = np.zeros_like(bins[1:]);
      full_transient[0:onsetInd] = onsFilt*my_truncnorm(*best_fit['x'], 1e3*onset_bins);
      if toNorm == 1:
         full_transient = np.divide(full_transient, np.max(full_transient));

   return full_transient;

def manual_fft(psth, tfs, onsetTransient=None, stimDur=1, binWidth=1e-3):
    ''' Compute the FFT in a manual way - must pass in np.array of TF values (as integer)
        - If you pass in onsetTransient (should be same length as psth), will include transient coefficient, too
    '''
    np = numpy;
    n_coeff = 1 + 2*len(tfs); # DC, onset transient, sin&cos for the two stim TFs
    if onsetTransient is not None:
      n_coeff += 1; # make sure we account for an extra column...

    input_mat = np.ones((n_coeff, len(psth))); # this way we don't need to manually enter the DC term...
    lower_bounds = []; upper_bounds = [];
    lower_bounds.append(0); upper_bounds.append(np.inf); # bounds for DC
    if onsetTransient is not None:
      input_mat[1,:] = onsetTransient #np.divide(full_transient, np.max(full_transient));
      start_ind = 2;
      lower_bounds.append(0); upper_bounds.append(np.inf); # bounds for transient (ensure it's >= 0)
    else:
      start_ind = 1;
    ### -- get the cos/sin?
    samps = np.linspace(0,stimDur, len(psth)) 
    cos_samp = lambda f: np.cos(2*np.pi*f*samps)
    sin_samp = lambda f: np.sin(2*np.pi*f*samps)
    for i,tf in enumerate(tfs):
      input_mat[start_ind + 2*i, :] = cos_samp(tf);
      input_mat[start_ind + 2*i+1, :] = -sin_samp(tf); # why negative? That's how np.fft has it, based on their coefficients...
      lower_bounds.append(-np.inf); upper_bounds.append(np.inf); # cos/sin coefficients are unbounded
      lower_bounds.append(-np.inf); upper_bounds.append(np.inf); # cos/sin coefficients are unbounded

    input_mat = np.transpose(input_mat);
    coeffs = opt.lsq_linear(input_mat, psth, bounds=(tuple(lower_bounds), tuple(upper_bounds)));
    sampFreq = len(psth)/stimDur; # how many samples per second?

    # Get the coefficients, and pack them as Fourier coefficients
    true_coeffs = coeffs['x']*sampFreq;
    asFFT = np.zeros((1+len(tfs), 1), dtype='complex');
    asFFT[0] = true_coeffs[0];
    amplitudes = np.zeros_like(asFFT, dtype='float32');
    amplitudes[0] = true_coeffs[0]; # FFT coefficient here as same as amplitude...
    for i,tf in enumerate(tfs):
      asFFT[1+i] = true_coeffs[start_ind + 2*i] + 1j* true_coeffs[start_ind + 2*i+1]; # * j to ensure the complex component...
      amplitudes[1+i] = np.abs(asFFT[1+i]);

    return input_mat, coeffs['x'], asFFT, amplitudes;

def fft_amplitude(fftSpectrum, stimDur):
    ''' given an fftSpectrum (and assuming all other normalization has taken place), we double the non-DC frequencies and return
        only the DC and positive frequencies; we also convert these values into rates (i.e. spikes or power per second)

        normalization: since this code is vectorized, the length of each signal passed into np.fft.fft is 1
        i.e. an array whose length is one, with the array[0] having length stimDur/binWidth
        Thus, the DC amplitude is already correct, i.e. not in need of normalization by nSamples

        But, remember that for a real signal like this, non-DC amplitudes need to be doubled - we take care of that here       
    '''
    nyquist = [numpy.int(len(x)/2) for x in fftSpectrum];
    correctFFT = [];
    for i, spect in enumerate(fftSpectrum):
      currFFT = numpy.abs(spect[0:nyquist[i]+1]); # include nyquist
      currFFT[1:nyquist[i]+1] = 2*currFFT[1:nyquist[i]+1];
      # note the divison by stimDur; our packaging of the psth when we call np.fft.fft means that the baseline is each trial
      # is one second; i.e. amplitudes are rates IFF stimDur = 1; here, we divide by stimDur to ensure all trials/psth
      # are true rates
      currFFT = numpy.divide(currFFT, stimDur);
      correctFFT.append(currFFT);
    
    return correctFFT;   

def spike_fft(psth, tfs = None, stimDur = None, binWidth=1e-3):
    ''' given a psth (and optional list of component TFs), compute the fourier transform of the PSTH
        if the component TFs are given, return the FT power at the DC, and at all component TFs
        NOTE: spectrum, rel_amp are rates (spks/s)
              full_fourier is unprocessed, in that regard
        
        normalization: since this code is vectorized, the length of each signal passed into np.fft.fft is 1
        i.e. an array whose length is one, with the array[0] having length stimDur/binWidth
        Thus, the DC amplitude is already correct, i.e. not in need of normalization by nSamples
        But, remember that for a real signal like this, non-DC amplitudes need to be doubled - we take care of that here 
        
        todo (make this--> happen) note: if only one TF is given, also return the power at f2 (i.e. twice f1, the stimulus frequency)
    '''
    np = numpy;

    full_fourier = [np.fft.fft(x) for x in psth];
    spectrum = fft_amplitude(full_fourier, stimDur);

    if tfs is not None:
      try:
        tf_as_ind = tf_to_ind(tfs, stimDur); # if 1s, then TF corresponds to index; if stimDur is 2 seconds, then we can resolve half-integer frequencies -- i.e. 0.5 Hz = 1st index, 1 Hz = 2nd index, ...; CAST to integer
        rel_amp = np.array([spectrum[i][tf_as_ind[i]] for i in range(len(tf_as_ind))]);
      except:
        warnings.warn('In spike_fft: if accessing power at particular frequencies, you must also include the stimulation duration!');
        rel_amp = [];
    else:
      rel_amp = [];

    return spectrum, rel_amp, full_fourier;

def compute_f1_byTrial(cellStruct, expInd, whichSpikes=1, binWidth=1e-3):
  ''' In the move to using the model code from model_responses.py to model_responses_pytorch.py, it's important
      to have a simple way to get [COMPLEX/VECTOR ]F1 responses per stimulus component, per trial
      This function does precisely that, returning the F1 response amp, phase as [nTr x nComp], each
      - by default (whichSpikes=1), we use the sorted spike times, if available

      Use: Will be used in conjunction with the adjust_f1_byTrial
  '''
  np = numpy;

  stimDur = get_exp_params(expInd).stimDur;
 
  if 'sfm' in cellStruct: 
    data = cellStruct['sfm']['exp']['trial'];
  else: # we've passed in sfm.exp.trial already
    data = cellStruct;

  # first, make the PSTH for each trial (only extract output [0], which is the psth)
  if whichSpikes == 1:
    try:
      psth = make_psth(data['spikeTimesGLX']['spikeTimes'], binWidth, stimDur)[0]; # CORRECTED ON 21.08.26
    except:
      whichSpikes = 0;
  if whichSpikes == 0:
    psth = make_psth(data['spikeTimes'], binWidth, stimDur)[0]; # CORRECTED ON 21.08.26
  # then, get the stimulus TF values
  all_tf = np.vstack(data['tf']);
  # with this, we can extract the F1 rates at those TF values
  stimDur = get_exp_params(expInd).stimDur;
  amps, _, full_fourier = spike_fft(psth, tfs = all_tf.transpose(), stimDur=stimDur);
  # get the phase of the response
  tf_as_ind = tf_to_ind(all_tf.transpose(), stimDur);
  resp_phase = np.array([np.angle(full_fourier[x][tf_as_ind[x]], True) for x in range(len(full_fourier))]); # true --> in degrees
  resp_amp = np.array([amps[tf_as_ind[ind]] for ind, amps in enumerate(amps)]); # after correction of amps (19.08.06)

  return resp_amp, resp_phase;

def adjust_f1_byTrial(cellStruct, expInd, dir=-1, whichSpikes=1, binWidth=1e-3, toSum=0):
  ''' Correct the F1 ampltiudes for each trial (in order) by: [akin to hf_sfBB.adjust_f1_byTrial)
      - Projecting the full, i.e. (r, phi) FT vector onto the (vector) mean phase
        across all trials of a given condition
      - NOTE: Will not work with expInd == 1, since we don't have integer cycles for computing F1, anyway
      - by default (whichSpikes=1), we use the sorted spike times, if available
      - if toSum, set the value for blank components to 0 and add up across components
      Return: return [nTr, nComp] of scalar F1 rates after vector adjustment
              OR     nTr of scalar rates if toSum==1
  '''
  np = numpy;
  conDig = 3; # round contrast to the thousandth

  if expInd == 1:
    warnings.warn('This function does not work with expInd=1, since that experiment does not have integer cycles for\
                   each stimulus component; thus, we will not analyze the F1 responses, anyway');
    return None;

  if 'sfm' in cellStruct: 
    data = cellStruct['sfm']['exp']['trial'];
  else: # we've passed in sfm.exp.trial already
    data = cellStruct;

  # 0. Get the r, phi for all trials (i.e. amplitude & phase) - and the stimulus phases
  r_byTrial, phi_byTrial = compute_f1_byTrial(data, expInd, whichSpikes, binWidth);
  stimPhase = np.vstack(data['ph']).transpose(); # [nTr x nComp], in degrees

  # 1. Get all possible stimulus conditions to cycle through

  # - set up the array for responses
  nTr = len(data['num']);
  nComps = get_exp_params(expInd).nStimComp;
  adjusted_f1_rate = np.nan * np.zeros((nTr, nComps));
  # - get the conditions so that we can use get_valid_trials quickly
  _, conds, _, val_by_stim_val, _ = tabulate_responses(data, expInd);

  nDisps = len(conds[0]);
  nCons = len(conds[1]);
  nSfs = len(conds[2]);

  for d in range(nDisps):
    for con in range(nCons):
      for sf in range(nSfs):
        val_trials = get_valid_trials(data, d, con, sf, expInd, stimVals=conds, validByStimVal=val_by_stim_val)[0][0];

        if np.all(np.unique(val_trials) == False):
          continue;

        phase_rel_stim = np.mod(np.multiply(dir, np.add(phi_byTrial[val_trials, :], stimPhase[val_trials, :])), 360);
        r_mean, phi_mean, r_sem, phi_var = polar_vec_mean(r_byTrial[val_trials, :].transpose(), phase_rel_stim.transpose(), sem=1); # transpose to ensure we get the r/phi average across components, not trials
        # finally, project the response as usual
        if np.any(np.isnan(phi_mean)): # hacky way of saying that there were no spikes! hence phi is undefined
          resp_proj = r_byTrial[val_trials, :]; # just set it equal to r value, which will be zero anyway
        else:
          resp_proj = np.multiply(r_byTrial[val_trials, :], np.cos(np.deg2rad(phi_mean) - np.deg2rad(phase_rel_stim)));

        adjusted_f1_rate[val_trials, :] = resp_proj;
          

  if toSum:
    # then, sum up the valid components per stimulus component
    allCons = np.vstack(data['con']).transpose();
    blanks = np.where(allCons==0);
    adjByTrialSum = np.copy(adjusted_f1_rate);
    adjByTrialSum[blanks] = 0; # just set it to 0 if that component was blank during the trial
    adjusted_f1_rate = np.sum(adjByTrialSum, axis=1);

  return adjusted_f1_rate;

def compute_f1f0(trial_inf, cellNum, expInd, loc_data, descrFitName_f0=None, descrFitName_f1=None):
  ''' Using the stimulus closest to optimal in terms of SF (at high contrast), get the F1/F0 ratio
      This will be used to determine simple versus complex
      Note that descrFitName_f1 is optional, i.e. we needn't pass this in
      
      As of 19.09.24, we can avoid passing in descrFitName at all, since we also manually calculate peak f0 and f1 SF, too
  '''
  np = numpy;

  ######
  # why are we keeping the trials with max response at F0 (always) and F1 (if present)? Per discussion with Tony, 
  # we should evaluate F1/F0 at the SF  which has the highest response as determined by comparing F0 and F1, 
  # i.e. F1 might be greater than F0 AND have a different than F0 - in the case, we ought to evalaute at the peak F1 frequency
  ######
  ## first, get F0 responses
  f0_counts = get_spikes(trial_inf, get_f0=1, expInd=expInd);
  f0_blank = blankResp(trial_inf, expInd)[0]; # we'll subtract off the f0 blank mean response from f0 responses
  stimDur = get_exp_params(expInd).stimDur;
  f0rates = np.divide(f0_counts - f0_blank, stimDur);
  f0rates_org = organize_resp(f0rates, trial_inf, expInd, respsAsRate=True)[2];
  ## then, get F1 - TODO: why cannot use get_spikes? what is saved in F1 of each structure? seems to be half of true F1 value
  all_trs = np.arange(trial_inf['num'][-1]); # i.e. all the trials
  spike_times = np.array([trial_inf['spikeTimes'][x] for x in all_trs]);
  psth, bins = make_psth(spike_times, stimDur=stimDur);
  all_tf = np.array([trial_inf['tf'][0][val_tr] for val_tr in all_trs]); # just take first grating (only will ever analyze single gratings)

  f1rates = adjust_f1_byTrial(trial_inf, expInd, toSum=1)
  if f1rates is None: # for V1_orig/, we'll compute the old way, since adjust_f1_byTrial will return None
    power, rel_power, full_ft = spike_fft(np.array(psth), tfs=all_tf, stimDur=stimDur);
    f1rates = rel_power; # f1 is already a rate (i.e. spks [or power] / sec); just unpack

  f1rates_org = organize_resp(f1rates, trial_inf, expInd, respsAsRate=True)[2];
  rates_org = [f0rates_org, f1rates_org];

  # get prefSfEst from f0 descrFits - NOTE: copied code from model_respsonses.set_model -- reconsider as a hf?
  f0f1_dfits = [descrFitName_f0, descrFitName_f1];
  prefSfEst = np.nan * np.zeros((len(f0f1_dfits), ));
  for i, ft in enumerate(f0f1_dfits):
    if ft is not None:
      dfits = np_smart_load(loc_data + ft);
      if expInd == 1:
        hiCon = 0; # holdover from hf.organize_resp (with expInd==1, sent to V1_orig/helper_fcns.organize_modResp
      else:
        hiCon = -1;
      prefSfEst[i] = dfits[cellNum-1]['prefSf'][0][hiCon]; # get high contrast, single grating prefSf
  # now "trim" prefSfEst (i.e. remove the second entry if dfn_f1 is None)
  prefSfEst = prefSfEst[~np.isnan(prefSfEst)];
  man_prefSfEst = np.array([np.argmax(resps[0, :, -1]) for resps in rates_org]); # get peak resp for f0 and f1
  _, stimVals, val_con_by_disp, val_byTrial, _ = tabulate_responses(trial_inf, expInd);
  all_sfs = stimVals[2];
  man_prefSfEst = all_sfs[man_prefSfEst];
  prefSfEst = np.hstack((prefSfEst, man_prefSfEst));

  ### now, figure out which SF value to evaluate at, get corresponding trials
  sf_match_inds = [np.argmin(np.square(all_sfs - psfEst)) for psfEst in prefSfEst]; # matching inds
  disp = 0; con = val_con_by_disp[disp][-1]; # i.e. highest con, single gratings
  val_trs = [get_valid_trials(trial_inf, disp=disp, con=con, sf=match_ind, expInd=expInd)[0][0] for match_ind in sf_match_inds]; # unpack - first 0 for first output argument, 2nd one to unpack into array rather than list of array(s)
  stimDur = get_exp_params(expInd).stimDur;

  ######
  # make the comparisons 
  ######
  f0_subset = [f0rates[val_tr] for val_tr in val_trs];
  f1_subset = [f1rates[val_tr] for val_tr in val_trs];
  f0f1_resps = [f0_subset, f1_subset];
  # now, we'll find out which of F0 or F1 peak inds has highest response for F0 and F1 separately 
  f0f1_max = [[numpy.nanmean(x) for x in resps] for resps in f0f1_resps]; # between f0 and f1 inds, which gives higher response?
  f0f1_ind = [np.argmax(x) for x in f0f1_max]; # and get the corresponding index of that highest response
  # finally, figure out which of the peakInd X F0/F1 combinations has the highest overall response
  peakRespInd = np.argmax([np.nanmean(x[y]) for x,y in zip(f0f1_resps, f0f1_ind)]);
  indToAnalyze = f0f1_ind[peakRespInd];

  f0rate, f1rate = [x[indToAnalyze] for x in f0f1_resps];
  # note: the below lines will help us avoid including trials for which the f0 is negative (after baseline subtraction!)
  # i.e. we will not include trials with below-baseline f0 responses in our f1f0 calculation (or negative f1)
  rate_posInd = np.where(np.logical_and(f0rate>0, f1rate>0));
  f0rate_pos = f0rate[rate_posInd];
  f1rate_pos = f1rate[rate_posInd];

  return np.nanmean(np.divide(f1rate_pos, f0rate_pos)), f0rate, f1rate, f0_counts, f1rates;

##################################################################
##################################################################
##################################################################
## III. PHASE/MORE PSTH
##################################################################
##################################################################
##################################################################

def project_resp(amp, phi_resp, phAdv_model, phAdv_params, disp, allCompSf=None, allSfs=None):
  ''' Using our model fit of (expected) response phase as a function of response amplitude, we can
      determine the difference in angle between the expected and measured phase and then project the
      measured response vector (i.e. amp/phase in polar coordinates) onto the expected phase line
      eq: adjResp = measuredResp * cos(expPhi - measuredPhi)
      vectorized: expects/returns lists of amplitudes/phis
  '''
  np = numpy;
  sfDig = 2; # round SFs to the hundredth  when comparing for equality
  all_proj = [];

  for i in range(len(amp)):
    if disp == 0:
      if amp[i] == []: # this shouldn't ever happen for single gratings, but just in case...
        continue;
      # why list comprehension with numpy array around? we want numpy array as output, but some of the 
      # sub-arrays (i.e. amp[i] is list of lists, or array of arrays) are of unequal length, so cannot
      # just compute readily 
      phi_true = np.array([phAdv_model(phAdv_params[i][0], phAdv_params[i][1], x) for x in amp[i]]);
      proj = np.array([np.multiply(amp[i][j], np.cos(np.deg2rad(phi_resp[i][j])-np.deg2rad(phi_true[j]))) for j in range(len(amp[i]))]);
      #proj = np.multiply(amp[i], np.cos(np.deg2rad(phi_resp[i])-np.deg2rad(phi_true)))
      all_proj.append(proj);
    elif disp > 0: # then we'll need to use the allCompSf to get the right phase advance fit for each component
      if amp[i] == []: # 
        all_proj.append([]);
        continue;
      # now, for each valid amplitude, there are responses for each component for each total stim contrast
      all_proj.append([]);
      for con_ind in range(len(amp[i])):
        curr_proj_con = [];
        for comp_ind in range(len(amp[i][con_ind])):
          curr_amp = amp[i][con_ind][comp_ind];
          curr_phi = phi_resp[i][con_ind][comp_ind];
          # now, for that component, find out the SF and get the right phase advance fit
          # note: where is array, so unpack one level to get  
          #   adjusted on 19.08.29 to find difference l.t. 0.02 rather than equality (rounding issues; came up first with V1/, not in LGN/ data)
          sf_ind = np.where(np.abs(np.round(allSfs, sfDig) - np.round(allCompSf[i][con_ind][comp_ind], sfDig))<0.02)[0][0];
          phi_true = phAdv_model(phAdv_params[sf_ind][0], phAdv_params[sf_ind][1], curr_amp);
          if isinstance(phi_true, np.ndarray): # i.e. array
            if isinstance(phi_true[0], np.ndarray): # i.e. nested array
            # flatten into array of numbers rather than array of arrays (of one number) 
              phi_true = phi_true.flatten(); # should be flatten_list? was previously flatten(x), but likely uncalled!
          # finally, project the response as usual
          proj = np.multiply(curr_amp, np.cos(np.deg2rad(curr_phi)-np.deg2rad(phi_true)));
          curr_proj_con.append(proj);
        all_proj[i].append(curr_proj_con);
  
  return all_proj;

def project_resp_cond(data, disp, expInd, con, sf, phAdv_model, phAdv_params, dir=-1):
  ''' NOTE: Not currently used, incomplete... 11.01.18
      Input: data structure, disp/con/sf (as indices, relative to the list of con/sf for that dispersion)
      Using our model fit of (expected) response phase as a function of response amplitude, we can
      determine the difference in angle between the expected and measured phase and then project the
      measured response vector (i.e. amp/phase in polar coordinates) onto the expected phase line
      eq: adjResp = measuredResp * cos(expPhi - measuredPhi)
  '''
  val_trials, allDisps, allCons, allSfs = get_valid_trials(data, disp, con, sf, expInd);

  if not numpy.any(val_trials[0]): # val_trials[0] will be the array of valid trial indices --> if it's empty, leave!
    warnings.warn('this condition is not valid');
    return [];

  allAmp, allPhi, _, allCompCon, allCompSf = get_all_fft(data, disp, expInd, dir=dir, all_trials=1);
  ampByTrial = allAmp[sf][con][2];
  phiByTrial = allPhi[sf][con][2];

  adjAmpAll = project_resp([ampByTrial], [phiByTrial], phAdv_model, [phAdv_params[sf]], disp, allCompSf, allSfs)[0];
  adjAmpMean = numpy.mean(adjAmpAll);
  adjAmpSem = sem(adjMeans);

  allPhiMean = numpy.mean(allPhi);
  allPhiSem = sem(allPhi);

  return [adjAmpAll, adjAmpMean, adjAmpSem], [allPhi, allPhiMean, allPhiSem];
  
def first_ph0(start_phase, stim_tf, dir=-1):
    ''' returns fraction of cycle until ph=0 and time until ph=0 
    use this function to determine how much of the cycle needs to be completed before the phase reaches 0 again
    given the start phase of the stimulus --> the same comments in the section with "get_phase" above 
    explain why we simply take the start phase as an indicator of the "cycle-distance until ph=0"
    if dir = -1, then we assume we have "start_phase" degrees to go before ph = 0
    if dir = 1, then we have 360-"start_phase" deg to go before ph = 0
    '''
    if dir == -1:
      cycle_until_ph0 = numpy.mod(start_phase, 360.0)/360.0;
    if dir == 1:
       cycle_until_ph0 = numpy.mod(numpy.subtract(360, start_phase), 360.0)/360.0;
    stim_period = numpy.divide(1.0, stim_tf); # divide by 1.0 so that stimPeriod is a float (and not just an int!)
    time_until_ph0 = cycle_until_ph0 * stim_period;
    return cycle_until_ph0, time_until_ph0;

def fold_psth(spikeTimes, stimTf, stimPh, n_cycles, n_bins, dir=-1):
    ''' Returns the folded_psth (bin counts), the bin edges of the folded psth, and a normalized folded psth
        The psth is centered relative to the 0 phase of the stimulus cycle
        Compute the psth and fold over a given number of cycles, with a set number of bins per cycle 
        For now, works only for single components...
    '''
    np = numpy;

    stimPeriod = np.divide(1.0, stimTf); # divide by 1.0 so that stimPeriod is a float (and not just an int!)
    _, ph0 = first_ph0(stimPh, stimTf, dir);
    folded = np.mod(spikeTimes-ph0[0], np.multiply(n_cycles, stimPeriod[0])); # center the spikes relative to the 0 phase of the stim
    bin_edges = np.linspace(0, n_cycles*stimPeriod[0], 1+n_cycles*n_bins);
    psth_fold = np.histogram(folded, bin_edges, normed=False)[0];
    psth_norm = np.divide(psth_fold, np.max(psth_fold));
    return psth_fold, bin_edges, psth_norm;

def get_true_phase(data, val_trials, expInd, dir=-1, psth_binWidth=1e-3):
    ''' Returns resp-phase-rel-to-stim, stimulus phase, response phase, and stimulus tf
        Given the data and the set of valid trials, first compute the response phase
        and stimulus phase - then determine the response phase relative to the stimulus phase
    '''
    np = numpy;

    exper = get_exp_params(expInd);
    stimDur = exper.stimDur;
    nComps  = exper.nStimComp;

    # get mask for tfs/phi - we only take components with con > 0
    all_cons = np.vstack([data['con'][i] for i in range(nComps)]);
    all_cons = np.transpose(all_cons)[val_trials];
    con_mask = np.ma.masked_greater(all_cons, 0);

    # prepare the TF information for each component - we know there are N components per stimulus
    all_tfs = np.vstack([data['tf'][i] for i in range(nComps)]);
    all_tfs = np.transpose(all_tfs)[val_trials];
    all_tf = [all_tfs[i, con_mask.mask[i, :]] for i in range(con_mask.mask.shape[0])] # mask shape (and alltfs/cons) is [nTrials x nCons]

    # and the phase...
    all_phis = np.vstack([data['ph'][i] for i in range(nComps)]);
    all_phis = np.transpose(all_phis)[val_trials];
    # only get PHI for the components we need - use the length of all_tf as a guide
    stim_phase = [all_phis[i, con_mask.mask[i, :]] for i in range(con_mask.mask.shape[0])] # mask shape (and alltfs/cons) is [nTrials x nCons]

    # perform the fourier analysis we need
    psth_val, _ = make_psth(data['spikeTimes'][val_trials], psth_binWidth, stimDur)
    all_amp, rel_amp, full_fourier = spike_fft(psth_val, all_tf, stimDur)
    # and finally get the stimulus-relative phase of each response
    tf_as_ind  = tf_to_ind(all_tf, stimDur);
    resp_phase = [np.angle(full_fourier[x][tf_as_ind[x]], True) for x in range(len(full_fourier))]; # true --> in degrees
    resp_amp = [amps[tf_as_ind[ind]] for ind, amps in enumerate(all_amp)]; # after correction of amps (19.08.06)
    #resp_amp = [np.abs(full_fourier[x][tf_as_ind[x]]) for x in range(len(full_fourier))]; # before correction of amps (19.08.06)
    phase_rel_stim = np.mod(np.multiply(dir, np.add(resp_phase, stim_phase)), 360);

    return phase_rel_stim, stim_phase, resp_phase, all_tf;

def polar_vec_mean(amps, phases, sem=0):
   ''' Given a set of amplitudes ("r") and phases ("theta"; in degrees) for a given stimulus condition (or set of conditions)
       RETURN the mean amplitude and phase (in degrees) computed by vector summation/averaging
       Note: amps/phases must be passed in as arrays of arrays, so that we can compute the vec mean for multiple different
             stimulus conditions just by calling this function once
       --- IF sem=1, then we give s.e.m. rather than std for "r"
   '''
   np = numpy;
  
   n_conds = len(amps);
   if len(phases) != n_conds:
     print('the number of conditions in amps is not the same as the number of conditions in phases --> giving up');
     return [], [], [], [];

   def circ_var(deg_phi): # compute and return a measure of circular variance [0, 1]
     s = np.sum(np.sin(np.deg2rad(deg_phi)));
     c = np.sum(np.cos(np.deg2rad(deg_phi)));
     return 1 - np.sqrt(np.square(s) + np.square(c))/len(deg_phi);

   all_r = []; all_phi = [];
   all_r_std = []; all_phi_var = [];

   for cond in range(n_conds):
     curr_amps = amps[cond];
     curr_phis = phases[cond];

     n_reps = len(curr_amps);
     # convert each amp/phase value to x, y
     [x_polar, y_polar] = [curr_amps*np.cos(np.radians(curr_phis)), curr_amps*np.sin(np.radians(curr_phis))]
     # take the mean/std
     x_avg, y_avg = [np.mean(x_polar), np.mean(y_polar)]
     x_std, y_std = [np.std(x_polar), np.std(y_polar)]
     # now compute (and return) r and theta
     r = np.sqrt(np.square(x_avg) + np.square(y_avg));
     r_std = np.sqrt(np.square(x_std) + np.square(y_std));
     if sem:
        r_std = r_std/len(x_polar); # now it's not really "std" --> it's s.e.m.
     # now the angle
     theta = angle_xy([x_avg], [y_avg])[0]; # just get the one value (will be packed in list)
     theta_var = circ_var(curr_phis); # compute on the original phases

     all_r.append(r);
     all_phi.append(theta);
     all_r_std.append(r_std);
     all_phi_var.append(theta_var);

   return all_r, all_phi, all_r_std, all_phi_var;

def get_all_fft(data, disp, expInd, cons=[], sfs=[], dir=-1, psth_binWidth=1e-3, all_trials=0):
  ''' for a given cell and condition or set of conditions, compute the mean amplitude and phase
      also return the temporal frequencies which correspond to each condition
      if all_trials=1, then return the individual trial responses (i.e. not just avg over all repeats for a condition)
  '''
  stimDur = get_exp_params(expInd).stimDur;

  _, stimVals, val_con_by_disp, validByStimVal, _ = tabulate_responses(data, expInd);

  # gather the sf indices in case we need - this is a dictionary whose keys are the valid sf indices
  valSf = validByStimVal[2];

  if cons == []: # then get all valid cons for this dispersion
    cons = val_con_by_disp[disp];
  if sfs == []: # then get all valid sfs for this dispersion
    sfs = list(valSf.keys());

  all_r = []; all_ph = []; all_tf = []; 
  # the all_..Comp will be used only if disp=1 (i.e. mixture stimuli)
  all_conComp = []; all_sfComp = [];

  for s in sfs:
    curr_r = []; curr_ph = []; curr_tf = [];
    curr_conComp = []; curr_sfComp = [];
    for c in cons:
      val_trials, allDisps, allCons, allSfs = get_valid_trials(data, disp, c, s, expInd, stimVals, validByStimVal);

      if not len(val_trials[0]) > 0: # val_trials[0] will be the array of valid trial indices --> if it's empty, leave!
        warnings.warn('this condition is not valid');
        continue;

      # get the phase relative to the stimulus
      ph_rel_stim, stim_ph, resp_ph, curr_tf = get_true_phase(data, val_trials, expInd, dir, psth_binWidth);
      # compute the fourier amplitudes
      psth_val, _ = make_psth(data['spikeTimes'][val_trials], binWidth=psth_binWidth, stimDur=stimDur);
      _, rel_amp, full_fourier = spike_fft(psth_val, curr_tf, stimDur)

      if disp == 0:
        # compute mean, gather
        avg_r, avg_ph, std_r, std_ph = polar_vec_mean([rel_amp], [ph_rel_stim]);
        if all_trials == 1:
          curr_r.append([avg_r[0], std_r[0], rel_amp, sem(rel_amp)]); # we can just grab 0 element, since we're going one value at a time, but it's packed in array
          curr_ph.append([avg_ph[0], std_ph[0], ph_rel_stim, sem(ph_rel_stim)]); # same as above
        elif all_trials == 0:
          curr_r.append([avg_r[0], std_r[0]]); # we can just grab 0 element, since we're going one value at a time, but it's packed in array 
          curr_ph.append([avg_ph[0], std_ph[0]]); # same as above
        curr_tf.append(curr_tf);
      elif disp>0: # for mixtures
        # need to switch rel_amp to be lists of amplitudes by component (rather than list of amplitudes by trial)
        rel_amp = switch_inner_outer(rel_amp);
        rel_amp_sem = [sem(x) for x in rel_amp];
        # call get_isolated_response just to get contrast/sf per component
        _, _, _, _, conByComp, sfByComp = get_isolated_response(data, val_trials);
        if numpy.array_equal(sfByComp, []):
          pdb.set_trace();
        # need to switch ph_rel_stim (and resp_phase) to be lists of phases by component (rather than list of phases by trial)
        ph_rel_stim = switch_inner_outer(ph_rel_stim);
        ph_rel_stim_sem = [sem(x) for x in ph_rel_stim];

        # compute vector mean, gather/organize
        avg_r, avg_ph, std_r, std_ph = polar_vec_mean(rel_amp, ph_rel_stim);
        if all_trials == 1:
          curr_r.append([avg_r, std_r, rel_amp, rel_amp_sem]);
          curr_ph.append([avg_ph, std_ph, ph_rel_stim, ph_rel_stim_sem])
        elif all_trials == 0:
          curr_r.append([avg_r, std_r]);
          curr_ph.append([avg_ph, std_ph]);
        curr_tf.append(curr_tf);
        curr_conComp.append(conByComp);
        curr_sfComp.append(sfByComp);

    all_r.append(curr_r);
    all_ph.append(curr_ph);
    all_tf.append(curr_tf);
    if disp > 0:
      all_conComp.append(curr_conComp);
      all_sfComp.append(curr_sfComp);

  return all_r, all_ph, all_tf, all_conComp, all_sfComp;

def get_phAdv_model():
  ''' simply return the phase advance model used in the fits
  '''
  # phAdv_model = [numpy.mod(phi0 + numpy.multiply(slope, x), 360) for x in amp] # because the sub-arrays of amp occasionally have
  phAdv_model = lambda phi0, slope, amp: numpy.mod(phi0 + numpy.multiply(slope, amp), 360);
  # must mod by 360! Otherwise, something like 340-355-005 will be fit poorly
  return phAdv_model;

def get_recovInfo(cellStruct, normType):
  ''' Given a cell structure and normalization type, return the (simulated) spikes and model recovery set parameters
  '''
  spks, prms = [], [];
  try:
    base = cellStruct['sfm']['mod']['recovery'];
    if normType == 1: # i.e. flat
      spks = base['respFlat'];
      prms = base['paramsFlat'];
    elif normType == 2: # i.e. weighted
      spks = base['respWght'];
      prms = base['paramsWght'];
    elif normType == 4: # i.e. two-halved, weighted gaussian
      spks = base['respFlex'];
      prms = base['paramsFlex'];
    else:
      warnings.warn('You have not set up an access for model recovery with this normalization type');
  except:
    warnings.warn('You likely do not have a recovery set up for this cell/file');
  return prms, spks;

def phase_advance(amps, phis, cons, tfs, n_repeats=100):
   ''' Given the mean amplitude/phase of responses over a range of contrasts, compute the linear model
       fit which describes the phase advance per unit contrast as described in Eq. 4 of
       Movshon, Kiorpes, Hawken, Cavanaugh; 2005
       RETURNS: phAdv_model (the model equation), the list of the optimal parameters, and the phase advance (in milliseconds)
       "Vectorized" - i.e. accepts arrays of amp/phi arrays
   '''
   np = numpy;

   get_mean = lambda y: [x[0] for x in y]; # for curr_phis and curr_amps, loc [0] is the mean, [1] is variance measure
   get_var = lambda y: [x[1] for x in y]; # for curr_phis and curr_amps, loc [0] is the mean, [1] is variance measure

   phAdv_model = get_phAdv_model()
   all_opts = []; all_loss = []; all_phAdv = [];

   abs_angle_diff = lambda deg1, deg2: np.arccos(np.cos(np.deg2rad(deg1) - np.deg2rad(deg2)));

   for i in range(len(amps)):
     #print('\n#######%d#######\n' % i);
     curr_amps = amps[i]; # amp for each of the different contrast conditions
     curr_ampMean = get_mean(curr_amps);
     curr_phis = phis[i]; # phase for ...
     curr_phiMean = get_mean(curr_phis);
     obj = lambda params: np.sum(np.square(abs_angle_diff(curr_phiMean, phAdv_model(params[0], params[1], curr_ampMean)))); 
     # just least squares...
     #obj = lambda params: np.sum(np.square(curr_phiMean - phAdv_model(params[0], params[1], curr_ampMean))); # just least squares...
     # phi0 (i.e. phase at zero response) --> just guess the phase at the lowest amplitude response
     # slope --> just compute the slope over the response range
     min_resp_ind = np.argmin(curr_ampMean);
     max_resp_ind = np.argmax(curr_ampMean);
     diff_sin = np.arcsin(np.sin(np.deg2rad(curr_phiMean[max_resp_ind]) - np.deg2rad(curr_phiMean[min_resp_ind])));
     init_slope = (np.rad2deg(diff_sin))/(curr_ampMean[max_resp_ind]-curr_ampMean[min_resp_ind]);
     init_params = [random_in_range([0.8, 1.2])[0]*curr_phiMean[min_resp_ind], random_in_range([0.5, 1.5])[0]*init_slope];
     best_loss = np.nan; best_params = [];
     for rpt in range(n_repeats):
        to_opt = opt.minimize(obj, init_params);
        if np.isnan(best_loss) or to_opt['fun'] < best_loss:
           best_loss = to_opt['fun'];
           best_params = to_opt['x'];
     opt_params = best_params;
     opt_loss = best_loss;
     #print(opt_params);
     all_opts.append(opt_params);
     all_loss.append(opt_loss);

     # now compute phase advance (in ms)
     curr_cons = cons[i];
     curr_tfs = tfs[i][0];
     #curr_sfs = sfs[i]; # TODO: Think about using the spatial frequency in the phase_adv calculation - if [p] = s^2/cycles, then we have to multiply by cycles/deg?
     cycle_fraction = opt_params[1] * curr_ampMean[max_resp_ind] / 360; # slope*respAmpAtMaxCon --> phase shift (in degrees) from 0 to responseAtMaxCon
     # then, divide this by 360 to get fractions of a cycle
     #phase_adv = 1e3*opt_params[1]/curr_tfs[0]; # get just the first grating's TF...
     phase_adv = 1e3*cycle_fraction/curr_tfs[0]; # get just the first grating's TF...
     # 1e3 to get into ms;

     all_phAdv.append(phase_adv);

   return phAdv_model, all_opts, all_phAdv, all_loss;

def tf_to_ind(tfs, stimDur):
  ''' simple conversion from temporal frequency to index into the fourier spectrum 
      we simply cast the result to integer, though this is not quite right for older versions
      of the experiment with non-integer number of stimulus cycles

      Note that we multiply, then round, and finally cast - if we simply cast right after the multiply,
      then decimals are just truncated; for older experiments, we want 5.8, e.g., to be treated as 6, not 5!
  '''
  try: # if tfs is an array, then we do it this way...
    return [numpy.round(numpy.multiply(tf, stimDur)).astype(numpy.int16) for tf in tfs];
  except: # otherwise, just the simple way
    return numpy.round(numpy.multiply(tfs, stimDur)).astype(numpy.int16);


##################################################################
##################################################################
##################################################################
### descriptive fits to sf tuning/basic data analyses
### IV. Descriptive functions - fits to spatial frequency tuning, other related calculations
##################################################################
##################################################################
##################################################################

def get_rvc_model():
  ''' simply return the rvc model used in the fits (type 0; should be used only for LGN)
      --- from Eq. 3 of Movshon, Kiorpes, Hawken, Cavanaugh; 2005
  '''
  rvc_model = lambda b, k, c0, cons: b + k*numpy.log(1+numpy.divide(cons, c0));

  return rvc_model;

def naka_rushton(con, params):
    ''' this is the classic naka rushton form of RVC - type 1
        but, if incl. optional 5th param "s", this is the 2007 Peirce super-saturating RVC (type 2)
    '''
    np = numpy;
    base = params[0];
    gain = params[1];
    expon = params[2];
    c50 = params[3];
    if len(params) > 4: # optionally, include "s" - the super-saturating parameter from Peirce, JoV (2007)
      sExp = params[4];
    else:
      sExp = 1; # otherwise, it's just 1

    return base + gain*np.divide(np.power(con, expon), np.power(con, expon*sExp) + np.power(c50, expon*sExp));

def rvc_fit(amps, cons, var = None, n_repeats = 100, mod=0, fix_baseline=False, prevFits=None):
   ''' Given the mean amplitude of responses (by contrast value) over a range of contrasts, compute the model
       fit which describes the response amplitude as a function of contrast as described in Eq. 3 of
       Movshon, Kiorpes, Hawken, Cavanaugh; 2005
       Optionally, can include a measure of variability in each response to perform weighted least squares
       Optionally, can include mod = 0 (as above) or 1 (Naka-Rushton) or 2 (Peirce 2007 modification of Naka-Rushton)
       RETURNS: rvc_model (the model equation), list of the optimal parameters, and the contrast gain measure
       Vectorized - i.e. accepts arrays of amp/con arrays
   '''

   np = numpy;

   rvc_model = get_rvc_model(); # only used if mod == 0
   
   all_opts = []; all_loss = [];
   all_conGain = [];
   n_amps = len(amps);
  
   for i in range(n_amps):
     curr_amps = amps[i];
     curr_cons = cons[i];
     
     if curr_amps == [] or curr_cons == []:
       # nothing to do - set to blank and move on
       all_opts.append([]);
       all_loss.append([]);
       all_conGain.append([]);
       continue;

     if var:
       loss_weights = np.divide(1, var[i]);
       if np.any(np.isnan(loss_weights)):
         loss_weights = np.ones_like(curr_amps); # if we have NaN, then we ignore the var...
     else:
       loss_weights = np.ones_like(curr_amps);
     if mod == 0:
       obj = lambda params: np.sum(np.multiply(loss_weights, np.square(curr_amps - rvc_model(params[0], params[1], params[2], curr_cons))));
     elif mod == 1:
       obj = lambda params: np.sum(np.multiply(loss_weights, np.square(curr_amps - naka_rushton(curr_cons, params))));
     elif mod == 2: # we also want a regularization term for the "s" term
       lam1 = 5; # lambda parameter for regularization
       obj = lambda params: np.sum(np.multiply(loss_weights, np.square(curr_amps - naka_rushton(curr_cons, params)))) + lam1*(params[-1]-1); # params[-1] is "sExp"

     if prevFits is None:
       best_loss = 1e6; # start with high value
       best_params = []; conGain = [];
     else: # load the previous best_loss/params/conGain
       best_loss = prevFits['loss'][i];
       best_params = prevFits['params'][i];
       conGain = prevFits['conGain'][i];

     for rpt in range(n_repeats):

       if mod == 0:
         if fix_baseline:
           b_rat = 0;
         else:
           b_rat = random_in_range([0.0, 0.2])[0];
         # for mod==0, params are [b {offset}, k {gain}, c0 {c50}]
         k_random_factor = random_in_range([0.5,2])[0];
         init_params = [b_rat*np.max(curr_amps), k_random_factor*(2+3*b_rat)*np.max(curr_amps), random_in_range([0.05, 0.5])[0]]; 
         if fix_baseline:
           b_bounds = (0, 0);
         else:
           b_bounds = (None, 0);
         k_bounds = (0, None);
         c0_bounds = (5e-3, 1);
         all_bounds = (b_bounds, k_bounds, c0_bounds); # set all bounds
       elif mod == 1 or mod == 2: # bad initialization as of now...
         if fix_baseline: # correct if we're fixing the baseline at 0
           i_base = 0;
         else:
           i_base = np.min(curr_amps) + random_in_range([-2.5, 2.5])[0];
         i_gain = random_in_range([0.8, 1.2])[0] * (np.max(curr_amps) - i_base);
         i_expon = random_in_range([1, 2])[0];
         i_c50 = random_in_range([0.05, 0.6])[0];
         i_sExp = 1;
         init_params = [i_base, i_gain, i_expon, i_c50, i_sExp];
         if fix_baseline:
           b_bounds = (0, 0);
         else:
           b_bounds = (None, None);
         g_bounds = (0, None);
         e_bounds = (0.75, None);
         c_bounds = (0.01, 1);
         if mod == 1:
           s_bounds = (1, 1);
         elif mod == 2:
           s_bounds = (1, 2); # for now, but can adjust as needed (TODO)
         all_bounds = (b_bounds, g_bounds, e_bounds, c_bounds, s_bounds);
       # now optimize
       try:
         to_opt = opt.minimize(obj, init_params, bounds=all_bounds);
       except:
         continue; # this fit failed; try again
       opt_params = to_opt['x'];
       opt_loss = to_opt['fun'];

       if opt_loss > best_loss and ~np.isnan(best_loss):
         continue;
       else:
         best_loss = opt_loss;
         best_params = opt_params;

       # now determine the contrast gain
       if mod == 0:
         b = opt_params[0]; k = opt_params[1]; c0 = opt_params[2];
         if b < 0: 
           # find the contrast value at which the rvc_model crosses/reaches 0
           obj_whenR0 = lambda con: np.square(0 - rvc_model(b, k, c0, con));
           con_bound = (0, 1);
           init_r0cross = 0;
           r0_cross = opt.minimize(obj_whenR0, init_r0cross, bounds=(con_bound, ));
           con_r0 = r0_cross['x'];
           conGain = k/(c0*(1+con_r0/c0));
         else: # i.e. if b = 0
           conGain = k/c0;
       else:
         conGain = -100;

     all_opts.append(best_params);
     all_loss.append(best_loss);
     all_conGain.append(conGain);

   return rvc_model, all_opts, all_conGain, all_loss;

def DiffOfGauss(gain, f_c, gain_s, j_s, stim_sf, baseline=0, computeNorm=0):
  ''' Difference of gaussians - as formulated in Levitt et al, 2001
  gain      - overall gain term
  f_c       - characteristic frequency of the center, i.e. freq at which response is 1/e of maximum
  gain_s    - relative gain of surround (e.g. gain_s of 0.5 says peak surround response is half of peak center response
  j_s       - relative characteristic freq. of surround (i.e. char_surround = f_c * j_s)
  --- Note that if baseline is non-zero, we'll add this to the response but it is NOT optimized, as of 21.05.03
  '''
  np = numpy;
  if computeNorm == 1:
    dog = lambda f: baseline + np.maximum(0, gain*(np.exp(-np.square(f/f_c)) - gain_s * np.exp(-np.square(f/(f_c*j_s))))); 

    norm = np.max(dog(stim_sf));

    dog_norm = lambda f: dog(f) / norm;
    return dog(stim_sf), dog_norm(stim_sf);
  else:
    tune = baseline + np.maximum(0, gain*(np.exp(-np.square(stim_sf/f_c)) - gain_s * np.exp(-np.square(stim_sf/(f_c*j_s))))); 
    return tune, [];

def DoGsach(gain_c, r_c, gain_s, r_s, stim_sf, baseline=0, computeNorm=0):
  ''' Difference of gaussians as described in Sach's thesis
  [0] gain_c    - gain of the center mechanism
  [1] r_c       - radius of the center
  [2] gain_s    - gain of surround mechanism
  [3] r_s       - radius of surround
  --- Note that if baseline is non-zero, we'll add this to the response but it is NOT optimized, as of 21.05.03
  '''
  np = numpy;
  if computeNorm == 1:
    dog = lambda f: baseline + np.maximum(0, gain_c*np.pi*np.square(r_c)*np.exp(-np.square(f*np.pi*r_c)) - gain_s*np.pi*np.square(r_s)*np.exp(-np.square(f*np.pi*r_s)));

    norm = np.max(dog(stim_sf));
    dog_norm = lambda f: dog(f) / norm;

    return dog(stim_sf), dog_norm(stim_sf);
  else:
    tune = baseline + np.maximum(0, gain_c*np.pi*np.square(r_c)*np.exp(-np.square(stim_sf*np.pi*r_c)) - gain_s*np.pi*np.square(r_s)*np.exp(-np.square(stim_sf*np.pi*r_s)));
    return tune, [];
 
def var_explained(data_resps, modParams, sfVals, dog_model = 2, baseline=0):
  ''' given a set of responses and model parameters, compute the variance explained by the model 
      UPDATE: If sfVals is None, then modParams is actually modResps, so we can just skip to the end...
  '''
  np = numpy;
  resp_dist = lambda x, y: np.sum(np.square(x-y))/np.maximum(len(x), len(y))
  var_expl = lambda m, r, rr: 100 * (1 - resp_dist(m, r)/resp_dist(r, rr));

  # organize data responses (adjusted)
  data_mean = np.mean(data_resps) * np.ones_like(data_resps);

  if sfVals is None:
    mod_resps = modParams;
  else:
    # compute model responses # TODO: Replace with get_descrResp() call
    if dog_model == 0:
      mod_resps = flexible_Gauss_np(modParams, stim_sf=sfVals);
    if dog_model == 1:
      mod_resps = DoGsach(*modParams, stim_sf=sfVals, baseline=baseline)[0];
    if dog_model == 2:
      mod_resps = DiffOfGauss(*modParams, stim_sf=sfVals, baseline=baseline)[0];

  try:
    return var_expl(mod_resps, data_resps, data_mean);
  except:
    return np.nan;

def chiSq(data_resps, model_resps, stimDur=1, kMult = 0.10):
  ''' given a set of measured and model responses, compute the chi-squared (see Cavanaugh et al '02a)
      Cavanaugh uses a multiplier of 0.01 for K, but our default is 0.1 (see modCompare.ipynb analysis)
      assumes: resps are mean/variance for each stimulus condition (e.g. like a tuning curve)
        with each condition a tuple (or 2-array) with [mean, var]
  '''
  np = numpy;

  # particularly for adjusted responses, a few values might be negative; remove these from the rho calculation 
  nonneg = np.where(np.logical_and(data_resps[0]>0, ~np.isnan(data_resps[0])))[0];
  rats = np.divide(data_resps[1][nonneg], data_resps[0][nonneg]);
  rho = geomean(rats);
  k   = kMult * rho * np.nanmax(data_resps[0]) # default kMult from Cavanaugh is 0.01

  # some conditions might be blank (and therefore NaN) - remove them!
  num = data_resps[0] - model_resps[0];
  valid = ~np.isnan(num);
  data_resp_recenter = data_resps[0][valid] - np.min(data_resps[0][valid]);
  # the numerator is (.)^2, and therefore always >=0; the denominator is now "recentered" so that the values are >=0
  # thus, chi will always be >=0, avoiding a fit which maximizes the numerator to reduce the loss (denom was not strictly >0)   
  chi = np.sum(np.divide(np.square(num[valid]), k + data_resp_recenter*rho/stimDur));

  # - now, again, but keep everything and don't sum
  data_resp_recenter = data_resps[0] - np.nanmin(data_resps[0]);
  notSum = np.divide(np.square(num), k + data_resp_recenter*rho/stimDur);

  return chi, notSum;

def get_c50(rvcMod, params):
  ''' get the c50 for a given rvcModel and parameter list
  '''
  try:
    if rvcMod == 1 or rvcMod == 2: # naka-rushton/peirce
      c50 = params[3];
    elif rvcMod == 0: # i.e. movshon form
      c50 = params[2]
  except:
    c50 = numpy.nan;
    
  return c50;

def c50_empirical(rvcMod, params):
  # now, by optimization and discrete numerical evaluation, get the c50

  con_bound = (5e-2, 1);
  # first, find the maximum response
  if rvcMod == 1 or rvcMod == 2: # naka-rushton/peirce
    obj = lambda con: -naka_rushton(con, params)
  elif rvcMod == 0: # i.e. movshon form
    rvcModel = get_rvc_model();
    obj = lambda con: -rvcModel(*params, con);
  max_opt = opt.minimize(obj, 0.8, bounds=(con_bound, ));
  max_response = -max_opt['fun'];
  max_con = max_opt['x'];

  # now, find out the contrast with 50% of max response
  con_bound = (5e-2, max_con); # i.e. ensure the c50_emp contrast is less than the max (relevant for super-saturating curves)
  c50_obj = lambda con: numpy.square(max_response/2 + obj(con));
  c50_opt = opt.minimize(c50_obj, max_con/2, bounds=(con_bound, ));
  c50_lsq = c50_opt['x'][0]; # not as array, instead unpack
  
  # now, let's evaluate in a second method: just sample, find the value closest to max_response/2
  con_vals = numpy.geomspace(0.05, max_con, 500);
  if rvcMod == 1 or rvcMod == 2:
    rvc_evals = naka_rushton(con_vals, params);
  elif rvcMod == 0: # mov form
    rvc_evals = rvcModel(*params, con_vals);
  ind_min = numpy.argmin(numpy.square(rvc_evals-max_response/2));
  c50_eval = con_vals[ind_min];

  return c50_lsq, c50_eval;

def descr_prefSf(modParams, dog_model=2, all_sfs=numpy.logspace(-1, 1, 11), baseline=0, nSamps=500):
  ''' Compute the preferred SF given a set of DoG [or in the case of dog_model==0, not DoG...) parameters
  '''
  np = numpy;
  sf_bound = (numpy.min(all_sfs), numpy.max(all_sfs));
  if dog_model == 0:
    return modParams[2]; # direct read out in this model!
  # if the solution is not analytical, then we compute
  sf_samps = np.geomspace(all_sfs[0], all_sfs[-1], nSamps);
  tuning = get_descrResp(modParams, sf_samps, dog_model);
  sf_evals = np.argmax(tuning);
  sf_peak = sf_samps[sf_evals];

  return sf_peak

def dog_prefSfMod(descrFit, allCons, disp=0, varThresh=65, dog_model=2, prefMin=0.1, highCut=None, base_sub=None):
  ''' Given a descrFit dict for a cell, compute a fit for the prefSf as a function of contrast
      Return ratio of prefSf at highest:lowest contrast, lambda of model, params
      - update! if highCut is None, get the peakSF; otherwise, get that fraction of the peak and fit that as a f'n of contrast
  '''
  np = numpy;
  # the model
  psf_model = lambda offset, slope, alpha, con: np.maximum(prefMin, offset + slope*np.power(con-con[0], alpha));
  # gather the values
  #   only include prefSf values derived from a descrFit whose variance explained is gt the thresh
  validInds = np.where(descrFit['varExpl'][disp, :] > varThresh)[0];
  if len(validInds) == 0: # i.e. no good fits...
    return np.nan, [], [];
  if highCut is None:
    if 'prefSf' in descrFit:
      prefSfs = descrFit['prefSf'][disp, validInds];
    else:
      prefSfs = [];
      for i in validInds:
        psf_curr = descr_prefSf(descrFit['params'][disp, i], dog_model);
        prefSfs.append(psf_curr);
  else:
    prefSfs = []; # but they're not really prefSfs - it's sfX, where is is the fraction of the peak!
    for i in validInds:
      psf_curr = sf_highCut(descrFit['params'][disp, i], dog_model, frac=highCut, sfRange=(0.1, 15), baseline_sub=base_sub);
      prefSfs.append(psf_curr);

  conVals = allCons[validInds];
  weights = descrFit['varExpl'][disp, validInds];
  # set up the optimization
  obj = lambda params: np.sum(np.multiply(weights,
        np.square(psf_model(params[0], params[1], params[2], conVals) - prefSfs)))
  conRange = conVals[-1] - conVals[0];
  init_offset = prefSfs[0];
  init_slope = (prefSfs[-1] - prefSfs[0]) / conRange;
  init_alpha = 0.4; # most tend to be saturation (i.e. contrast exp < 1)
  # run
  optz = opt.minimize(obj, [init_offset, init_slope, init_alpha], bounds=((None, None), (None, None), (0.1, 10)));
  opt_params = optz['x'];
  # ratio:
  extrema = psf_model(*opt_params, con=(conVals[0], conVals[-1]))
  pSfRatio = extrema[-1] / extrema[0]

  return pSfRatio, psf_model, opt_params;

def dog_charFreq(prms, DoGmodel=1):
  ''' Compute the characteristic frequency given a set of DoG model parameters
      Here, we use the Enroth-Cugell, Robson (1966) definition of charFreq (1/e fall off in the center mechanism strength)
  '''
  if DoGmodel == 0:
      f_c = numpy.nan; # Cannot compute charFreq without DoG model fit (see sandbox_careful.ipynb)
  elif DoGmodel == 1: # sach
      r_c = prms[1];
      f_c = 1/(numpy.pi*r_c) # TODO: might need a 2* in the denom???
  elif DoGmodel == 2: # tony
      f_c = prms[1];

  return f_c;

def dog_charFreqMod(descrFit, allCons, val_inds, varThresh=70, DoGmodel=1, lowConCut = 0.1, disp=0, fixPower=False):
  ''' Given a descrFit dict for a cell, compute a fit for the charFreq as a function of contrast
      Return ratio of charFreqat highest:lowest contrast, lambda of model, params, the value of the charFreq at the valid contrasts, the corresponding valid contrast
      Note: valid contrast means a contrast which is greater than the lowConCut and one for which the Sf tuning fit has a variance explained gerat than varThresh

      NOTE: Fitting in log(2)--log(2) coordinates // i.e. log2 the charFreqs and contrasts before fitting
  '''
  np = numpy;
  # the model
  fc_model = lambda offset, slope, alpha, con: offset + slope*np.power(np.log2(con)-np.log2(con[0]), alpha);
  # gather the values
  #   only include prefSf values derived from a descrFit whose variance explained is gt the thresh
  validInds = np.where((descrFit['varExpl'][disp, val_inds] > varThresh) & (allCons > lowConCut))[0];
  conVals = allCons[validInds];

  if len(validInds) == 0: # i.e. no good fits...
    return np.nan, None, None, None, None;
  if 'charFreq' in descrFit:
    charFreqs = np.log2(descrFit['charFreq'][disp, val_inds[validInds]]);
  else:
    charFreqs = [];
    for i in validInds:
      cf_curr = np.log2(dog_charFreq(descrFit['params'][disp, i], DoGmodel));
      charFreqs.append(cf_curr);
  weights = descrFit['varExpl'][disp, val_inds[validInds]];
  # set up the optimization
  obj = lambda params: np.sum(np.multiply(weights,
        np.square(fc_model(params[0], params[1], params[2], conVals) - charFreqs)))
  init_offset = charFreqs[0];
  conRange = np.log2(conVals[-1]) - np.log2(conVals[0]);
  init_slope = (charFreqs[-1] - charFreqs[0]) / conRange;
  if fixPower == True:
    init_alpha = 1; # assume a power law of 1 
    # run
    optz = opt.minimize(obj, [init_offset, init_slope, init_alpha], bounds=((None, None), (None, None), (1, 1))); # (1, None)
  else:
    init_alpha = 0.4; # most tend to be saturation (i.e. contrast exp < 1)
    # run
    optz = opt.minimize(obj, [init_offset, init_slope, init_alpha], bounds=((None, None), (None, None), (0.25, None)));
  #optz = opt.minimize(obj, [init_offset, init_slope, init_alpha], bounds=((0, None), (None, None), (0.25, 4)));
  opt_params = optz['x'];
  # ratio:
  extrema = fc_model(*opt_params, con=(conVals[0], conVals[-1]))
  fcRatio = extrema[-1] / extrema[0]

  return fcRatio, fc_model, opt_params, np.power(2, charFreqs), conVals;

def dog_get_param(params, DoGmodel, metric):
  ''' given a code for which tuning metric to get, and the model/parameters used, return that metric
      note: when comparing the two formulations for DoG (i.e. Sach and Tony), we use Sach values as the reference
        to this end, we make the following transformations of the Tony parameters
        - gain:   gain/(pi*r^2)
        - radius: 1/(pi*fc)
  '''
  np = numpy;

  if DoGmodel == 0:
    return np.nan; # we cannot compute from that form of the model!
  if metric == 'gc': # i.e. center gain
    if DoGmodel == 1: # sach
      return params[0];
    elif DoGmodel == 2: # tony
      fc = params[1];
      rc = np.divide(1, np.pi*fc);
      return np.divide(params[0], np.pi*np.square(rc));
  if metric == 'gs': # i.e. surround gain
    if DoGmodel == 1: # sach
      return params[2];
    elif DoGmodel == 2: # tony
      fc = params[1];
      rs = np.divide(1, np.pi*fc*params[3]); # params[3] is the multiplier on fc to get fs
      return np.divide(params[0]*params[2], np.pi*np.square(rs));
  if metric == 'rc': # i.e. center radius
    if DoGmodel == 1: # sach
      return params[1];
    elif DoGmodel == 2: # tony
      fc = params[1];
      return np.divide(1, np.pi*fc);
  if metric == 'rs': # i.e. surround radius
    if DoGmodel == 1: # sach
      return params[3];
    elif DoGmodel == 2: # tony
      fc = params[1];
      rs = np.divide(1, np.pi*fc*params[3]); # params[3] is the multiplier on fc to get fs
      return rs;

## - for fitting DoG models

def DoG_loss(params, resps, sfs, loss_type = 3, DoGmodel=1, dir=-1, resps_std=None, gain_reg = 0, minThresh=0.1, joint=False, baseline=0, fracSig=1, enforceMaxPenalty=1):
  '''Given the model params (i.e. sach or tony formulation)), the responses, sf values
  return the loss
  loss_type: 1 - lsq
             2 - sqrt
             3 - poiss
             4 - Sach sum{[(exp-obs)^2]/[k+sigma^2]} where
                 k := 0.01*max(obs); sigma := measured variance of the response
  DoGmodel: 0 - flexGauss (not DoG...)
            1 - sach
            2 - tony

    - if joint=True, then resps, resps_std will be arrays in which to index
    - if enforceMaxPenalty, then we add a penalty if the maximum model response is more than 50% larger than the maximum data response
    - params will be 2*N+2, where N is the number of contrasts;
    --- "2" is for a shared (across all contrasts) ratio of gain/[radius/freq]
    --- then, there are two parameters fit uniquely to each contrast - center gain & [radius/freq]
  '''
  np = numpy;

  # we'll use the "joint" flag to determine how many DoGs we are fitting
  if joint==True:
    n_fits = len(resps);
    gain_rat = params[0]; # the ratio of center::surround gain is shared across all fits
    shape_rat = params[1]; # the ratio of ctr::surr freq (or radius) is shared across all fits
  else:
    n_fits = 1;

  totalLoss = 0;

  for i in range(n_fits):
    if n_fits == 1: # i.e. joint is False!
      curr_params = params;
      curr_resps = resps;
      curr_std = resps_std;

      if enforceMaxPenalty:
        max_data = np.max(curr_resps);

    else:
      curr_resps = resps[i];
      curr_std = resps_std[i];
      local_gain = params[2+i*2]; 
      local_shape = params[3+i*2]; # shape, as in radius/freq, depending on DoGmodel
      if DoGmodel == 1: # i.e. sach
        curr_params = [local_gain, local_shape, local_gain*gain_rat, local_shape*shape_rat];
      elif DoGmodel == 2: # i.e. Tony
        curr_params = [local_gain, local_shape, gain_rat, shape_rat];

      if enforceMaxPenalty: # check if this gives separate max for each condition
        print('NOTICE: check if this gives separate max for each condition (dog_loss in helper_fcns');
        max_data = np.max(curr_resps);

    pred_spikes = get_descrResp(curr_params, sfs, DoGmodel, minThresh, baseline, fracSig);
    if enforceMaxPenalty: # check if this gives separate max for each condition
      max_mod = get_descrResp(curr_params, np.array([descr_prefSf(curr_params, DoGmodel)]), DoGmodel, minThresh, baseline, fracSig)[0];
      applyPen = 1 if (max_mod-1.40*max_data)>0 else 0;
      # TODO: Make this penalty smooth/continuous rather than discrete...
      maxPen = applyPen*1*(max_mod-1.4*max_data); # scale factor of 1 chosen to be within the typical loss values (O(10),O(100), at least for loss_type=2) so this regularization does not overwhelm
    else:
      maxPen = 0;

    if loss_type == 1: # lsq
      loss = np.sum(np.square(curr_resps - pred_spikes)) + maxPen;
      totalLoss = totalLoss + loss;
    elif loss_type == 2: # sqrt - now handles negative responses by first taking abs, sqrt, then re-apply the sign 
      loss = np.sum(np.square(np.sign(curr_resps)*np.sqrt(np.abs(curr_resps)) - np.sign(pred_spikes)*np.sqrt(np.abs(pred_spikes)))) + maxPen
      #loss = np.sum(np.square(np.sqrt(curr_resps) - np.sqrt(pred_spikes)));
      totalLoss = totalLoss + loss;
    elif loss_type == 3: # poisson model of spiking
      poiss = poisson.pmf(np.round(curr_resps), pred_spikes); # round since the values are nearly but not quite integer values (Sach artifact?)...
      ps = np.sum(poiss == 0);
      if ps > 0:
        poiss = np.maximum(poiss, 1e-6); # anything, just so we avoid log(0)
      totalLoss = totalLoss + sum(-np.log(poiss)) + maxPen;
    elif loss_type == 4: # sach's loss function
      k = 0.01*np.max(curr_resps);
      if resps_std is None:
        sigma = np.ones_like(curr_resps);
      else:
        sigma = curr_std;
      sq_err = np.square(curr_resps-pred_spikes);
      totalLoss = totalLoss + np.sum((sq_err/(k+np.square(sigma)))) + gain_reg*(params[0] + params[2]) + maxPen; # regularize - want gains as low as possible

  return totalLoss;

def dog_init_params(resps_curr, base_rate, all_sfs, valSfVals, DoGmodel, bounds=None, fracSig=1):
  ''' return the initial parameters for the DoG model, given the model choice and responses
      --- if bounds is not None, then we'll ensure that each parameter is within the specified bounds
  '''
  np = numpy;

  maxResp       = np.max(resps_curr);
  freqAtMaxResp = all_sfs[np.argmax(resps_curr)];

  ## FLEX (not difference of gaussian)
  if DoGmodel == 0:
    # set initial parameters - a range from which we will pick!
    if base_rate <= 3 or np.isnan(base_rate): # will be NaN if getting f1 responses
        range_baseline = (0, 3);
    else:
        range_baseline = (0.5 * base_rate, 1.5 * base_rate);
    range_amp = (0.4 * maxResp, 0.8 * maxResp); # was [0.5, 1.25]

    max_sf_index = np.argmax(resps_curr); # what sf index gives peak response?
    mu_init = valSfVals[max_sf_index];

    if max_sf_index == 0: # i.e. smallest SF center gives max response...
        range_mu = (mu_init/2, valSfVals[max_sf_index + np.minimum(3, len(valSfVals)-1)]);
    elif max_sf_index+1 == len(valSfVals): # i.e. highest SF center is max
        range_mu = (valSfVals[max_sf_index-np.minimum(3, len(valSfVals)-1)], mu_init);
    else:
        range_mu = (valSfVals[max_sf_index-1], valSfVals[max_sf_index+1]); # go +-1 indices from center

    denom_lo = 0.2; # better to start narrow...
    denom_hi = 0.7; # ... in order to avoid broad tuning curve which optimizes to flat
    range_denom = (denom_lo, denom_hi); # don't want 0 in sigma 
    #log_bw_lo = 0.2; # previously, was 0.75 octave bandwidth...
    #log_bw_hi = 0.7; # previously, was 2 octave bandwidth...
    #denom_lo = bw_log_to_lin(log_bw_lo, mu_init)[0]; # get linear bandwidth
    #denom_hi = bw_log_to_lin(log_bw_hi, mu_init)[0]; # get lin. bw (cpd)
    #range_denom = (denom_lo, denom_hi); # don't want 0 in sigma 
    if fracSig:
      range_sigmaHigh = (0.2, 0.75); # allow the fracSig value to go above the bound used for V1, since we adjust if bound is there

    init_base = random_in_range(range_baseline)[0]; # NOTE addition of [0] to "unwrap" random_in_range value
    init_amp = random_in_range(range_amp)[0];
    init_mu = random_in_range(range_mu)[0];
    init_sig_left = random_in_range(range_denom)[0];
    init_sig_right = random_in_range(range_sigmaHigh)[0] if fracSig else random_in_range(range_denom)[0];
    init_params = [init_base, init_amp, init_mu, init_sig_left, init_sig_right];

  ############
  ## SACH
  ############
  elif DoGmodel == 1:
    init_gainCent = 5e2*random_in_range((1, 300))[0];
    #init_gainCent = random_in_range((maxResp, 100*maxResp))[0];
    init_radiusCent = random_in_range((0.02, 0.5))[0];
    init_gainSurr = init_gainCent * random_in_range((0.01, 0.5))[0];
    init_radiusSurr = init_radiusCent * random_in_range((0.75, 7))[0];
    init_params = [init_gainCent, init_radiusCent, init_gainSurr, init_radiusSurr];
  ############
  ## TONY
  ############
  elif DoGmodel == 2:
    init_gainCent = maxResp * random_in_range((0.9, 1.2))[0];
    init_freqCent = np.maximum(all_sfs[2], freqAtMaxResp * random_in_range((1.2, 1.5))[0]); # don't pick all_sfs[0] -- that's zero (we're avoiding that)
    init_gainFracSurr = random_in_range((0.7, 1))[0];
    init_freqFracSurr = random_in_range((.25, .35))[0];
    init_params = [init_gainCent, init_freqCent, init_gainFracSurr, init_freqFracSurr];

  # For all -- try 
  if bounds is not None:
    try:
      for (ii,prm),bound in zip(enumerate(init_params), bounds):
        if prm < bound[0] or prm > bound[1]:
          init_params[ii] = (bound[0]+bound[1])*random_in_range([0.25, 0.75])[0] # some value in-between the two bounds
    except: # we end up here if bounds is somehow not equal in # of entries to init_params
      pass; # not ideal, but the parent function should handle failures of initialization by trying again, anyway

  return init_params

def dog_fit(resps, DoGmodel, loss_type, disp, expInd, stimVals, validByStimVal, valConByDisp, n_repeats=100, joint=False, gain_reg=0, ref_varExpl=None, veThresh=70, prevFits=None, baseline_DoG=True, fracSig=1, noDisp=0, debug=0):
  ''' Helper function for fitting descriptive funtions to SF responses
      if joint=True, (and DoGmodel is 1 or 2, i.e. not flexGauss), then we fit assuming
      a fixed ratio for the center-surround gains and [freq/radius]
      - i.e. of the 4 DoG parameters, 2 are fit separately for each contrast, and 2 are fit 
        jointly across all contrasts!
      - note that ref_varExpl (optional) will be of the same form that the output for varExpl will be
      --- fracSig: if on, then the right-half (high SF) of the flex. gauss tuning curve is expressed as a fraction of the lower half

      inputs: self-explanatory, except for resps, which should be [resps_mean, resps_all, resps_sem, base_rate]
      outputs: bestNLL, currParams, varExpl, prefSf, charFreq, [overallNLL, paramList; if joint=True]
  '''
  np = numpy;

  if DoGmodel == 0:
    joint=False; # we cannot fit the flex gauss model jointly!
    nParam = 5;
  else: # and joint can stay as specified
    nParam = 4;

  ### organize stimulus information, responses
  all_disps = stimVals[0];
  all_cons = stimVals[1];
  all_sfs = stimVals[2];

  nDisps = len(all_disps);
  nCons = len(all_cons);

  # unpack responses
  resps_mean, resps_all, resps_sem, base_rate = resps;
  baseline = 0 if base_rate is None else base_rate; # what's the baseline to add 
  base_rate = base_rate if base_rate is not None else np.nanmin(resps_mean)

  # next, let's compute some measures about the responses
  if resps_all is not None:
    max_resp = np.nanmax(resps_all);
  else: # we don't really need to pass in resps_all
    max_resp = np.nanmax(resps_mean);

  # and set up initial arrays
  if prevFits is None:
    bestNLL = np.ones((nCons, )) * np.nan;
    currParams = np.ones((nCons, nParam)) * np.nan;
    varExpl = np.ones((nCons, )) * np.nan;
    prefSf = np.ones((nCons, )) * np.nan;
    charFreq = np.ones((nCons, )) * np.nan;
    if joint==True:
      overallNLL = np.nan;
      params = np.nan;
  else: # we've passed in existing fits!
    if noDisp:
      bestNLL, currParams, varExpl, prefSf, charFreq = prevFits['NLL'], prevFits['params'], prevFits['varExpl'], prevFits['prefSf'], prevFits['charFreq'];
    else:
      bestNLL, currParams, varExpl, prefSf, charFreq = prevFits['NLL'][disp,:], prevFits['params'][disp,:], prevFits['varExpl'][disp,:], prevFits['prefSf'][disp,:], prevFits['charFreq'][disp,:];
    if joint==True:
      overallNLL = prevFits['totalNLL'];
      params = prevFits['paramList'];

  ### set bounds
  if DoGmodel == 0: # FLEX - flexible gaussian (i.e. two halves)
    min_bw = 1/4; max_bw = 10; # ranges in octave bandwidth
    bound_baseline = (0, max_resp);
    bound_range = (0, 1.5*max_resp);
    bound_mu = (0.01, 10);
    bound_sig = (np.maximum(0.1, min_bw/(2*np.sqrt(2*np.log(2)))), max_bw/(2*np.sqrt(2*np.log(2)))); # Gaussian at half-height
    if fracSig:
      bound_sigFrac = (0.2, 2);
    else:
      bound_sigFrac = (1e-4, None); # arbitrarily small, to None // TRYING
    allBounds = (bound_baseline, bound_range, bound_mu, bound_sig, bound_sigFrac);
  elif DoGmodel == 1: # SACH
    bound_gainCent = (1e-3, None);
    bound_radiusCent= (1e-3, None);
    if joint==True:
      bound_gainRatio = (1e-3, 1); # the surround gain will always be less than the center gain
      bound_radiusRatio= (1, None); # the surround radius will always be greater than the ctr r
      # we'll add to allBounds later, reflecting joint gain/radius ratios common across all cons
      allBounds = (bound_gainRatio, bound_radiusRatio);
    elif joint==False:
      bound_gainSurr = (1e-3, None);
      bound_radiusSurr= (1e-3, None);
      allBounds = (bound_gainCent, bound_radiusCent, bound_gainSurr, bound_radiusSurr);
  elif DoGmodel == 2: # TONY
    bound_gainCent = (1e-3, None);
    bound_freqCent= (1e-1, 2e1); # let's set the charFreq upper bound at 20 cpd (is that ok?)
    if joint==True:
      bound_gainRatio = (1e-3, 3);
      bound_freqRatio = (1e-1, 1); 
      # we'll add to allBounds later, reflecting joint gain/radius ratios common across all cons
      allBounds = (bound_gainRatio, bound_freqRatio);
    elif joint==False:
      bound_gainFracSurr = (1e-3, 2); # surround gain always less than center gain NOTE: SHOULD BE (1e-3, 1)
      bound_freqFracSurr = (5e-2, 1); # surround freq always less than ctr freq NOTE: SHOULD BE (1e-1, 1)
      allBounds = (bound_gainCent, bound_freqCent, bound_gainFracSurr, bound_freqFracSurr);

  ### organize responses -- and fit, if joint=False
  allResps = []; allRespsSem = []; start_incl = 0; incl_inds = [];

  for con in range(nCons):
    if con not in valConByDisp[disp]:
      continue;

    if validByStimVal is not None:
      valSfInds = get_valid_sfs(None, disp, con, expInd, stimVals, validByStimVal); # we pass in None for data, since we're giving stimVals/validByStimVal, anyway
    else: # then we aren't skipping any
      valSfInds = np.arange(0,len(all_sfs));
    valSfVals = all_sfs[valSfInds];
    # ensure all are strictly GT 0
    valSfVals = valSfVals[valSfVals>0];

    respConInd = np.where(np.asarray(valConByDisp[disp]) == con)[0];
    resps_curr = resps_mean[disp, valSfInds, con];
    sem_curr   = resps_sem[disp, valSfInds, con];

    ### prepare for the joint fitting, if that's what we've specified!
    if joint==True:
      if ref_varExpl is None:
        start_incl = 1; # hacky...
      if start_incl == 0:
        if ref_varExpl[con] < veThresh:
          continue; # i.e. we're not adding this; yes we could move this up, but keep it here for now
        else:
          start_incl = 1; # now we're ready to start adding to our responses that we'll fit!

      incl_inds.append(con); # keep note of which contrast indices are included
      allResps.append(resps_curr);
      allRespsSem.append(sem_curr);
      # and add to the parameter list!
      if DoGmodel == 1:
        allBounds = (*allBounds, bound_gainCent, bound_radiusCent);
      elif DoGmodel == 2:
        allBounds = (*allBounds, bound_gainCent, bound_freqCent);
      continue;

    ### otherwise, we're really going to fit here! [i.e. if joint is False]
    if debug:
      save_all = [];
    import timeit

    for n_try in range(n_repeats):
      ###########
      ### pick initial params
      ###########
      init_params = dog_init_params(resps_curr, base_rate, all_sfs, valSfVals, DoGmodel, allBounds)

      # choose optimization method
      if np.mod(n_try, 2) == 0:
          methodStr = 'L-BFGS-B';
      else:
          methodStr = 'TNC';

      #stmt = '''DoG_loss(init_params, resps_curr, valSfVals, resps_std=sem_curr, loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, gain_reg=gain_reg, joint=joint, baseline=baseline, enforceMaxPenalty=0)'''
      #isitok = timeit.timeit(stmt, globals={'DoG_loss': DoG_loss, 'init_params': init_params, 'resps_curr': resps_curr, 'valSfVals': valSfVals, 'sem_curr': sem_curr, 'loss_type': loss_type, 'DoGmodel': DoGmodel, 'dir': dir, 'gain_reg': gain_reg, 'joint': joint, 'baseline': baseline}, number=1000);
 
      #pdb.set_trace();

      obj = lambda params: DoG_loss(params, resps_curr, valSfVals, resps_std=sem_curr, loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, gain_reg=gain_reg, joint=joint, baseline=baseline, enforceMaxPenalty=1);
      try:
        wax = opt.minimize(obj, init_params, method=methodStr, bounds=allBounds);
      except:
        continue; # the fit has failed (bound issue, for example); so, go back to top of loop, try again

      # compare
      NLL = wax['fun'];
      params = wax['x'];

      if debug:
        save_all.append([wax, init_params]);

      if np.isnan(bestNLL[con]) or NLL < bestNLL[con]:
        bestNLL[con] = NLL;
        currParams[con, :] = params;
        varExpl[con] = var_explained(resps_curr, params, valSfVals, DoGmodel, baseline=baseline);
        prefSf[con] = descr_prefSf(params, dog_model=DoGmodel, all_sfs=valSfVals, baseline=baseline);
        charFreq[con] = dog_charFreq(params, DoGmodel=DoGmodel);

  if joint==False: # then we're DONE
    if debug:
      return bestNLL, currParams, varExpl, prefSf, charFreq, None, None, save_all; # placeholding None for overallNLL, params [full list]
    else:
      return bestNLL, currParams, varExpl, prefSf, charFreq, None, None; # placeholding None for overallNLL, params [full list]

  ### NOW, we do the fitting if joint=True
  if joint==True:
    ### now, we fit!
    for n_try in range(n_repeats):
      # we'll also add to allInitParams later, to estimate the center gain/radius
      if DoGmodel == 1:
        # -- but first, we estimate the gain ratio and the radius ratio
        allInitParams = [random_in_range((0.2, 0.8))[0], random_in_range((2, 3.5))[0]];
      elif DoGmodel == 2:
        # -- but first, we estimate the gain ratio and the cfreq ratio
        allInitParams = [random_in_range((0.2, 0.8))[0], random_in_range((0.2, 0.4))[0]];
      for resps_curr in allResps:
        # now, we need to guess the local parameters - 0:2 will give us center gain/shape
        curr_init = dog_init_params(resps_curr, base_rate, all_sfs, valSfVals, DoGmodel)[0:2];
        allInitParams = [*allInitParams, curr_init[0], curr_init[1]];

      # choose optimization method
      if np.mod(n_try, 2) == 0:
          methodStr = 'L-BFGS-B';
      else:
          methodStr = 'TNC';

      obj = lambda params: DoG_loss(params, allResps, valSfVals, resps_std=allRespsSem, loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, gain_reg=gain_reg, joint=joint);
      try:
        wax = opt.minimize(obj, allInitParams, method=methodStr, bounds=allBounds);
      except:
        continue; # if that particular fit fails, go back and try again

      # compare
      NLL = wax['fun'];
      params_curr = wax['x'];

      if np.isnan(overallNLL) or NLL < overallNLL:
        overallNLL = NLL;
        params = params_curr;

    ### then, we must unpack the fits to actually fill in the "true" parameters for each contrast
    gain_rat, shape_rat = params[0], params[1];
    for con in range(len(allResps)):
      # get the current parameters and responses by unpacking the associated structures
      local_gain = params[2+con*2]; 
      local_shape = params[3+con*2]; # shape, as in radius/freq, depending on DoGmodel
      if DoGmodel == 1: # i.e. sach
        curr_params = [local_gain, local_shape, local_gain*gain_rat, local_shape*shape_rat];
      elif DoGmodel == 2: # i.e. Tony
        curr_params = [local_gain, local_shape, gain_rat, shape_rat];
      # -- then the responses, and overall contrast index
      resps_curr = allResps[con];
      sem_curr   = allRespsSem[con];
      respConInd = incl_inds[con];
      #respConInd = valConByDisp[disp][con]; 
      
      # now, compute!
      bestNLL[respConInd] = DoG_loss(curr_params, resps_curr, valSfVals, resps_std=sem_curr, loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, gain_reg=gain_reg, joint=False); # not joint, now!
      currParams[respConInd, :] = curr_params;
      varExpl[respConInd] = var_explained(resps_curr, curr_params, valSfVals, DoGmodel);
      prefSf[respConInd] = descr_prefSf(curr_params, dog_model=DoGmodel, all_sfs=valSfVals);
      charFreq[respConInd] = dog_charFreq(curr_params, DoGmodel=DoGmodel);    

    # and NOW, we can return!
    if debug:
      return bestNLL, currParams, varExpl, prefSf, charFreq, overallNLL, params, None; # None should be save_all
    else:
      return bestNLL, currParams, varExpl, prefSf, charFreq, overallNLL, params;
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

def get_prefSF(flexGauss_fit, DoGmodel=0): # DEPRECATE [SAME AS descr_prefSf]
   ''' Given a set of parameters for a descriptive SF fit, return the preferred SF
       -- analytic solution if two-half gauss; otherwise, numerical solution
   '''
   if DoGmodel==0:
     return flexGauss_fit[2];
   else:
     return descr_prefSf(flexGauss_fit, dog_model=DoGmodel);

def compute_LSFV(fit, sfMod=0):
    ''' Low spatial-frequency variance [LSFV] as computed in Xing et al 2004 '''
    
    np = numpy;

    logBase = 16
    nSteps = 1000;

    # get descrFit, stim_sf, and descriptive responses
    prefSf = descr_prefSf(fit, sfMod);
    lowBound = prefSf/logBase
    stim_sf = np.geomspace(lowBound, 10, nSteps) # go from prefSf/[logBase=16] up to 10 cpd0
    resps = get_descrResp(fit, stim_sf, sfMod)

    # lsfv calculation
    wherePref = np.argmin(np.square(prefSf - stim_sf)); # i.e. which arg has prefSf?
    lsfv_num = [resps[x]*np.square(np.log(stim_sf[x])/np.log(logBase) - np.log(stim_sf[wherePref])/np.log(logBase)) for x in np.arange(0, wherePref)]
    lsfv_denom = [resps[x] for x in np.arange(0, wherePref)]
    lsfv = np.sum(lsfv_num)/np.sum(lsfv_denom);

    return lsfv;  

def compute_SF_BW(fit, height, sf_range, which_half=0, sfMod=0, baseline=None, fracSig=1):
    ''' if which_half = 0, use both-halves; if +1/-1, only return bandwidth at higher/lower SF end
     Height is defined RELATIVE to baseline
     i.e. baseline = 10, peak = 50, then half height is NOT 25 but 30
     --- until 21.05.03, we only used this function with flexible gaussian model, but we'll adjust now to all descr. mods
     --- --- baseline only used if sfMod != 0 (will be None if F1 response)
    '''
    # predefine as NaN, in case bandwidth is undefined
    np = numpy;
    bw_log = np.nan;
    SF = np.empty((2, 1));
    SF[:] = np.nan;
    
    if sfMod==0: # the default, and until 21.05.03, the only method
      prefSf = get_prefSF(fit);

      # left-half
      left_full_bw = 2 * (fit[3] * sqrt(2*log(1/height)));
      left_cpd = fit[2] * exp(-(fit[3] * sqrt(2*log(1/height))));

      # right-half
      if fracSig:
        sigRight = fit[3]*fit[4];
      else:
        sigRight = fit[3];
      right_full_bw = 2 * (sigRight * sqrt(2*log(1/height)));
      right_cpd = fit[2] * exp((sigRight * sqrt(2*log(1/height))));

    elif sfMod==1 or sfMod==2: # we'll do this numerically rather than in closed form
      prefSf = descr_prefSf(fit, dog_model=sfMod, all_sfs=sf_range)
      peakResp = get_descrResp(fit, prefSf, sfMod);
      targetResp = peakResp*height;
      if sfMod == 1:
         obj = lambda sf: np.square(targetResp - DoGsach(*fit, stim_sf=sf)[0]);
      elif sfMod == 2:
         obj = lambda sf: np.square(targetResp - DiffOfGauss(*fit, stim_sf=sf)[0]);
      # lower half, first
      sf_samps = np.geomspace(sf_range[0], prefSf, 500);
      sf_evals = np.argmin([obj(x) for x in sf_samps]);
      left_cpd = sf_samps[sf_evals];
      # then, upper half
      sf_samps = np.geomspace(prefSf, sf_range[1], 500);
      sf_evals = np.argmin([obj(x) for x in sf_samps]);
      right_cpd = sf_samps[sf_evals];

    # Now, these calculations applies to all types of models
    if which_half == 0:
      if left_cpd > sf_range[0] and right_cpd < sf_range[-1]:
          SF = [left_cpd, right_cpd];
          bw_log = log(right_cpd / left_cpd, 2);
    elif which_half == 1:
      if right_cpd < sf_range[-1]:
        SF = [prefSf, right_cpd]
        bw_log = log(right_cpd / prefSf, 2);
    elif which_half == -1:
      if left_cpd > sf_range[0]:
        SF = [left_cpd, prefSf]
        bw_log = log(prefSf / left_cpd, 2);

    # otherwise we don't have defined BW!
    
    return SF, bw_log;

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

def flexible_Gauss_np(params, stim_sf, minThresh=0.1, fracSig=1):
    # REPLACEMENT for flexible_Gauss in numpy (i.e. written with numpy funcs rather than math/python-default funcs]
    # The descriptive model used to fit cell tuning curves - in this way, we
    # can read off preferred SF, octave bandwidth, and response amplitude

    respFloor       = params[0];
    respRelFloor    = params[1];
    sfPref          = params[2];
    sigmaLow        = params[3];
    sigmaHigh       = params[4];
    if fracSig:
      sigmaHigh = sigmaHigh*params[3]; # i.e. the true sigmaHigh value is params[4]*params[3]

    np = numpy;
    # Tuning function
    sf0   = np.divide(stim_sf, sfPref);

    sigma = np.full_like(sf0, sigmaLow);
    whereSigHigh = np.where(sf0>1);
    sigma[whereSigHigh] = sigmaHigh;

    shape = np.exp(-np.divide(np.square(np.log(sf0)), 2*np.square(sigma)))
                
    return np.maximum(minThresh, respFloor + respRelFloor*shape);

def get_descrResp(params, stim_sf, DoGmodel, minThresh=0.1, baseline=0, fracSig=1):
  ''' returns only pred_spikes; 0 is flexGauss.; 1 is DoG sach; 2 is DoG (tony)
      --- baseline is a non-optimized for additive constant that we can optionally use for diff. of gauss fits
      --- i.e. if we use it, we're simply adding the baseline response to the data, so the model fit is on top of that
  '''
  if DoGmodel == 0:
    #pred_spikes = flexible_Gauss(params, stim_sf=stim_sf, minThresh=minThresh);
    pred_spikes = flexible_Gauss_np(params, stim_sf=stim_sf, minThresh=minThresh, fracSig=fracSig);
  elif DoGmodel == 1:
    pred_spikes, _ = DoGsach(*params, stim_sf=stim_sf, baseline=baseline);
  elif DoGmodel == 2:
    pred_spikes, _ = DiffOfGauss(*params, stim_sf=stim_sf, baseline=baseline);
  return pred_spikes;

def get_rvcResp(params, curr_cons, rvcMod):
  # returns only pred_spikes
  if rvcMod == 0:
    mod = get_rvc_model();
    pred_spikes = mod(*params, curr_cons);
  elif rvcMod == 1 or rvcMod == 2:
    pred_spikes = naka_rushton(curr_cons, params);
  return pred_spikes;


##################################################################
##################################################################
##################################################################
### V. JOINT LIST ANALYSES (pref "jl_")
##################################################################
##################################################################
##################################################################

def jl_perCell(cell_ind, dataList, descrFits, dogFits, rvcFits, expDir, data_loc, dL_nm, fLW_nm, fLF_nm, dF_nm, dog_nm, rv_nm, superAnalysis=None, conDig=1, sf_range=[0.1, 10], rawInd=0, muLoc=2, varExplThresh=75, dog_varExplThresh=60, descrMod=0, dogMod=1, isSach=0, isBB=0, rvcMod=1):

   np = numpy;
   print('%s/%d' % (expDir, 1+cell_ind));

   ###########
   ### meta parameters      
   ###########
   # get experiment name, load cell
   if isSach or isBB: # if it's Sach or Bauman+Bonds
     if isSach:
       expName = dataList[cell_ind]['progName'];
       expInd = np.nan; # not applicable...
       cell = dataList[cell_ind];
       data = cell['data'];
       cellTypeOrig = dataList[cell_ind]['cellType'];
       if cellTypeOrig == 'M-cell':
          cellType = 'magno';
       elif cellTypeOrig == 'P-cell':
          cellType = 'parvo';
       else: # all other types, just keep as is...
          cellType = cellTypeOrig
       from LGN.sach.helper_fcns import tabulateResponses as sachTabulate
       tabulated = sachTabulate(data);
       stimVals = [[1], tabulated[1][0], tabulated[1][1]] # disp X con X SF
       val_con_by_disp = [range(0,len(stimVals[1]))]; # all cons are valid, since we don't have dispersions
       # F1 means, expanded to have disp dimension at front; we also transpose so that it's [disp X sf X con], as it is in other expts
       sfTuning = np.expand_dims(np.transpose(tabulated[0][1]['mean']), axis=0);
     if isBB:
       ### As of 21.05.10, we will only consider the maskOnly responses, at the corresponding response measure (DC or F1, by f1:f0 ratio)
       expName = dataList['unitName'][cell_ind]
       cellType = dataList['unitArea'][cell_ind];
       expInd = -1;
       cell = np_smart_load(data_loc + expName + '_sfBB.npy');
       expInfo = cell['sfBB_core']; # we ONLY analyze sfBB_core in this jointList (as of 21.05.10)
       tr = expInfo['trial'];
       from helper_fcns_sfBB import compute_f1f0 as bb_compute_f1f0
       from helper_fcns_sfBB import get_mask_resp
       f1f0_ind = bb_compute_f1f0(expInfo)[0] > 1; # if simple, then index with 1; else, 0
       respKey = get_resp_str(f1f0_ind); # as string, either 'dc', or 'f1'
       maskSf, maskCon = expInfo['maskSF'], expInfo['maskCon'];
       stimVals = [[1], maskCon, maskSf] # disp X con X SF
       val_con_by_disp = [range(0, len(stimVals[1]))]; # all cons are valid, since we don't have dispersions
       # the following returns [con, sf, [mn,sem]]
       maskResps = get_mask_resp(expInfo, withBase=0, vecCorrectedF1=1, returnByTr=1);
       if f1f0_ind: # i.e. if it's f1, then maskMeans/maskAll have an extra dim (len 2) at the end for R, phi (separately)
         maskMeans = maskResps[f1f0_ind][:,:,0,0]; # only get the mean "R" response (i.e. ignore PHI and "R" SEM), access only DC or F1
         maskAll = maskResps[f1f0_ind + 2][:,:,:,0]; # this gets all, organized as [con,sf,trial], just for DC or F1, appropriately
       else:
         maskMeans = maskResps[f1f0_ind][:,:,0]; # only get the mean response (i.e. ignore SEM), access only DC or F1
         maskAll = maskResps[f1f0_ind + 2]; # this gets all, organized as [con,sf,trial], just for DC or F1, appropriately
       # means by condition, expanded to have disp dimension at front; we also transpose so that it's [disp X sf X con], as it is in other expts
       sfTuning = np.expand_dims(np.transpose(maskMeans), axis=0);
   else:
     expName = dataList['unitName'][cell_ind];
     expInd = get_exp_ind(data_loc, expName)[0];
     cell = np_smart_load(data_loc + expName + '_sfm.npy');
     # get stimlus values
     resps, stimVals, val_con_by_disp, validByStimVal, _ = tabulate_responses(cell, expInd);
     # get SF responses (for model-free metrics)
     tr = cell['sfm']['exp']['trial']
     spks = get_spikes(tr, get_f0=1, expInd=expInd, rvcFits=None); # just to be explicit - no RVC fits right now
     sfTuning = organize_resp(spks, tr, expInd=expInd)[2]; # responses: nDisp x nSf x nCon
     try:
        cellType = dataList['unitType'][cell_ind];
     except: 
        # TODO: note, this is dangerous; thus far, only V1 cells don't have 'unitType' field in dataList, so we can safely do this
        cellType = 'V1'; 

   meta = dict([('fullPath', data_loc),
               ('cellNum', cell_ind+1),
               ('dataList', dL_nm),
               ('fitListWght', fLW_nm),
               ('fitListFlat', fLF_nm),
               ('descrFits', dF_nm),
               ('dogFits', dog_nm),
               ('rvcFits', rv_nm),
               ('expName', expName),
               ('expInd', expInd),
               ('stimVals', stimVals),
               ('cellType', cellType),
               ('val_con_by_disp', val_con_by_disp)]);

   ###########
   ### superposition analysis
   ###########
   suppr = None;
   if superAnalysis is not None:
     try:
       super_curr = superAnalysis[cell_ind];
       suppr = dict([('byDisp', super_curr['supr_disp']),
                     ('bySf', super_curr['supr_sf']),
                     ('sfErrsInd_var', super_curr['sfErrsInd_VAR']),
                     ('errsRat_var', super_curr['sfRat_VAR']),
                     ('corr_derivWithErr', super_curr['corr_derivWithErr']),
                     ('corr_derivWithErrsInd', super_curr['corr_derivWithErrsInd']),
                     ('supr_index', super_curr['supr_index'])]);
     except:
       pass;

   ###########
   ### basics (ori16, tf11, rfsize10, rvc10, sf11)
   ###########
   try: # try to get the basic list from superposition analysis
     basics_list = superAnalysis[cell_ind]['basics'];
   except: # but if it's not done, try to do those analyses here
     try:
       basic_names, basic_order = dataList['basicProgName'][cell_ind], dataList['basicProgOrder'];
       basics_list = get_basic_tunings(basic_names, basic_order);
     except:
       try:
         # we've already put the basics in the data structure... (i.e. post-sorting 2021 data)
         basic_names = ['','','','',''];
         basic_order = ['rf', 'sf', 'tf', 'rvc', 'ori']; # order doesn't matter if they are already loaded
         basics_list = get_basic_tunings(basic_names, basic_order, preProc=cell)
       except:
         basics_list = None;

   ###########
   ### metrics (inferred data measures)
   ###########
   disps, cons, sfs = stimVals;
   nDisps, nCons, nSfs = [len(x) for x in stimVals];

   # compute the set of SF which appear at all dispersions: highest dispersion, pick a contrast (all same)
   maxDisp = nDisps-1;
   try:
     cut_sf = np.array(get_valid_sfs(tr, disp=maxDisp, con=val_con_by_disp[maxDisp][0], expInd=expInd, stimVals=stimVals, validByStimVal=validByStimVal))
   except:
     cut_sf = None;

   ####
   # set up the arrays we need to store analyses
   ####
   # first, model-free
   sfVar = np.zeros((nDisps, nCons)) * np.nan; # variance calculation
   sfCom = np.zeros((nDisps, nCons)) * np.nan; # center of mass
   sfComCut = np.zeros((nDisps, nCons)) * np.nan; # center of mass, but with a restricted ("cut") set of SF
   f1f0_ratio = np.nan;
   # then, inferred from descriptive fits
   lsfv = np.zeros((nDisps,nCons)) * np.nan; # LSFV as from Xing et al, 2004
   bw_sigma = np.zeros((nDisps, nCons, 2)) * np.nan; # what's the left/right sigma parameter from the flex gauss fit?
   bwHalf = np.zeros((nDisps, nCons)) * np.nan;
   bwHalf_split = np.zeros((nDisps, nCons, 2)) * np.nan; # [:,:,[lower,upper]]
   bw34 = np.zeros((nDisps, nCons)) * np.nan;
   bw34_split = np.zeros((nDisps, nCons, 2)) * np.nan;
   pSf = np.zeros((nDisps, nCons)) * np.nan;
   sf70 = np.zeros((nDisps, nCons)) * np.nan;
   dog_sf70 = np.zeros((nDisps, nCons)) * np.nan;
   sf75 = np.zeros((nDisps, nCons)) * np.nan;
   dog_sf75 = np.zeros((nDisps, nCons)) * np.nan;
   sfE = np.zeros((nDisps, nCons)) * np.nan;
   dog_sfE = np.zeros((nDisps, nCons)) * np.nan;
   sfVarExpl = np.zeros((nDisps, nCons)) * np.nan;
   conGain = np.zeros((nDisps, nSfs)) * np.nan;
   c50 = np.zeros((nDisps, nSfs)) * np.nan;
   c50_emp = np.zeros((nDisps, nSfs)) * np.nan;
   c50_eval = np.zeros((nDisps, nSfs)) * np.nan;
   c50_varExpl = np.zeros((nDisps, nSfs)) * np.nan;
   # including from the DoG fits
   dog_pSf = np.zeros((nDisps, nCons)) * np.nan;
   dog_bwHalf = np.zeros((nDisps, nCons)) * np.nan;
   dog_bwHalf_split = np.zeros((nDisps, nCons, 2)) * np.nan;
   dog_bw34 = np.zeros((nDisps, nCons)) * np.nan;
   dog_bw34_split = np.zeros((nDisps, nCons, 2)) * np.nan;
   dog_varExpl = np.zeros((nDisps, nCons)) * np.nan;
   dog_charFreq = np.zeros((nDisps, nCons)) * np.nan;
   # including the difference/ratio arrays; where present, extra dim of len=2 is for raw/normalized-to-con-change values
   sfVarDiffs = np.zeros((nDisps, nCons, nCons, 2)) * np.nan;
   sfComRats = np.zeros((nDisps, nCons, nCons, 2)) * np.nan;
   lsfvRats = np.zeros((nDisps, nCons, nCons, 2)) * np.nan;
   bwHalfDiffs = np.zeros((nDisps, nCons, nCons, 2)) * np.nan;
   bwHalfDiffs_split = np.zeros((nDisps, nCons, nCons, 2, 2)) * np.nan; # 2nd to last dim is [lower,upper] half rel. to peak
   bw34Diffs = np.zeros((nDisps, nCons, nCons, 2)) * np.nan;
   bw34Diffs_split = np.zeros((nDisps, nCons, nCons, 2, 2)) * np.nan; # 2nd to last dim is [lower,upper] half rel. to peak
   pSfRats = np.zeros((nDisps, nCons, nCons, 2)) * np.nan;
   pSfModRat = np.zeros((nDisps, 2)) * np.nan; # derived measure from descrFits (see descr_prefSf)
   dog_bwHalfDiffs = np.zeros((nDisps, nCons, nCons, 2)) * np.nan;
   dog_bwHalfDiffs_split = np.zeros((nDisps, nCons, nCons, 2, 2)) * np.nan; # 2nd to last dim is [lower,upper] half rel. to peak
   dog_bw34Diffs = np.zeros((nDisps, nCons, nCons, 2)) * np.nan;
   dog_bw34Diffs_split = np.zeros((nDisps, nCons, nCons, 2, 2)) * np.nan; # 2nd to last dim is [lower,upper] half rel. to peak
   dog_pSfRats = np.zeros((nDisps, nCons, nCons, 2)) * np.nan;
   dog_pSfModRat = np.zeros((nDisps, 2)) * np.nan; # derived measure from descrFits (see descr_prefSf)

   sf70Rats = np.zeros((nDisps, nCons, nCons, 2)) * np.nan;
   sf70ModRat = np.zeros((nDisps, 2)) * np.nan; # derived measure from descrFits (see descr_prefSf)
   dog_sf70Rats = np.zeros((nDisps, nCons, nCons, 2)) * np.nan;
   dog_sf70ModRat = np.zeros((nDisps, 2)) * np.nan; # derived measure from descrFits (see descr_prefSf)
   sf75Rats = np.zeros((nDisps, nCons, nCons, 2)) * np.nan;
   sf75ModRat = np.zeros((nDisps, 2)) * np.nan; # derived measure from descrFits (see descr_prefSf)
   dog_sf75Rats = np.zeros((nDisps, nCons, nCons, 2)) * np.nan;
   dog_sf75ModRat = np.zeros((nDisps, 2)) * np.nan; # derived measure from descrFits (see descr_prefSf)
   sfERats = np.zeros((nDisps, nCons, nCons, 2)) * np.nan;
   sfEModRat = np.zeros((nDisps, 2)) * np.nan; # derived measure from descrFits (see descr_prefSf)
   dog_sfERats = np.zeros((nDisps, nCons, nCons, 2)) * np.nan;
   dog_sfEModRat = np.zeros((nDisps, 2)) * np.nan; # derived measure from descrFits (see descr_prefSf)

   c50Rats = np.zeros((nDisps, nSfs, nSfs)) * np.nan;
   # bwHalf, bw34, pSf, sfVar, sfCom, sf75, dog_sf75, sf70, dog_sf70, sfE, dog_sfE, dog_bwHalfDiffs, dog_bw34Diffs, dog_pSfRats
   # -- evaluated from data at 1:.33 contrast (only for single gratings)
   diffsAtThirdCon = np.zeros((nDisps, 14, )) * np.nan;
   # nDisps, half/three-fourths, lower/upper
   diffsAtThirdCon_bwSplit = np.zeros((nDisps, 2, 2, )) * np.nan;
   # just a separate one for LSFV, since we don't know if it will be kept
   diffsAtThirdCon_lsfv = np.zeros((nDisps, )) * np.nan;

   # let's also keep a simple array for the index for full, one-third, and lowest valid contrast given descr. sf fit
   # -- why nDisps, 4? full -- one-third -- lowest (with descr.) -- lowest (with DoG)
   relDescr_inds = -1 * np.ones((nDisps, 4, ), dtype=int); # we'll only count indices >=0 as valid

   ## first, get mean/median/maximum response over all conditions
   if isSach or isBB: # we can use ternary, since we can safely assume that ONLY one of isSach or isBB is true
     respByCond = data['f1'] if isSach else maskAll.flatten();
   else:
     respByCond = resps[0].flatten();
   mn_med_max = np.array([np.nanmean(respByCond), np.nanmedian(respByCond), np.nanmax(respByCond)])

   ## then, get the superposition ratio
   ### WARNING: resps is WITHOUT any rvcAdjustment (computed above)
   if expInd > 2:
     rvcFitsCurr = rvcFits[cell_ind];
     supr_ind = np.nan;
   else:
     rvcFitsCurr = None;
     supr_ind = np.nan;

   # let's figure out if simple or complex
   if isSach or isBB:
     if isSach:
       f1f0_ratio = np.nan
       respMeasure = 1; # we look at F1 for sach (LGN data)
     if isBB:
       f1f0_ratio = bb_compute_f1f0(expInfo)[0]; # yes, we're duplicating this (have already computed by this point)
       respMeasure = f1f0_ratio > 1;
   else:
     f1f0_ratio = compute_f1f0(tr, cell_ind+1, expInd, data_loc, dF_nm)[0]; # f1f0 ratio is 0th output
     respMeasure = 0; # assume it's DC by default
   force_dc = False; force_f1 = False;
   if expDir == 'LGN/' or isSach:
     force_f1 = True;
   if expDir == 'V1_orig/' or expDir == 'altExp/':
     force_dc = True;
   if f1f0_ratio < 1 or force_dc is True: # i.e. if we're in LGN, DON'T get baseline, even if f1f0 < 1 (shouldn't happen)
     # NOTE: rvcMod=-1 since we're passing in loaded rvcFits (i.e. rvcFitsCurr is the fits, not just the name of the file)
     if isBB: # must do this differently for Bauman+Bonds experiments
        baseline_resp = expInfo['blank']['mean']
     else:
        spikes_rate = get_adjusted_spikerate(tr, cell_ind, expInd, data_loc, rvcFitsCurr, rvcMod=-1, baseline_sub=False, force_dc=force_dc, force_f1=force_f1); # and get spikes_rate so we can do 
        baseline_resp = blankResp(tr, expInd, spikes=spikes_rate, spksAsRate=True)[0];
   else:
     baseline_resp = None;
     respMeasure = 1; # then it's actually F1!
   # set the respMeasure in the meta dictionary
   meta['respMeasure'] = respMeasure;

   eFrac = 1-np.divide(1, np.e); # will be used to compute the SF at which response has reduced by 1/e (about 63% of peak)

   if isSach or isBB: # as before, this means Sach's data, not the Sach formulation of the DoG
     # we'll write a simple function that helps expand the dimensions of each dictionary key to [nDisp, X, Y]
     def expand_sach(currDict):
       for key in currDict: # just expand along first dim
          try:
             currDict[key] = np.expand_dims(currDict[key], axis=0);
          except:
             pass;
       return currDict;

   if isSach:
      dogFits[cell_ind] = expand_sach(dogFits[cell_ind]);
      descrFits[cell_ind] = expand_sach(descrFits[cell_ind]);
      rvcFits[cell_ind] = expand_sach(rvcFits[cell_ind]);
   if isBB: # then we have to 1) unpack the DC OR F1; 2) get just the mask response; 3) also expand
      try:
        dogFits[cell_ind] = expand_sach(dogFits[cell_ind][respKey]['mask']);
        descrFits[cell_ind] = expand_sach(descrFits[cell_ind][respKey]['mask']);
        rvcFits[cell_ind] = expand_sach(rvcFits[cell_ind][respKey]['mask']);
      except:
        print('****Cell ind %d****' % cell_ind);

   for d in range(nDisps):

     #######
     ## spatial frequency stuff
     #######
     for c in range(nCons):

       # zeroth...model-free metrics
       if isSach:
         curr_sfInd = np.arange(0, len(stimVals[2])); # all SFS are valid for Sach
       elif isBB:
         curr_sfInd = np.arange(0, len(stimVals[2]));
       else:
         curr_sfInd = get_valid_sfs(tr, d, c, expInd=expInd, stimVals=stimVals, validByStimVal=validByStimVal)
       curr_sfs   = stimVals[2][curr_sfInd];
       curr_resps = sfTuning[d, curr_sfInd, c];
       sf_gt0 = np.where(curr_sfs>0)[0]; # if we include a zero-SF condition, then everything goes to zero!
       sfCom[d, c] = sf_com(curr_resps[sf_gt0], curr_sfs[sf_gt0])
       sfVar[d, c] = sf_var(curr_resps[sf_gt0], curr_sfs[sf_gt0], sfCom[d, c]);
       # get the c.o.m. based on the restricted set of SFs, only
       if cut_sf is not None:
         cut_sfs, cut_resps = np.array(stimVals[2])[cut_sf], sfTuning[d, cut_sf, c];
         sfComCut[d, c] = sf_com(cut_resps, cut_sfs)

       # first, DoG fit
       if cell_ind in dogFits:
         try:
           varExpl = dogFits[cell_ind]['varExpl'][d, c];
           if varExpl > dog_varExplThresh:
             # on data
             dog_pSf[d, c] = dogFits[cell_ind]['prefSf'][d, c]
             dog_charFreq[d, c] = dogFits[cell_ind]['charFreq'][d, c]
             dog_varExpl[d, c] = varExpl;
             # get the params and do bandwidth, high-freq. cut-off measures
             dog_params_curr = dogFits[cell_ind]['params'][d, c];
             dog_bwHalf[d, c] = compute_SF_BW(dog_params_curr, 0.5, sf_range=sf_range, sfMod=dogMod, baseline=baseline_resp)[1]
             dog_bw34[d, c] = compute_SF_BW(dog_params_curr, 0.75, sf_range=sf_range, sfMod=dogMod, baseline=baseline_resp)[1]
             for splitInd,splitHalf in enumerate([-1,1]):
                dog_bwHalf_split[d, c, splitInd] = compute_SF_BW(dog_params_curr, height=0.5, sf_range=sf_range, which_half=splitHalf, sfMod=dogMod, baseline=baseline_resp)[1];
                dog_bw34_split[d, c, splitInd] = compute_SF_BW(dog_params_curr, height=0.75, sf_range=sf_range, which_half=splitHalf, sfMod=dogMod, baseline=baseline_resp)[1]
             # -- note that for sf_highCut with the Diff. of Gauss models, we do NOT need to subtract the baseline
             # -- why? becaue the descr. model is already fit on top of the baseline (i.e. the descr. fit response does not include baseline)
             dog_sf70[d, c] = sf_highCut(dog_params_curr, sfMod=dogMod, frac=0.7, sfRange=(0.1, 15));
             dog_sf75[d, c] = sf_highCut(dog_params_curr, sfMod=dogMod, frac=0.75, sfRange=(0.1, 15));
             dog_sfE[d, c] = sf_highCut(dog_params_curr, sfMod=dogMod, frac=eFrac, sfRange=(0.1, 15));
         except: # then this dispersion does not have that contrast value, but it's ok - we already have nan
           pass 

       # then, non-DoG descr fit
       if cell_ind in descrFits:
         try:
           varExpl = descrFits[cell_ind]['varExpl'][d, c];
           if varExpl > varExplThresh:
             # on data
             lsfv[d, c] = compute_LSFV(descrFits[cell_ind]['params'][d, c, :]);
             bwHalf[d, c] = compute_SF_BW(descrFits[cell_ind]['params'][d, c, :], height=0.5, sf_range=sf_range)[1]
             bw34[d, c] = compute_SF_BW(descrFits[cell_ind]['params'][d, c, :], height=0.75, sf_range=sf_range)[1]
             for splitInd,splitHalf in enumerate([-1,1]):
                bw_sigma[d, c, splitInd] = descrFits[cell_ind]['params'][d, c, 3+splitInd]
                bwHalf_split[d, c, splitInd] = compute_SF_BW(descrFits[cell_ind]['params'][d, c, :], height=0.5, sf_range=sf_range, which_half=splitHalf)[1];
                bw34_split[d, c, splitInd] = compute_SF_BW(descrFits[cell_ind]['params'][d, c, :], height=0.75, sf_range=sf_range, which_half=splitHalf)[1]
             pSf[d, c] = descrFits[cell_ind]['params'][d, c, muLoc]
             curr_params = descrFits[cell_ind]['params'][d, c, :];
             sf70[d, c] = sf_highCut(curr_params, sfMod=0, frac=0.7, sfRange=(0.1, 15), baseline_sub=baseline_resp);
             sf75[d, c] = sf_highCut(curr_params, sfMod=0, frac=0.75, sfRange=(0.1, 15), baseline_sub=baseline_resp);
             sfE[d, c] = sf_highCut(curr_params, sfMod=0, frac=eFrac, sfRange=(0.1, 15), baseline_sub=baseline_resp);
             sfVarExpl[d, c] = varExpl;
         except: # then this dispersion does not have that contrast value, but it's ok - we already have nan
             pass 

     # Now, compute the derived pSf Ratio
     try:
       _, psf_model, opt_params = dog_prefSfMod(descrFits[cell_ind], allCons=cons, disp=d, varThresh=varExplThresh, dog_model=descrMod)
       valInds = np.where(descrFits[cell_ind]['varExpl'][d, :] > varExplThresh)[0];
       if len(valInds) > 1:
         extrema = [cons[valInds[0]], cons[valInds[-1]]];
         logConRat = np.log2(extrema[1]/extrema[0]);
         evalPsf = psf_model(*opt_params, con=extrema);
         evalRatio = evalPsf[1]/evalPsf[0];
         pSfModRat[d] = [np.log2(evalRatio), np.log2(evalRatio)/logConRat];
     except: # then likely, no rvc/descr fits...
       pass 
     # and likewise for DoG
     try:
       _, psf_model, opt_params = dog_prefSfMod(dogFits[cell_ind], allCons=cons, disp=d, varThresh=varExplThresh, dog_model=descrMod)
       valInds = np.where(dogFits[cell_ind]['varExpl'][d, :] > dog_varExplThresh)[0];
       if len(valInds) > 1:
         extrema = [cons[valInds[0]], cons[valInds[-1]]];
         logConRat = np.log2(extrema[1]/extrema[0]);
         evalPsf = psf_model(*opt_params, con=extrema);
         evalRatio = evalPsf[1]/evalPsf[0];
         dog_pSfModRat[d] = [np.log2(evalRatio), np.log2(evalRatio)/logConRat];
     except: # then likely, no rvc/descr fits...
       pass 

     for cutInd, cutVal in enumerate([0.7, 0.75, eFrac]):
        for (fitInd, currFit), whichMod in zip(enumerate([descrFits, dogFits]), [descrMod, dogMod]):
           sfModRatCurr = [np.nan, np.nan]
           try:
              _, psf_model, opt_params = dog_prefSfMod(currFit[cell_ind], allCons=cons, disp=d, varThresh=varExplThresh, dog_model=whichMod, highCut=cutVal, base_sub=baseline_resp)
              valInds = np.where(currFit[cell_ind]['varExpl'][d, :] > varExplThresh)[0];
              if len(valInds) > 1:
                 extrema = [cons[valInds[0]], cons[valInds[-1]]];
                 logConRat = np.log2(extrema[1]/extrema[0]);
                 evalPsf = psf_model(*opt_params, con=extrema);
                 evalRatio = evalPsf[1]/evalPsf[0];
                 sfModRatCurr = [np.log2(evalRatio), np.log2(evalRatio)/logConRat];
           except: # then likely, no rvc/descr fits...
              pass 
           ### then, organize for the outputs
           if cutInd == 0:
             if fitInd == 0:
                sf70ModRat[d] = np.copy(sfModRatCurr);
             else:
                dog_sf70ModRat[d] = np.copy(sfModRatCurr);
           elif cutInd == 1:
             if fitInd == 0:
                sf75ModRat[d] = np.copy(sfModRatCurr);
             else:
                dog_sf75ModRat[d] = np.copy(sfModRatCurr);
           elif cutInd == 2:
             if fitInd == 0:
                sfEModRat[d] = np.copy(sfModRatCurr);
             else:
                dog_sfEModRat[d] = np.copy(sfModRatCurr);

     #######
     ## RVC stuff
     #######
     for s in range(nSfs):
       if cell_ind in rvcFits:
         # on data
         try: # if from fit_RVC_F0
             conGain[d, s] = rvcFits[cell_ind]['conGain'][d,s];
             c50[d, s] = get_c50(rvcMod, rvcFits[cell_ind]['params'][d, s, :]);
             c50_emp[d, s], c50_eval[d, s] = c50_empirical(rvcMod, rvcFits[cell_ind]['params'][d, s, :]);
             c50_varExpl[d, s] = rvcFits[cell_ind]['varExpl'][d,s];
         except: # might just be arranged differently...(not fit_rvc_f0)
             try: # TODO: investigate why c50 param is saving for nan fits in hf.fit_rvc...
               if ~np.isnan(rvcFits[cell_ind][d]['loss'][s]): # only add it if it's a non-NaN loss value...
                 conGain[d, s] = rvcFits[cell_ind][d]['conGain'][s];
                 c50[d, s] = get_c50(rvcMod, rvcFits[cell_ind][d]['params'][s]);
                 c50_emp[d, s], c50_eval[d, s] = c50_empirical(rvcMod, rvcFits[cell_ind][d]['params'][s]);
                 c50_varExpl[d, s] = rvcFits[cell_ind][d]['varExpl'][s];
             except: # then this dispersion does not have that SF value, but it's ok - we already have nan
               pass;

     ## Now, after going through all cons/sfs, compute ratios/differences
     # first, with contrast
     for comb in itertools.combinations(range(nCons), 2):
       # first, in raw values [0] and per log2 contrast change [1] (i.e. log2(highCon/lowCon))
       conChange = np.log2(cons[comb[1]]/cons[comb[0]]);

       diff = bwHalf[d,comb[1]] - bwHalf[d,comb[0]];
       bwHalfDiffs[d,comb[0],comb[1]] = [diff, diff/conChange];
       diff_dog = dog_bwHalf[d,comb[1]] - dog_bwHalf[d,comb[0]];
       dog_bwHalfDiffs[d,comb[0],comb[1]] = [diff_dog, diff_dog/conChange];
       # -- BW split, first lower, then upper 
       for sideInd in [0,1]:
         diff = bwHalf_split[d,comb[1],sideInd] - bwHalf_split[d,comb[0],sideInd]
         bwHalfDiffs_split[d,comb[0],comb[1], sideInd, :] = [diff, diff/conChange];
         diff_dog = dog_bwHalf_split[d,comb[1],sideInd] - dog_bwHalf_split[d,comb[0],sideInd]
         bwHalfDiffs_split[d,comb[0],comb[1], sideInd, :] = [diff_dog, diff_dog/conChange];
       diff = bw34[d,comb[1]] - bw34[d,comb[0]];
       bw34Diffs[d,comb[0],comb[1]] = [diff, diff/conChange];
       diff_dog = dog_bw34[d,comb[1]] - dog_bw34[d,comb[0]];
       dog_bw34Diffs[d,comb[0],comb[1]] = [diff_dog, diff_dog/conChange];
       # -- BW split, first lower, then upper 
       for sideInd in [0,1]:
         diff = bw34_split[d,comb[1],sideInd] - bw34_split[d,comb[0],sideInd]
         bw34Diffs_split[d,comb[0],comb[1], sideInd, :] = [diff, diff/conChange];
         diff_dog = dog_bw34_split[d,comb[1],sideInd] - dog_bw34_split[d,comb[0],sideInd]
         dog_bw34Diffs_split[d,comb[0],comb[1], sideInd, :] = [diff_dog, diff_dog/conChange];

       # NOTE: For pSf, we will log2 the ratio, such that a ratio of 0 
       # reflects the prefSf remaining constant (i.e. log2(1/1)-->0)
       rat = pSf[d,comb[1]] / pSf[d,comb[0]];
       pSfRats[d,comb[0],comb[1]] = [np.log2(rat), np.log2(rat)/conChange];
       dog_rat = dog_pSf[d,comb[1]] / dog_pSf[d,comb[0]];
       dog_pSfRats[d,comb[0],comb[1]] = [np.log2(dog_rat), np.log2(dog_rat)/conChange];
       # -- and we'll do the same for LSFV
       rat = lsfv[d,comb[1]] / lsfv[d,comb[0]];
       lsfvRats[d,comb[0],comb[1]] = [np.log2(rat), np.log2(rat)/conChange];

       ## now, model-free metrics
       sfVarDiffs[d,comb[0],comb[1]] = sfVar[d,comb[1]] - sfVar[d,comb[0]]
       rat = sfCom[d, comb[1]] / sfCom[d, comb[0]];
       sfComRats[d,comb[0],comb[1]] = [np.log2(rat), np.log2(rat)/conChange]

       ## and sf70 and dog_sf70 ratios
       rat = sf70[d, comb[1]] / sf70[d, comb[0]];
       sf70Rats[d,comb[0],comb[1]] = [np.log2(rat), np.log2(rat)/conChange]
       rat = dog_sf70[d, comb[1]] / dog_sf70[d, comb[0]];
       dog_sf70Rats[d,comb[0],comb[1]] = [np.log2(rat), np.log2(rat)/conChange]

       ## and sf75 and dog_sf75 ratios
       rat = sf75[d, comb[1]] / sf75[d, comb[0]];
       sf75Rats[d,comb[0],comb[1]] = [np.log2(rat), np.log2(rat)/conChange]
       rat = dog_sf75[d, comb[1]] / dog_sf75[d, comb[0]];
       dog_sf75Rats[d,comb[0],comb[1]] = [np.log2(rat), np.log2(rat)/conChange]

       ## and sfE and dog_sfE ratios (reduction by 1/e)
       rat = sfE[d, comb[1]] / sfE[d, comb[0]];
       sfERats[d,comb[0],comb[1]] = [np.log2(rat), np.log2(rat)/conChange]
       rat = dog_sfE[d, comb[1]] / dog_sfE[d, comb[0]];
       dog_sfERats[d,comb[0],comb[1]] = [np.log2(rat), np.log2(rat)/conChange]

     # then, as function of SF
     for comb in itertools.permutations(range(nSfs), 2):
       c50Rats[d,comb[0],comb[1]] = c50[d,comb[1]] / c50[d,comb[0]]

     # finally, just get the straight-from-data ratio/diff evaluated from highest to one-third-of-max contrast
     hiConInd = np.argmax(cons[val_con_by_disp[d]]); # the highest contrast for this dispersion
     hiConValue = cons[val_con_by_disp[d][hiConInd]];
     # -- what index has the contrast closest to one-third of the max?
     thirdConInd = np.argmin(np.square(cons[val_con_by_disp[d]] - (hiConValue/ 3)));
     thirdConValue = cons[val_con_by_disp[d][thirdConInd]];
     if np.abs(thirdConValue - hiConValue/3) < .1*hiConValue: # should be within 10% of the high contrast value to count...
       relDescr_inds[d, 0] = val_con_by_disp[d][hiConInd]; # highest
       relDescr_inds[d, 1] = val_con_by_disp[d][thirdConInd]; # one-third
       for indAdd, metr in enumerate([pSf, dog_pSf]):
         try:
           valDescrFits = np.where(~np.isnan(metr[d, :]))[0]; # might be empty (i.e. no fits which passed varExpl_thresh/dog_varExpl_thresh)
           lowInd, hiInd = valDescrFits[0], valDescrFits[-1]; # get the lowest/highest indicies with a still-valid descriptive fit (if cI invalid, then pSf[d,cI] will be nan)
           relDescr_inds[d, 2+indAdd] = lowInd if cons[lowInd] < cons[hiInd] else hiInd;
         except:
           pass
       diffsAtThirdCon[d, :] = np.array([bwHalfDiffs[d, relDescr_inds[d, 1], relDescr_inds[d, 0], rawInd],
                                bw34Diffs[d, relDescr_inds[d, 1], relDescr_inds[d, 0], rawInd],
                                pSfRats[d, relDescr_inds[d, 1], relDescr_inds[d, 0], rawInd],
                                sfVarDiffs[d, relDescr_inds[d, 1], relDescr_inds[d, 0], rawInd],
                                sfComRats[d, relDescr_inds[d, 1], relDescr_inds[d, 0], rawInd],
                                sf70Rats[d, relDescr_inds[d, 1], relDescr_inds[d, 0], rawInd],
                                dog_sf70Rats[d, relDescr_inds[d, 1], relDescr_inds[d, 0], rawInd],
                                sf75Rats[d, relDescr_inds[d, 1], relDescr_inds[d, 0], rawInd],
                                dog_sf75Rats[d, relDescr_inds[d, 1], relDescr_inds[d, 0], rawInd],
                                sfERats[d, relDescr_inds[d, 1], relDescr_inds[d, 0], rawInd],
                                dog_sfERats[d, relDescr_inds[d, 1], relDescr_inds[d, 0], rawInd],
                                dog_bwHalfDiffs[d, relDescr_inds[d, 1], relDescr_inds[d, 0], rawInd],
                                dog_bw34Diffs[d, relDescr_inds[d, 1], relDescr_inds[d, 0], rawInd],
                                dog_pSfRats[d, relDescr_inds[d, 1], relDescr_inds[d, 0], rawInd]]);

       for metrInd,bw_split_curr in enumerate([bwHalfDiffs_split, bw34Diffs_split]):
         diffsAtThirdCon_bwSplit[d,metrInd,:] = bw_split_curr[d, relDescr_inds[d, 1], relDescr_inds[d, 0], rawInd];
       diffsAtThirdCon_lsfv[d] = lsfvRats[d, relDescr_inds[d, 1], relDescr_inds[d, 0], rawInd];


   print('\tdiffsAtThirdCon pSf||sf70||dogSf70 ...  = (%.2f, %.2f, %.2f)' % (diffsAtThirdCon[0, 2], diffsAtThirdCon[0, 5], diffsAtThirdCon[0, 6]));
   print('\tmodRat pSf||sf70||dogSf70 ...  = (%.2f, %.2f, %.2f)' % (pSfModRat[0, 1], sf70ModRat[0, 1], dog_sf70ModRat[0, 1]));

   dataMetrics = dict([('sfCom', sfCom),
                      ('sfComCut', sfComCut),
                      ('sfVar', sfVar),
                      ('f1f0_ratio', f1f0_ratio),
                      ('lsfv', lsfv),
                      ('bw_sigma', bw_sigma),
                      ('bwHalf', bwHalf),
                      ('dog_bwHalf', dog_bwHalf),
                      ('bwHalf_split', bwHalf_split),
                      ('dog_bwHalf_split', dog_bwHalf_split),
                      ('bw34', bw34),
                      ('dog_bw34', dog_bw34),
                      ('bw34_split', bw34_split),
                      ('dog_bw34_split', dog_bw34_split),
                      ('pSf', pSf),
                      ('sf70', sf70),
                      ('dog_sf70', dog_sf70),
                      ('sf75', sf75),
                      ('dog_sf75', dog_sf75),
                      ('sfE', sfE),
                      ('dog_sfE', dog_sfE),
                      ('conGain', conGain),
                      ('c50', c50),
                      ('c50_emp', c50_emp),
                      ('c50_eval', c50_eval),
                      ('c50_varExpl', c50_varExpl),
                      ('dog_pSf', dog_pSf),
                      ('dog_charFreq', dog_charFreq),
                      ('dog_varExpl', dog_varExpl),
                      ('bwHalfDiffs', bwHalfDiffs),
                      ('bwHalfDiffs_split', bwHalfDiffs_split),
                      ('bw34Diffs', bw34Diffs),
                      ('bw34Diffs_split', bw34Diffs_split),
                      ('lsfvRats', lsfvRats),
                      ('pSfRats', pSfRats),
                      ('pSfModRat', pSfModRat),
                      ('dog_pSfRats', dog_pSfRats),
                      ('dog_pSfModRat', dog_pSfModRat),
                      ('sf70Rats', sf70Rats),
                      ('sf70ModRat', sf70ModRat),
                      ('dog_sf70Rats', dog_sf70Rats),
                      ('dog_sf70ModRat', dog_sf70ModRat),
                      ('sf75Rats', sf75Rats),
                      ('sf75ModRat', sf75ModRat),
                      ('dog_sf75Rats', dog_sf75Rats),
                      ('dog_sf75ModRat', dog_sf75ModRat),
                      ('sfERats', sfERats),
                      ('sfEModRat', sfEModRat),
                      ('dog_sfERats', dog_sfERats),
                      ('dog_sfEModRat', dog_sfEModRat),
                      ('sfVarDiffs', sfVarDiffs),
                      ('sfComRats', sfComRats),
                      ('sfVarExpl', sfVarExpl),
                      ('c50Rats', c50Rats),
                      ('suppressionIndex', supr_ind),
                      ('diffsAtThirdCon', diffsAtThirdCon),
                      ('diffsAtThirdCon_bwSplit', diffsAtThirdCon_bwSplit),
                      ('diffsAtThirdCon_lsfv', diffsAtThirdCon_lsfv),
                      ('relDescr_inds', relDescr_inds),
                      ('mn_med_max', mn_med_max)
                      ]);

   ###########
   ### model
   ###########
   if 'pyt' in fLW_nm: # i.e. it's a pytorch fit; we'll assume that if one is pytorch fit, the other is, too!
      try: # F1 or F0
         respStr = get_resp_str(respMeasure);
         nllW, paramsW = [fitListWght[cell_ind][respStr]['NLL'], fitListWght[cell_ind][respStr]['params']];
         nllF, paramsF = [fitListFlat[cell_ind][respStr]['NLL'], fitListFlat[cell_ind][respStr]['params']];
         try:
            # first, for weighted model
            varExplW = fitListWght[cell_ind][respStr]['varExpl'];
            varExplW_SF = fitListWght[cell_ind][respStr]['varExpl_SF'];
            varExplW_con = fitListWght[cell_ind][respStr]['varExpl_con'];
            # then, flat model
            varExplF = fitListFlat[cell_ind][respStr]['varExpl'];
            varExplF_SF = fitListFlat[cell_ind][respStr]['varExpl_SF'];
            varExplF_con = fitListFlat[cell_ind][respStr]['varExpl_con'];
         except:
            varExplW, varExplW_SF, varExplW_con = None, None, None
            varExplF, varExplF_SF, varExplF_con = None, None, None

         model = dict([('NLL_wght', nllW),
                     ('params_wght', paramsW),
                     ('NLL_flat', nllF),
                     ('params_flat', paramsF),
                     ('varExplW', varExplW),
                     ('varExplW_SF', varExplW_SF),
                     ('varExplW_con', varExplW_con),
                     ('varExplF', varExplF),
                     ('varExplF_SF', varExplF_SF),
                     ('varExplF_con', varExplF_con)
                    ])
      except:
         model = dict([('NLL_wght', np.nan),
                       ('params_wght', []),
                       ('NLL_flat', np.nan),
                       ('params_flat', []),
                       ('varExplW', None),
                       ('varExplW_SF', None),
                       ('varExplW_con', None),
                       ('varExplF', None),
                       ('varExplF_SF', None),
                       ('varExplF_con', None)
                    ])
   else:
     try:
       nllW, paramsW = [fitListWght[cell_ind]['NLL'], fitListWght[cell_ind]['params']];
       nllF, paramsF = [fitListFlat[cell_ind]['NLL'], fitListFlat[cell_ind]['params']];
       # and other other future measures?

       model = dict([('NLL_wght', nllW),
                    ('params_wght', paramsW),
                    ('NLL_flat', nllF),
                    ('params_flat', paramsF)
                   ])
     except: # then no model fit!
       model = dict([('NLL_wght', np.nan),
                    ('params_wght', []),
                    ('NLL_flat', np.nan),
                    ('params_flat', [])
                   ])

   ###########
   ### now, gather all together in one dictionary
   ###########
   cellSummary = dict([('metadata', meta),
                      ('metrics', dataMetrics),
                      ('model', model),
                      ('superpos', suppr),
                      ('basics', basics_list)]);

   return cellSummary;

def jl_create(base_dir, expDirs, expNames, fitNamesWght, fitNamesFlat, descrNames, dogNames, rvcNames, rvcMods,
              conDig=1, sf_range=[0.1, 10], rawInd=0, muLoc=2, varExplThresh=75, dog_varExplThresh=60, descrMod=0, dogMod=1, toPar=1):
  ''' create the "super structure" that we use to analyze data across multiple versions of the experiment
      TODO: update this to get proper spikes/tuning measures based on f1/f0 ratio (REQUIRES descrFits to be like rvcFits, i.e. fit F1 or F0 responses, accordingly)
      inputs:
        baseDir      - what is the base directory?
        expDirs      - what are the directories of the experiment directory
        expNames     - names of the dataLists
        fitNamesWght - names of the model fitList with weighted normalization
        fitNamesFlat - as above
        descrNames   - names of the non-DoG descriptive SF fits
        dogNames     - names of the DoG descriptive SF fits
        rvcNames     - names of the response-versus-contrast fits
        rvcMods

        [default inputs]
        [conDig]     - how many decimal places to round contrast value when testing equality
        [sf_range]   - what bounds to use when computing SF bandwidth
        [rawInd]     - for accessing ratios/differences that we pass into diffsAtThirdCon
        [mu loc]     - what index into corresponding parameter array is the peak SF value?
        [{dog_}vaExplThresh] - only fits with >= % variance explained have their paramter values added for consideration
  '''

  np = numpy;
  jointList = [];
  jointListAsDict = dict();
  totCells = 0;

  for expDir, dL_nm, fLW_nm, fLF_nm, dF_nm, dog_nm, rv_nm, rvcMod in zip(expDirs, expNames, fitNamesWght, fitNamesFlat, descrNames, dogNames, rvcNames, rvcMods):

    #if 'BB' not in expDir:
    #   continue;

    # get the current directory, load data list
    data_loc = base_dir + expDir + 'structures/';    
    dataList = np_smart_load(data_loc + dL_nm);
    fitListWght = np_smart_load(data_loc + fLW_nm);
    fitListFlat = np_smart_load(data_loc + fLF_nm);
    descrFits = np_smart_load(data_loc + dF_nm);
    dogFits = np_smart_load(data_loc + dog_nm);
    rvcFits = np_smart_load(data_loc + rv_nm);
    try:
      superAnalysis = np_smart_load(data_loc + 'superposition_analysis_210824.npy');
    except:
      superAnalysis = None;

    # Now, go through for each cell in the dataList
    if 'sach' in expDir: # why? For sach data, the keys are first and all information is contained within each cell
      nCells = len(dataList);
      isSach = 1;
    else: 
      nCells = len(dataList['unitName']);
      isSach = 0;
    isBB = 1 if 'BB' in expDir else 0;

    if toPar:
      perCell_summary = partial(jl_perCell, dataList=dataList, descrFits=descrFits, dogFits=dogFits, rvcFits=rvcFits, expDir=expDir, data_loc=data_loc, dL_nm=dL_nm, fLW_nm=fLW_nm, fLF_nm=fLF_nm, dF_nm=dF_nm, dog_nm=dog_nm, rv_nm=rv_nm, superAnalysis=superAnalysis, conDig=conDig, sf_range=sf_range, rawInd=rawInd, muLoc=muLoc, varExplThresh=varExplThresh, dog_varExplThresh=dog_varExplThresh, descrMod=descrMod, dogMod=dogMod, isSach=isSach, rvcMod=rvcMod, isBB=isBB)
      
      nCpu = mp.cpu_count();
      with mp.Pool(processes = nCpu) as pool:
         cellSummaries = pool.map(perCell_summary, range(nCells));
      for cell_ind, cellSummary in enumerate(cellSummaries):
        currKey = totCells + cell_ind
        jointListAsDict[currKey] = cellSummary;

      totCells += nCells;
    else:
      for cell_ind in range(nCells):

        print('%s/%d' % (expDir, 1+cell_ind));
        cellSummary = jl_perCell(cell_ind, dataList, descrFits, dogFits, rvcFits, expDir, data_loc, dL_nm, fLW_nm, fLF_nm, dF_nm, dog_nm, rv_nm, superAnalysis, conDig=conDig, sf_range=sf_range, rawInd=rawInd, muLoc=muLoc, varExplThresh=varExplTresh, dog_varExplThresh=dog_varExplThresh, descrMod=descrMod, dogMod=dogMod)
        jointList.append(cellSummary);

  if toPar:
    return jointListAsDict;
  else:
    return jointList;

def jl_get_metric_byCon(jointList, metric, conVal, disp, conTol=0.02):
  ''' given a "jointList" structure, get the specified metric (as string) for a given conVal & dispersion
      returns: array of metric value for a given con X disp
      inputs:
        jointList - see above (jl_create)
        metric    - as a string, which metric are you querying (e.g. 'pSf', 'sfCom', etc)
        conVal    - what contrast (e.g. 33% or 99% or ...)
        disp      - which dispersion (0, 1, ...)
        [conTol]  - we consider the contrast to match conVal if within +/- 2% (given experiment, this is satisfactory to match con level across dispersions, versions)
  '''
  np = numpy;
  nCells = len(jointList);
  output = np.nan * np.zeros((nCells, ));

  # how to handle contrast? for each cell, find the contrast that is valid for that dispersion and matches the con_lvl
  # to within some tolerance (e.g. +/- 0.01, i.e. 1% contrast)

  for ind, i in enumerate(jointList.keys()): #range(nCells): # holdover from when jointList was list rather than dict
      #######
      # get structure, metadata, etc
      #######
      curr_cell = jointList[i]
      curr_metr = curr_cell['metrics']['%s' % metric];
      curr_meta = curr_cell['metadata'];
      curr_cons = curr_meta['stimVals'][1]; # stimVals[1] is list of contrasts
      curr_byDisp = curr_meta['val_con_by_disp'];
      if disp < len(curr_byDisp):
          curr_inds = curr_byDisp[disp];
      else:
          continue; # i.e. this dispersion isn't there...

      #######
      # get the specified contrast, package, and return
      #######
      curr_conVals = curr_cons[curr_inds];
      match_ind = np.where(np.abs(curr_conVals-conVal)<=conTol)[0];
      if np.array_equal(match_ind, []):
        continue; # i.e. didn't find the right contrast match

      full_con_ind = curr_byDisp[disp][match_ind[0]];
      output[ind] = curr_metr[disp][full_con_ind];

  return output;

def jl_get_metric_highComp(jointList, metric, whichMod, atLowest, disp=0, extraInds=None):
    ''' This should work for most (all?) descriptive metrics in the jointList
        Will return the two metrics (high, comp) and the corresponding comparison contrasts

        jointList -- the joint list (dictionary)
        metric    -- string (e.g. pSf or dog_varExpl)
        whichMod  -- 0 for flex. gauss, 1 for DoG
        atLowest  -- 0 if at fixed one-third contrast; 1 if at lowest contrast
        disp      -- integer, 0 for single gratings
        extraInds -- is there an extra dimension to index? if so, pass it in
    '''
    np = numpy;

    dogAdd = whichMod & atLowest
    comp = 1 + atLowest + dogAdd
    
    highInd = np.array([jointList[x]['metrics']['relDescr_inds'][disp,0] for x in jointList.keys()]);
    # compInd will first be at one third (when i==0), then at lowest valid contrast...
    compInd = np.array([jointList[x]['metrics']['relDescr_inds'][disp,comp] for x in jointList.keys()]);
    
    if extraInds is None:
        highSf = np.array([jointList[key]['metrics'][metric][disp, hI] for key,hI in zip(jointList.keys(), highInd)])
        compSf = np.array([jointList[key]['metrics'][metric][disp, cI] for key,cI in zip(jointList.keys(), compInd)])
    else:
        highSf = np.array([jointList[key]['metrics'][metric][disp, hI, extraInds] for key,hI in zip(jointList.keys(), highInd)])
        compSf = np.array([jointList[key]['metrics'][metric][disp, cI, extraInds] for key,cI in zip(jointList.keys(), compInd)])
    
    compCons = np.array([jointList[x]['metadata']['stimVals'][1][conInd] for x,conInd in zip(jointList.keys(), compInd)]);
    
    return highSf, compSf, compCons

##################################################################
##################################################################
##################################################################
### IV. RETURN TO DESCRIPTIVE FITS/ANALYSES
##################################################################
##################################################################
##################################################################

def blankResp(cellStruct, expInd, spikes=None, spksAsRate=False, returnRates=False):
    ''' optionally, pass in array of spikes (by trial) and flag for whether those spikes are rates or counts over the whole trial
        mu/std_err are ALWAYS rates
    '''
    # works for all experiment variants (checked 08.20.19)
    if 'sfm' in cellStruct:
      tr = cellStruct['sfm']['exp']['trial'];
    else:
      tr = cellStruct;
    if spikes is None: # else, we could've passed in adjusted spikes
      spikes = tr['spikeCount']; 
      spksAsRate = False; # NOTE: these are f0, only...
    if spksAsRate is True:
      divFactor = 1;
    else:
      divFactor = get_exp_params(expInd).stimDur;

    blank_tr = spikes[numpy.isnan(tr['con'][0])];
    mu = numpy.mean(numpy.divide(blank_tr, divFactor));
    std_err = sem(numpy.divide(blank_tr, divFactor));
    if returnRates:
      blank_tr = numpy.divide(blank_tr, divFactor);
    
    return mu, std_err, blank_tr;
    
def get_valid_trials(data, disp, con, sf, expInd, stimVals=None, validByStimVal=None):
  ''' Given a data and the disp/con/sf indices (i.e. integers into the list of all disps/cons/sfs
      Determine which trials are valid (i.e. have those stimulus criteria)
      RETURN list of valid trials, lists for all dispersion values, all contrast values, all sf values
  '''
  if stimVals is None or validByStimVal is None:
    _, stimVals, _, validByStimVal, _ = tabulate_responses(data, expInd);
  # otherwise, we've passed this in, so no need to call tabulate_responses again!

  # gather the conditions we need so that we can index properly
  valDisp = validByStimVal[0];
  valCon = validByStimVal[1];
  valSf = validByStimVal[2];

  allDisps = stimVals[0];
  allCons = stimVals[1];
  allSfs = stimVals[2];

  val_trials = numpy.where(valDisp[disp] & valCon[con] & valSf[sf]);

  return val_trials, allDisps, allCons, allSfs;

def get_valid_sfs(data, disp, con, expInd, stimVals=None, validByStimVal=None):
  ''' Self explanatory, innit? Returns the indices (into allSfs) of valid sfs for the given condition
      As input, disp/con should be indices into the valDisp/Con arrays (i.e. not values)
  '''
  if stimVals is None or validByStimVal is None:
    _, stimVals, _, validByStimVal, _ = tabulate_responses(data, expInd);
  # otherwise, we've passed this in, so no need to call tabulate_responses again!

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

## 

def get_condition(data, n_comps, con, sf):
    ''' Returns the trial responses (f0 and f1) and corresponding trials for a given 
        dispersion level (note: # of components), contrast, and spatial frequency
    '''
    np = numpy;
    conDig = 3; # default...

    val_disp = data['num_comps'] == n_comps;
    val_con = np.round(data['total_con'], conDig) == con;
    val_sf = data['cent_sf'] == sf;

    val_trials = np.where(val_disp & val_con & val_sf)[0]; # get as int array of indices rather than boolean array
 
    f0 = data['spikeCount'][val_trials];
    try:
      f1 = data['power_f1'][val_trials];
    except: # not every experiment/cell will have measured f1 responses
      f1 = np.nan * np.zeros_like(f0);

    return f0, f1, val_trials;

def get_conditionAdj(data, n_comps, con, sf, adjByTrial):
  ''' Returns the trial responses (f0 and f1) and corresponding trials for a given 
      dispersion level (note: # of components), contrast, and spatial frequency
      Note: Access trial responses from a vector of adjusted (i.e. "projected") responses
  '''
  np = numpy;
  conDig = 3; # default...

  val_disp = data['num_comps'] == n_comps;
  val_con = np.round(data['total_con'], conDig) == con;
  val_sf = np.round(data['cent_sf'], conDig) == np.round(sf, conDig);

  val_trials = np.where(val_disp & val_con & val_sf)[0]; # get as int array of indices rather than boolean array
  resps = adjByTrial[val_trials];

  return resps, val_trials;
    
def get_isolated_response(data, trials):
   ''' Given a set of trials (assumed to be all from one unique disp-con-sf set), collect the responses to the components of the 
       stimulus when presented in isolation - returns the mean/sem and individual trial responses
       Assumed to be for mixture stimuli
   '''
   np = numpy; conDig = 3;
   n_comps = np.unique(data['num_comps'][trials]);
   if len(n_comps) > 1:
     warnings.warn('must have only one level of dispersion for the requested trials');
     return [], [], [], [];
   n_comps = n_comps[0]; # get just the value so it's an integer rather than array

   # assumption is that #trials of mixture stimulus will be >= the number of repetitions of the isolated presentations of that stimulus component
   f0all = np.array(np.nan * np.zeros((n_comps, )), dtype='O'); # might have different number of responses for each component, so create object/flexible array
   f1all = np.array(np.nan * np.zeros((n_comps, )), dtype='O');
   f0summary = np.nan * np.zeros((n_comps, 2)); # mean/std in [:, 0 or 1], respectively
   f1summary = np.nan * np.zeros((n_comps, 2));

   cons = []; sfs = [];
   for i in range(n_comps):

     # now go through for each component and get the response to that stimulus component when presented alone
     con = np.unique(data['con'][i][trials]); cons.append(np.round(con, conDig));
     sf = np.unique(data['sf'][i][trials]); sfs.append(sf);

     if len(con)>1 or len(sf)>1:
       warnings.warn('the trials requested must have only one sf/con for a given stimulus component');
       return [], [], [], [];
     
     f0curr, f1curr, _ = get_condition(data, 1, np.round(con, conDig), sf); # 1 is for #components - we're looking for single component trials/responses
     f0all[i] = f0curr;
     f1all[i] = f1curr;

     f0summary[i, :] = [np.nanmean(f0all[i]), sem(f0all[i])]; # nanmean/std in case fewer presentations of individual component than mixture
     f1summary[i, :] = [np.nanmean(f1all[i]), sem(f1all[i])];

   return f0summary, f1summary, f0all, f1all, cons, sfs;

def get_isolated_responseAdj(data, trials, adjByTrial):
   ''' Given a set of trials (assumed to be all from one unique disp-con-sf set), collect the responses to the components of the 
       stimulus when presented in isolation - returns the mean/std and individual trial responses
       Assumed to be for mixture stimuli
   '''
   np = numpy; conDig = 3;
   n_comps = np.unique(data['num_comps'][trials]);
   if len(n_comps) > 1:
     warnings.warn('must have only one level of dispersion for the requested trials');
     return [], [], [], [];
   n_comps = n_comps[0]; # get just the value so it's an integer rather than array

   f1all = np.array(np.nan * np.zeros((n_comps, )), dtype='O');
   f1summary = np.nan * np.zeros((n_comps, 2));

   cons = []; sfs = [];
   for i in range(n_comps):
     # now go through for each component and get the response to that stimulus component when presented alone
     con = np.round(np.unique(data['con'][i][trials]), conDig); cons.append(con);
     sf = np.unique(data['sf'][i][trials]); sfs.append(sf);

     if len(con)>1 or len(sf)>1:
       warnings.warn('the trials requested must have only one sf/con for a given stimulus component');
       return [], [], [], [];
     # always getting for a single component (hence "1")
     curr_resps, _ = get_conditionAdj(data, 1, con, sf, adjByTrial);

     f1all[i] = curr_resps;
     f1summary[i, :] = [np.nanmean(f1all[i]), sem(f1all[i])];

   return f1summary, f1all, cons, sfs;

##

def tabulate_responses(cellStruct, expInd, modResp = [], mask=None, overwriteSpikes=None, respsAsRates=False, modsAsRate=False):
    ''' Given cell structure (and opt model responses), returns the following:
        (i) respMean, respStd, predMean, predStd, organized by condition; pred is linear prediction
        (ii) all_disps, all_cons, all_sfs - i.e. the stimulus conditions of the experiment
        (iii) the valid contrasts for each dispersion level
        (iv) valid_disp, valid_con, valid_sf - which conditions are valid for this particular cell
        (v) modRespOrg - the model responses organized as in (i) - only if modResp argument passed in
        NOTE: We pass in the overall spike counts (modResp; either real or predicted), and compute 
          the spike *rates* (i.e. spikes/s)
  
        overwriteSpikes: optional argument - if None, simply use F0 as saved in cell (i.e. spikeCount)
                         otherwise, pass in response by trial (i.e. not organized by condition; MUST be one value per trial, not per component)
                           e.g. F1, or adjusted F1 responses
        respsAsRates/modsAsRate: optional argument - if False (or if overwriteSpikes is None), then divide response by stimDur
    '''
    np = numpy;
    conDig = 3; # round contrast to the thousandth
    exper = get_exp_params(expInd);
    stimDur = exper.stimDur;
    
    if 'sfm' in cellStruct: 
      data = cellStruct['sfm']['exp']['trial'];
    else: # we've passed in sfm.exp.trial already
      data = cellStruct;

    if overwriteSpikes is None:
      respToUse = data['spikeCount'];
      respsAsRates = False; # ensure that we divide by stimDur
    else:
      respToUse = overwriteSpikes;

    if respsAsRates is True:
      respDiv = 1; # i.e. we don't need to divide by stimDur, since values are already in spikes/sec
    elif respsAsRates is False:
      respDiv = stimDur; # responses are NOT rates, yet, so divide by stimDur

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

    if mask is not None:
      inval[~mask] = 1; # these ones are invalid, i.e. the mask tells us where we should ook

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
        # WHY BREAK SOMETIMES?
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
                if expInd == 1: # also take care of excluding ori/rvc trials
                  valid_tr = valid_tr & ~inval;
                if np.all(np.unique(valid_tr) == False):
                    continue;
                    
                respMean[d, sf, con] = np.mean(respToUse[valid_tr]/respDiv);
                respStd[d, sf, con] = np.std(respToUse[valid_tr]/respDiv);
                #respMean[d, sf, con] = np.mean(data['spikeCount'][valid_tr]/stimDur);
                #respStd[d, sf, con] = np.std((data['spikeCount'][valid_tr]/stimDur));
                
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
                        continue; # empty ....
                    
                    curr_pred = curr_pred + np.mean(respToUse[val_tr]/respDiv);
                    curr_var = curr_var + np.var(respToUse[val_tr]/respDiv);
                    #curr_pred = curr_pred + np.mean(data['spikeCount'][val_tr]/stimDur);
                    #curr_var = curr_var + np.var(data['spikeCount'][val_tr]/stimDur);
                    
                predMean[d, sf, con] = curr_pred;
                predStd[d, sf, con] = np.sqrt(curr_var);
                
                if mod: # if needed, convert spike counts in each trial to spike rate (spks/s)
                    nTrCurr = sum(valid_tr); # how many trials are we getting?
                    if modsAsRate == True: # i.e. we passed on the model repsonses as rates already!
                      divFactor = 1;
                    else: # default behavior
                      divFactor = stimDur;
                    modRespOrg[d, sf, con, 0:nTrCurr] = np.divide(modResp[valid_tr], divFactor);

            if np.any(~np.isnan(respMean[d, :, con])):
                if ~np.isnan(np.nanmean(respMean[d, :, con])):
                    val_con_by_disp[d].append(con);
                    
    return [respMean, respStd, predMean, predStd], [all_disps, all_cons, all_sfs], val_con_by_disp, [valid_disp, valid_con, valid_sf], modRespOrg;

def organize_adj_responses(data, rvcFits, expInd, vecF1=0):
  ''' Used as a wrapper to call the organize_adj_responses function for a given experiment
      BUT, also has organize_adj_responses for newer experiments ( see "except" assoc. with main try)
      We set the organize_adj_responses separately for each experiment since some versions don't have adjusted responses
        and for those that do, some aspects of the calculation may differ
  '''
  ### First, we'll see if there is a direct helper_fcns method for this
  dir = get_exp_params(expInd).dir;
  to_import = dir.replace('/', '.') + 'helper_fcns';
  if os.path.isfile(dir + 'helper_fcns'): # i.e. what if we don't have an associated helper_fcns? then do "except"
    new_hf = il.import_module(to_import);
    if hasattr(new_hf, 'organize_adj_responses'):
      return new_hf.organize_adj_responses(data, rvcFits)[1]; # 2nd returned argument (pos 1) is responses by trial

  ### otherwise...
  # a "simple" adj responses
  nTr = len(data['num']);
  adjResps = numpy.nan * numpy.zeros((nTr, ), dtype='O');
  # first, get all of the stimulus conds
  _, conds, val_con_by_disp, val_by_stim_val, _ = tabulate_responses(data, expInd);
  all_d = conds[0];

  for d_ind, d in enumerate(all_d):
    val_cons = val_con_by_disp[d_ind];
    for c_ind_val, c_ind_total in enumerate(val_cons):
      val_sfs = get_valid_sfs(data, d_ind, c_ind_total, expInd, stimVals=conds, validByStimVal=val_by_stim_val);
      for s_ind_val, s_ind_total in enumerate(val_sfs):
        val_trials = get_valid_trials(data, d_ind, c_ind_total, s_ind_total, expInd, stimVals=conds, validByStimVal=val_by_stim_val)[0];
        if vecF1: # adjByTr are organized differently in these cases...
          # NOTE: We will only make it to this code if we're getting F1 responses and they are vecF1 corrected
          for vT in val_trials[0]: # cannot be done with list comprehension? I think, bc we're doing assignment
            adjResps[vT] = rvcFits[d_ind]['adjByTr'][vT] # non-valid stimComps will be zero'd anyway!
        else:
          # this is why we enumerate val_cons above - the index into val_cons is how we index into rvcFits
          curr_resps = rvcFits[d_ind]['adjByTr'][s_ind_total][c_ind_val];
          if d_ind > 0:
            try: # well, if the cell is simple & this is mixture stimulus, then we need to do this
              curr_flipped = switch_inner_outer(curr_resps, asnp=True);
              adjResps[val_trials] = curr_flipped;
            except: # otherwise, we "flatten" the incoming list
              adjResps[val_trials] = curr_resps.flatten();
          if d_ind == 0:
              adjResps[val_trials] = curr_resps.flatten();
     
  return adjResps;

def organize_resp(spikes, expStructure, expInd, mask=None, respsAsRate=False):
    ''' organizes the responses by condition given spikes, experiment structure, and expInd
        mask will be None OR list of trials to consider (i.e. trials not in mask/where mask is false are ignored)
        - respsAsRate: are "spikes" already in rates? if yes, pass in "True"; otherwise, we'll divide by stimDur to get rate
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

      if mask is None:
        mask = numpy.ones((len(data['blockID']), ), dtype=bool); # i.e. look at all trials
   
      rateOr = numpy.empty((0,));
      for iB in oriBlockIDs:
          indCond = numpy.where(data['blockID'][mask] == iB);
          if len(indCond[0]) > 0:
              rateOr = numpy.append(rateOr, numpy.mean(spikes[mask][indCond]));
          else:
              rateOr = numpy.append(rateOr, numpy.nan);

      # Analyze the stimulus-driven responses for the contrast response function
      conBlockIDs = numpy.arange(138, 156+1, 2);
      iC = 0;

      rateCo = numpy.empty((0,));
      for iB in conBlockIDs:
          indCond = numpy.where(data['blockID'][mask] == iB);   
          if len(indCond[0]) > 0:
              rateCo = numpy.append(rateCo, numpy.mean(spikes[mask][indCond]));
          else:
              rateCo = numpy.append(rateCo, numpy.nan);
    else:
      rateOr = None;
      rateCo = None;

    # Analyze the stimulus-driven responses for the spatial frequency mixtures
    if expInd == 1: # have to do separate work, since the original V1 experiment has trickier stimuli (i.e. not just sfMix, but also ori and rvc measurements done separately)
      # TODO: handle non-"none" mask in v1_hf.organize_modResp!
      v1_dir = exper.dir.replace('/', '.');
      v1_hf = il.import_module(v1_dir + 'helper_fcns');
      _, _, rateSfMix, allSfMix = v1_hf.organize_modResp(spikes, data, mask);
    else:
      # NOTE: we are getting the modRespOrg output of tabulate_responses, and ensuring the spikes are treated as rates (or raw counts) based on how they are passed in here
      allSfMix  = tabulate_responses(expStructure, expInd, spikes, mask, modsAsRate = respsAsRate)[4];
      rateSfMix = numpy.nanmean(allSfMix, -1);

    return rateOr, rateCo, rateSfMix, allSfMix;  

def get_spikes(data, get_f0 = 1, rvcFits = None, expInd = None, overwriteSpikes = None, vecF1=0):
  ''' Get trial-by-trial spike count
      Given the data (S.sfm.exp.trial), if rvcFits is None, simply return saved spike count;
                                        else return the adjusted spike counts (e.g. LGN, expInd 3)
      --- If we pass in rvcFits, we can optionally specify if it's a vecF1 fit (will need to access...
      --- -- the responses differently in that case)
  '''

  if overwriteSpikes is not None: # as of 19.05.02, used for fitting model recovery spikes
    return overwriteSpikes;
  if rvcFits is None:
    if get_f0 == 1:
      spikes = data['spikeCount'];
    elif get_f0 == 0:
      spikes = data['f1']
  else:
    if expInd is None:
      warnings.warn('Should pass in expInd; defaulting to 3');
      expInd = 3; # should be specified, but just in case
    try:
      spikes = organize_adj_responses(data, rvcFits, expInd, vecF1);
    except: # in case this does not work...
      warnings.warn('Tried to access f1 adjusted responses, defaulting to F1/F0 request');
      if get_f0 == 1:
        spikes = data['spikeCount'];
      elif get_f0 == 0:
        spikes = data['f1']
        # Now, update to reflect newer .npy which have spikes per component (i.e. not summed)
        # -- note that in this per-component case, we assume the non-valid components have zero amplitude
        if len(data['spikeCount']) == len(data['f1']):
           spikes = data['f1'];
        else: # we assume it's arranged (nComp, nTr), so sum!
           spikes = numpy.sum(data['f1'], axis=0); # i.e. sum over components

  return spikes;

def get_rvc_fits(loc_data, expInd, cellNum, rvcName='rvcFits', rvcMod=0, direc=1, vecF1=None):
  ''' simple function to return the rvc fits needed for adjusting responses
  '''
  if expInd > 2: # no adjustment for V1, altExp as of now (19.08.28)
    fitName = str(loc_data + rvc_fit_name(rvcName, rvcMod, direc, vecF1=vecF1));
    rvcFits = np_smart_load(fitName);
    try:
      rvcFits = rvcFits[cellNum-1];
    except: # if the RVC fits haven't been done...
      warnings.warn('This experiment type (expInd=3) usually has associated RVC fits for response adjustment');
      rvcFits = None;
  else:
    rvcFits = None;

  return rvcFits;

def get_adjusted_spikerate(expData, which_cell, expInd, dataPath, rvcName, rvcMod=0, vecF1=0, descrFitName_f0=None, descrFitName_f1=None, force_dc=False, force_f1=False, baseline_sub=True, return_measure=False):
  ''' wrapper function which will call needed subfunctions to return dc-subtracted spikes by trial
      note: rvcMod = -1 is the code indicating that rvcName is actually the fits, already!
      OUTPUT: SPIKES (as rate, per s), baseline subtracted (default, if DC); responses are per stimulus, not per component
        note: user can override f1f0 calculation to force return of DC values only (set force_dc=TRUE) or F! values only (force_f1=TRUE)
  '''
  f1f0_rat = compute_f1f0(expData, which_cell, expInd, dataPath, descrFitName_f0=descrFitName_f0, descrFitName_f1=descrFitName_f1)[0];
  stimDur = get_exp_params(expInd).stimDur; # may need this value
  
  ### i.e. if we're looking at a simple cell, then let's get F1
  if (f1f0_rat > 1 and force_dc is False) or force_f1 is True:
      if rvcMod == -1: # then rvcName is the rvcFits, already!
        rvcFits = rvcName;
      else:
        if rvcName is not None:
          rvcFits = get_rvc_fits(dataPath, expInd, which_cell, rvcName=rvcName, rvcMod=rvcMod);
        else:
          rvcFits = None
      spikes_byComp = get_spikes(expData, get_f0=0, rvcFits=rvcFits, expInd=expInd, vecF1=vecF1);
      spikes = numpy.array([numpy.sum(x) for x in spikes_byComp]);
      rates = True; # when we get the spikes from rvcFits, they've already been converted into rates (in get_all_fft)
      baseline = None; # f1 has no "DC", yadig? 

      which_measure = 1; # i.e. F1 spikes
  ### then complex cell, so let's get F0
  else:
      spikes = get_spikes(expData, get_f0=1, rvcFits=None, expInd=expInd, vecF1=vecF1);
      rates = False; # get_spikes without rvcFits is directly from spikeCount, which is counts, not rates!
      if baseline_sub: # as of 19.11.07, this is optional, but default
        baseline = blankResp(expData, expInd)[0]; # we'll plot the spontaneous rate
        # why mult by stimDur? well, spikes are not rates but baseline is, so we convert baseline to count (i.e. not rate, too)
        spikes = spikes - baseline*stimDur;

      which_measure = 0; # i.e. DC spikes

  # now, convert to rate (could be just done in above if/else, but cleaner to have it explicit here)
  if rates == False:
    spikerate = numpy.divide(spikes, stimDur);
  else:
    spikerate = spikes;

  if return_measure:
    return spikerate, which_measure;
  else:
    return spikerate;

def mod_poiss(mu, varGain):
    np = numpy;
    var = mu + (varGain * np.power(mu, 2));                        # The corresponding variance of the spike count
    r   = np.power(mu, 2) / (var - mu);                           # The parameters r and p of the negative binomial distribution
    p   = r/(r + mu)

    return r, p

def fit_CRF(cons, resps, nr_c50, nr_expn, nr_gain, nr_base, v_varGain, loss_type):
    # loss_type (i.e. which loss function):
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
        
        if loss_type == 4:
	    # Get predicted spike count distributions
          mu  = pred; # The predicted mean spike count; respModel[iR]
          var = mu + (v_varGain * np.power(mu, 2));                        # The corresponding variance of the spike count
          r   = np.power(mu, 2) / (var - mu);                           # The parameters r and p of the negative binomial distribution
          p   = r/(r + mu);
	# no elif/else

        if loss_type == 1 or loss_type == 2:
          # error calculation
          if loss_type == 1:
            loss = lambda resp, pred: np.sum(np.power(resp-pred, 2)); # least-squares, for now...
          if loss_type == 2:
            loss = lambda resp, pred: np.sum(np.square(np.sqrt(resp) - np.sqrt(pred)));

          curr_loss = loss(resps[sf], pred);
          loss_by_sf[sf] = np.sum(curr_loss);

        else:
          # if likelihood calculation
          if loss_type == 3:
            loss = lambda resp, pred: poisson.logpmf(resp, pred);
            curr_loss = loss(resps[sf], pred); # already log
          if loss_type == 4:
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

## 

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

def makeStimulusRef(data, disp, con, sf, expInd, nRepeats=None):
  ''' Created 19.05.13: hacky method for creating artificial stimuli
      in: data structure (trialInf or whole is ok); disp index; con, sf; expInd; [nRepeats]
        One of con or sf will be an array, rather than index
        If nRepeats is NOT none (should be integer), then we'll take repeat one example N times, but randomize the phase
        This approach was borne out of phase-interaction issues for dispersions causing discontinuities in the simulated case
      out: trial structure with new stimuli
      In that case, we borrow from the existing stimuli but create new stimuli with the interpolated value

      For both contrast and sf, we find the true stimulus with the closest con/sf (so that TF is close to as it was in the stimulus...)
      Note that this approach is only coded to work when con/sf is simulated at one value, only
      For contrast, we assume the con array contains total contrast levels; we will scale the contrast of each component accordingly
      For SF, 
  '''
  np = numpy;

  if 'sfm' in data:
    trialInf = data['sfm']['exp']['trial'];
  else:
    trialInf = data; # i.e. we've passed in trial directly...

  _, stimVals, val_con_by_disp, validByStimVal, _ = tabulate_responses(data, expInd);
  all_cons, all_sfs = stimVals[1], stimVals[2];

  if isinstance(con, numpy.ndarray):
    # then, we are interpolating contrasts for a given disp/sf condition
    if len(con) == 1:
      curr_cons = all_cons[val_con_by_disp[disp]];
      conInd = np.argmin(np.square(curr_cons-con[0]));
      conIndToUse = val_con_by_disp[disp][conInd];
      refCon = all_cons[conIndToUse];
    else:
      conIndToUse = val_con_by_disp[disp][-1]; # let's use the highest contrast as our reference
      refCon = all_cons[conIndToUse];
    # first arg is validTr ([0]), then unpack array into indices ([0][0])
    ref_trials = get_valid_trials(data, disp, conIndToUse, sf, expInd, stimVals, validByStimVal)[0][0];
    interpSF = 0;
  elif isinstance(sf, numpy.ndarray):
    val_sfs = get_valid_sfs(data, disp, con, expInd, stimVals, validByStimVal)
    if len(sf) == 1:
      sfIndToUse = np.argmin(np.square(all_sfs[val_sfs] - sf[0]));
      sfIndToUse = val_sfs[sfIndToUse];
    else:
      sfIndToUse = val_sfs[0];
    refSf = all_sfs[sfIndToUse];
    # first arg is validTr ([0]), then unpack array into indices ([0][0])
    ref_trials = get_valid_trials(data, disp, con, sfIndToUse, expInd, stimVals, validByStimVal)[0][0];
    interpSF = 1;
  else:
    warnings.warn('Con or Sf must be array!; returning full, original experiment at trial level');
    return trialInf;

  if nRepeats is not None:
    ref_trials = [ref_trials[0]] * nRepeats; # just pick one...

  if interpSF == 1:
    interpVals = sf;
  elif interpSF == 0:
    interpVals = con;

  all_trials = dict();
  all_trCon = [];
  all_trSf  = [];
  # and the measures that we won't alter...
  all_trPh  = [];
  all_trOr  = [];
  all_trTf  = [];

  for vals in range(len(interpVals)):
    # for every value in the interpolated dimension, get the N (usually 5 or 10) ref_trials
    # then replace the interpolated dimension accordingly
    valCurr = interpVals[vals];
    if interpSF == 1:
      sfMult = valCurr/refSf;
    elif interpSF == 0:
      conMult = valCurr/refCon; 

    # remember, trialInf values are [nStimComp, ], where each stimComp is [nTrials]
    conCurr = np.array([x[ref_trials] for x in trialInf['con']]);  
    sfCurr = np.array([x[ref_trials] for x in trialInf['sf']]);
    if nRepeats is None:
      phCurr = np.array([x[ref_trials] for x in trialInf['ph']]);  
    else: # then we randomize phase...
      phCurr = np.array([random_in_range((0,360), len(ref_trials)) for x in trialInf['ph']]);  
    tfCurr = np.array([x[ref_trials] for x in trialInf['tf']]); 
    orCurr = np.array([x[ref_trials] for x in trialInf['ori']]); 

    if interpSF == 1:
      sfCurr = np.multiply(sfCurr, sfMult);
    elif interpSF == 0:
      conCurr = np.multiply(conCurr, conMult);

    if all_trCon == []: # just overwrite the blank
      all_trCon = conCurr;
      all_trSf = sfCurr;
      all_trPh = phCurr;
      all_trTf = tfCurr;
      all_trOr = orCurr;
    else:
      all_trCon = np.hstack((all_trCon, conCurr));
      all_trSf = np.hstack((all_trSf, sfCurr));
      all_trPh = np.hstack((all_trPh, phCurr));
      all_trTf = np.hstack((all_trTf, tfCurr));
      all_trOr = np.hstack((all_trOr, orCurr));

  # but now, all_trSf/Con are [nStimpComp, nTrials] - need to reorganize as [nStimComp, ] with each entry as [nTrials]
  nComps = all_trCon.shape[0];
  
  newCons = np.zeros((nComps, ), dtype='O');
  newSf = np.zeros((nComps, ), dtype='O');
  newPh = np.zeros((nComps, ), dtype='O');
  newTf = np.zeros((nComps, ), dtype='O');
  newOr = np.zeros((nComps, ), dtype='O');
  # for each component, pack as array, which is the default/working method
  for ci in range(nComps):
    newCons[ci] = np.array(all_trCon[ci, :])
    newSf[ci] = np.array(all_trSf[ci, :])
    newPh[ci] = np.array(all_trPh[ci, :])
    newTf[ci] = np.array(all_trTf[ci, :])
    newOr[ci] = np.array(all_trOr[ci, :])

  all_trials['con'] = newCons;
  all_trials['sf'] = newSf;
  all_trials['ph'] = newPh;
  all_trials['tf'] = newTf;
  all_trials['ori'] = newOr;

  return all_trials;

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

    if expInd == 1:
      spreadVec = numpy.logspace(math.log10(.125), math.log10(1.25), num_families);
      octSeries  = numpy.linspace(1.5, -1.5, num_gratings);
      spread     = spreadVec[stimFamily-1];
      profTemp = norm.pdf(octSeries, 0, spread);
      profile    = profTemp/sum(profTemp);

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

def getNormParams(params, normType, forceAsymZero=True):
  ''' pass in param list, normType (1=original "tilt'; 2=gaussian weighting (wght); 3=con-dep sigma)
  '''
  if normType == 1:
    if len(params) > 8:
      inhAsym = params[8];
    else:
      inhAsym = 0;

    if forceAsymZero == True: # overwrite any value passed in...
      inhAsym = 0;

    return inhAsym;
  elif normType == 2 or normType == 5:
    gs_mean = params[8];
    gs_std  = params[9];
    if normType == 2:
      return gs_mean, gs_std;
    elif normType == 5:
      gs_gain = params[10]; # just one parameter after gs_std
      return gs_mean, gs_std, gs_gain;
  elif normType == 3:
    # sigma calculation
    offset_sigma = params[8];  # c50 filter will range between [v_sigOffset, 1]
    stdLeft      = params[9];  # std of the gaussian to the left of the peak
    stdRight     = params[10]; # '' to the right '' 
    sfPeak       = params[11]; # where is the gaussian peak?
    return offset_sigma, stdLeft, stdRight, sfPeak;
  elif normType == 4:
    gs_mean = params[8];
    gs_std  = [params[9], params[10]];
    return gs_mean, gs_std;
  else:
    inhAsym = 0;
    return inhAsym;

def genNormWeightsSimple(cellStruct, gs_mean=None, gs_std=None, normType = 2, trialInf = None, lgnFrontParams = None):
  ''' simply evaluates the usual normalization weighting but at the frequencies of the stimuli directly
  i.e. in effect, we are eliminating the bank of filters in the norm. pool
  '''
  np = numpy;

  if trialInf is not None:
    trialInf = trialInf;
    sfs = np.vstack([comp for comp in trialInf['sf']]); # [nComps x nTrials]
    cons = np.vstack([comp for comp in trialInf['con']]);
    consSq = np.square(cons);
  else:
    try:
      trialInf = cellStruct['sfm']['exp']['trial'];
      sfs = np.vstack([comp for comp in trialInf['sf']]); # [nComps x nTrials]
      cons = np.vstack([comp for comp in trialInf['con']]);
    except: # we allow cellStruct to simply be an array of sfs...
      sfs = cellStruct;
      #warnings.warn('Your first argument is simply an array of spatial frequencies - it should include the full trial information, including SF and CON values');
      cons = np.ones_like(sfs);

  # apply LGN stage, if specified - we apply equal M and P weight, since this is across a population of neurons, not just the one one uron under consideration
  if lgnFrontParams is not None:
    mod = lgnFrontParams['dogModel'];
    dog_m = lgnFrontParams['dog_m'];
    dog_p = lgnFrontParams['dog_p'];

    resps_m = get_descrResp(dog_m, sfs, mod, minThresh=0.1)
    resps_p = get_descrResp(dog_p, sfs, mod, minThresh=0.1)
    # -- make sure we normalize by the true max response:
    sfTest = np.geomspace(0.1, 10, 1000);
    max_m = np.max(get_descrResp(dog_m, sfTest, mod, minThresh=0.1));
    max_p = np.max(get_descrResp(dog_p, sfTest, mod, minThresh=0.1));
    # -- then here's our selectivity per component for the current stimulus
    selSf_m = np.divide(resps_m, max_m);
    selSf_p = np.divide(resps_p, max_p);
    # - then RVC response: # rvcMod 0 (Movshon)
    params_m = lgnFrontParams['rvc_m'];
    params_p = lgnFrontParams['rvc_p'];
    rvc_mod = get_rvc_model();
    selCon_m = rvc_mod(*params_m, cons)
    selCon_p = rvc_mod(*params_p, cons)
    # now, sum the responses and divide by the sum of the max possible M and P responses
    # -- note that those values are just the max of the CRF/RVC, since the Sf is normalized already...
    lgnStage = np.divide(selSf_m*selCon_m + selSf_p*selCon_p, np.nanmax(selCon_m)+np.nanmax(selCon_p));
  else:
    lgnStage = np.ones_like(sfs);

  if gs_mean is None or gs_std is None: # we assume inhAsym is 0
    inhAsym = 0;
    new_weights = 1 + inhAsym*(np.log(sfs) - np.nanmean(np.log(sfs)));
    new_weights = np.multiply(lgnStage, new_weights);
  elif normType == 2:
    log_sfs = np.log(sfs);
    new_weights = norm.pdf(log_sfs, gs_mean, gs_std);
    new_weights = np.multiply(lgnStage, new_weights);
  elif normType == 4:
    log_sfs = np.log(sfs);
    sfs_l = log_sfs[log_sfs<gs_mean];
    wts_l = norm.pdf(sfs_l, gs_mean, gs_std[0]); # first gs_std entry is left-std
    sfs_r = log_sfs[log_sfs>=gs_mean];
    wts_r = norm.pdf(sfs_r, gs_mean, gs_std[1]); # and second is right-std
    # now, set up masks
    lt =  np.ma.masked_less(log_sfs, gs_mean);
    gte = np.ma.masked_greater_equal(log_sfs, gs_mean);
    new_weights = np.zeros_like(log_sfs);
    new_weights[lt.mask]  = wts_l;
    new_weights[gte.mask] = wts_r;
    new_weights = np.multiply(lgnStage, new_weights);

  return new_weights;

def genNormWeights(cellStruct, nInhChan, gs_mean, gs_std, nTrials, expInd, normType = 2):
  ''' Compute the weights for the normalization pool; default is standard log-gaussian
      new (19.05.08) - normType = 4 will be two-halved Gaussian (i.e like flexibleGauss)
        in that case, gs_std is actually a tuple/2-element array
  '''
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
    if normType == 2:
      new_weights = norm.pdf(log_sfs, gs_mean, gs_std);
    elif normType == 4:
      sfs_l = log_sfs[log_sfs<gs_mean];
      wts_l = norm.pdf(sfs_l, gs_mean, gs_std[0]); # first gs_std entry is left-std
      sfs_r = log_sfs[log_sfs>=gs_mean];
      wts_r = norm.pdf(sfs_r, gs_mean, gs_std[1]); # and second is right-std
      new_weights = np.hstack((wts_l, wts_r));
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
    filterShape = numpy.array(flexible_Gauss_np(params, evalSfs, 0)); # 0 is baseline/minimum value of flexible_Gauss
  elif filter['type'] == 2:
    filterShape = deriv_gauss(params, evalSfs)[0]; # take the first output argument only

  evalC50 = scale*filterShape + offset - scale 
  # scale*filterShape will be between [scale, 0]; then, -scale makes it [0, -scale], where scale <0 ---> -scale>0
  # finally, +offset means evalC50 is [offset, -scale+offset], where -scale+offset will typically = 1
  return evalC50;

def setNormTypeArr(params, normTypeArr = []):
  '''
  TODO: Deprecate or make normType == 4 case
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

def nParamsLGN_joint(): # how many front end lgn parameters are there in joint fitting?
  # As of 20.08.26, we are using a fixed RVC set, and only fitting:
  # f_c, k_s, j_s for M & P each
  return 6;

def nParamsByType(fitType, excType, lgnType=0):
  # 9, 10, 11, 10 -- before excType == 2, before any lgnType
  try:
    if fitType == 1:
      nParams = 9; 
    elif fitType == 2 or fitType == 4:
      nParams = 10;
    elif fitType == 3 or fitType == 5:
      nParams = 11;
    # add one extra parameter if it's excType == 2
    if excType == 2:
      nParams += 1;
    # add one extra parameter if there's an LGN front end (mWeight)
    if lgnType > 0:
      nParams += 1;
  except:
    nParams = numpy.nan;

  return nParams;

def getConstraints(fitType, excType = 1, fixRespExp = None):
        #   00 = preferred spatial frequency   (cycles per degree) || [>0.05]
        #   if excType == 1:
          #   01 = derivative order in space || [>0.1]
        #   elif excType == 2:
          #   01 = sigma for SF lower than sfPref
          #   -1-lgnFrontEnd = sigma for SF higher than sfPref
        #   02 = normalization constant (log10 basis) || unconstrained
        #   03 = response exponent || >1
        #   04 = response scalar || >1e-3
        #   05 = early additive noise || [0, 1]; was [0.001, 1] - see commented out line below
        #   06 = late additive noise || >0.01
        #   07 = variance of response gain || >1e-3
        # if fitType == 2
        #   08 = mean of normalization weights gaussian || [>-2]
        #   09 = std of ... || >1e-3 or >5e-1
        # if fitType == 3
        #   08 = the offset of the c50 tuning curve which is bounded between [v_sigOffset, 1] || [0, 0.75]
        #   09 = standard deviation of the gaussian to the left of the peak || >0.1
        #   10 = "" to the right "" || >0.1
        #   11 = peak (i.e. sf location) of c50 tuning curve 
        # if fitType == 4
        #   08 = mean of normalization weights gaussian || [>-2]
        #   09, 10 = std left/right ... || >1e-3 or >5e-1
        # USED ONLY IF lgnFrontEnd == 1
        # -1 = mWeight (with pWeight = 1-mWeight)

    np = numpy;

    zero = (0.05, 15);
    if excType == 1:
      one = (0.1, None);
    elif excType == 2:
      # sigma for flexGauss version (bandwidth)
      min_bw = 1/4; max_bw = 10; # ranges in octave bandwidth
      one = (np.maximum(0.1, min_bw/(2*np.sqrt(2*np.log(2)))), max_bw/(2*np.sqrt(2*np.log(2)))); # Gaussian at half-height
    two = (None, None);
    #three = (2.0, 2.0); # fix at 2 (addtl suffix B)
    if fixRespExp is None:
      three = (0.25, None); # trying, per conversation with Tony (03.01.19)
    else:
      three = (fixRespExp, fixRespExp); # fix at the value passed in (e.g. usually 2, or 1)
    #three = (1, None);
    four = (1e-3, None);
    five = (0, 1); # why? if this is always positive, then we don't need to set awkward threshold (See ratio = in GiveBof)
    six = (0.01, None); # if always positive, then no hard thresholding to ensure rate (strictly) > 0
    seven = (1e-3, None);
    minusOne = (0, 1); # mWeight must be bounded between 0 and 1
    if fitType == 1:
      eight = (0, 0); # flat normalization (i.e. no tilt)
      if excType == 1:
        return (zero,one,two,three,four,five,six,seven,eight,minusOne);
      elif excType == 2:
        nine = (np.maximum(0.1, min_bw/(2*np.sqrt(2*np.log(2)))), max_bw/(2*np.sqrt(2*np.log(2)))); # Gaussian at half-height
        return (zero,one,two,three,four,five,six,seven,eight,nine,minusOne);
    if fitType == 2:
      eight = (-2, None);
      nine = (5e-1, None);
      if excType == 1:
        return (zero,one,two,three,four,five,six,seven,eight,nine,minusOne);
      elif excType == 2:
        ten = (np.maximum(0.1, min_bw/(2*np.sqrt(2*np.log(2)))), max_bw/(2*np.sqrt(2*np.log(2)))); # Gaussian at half-height
        return (zero,one,two,three,four,five,six,seven,eight,nine,ten,minusOne);
    elif fitType == 3:
      eight = (0, 0.75);
      nine = (1e-1, None);
      ten = (1e-1, None);
      eleven = (0.05, None);
      return (zero,one,two,three,four,five,six,seven,eight,nine,ten,eleven);
    elif fitType == 4:
      eight = (-2, None);
      nine = (5e-1, None);
      ten = (5e-1, None);
      return (zero,one,two,three,four,five,six,seven,eight,nine,ten);
    else: # mistake!
      return [];

def getConstraints_joint(nCells, fitType, excType = 1, fixRespExp = None):
        # NOTE: The LGN front end constraints will be at the END of the list
        # AS OF 20.08.06, we will optimize the following LGN parameters:
        #   lgn0 - m_fc (characteristic frequency of the magno center) || [2, 10]
        #   lgn1 - p_fc (RELATIVE TO M, characteristic frequency of the parvo center) || [1,4]
        #   lgn2 - m_ks (relative gain of magno surround) || [0.1, 0.8]
        #   lgn3 - p_ks (relative gain of parvo surround) || [0.1, 0.8]
        #   lgn4 - m_js (relative char. freq of magno surround) || [0.1, 0.8]
        #   lgn5 - p_js (relative char. freq of magno surround) || [0.1, 0.8]

  all_constr = getConstraints(fitType, excType, fixRespExp) * nCells;
  # THEN, tack on the LGN constraints
  lgn0 = (2, 10);
  lgn1 = (1, 4);
  lgn2 = (1e-3, 0.9);
  lgn3 = (1e-3, 0.9);
  lgn4 = (0.1, 0.9);
  lgn5 = (0.1, 0.9);
  return all_constr + (lgn0, lgn1, lgn2, lgn3, lgn4, lgn5);

##################################################################
##################################################################
##################################################################
### VI. Basic characterization analysis
##################################################################
##################################################################
##################################################################

def oriCV(oriVals, oriResps, baselineSub = False):
  ''' From Xing, Ringach, Shapley, and Hawken (2004), compute the orientation circular variance
      defined as 1 - divide(magnitude of summmed response vector, sum of all magnitudes)
      where the above numerator takes into account the angle of the stimulus
      - Interpretation: low circular variance is high selectivity

      - NOTE: we assume that oriVals are in deg (and we will convert to radians)
              we also assume oriResps is just nX1, where n is #oriVals
  '''
  np = numpy;
  oriAsRad = np.deg2rad(oriVals);
  if baselineSub:
    oriResps = oriResps - np.min(oriResps);
  numer = np.abs(np.dot(oriResps, np.exp(2j*oriAsRad)));
  denom = np.sum(oriResps);

  return 1 - np.divide(numer, denom);

def get_ori_mod(extrema = None):
  ''' Double Von Mises function as in Wang and Movshon, 2014
      - with one modification: normalize by resp at pref ori to make "a" parameterization more clear
      IF norm_factor is not None, then we assume we've already computed the function and simply want to evaluate at a particular theta
      The von Mises function:
        - a scales the height of the tuning curve
        - ds scales the non-preferred direction of the tuning curve
        - w determines the tuning band-width
        - xc is the location of the tuning curve peak
        - θ is the direction (not a fitted parameter; based on data)
        - r0 is thespontaneous firing rate of the cell.
  '''
  np = numpy;

  vonMis = lambda w,xc,ds,theta: np.exp(np.cos(theta-xc)/w) + ds*np.exp(np.cos(theta-xc-np.pi)/w);
  if extrema is None:
    mod = lambda a,w,xc,ds,r0,theta: r0 + a*np.divide(vonMis(w,xc,ds,theta) - np.min(vonMis(w,xc,ds,theta)), vonMis(w,xc,ds,xc) - np.min(vonMis(w,xc,ds,theta)));
  else:
    minResp, maxResp = extrema[0], extrema[1];
    mod = lambda a,w,xc,ds,r0,theta: r0 + a*np.divide(vonMis(w,xc,ds,theta) - minResp, maxResp-minResp);

  return mod, vonMis;

def oriTune(oriVals, oriResps, oriResps_std=None, baselineSub = False, nOpts = 30):
  ''' Using double von mises (see get_ori_mod above)
  '''
  np = numpy;

  if baselineSub:
    oriResps = oriResps - np.min(oriResps);

  k = 0.01*np.max(oriResps);
  if oriResps_std is None:
    sigma = np.ones_like(oriResps);
  else:
    sigma = oriResps_std;

  oriAsRad = np.deg2rad(oriVals);
 
  # and set the bounds
  allBounds = ((0, None), (0, None), (0, 2*np.pi), (0,1), (0, None));

  # squared error - with the k+sig^2 adjustment
  best_params = []; best_loss = np.nan; 
  curr_mod, vonMises = get_ori_mod();
  obj = lambda params: np.sum(np.divide(np.square(oriResps - curr_mod(*params, oriAsRad)), (k+np.square(sigma))));

  baseline = np.min(oriResps);
  maxResp = np.max(oriResps);
  oriEst = oriAsRad[np.argmax(oriResps)];
  nonPD = np.where(oriVals == np.mod(180 + oriVals[np.argmax(oriResps)], 360))[0];
  dsEst = np.divide(oriResps[nonPD], maxResp);

  for i in np.arange(nOpts):
    init_a = random_in_range([0.75, 1.25])[0] * (maxResp - baseline);
    init_w = random_in_range([0.2, 0.7])[0];
    init_xc = random_in_range([-0.5, 0.5])[0] + oriEst;
    init_ds = dsEst + random_in_range([-0.1, 0.1])[0];
    init_r0 = random_in_range([0.5, 1])[0] * np.min(oriResps)
    init_params = [init_a, init_w, init_xc, init_ds, init_r0];
    # make sure the ints are within the bounds
    for (ii,prm),bound in zip(enumerate(init_params), allBounds):
      compLow = -np.Inf if bound[0] is None else bound[0];
      compHigh = np.Inf if bound[1] is None else bound[1];
      if prm < compLow or prm > compHigh:
        try:
          init_params[ii] = bound[0]*random_in_range([1.25, 1.75])[0]
        except:
          init_params[ii] = bound[1]*random_in_range([0.25, 0.75])[0]


    if np.mod(i, 2) == 0:
       wax = opt.minimize(obj, init_params, bounds=allBounds, method='TNC');
    else:
       wax = opt.minimize(obj, init_params, bounds=allBounds, method='L-BFGS-B');
    if best_loss is np.nan or wax['fun'] < best_loss:
       best_loss = wax['fun'];
       best_params = wax['x'];

  oriParams = best_params;
  oriPrefRad = oriParams[2];
  oriDS = 1-oriParams[3];
  # then, let's compute the prefOri and oriBandwidth
  oriPref = np.rad2deg(oriPrefRad);
  # now, compute the normalization factor as shown in get_ori_mod -- necessary, since we are evaluating only at one theta
  maxResp, minResp = vonMises(*oriParams[1:-1], oriPrefRad), np.min(vonMises(*oriParams[1:-1], oriAsRad));
  eval_mod, _ = get_ori_mod([minResp, maxResp]);
  halfHeight = oriParams[-1] + 0.5 * (eval_mod(*oriParams, oriPrefRad) - oriParams[-1]); # half-height relative to the baseline/pedestal
  bwObj = lambda x: np.square(eval_mod(*oriParams, x) - halfHeight);
  # only look within 180 deg (i.e. [-pi/2, +pi/2] relative to peak]
  bwLimsLow = opt.minimize(bwObj, oriPrefRad-0.1, bounds=((oriPrefRad - np.pi/2, oriPrefRad), ));
  bwLimsHigh = opt.minimize(bwObj, oriPrefRad+0.1, bounds=((oriPrefRad, oriPrefRad + np.pi/2), ));
  oriBW = np.abs(np.rad2deg(bwLimsHigh['x'] - bwLimsLow['x']));
  
  return oriPref, oriBW[0], oriDS, oriParams, curr_mod, wax;

def tfTune(tfVals, tfResps, tfResps_std=None, baselineSub=False, nOpts = 10, fracSig=1):
  ''' Temporal frequency tuning! (Can also be used for sfTune)
      Let's assume we use two-halved gaussian (flexible_Gauss, as above)
      - respFloor, respRelFloor, tfPref, sigmaLow, sigmaHigh
      --- fracSig: if on, then the right-half (high SF) of the flex. gauss tuning curve is expressed as a fraction of the lower half
  '''
  np = numpy;

  mod = lambda params, tfVals: flexible_Gauss_np(params, stim_sf=tfVals);
  # set bounds
  min_bw = 1/4; max_bw = 10; # ranges in octave bandwidth
  bound_baseline = (0, np.max(tfResps));
  bound_range = (0, 1.5*np.max(tfResps));
  bound_mu = (np.min(tfVals), np.max(tfVals));
  bound_sig = (np.maximum(0.1, min_bw/(2*np.sqrt(2*np.log(2)))), max_bw/(2*np.sqrt(2*np.log(2)))); # Gaussian at half-height
  if fracSig:
    bound_sigFrac = (0.2, 2);
  else:
    bound_sigFrac = (1e-4, None); # arbitrarily small, to None // TRYING
  allBounds = (bound_baseline, bound_range, bound_mu, bound_sig, bound_sigFrac);

  ######
  # now, run the optimization
  ######
  best_params = []; best_loss = np.nan; 
  modObj = lambda params: DoG_loss(params, tfResps, tfVals, loss_type=2, DoGmodel=0);
  #modObj = lambda params: numpy.sum(numpy.square(mod(params, tfVals) - tfResps));
  for i in np.arange(nOpts):
    init_params = dog_init_params(tfResps, np.min(tfResps), tfVals, tfVals, DoGmodel=0, bounds=allBounds);
    if np.mod(i, 2) == 0:
       wax = opt.minimize(modObj, init_params, bounds=allBounds, method='TNC');
    else:
       wax = opt.minimize(modObj, init_params, bounds=allBounds, method='L-BFGS-B');
    if best_loss is np.nan or wax['fun'] < best_loss:
       best_loss = wax['fun'];
       best_params = wax['x'];

  tfParams = best_params;
  
  tfPref = get_prefSF(tfParams);
  tfBWbounds, tfBWlog = compute_SF_BW(tfParams, 1/2.0, sf_range=bound_mu);

  ######
  # also fit a DoG and measure the characteristic frequency (high freq. cut-off)
  ######
  bound_gainCent = (1e-3, None);
  bound_radiusCent= (1e-3, None);
  bound_gainSurr = (1e-3, None);
  bound_radiusSurr= (1e-3, None);
  allBounds = (bound_gainCent, bound_radiusCent, bound_gainSurr, bound_radiusSurr);

  best_params = []; best_loss = np.nan; 
  mod = lambda params, tfVals: DoGsach(*params, tfVals);
  #modObj = lambda params: numpy.sum(numpy.square(mod(params, tfVals) - tfResps));
  modObj = lambda params: DoG_loss(params, tfResps, tfVals, loss_type=2, DoGmodel=1); # for sachx
  for i in np.arange(nOpts):
    init_params = dog_init_params(tfResps, np.min(tfResps), tfVals, tfVals, DoGmodel=1, bounds=allBounds); # sach is dogModel 1
    if np.mod(i, 2) == 0:
      wax = opt.minimize(modObj, init_params, bounds=allBounds, method='TNC');
    else:
      wax = opt.minimize(modObj, init_params, bounds=allBounds, method='L-BFGS-B');
    if best_loss is np.nan or wax['fun'] < best_loss:
      best_loss = wax['fun'];
      best_params = wax['x'];

  tfParamsDoG = best_params;

  return tfPref, tfBWlog, tfParams, tfParamsDoG;

def sizeTune(diskSize, annSize, diskResps, annResps, diskStd, annStd, stepSize=0.01, nOpts=30):
  ''' Analysis as in Cavanaugh, Bair, Movshon (2002a)
      Inputs:
      - stepSize is used to determine the steps (in diameter) we use to evaluate the model
        // because we are approximating an integral
      Returns:
      - suppression index (calculated from responses, alone)
      - 
  '''
  np = numpy;

  ###########  
  ### first, data-derived measures
  ###########
  metrics_data = dict();

  rOpt = np.max(diskResps);  
  rSupp = diskResps[-1];

  supprInd_data = np.divide(rOpt - rSupp, rOpt);
  gsf_data = diskSize[np.argmax(diskResps)]; # grating summation field

  metrics_data['sInd'] = supprInd_data;
  metrics_data['gsf'] = gsf_data;
  metrics_data['maxResp'] = rOpt;
  metrics_data['platResp'] = rSupp;

  ###########
  ### now, let's fit a ratio of gaussians (eq. 9, fig. 6)
  ###########
  const_term = 2/np.sqrt(np.pi);
  # extent is separate for center/surround; diam is/are the stimulus diameter/s
  gauss_form = lambda extent, diam: np.square(const_term * np.trapz(np.exp(-np.square(np.divide(diam, extent))), x=diam));
  full_resp = lambda kC, exC, kSrat, exSrat, diam: np.divide(kC*gauss_form(exC, diam), 1 + kSrat*kC*gauss_form(exC*exSrat, diam));

  ## set bounds
  kC_bound = (0, None);
  kSrat_bound = (0, None); # ensure kS is less than kC
  exC_bound = (0, 1);
  exSrat_bound = (1, None); # i.e. exS = exC*exSrat must always be > exC
  allBounds = (kC_bound, exC_bound, kSrat_bound, exSrat_bound);

  diams_up_to = lambda x: np.arange(0, x, stepSize);
  anns_from = lambda x: np.arange(x, np.max(diskSize), stepSize);

  all_disk_comb = [diams_up_to(x) for x in diskSize];
  all_resps = lambda params: [full_resp(*params, x) for x in all_disk_comb];
  all_ann_comb = [anns_from(x) for x in annSize];
  all_resps_ann = lambda params: [full_resp(*params, x) for x in all_ann_comb];

  ###########  
  ### LOSS FUNCTION
  ###########  
  ### weight by inverse std (including disk AND annulus points)
  #fit_wt = np.divide(1, np.maximum(2, np.hstack((diskStd, annStd))));
  ### weight by inverse std (disk ONLY)
  fit_wt = np.hstack((np.divide(1, np.maximum(5, diskStd)), np.zeros_like(annStd)));
  ### weight equally - disk ONLY
  #fit_wt = np.hstack((np.ones_like(diskStd), np.zeros_like(annStd))); # for now, assume equal weight for all points
  ### weight equally - disk AND annulus
  #fit_wt = np.hstack((np.ones_like(diskStd), np.ones_like(annStd))); # for now, assume equal weight for all points
  ###########  
  ###########  
  obj = lambda params: np.dot(fit_wt, np.hstack((np.square(all_resps(params) - diskResps), np.square(all_resps_ann(params) - annResps))));

  best_params = []; best_loss = np.nan;
  for i in np.arange(nOpts):
    ## initialize parameters
    kC_init = random_in_range([1.25, 4])[0] * rOpt;
    kSrat_init = random_in_range([0.1, 0.3])[0];
    exC_init = random_in_range([0.1, 0.75])[0];
    exSrat_init = random_in_range([2, 10])[0];
    init_params = [kC_init, exC_init, kSrat_init, exSrat_init];

    if np.mod(i, 2) == 0:
      wax = opt.minimize(obj, init_params, bounds=allBounds, method='TNC');
    else:
      wax = opt.minimize(obj, init_params, bounds=allBounds, method='L-BFGS-B');
    if best_loss is np.nan or wax['fun'] < best_loss:
      best_loss = wax['fun'];
      best_params = wax['x'];

  opt_params = best_params;

  # now, infer measures from the model fit
  plt_diams = np.arange(0, np.max(diskSize), stepSize);
  plt_diam_lists = [diams_up_to(x) for x in plt_diams];
  plt_resps = [full_resp(*opt_params, x) for x in plt_diam_lists];

  max_mod = np.max(plt_resps);
  gsf_mod = plt_diam_lists[np.argmax(plt_resps)][-1]; # get the list with the maximum response, and get the last elem of that list (i.e. max size)

  plat_cutoff = 1.05*plt_resps[-1];
  plt_min = np.argmin(np.square(plt_resps - plat_cutoff));
  plat_val = plt_resps[plt_min];
  surr_diam_mod = plt_diams[plt_min];

  supprInd_mod = np.divide(max_mod-plat_val, max_mod);

  ###########  
  ### model-derived annulus tuning?
  ###########
  anns_list = np.arange(0, np.max(annSize), stepSize);
  anns_from = lambda x: np.arange(x, np.max(diskSize), stepSize);
  plt_ann_lists = [anns_from(x) for x in anns_list];
  ann_resps = [full_resp(*opt_params, x) for x in plt_ann_lists];

  amrf_val = 0.05 * max_mod;
  amrf = anns_list[np.argmin(np.square(ann_resps - amrf_val))];

  ###########  
  ### now, save the model-derived measures
  ###########
  metrics_mod = dict();
  metrics_mod['sInd'] = supprInd_mod;
  metrics_mod['gsf'] = gsf_mod;
  metrics_mod['surrDiam'] = surr_diam_mod;
  metrics_mod['amrf'] = amrf;
  metrics_mod['maxResp'] = max_mod;
  metrics_mod['platResp'] = plat_val;

  # for plotting smooth tuning curve
  to_plot = dict();
  to_plot['diams'] = plt_diams;
  to_plot['resps'] = plt_resps;
  to_plot['ann'] = anns_list;
  to_plot['ann_resp'] = ann_resps;
 
  return metrics_data, metrics_mod, to_plot, opt_params;

def rvcTune(rvcVals, rvcResps, rvcResps_std, rvcMod=1):
  ''' Response versus contrast!
      Default form is naka_rushton (mod #1)
  '''

  # rvc_fit works on vectorized data, so wrap our single values...
  _, optParam, conGain, loss = rvc_fit([rvcResps], [rvcVals], var=[rvcResps_std], mod=rvcMod, n_repeats=50);
  opt_params = optParam[0];

  c50 = get_c50(rvcMod, opt_params);
  # now, by optimization and discrete numerical evaluation, get the c50
  c50_emp, c50_emp_eval = c50_empirical(rvcMod, opt_params);

  return c50, c50_emp, c50_emp_eval, conGain[0], opt_params;

####

def get_basic_tunings(basicPaths, basicProgNames, forceSimple=None, preProc=None):
  ''' wrapper function used to get the derived measures for the basic characterizations
      - basicPaths [the full path to each of the basic tuning program files (xml)]
      - basicProgNames [what order and name for the tunings]
      - forceSimple [1 to get F1, 0 to get DC, leave as None to get resp. based on F1/F0 ratio]
      - if preProc is not None, then we assume it's the "new" sf* structures which have:
      --- the keys 'rfsize', 'tf', 'sf1', 'rvc', 'ori' for the basic programs
  '''

  from build_basics_list import prog_name
  sys.path.append('ExpoAnalysisTools/python/');
  import readBasicCharacterization as rbc

  basic_outputs = dict();
  basic_outputs['rvc'] = None;
  basic_outputs['sf'] = None;
  basic_outputs['rfsize'] = None;
  basic_outputs['tf'] = None;
  basic_outputs['ori'] = None;

  # this var will be used to get f1 or f0 responses, correspondingly
  # - if available, we base it off of the SF f1/f0 ratio; otherwise, exp by exp
  sf_resp_ind = None;

  if numpy.any(['LGN' in y for y in basicPaths]): # automatically make LGN simple (sf_resp_ind = 1)
    sf_resp_ind = 1;
  if forceSimple is not None:
    sf_resp_ind = forceSimple;
 
  for curr_name, prog in zip(basicPaths, basicProgNames):
    try:
      prog_curr = prog_name(curr_name) if preProc is not None else ''; # we aren't loading

      if 'sf' in prog:
        sfLoaded = 0;
        if preProc is not None:
          try:
            sf = preProc['sf1'];
            sfLoaded = 1;
          except:
            pass;
        if sfLoaded == 0: # if it wasn't or couldn't be loaded already, run this:
          sf = rbc.readSf11(curr_name, prog_curr);
        if sf['f1f0_rat'] == []:
          f1f0 = -1; # ensure it's complex...
        else:
          f1f0 = sf['f1f0_rat'];
        if f1f0 > 1 and sf_resp_ind is None: # simple
          sf_resp_ind = 1;
        elif f1f0 < 1 and sf_resp_ind is None: # complex - let's subtract baseline
          sf_resp_ind = 0;
        resps = sf['counts_mean'][:,0,sf_resp_ind];
        if sf_resp_ind == 0:
          baseline = sf['blank']['mean'];
          resps = resps - baseline;

        sfPref, sfBW, sfParams, sfParamsDoG = tfTune(sf['sfVals'], resps); # ignore other direction, if it's there..
        sf_dict = dict();
        sf_dict['isSimple'] = sf_resp_ind;
        sf_dict['sfPref'] = sfPref;
        sf_dict['sfBW_oct'] = sfBW;
        sf_dict['sfParams'] = sfParams;
        sf_dict['sfParamsDoG'] = sfParamsDoG;
        sf_dict['charFreq'] = dog_charFreq(sfParamsDoG, DoGmodel=1); # 1 is Sach, that's what is used in tfTune
        sf_dict['sf_exp'] = sf;
        basic_outputs['sf'] = sf_dict; # TODO: for now, we don't have SF basic tuning (redundant...but should add)

      if 'rv' in prog:
        rvcLoaded = 0;
        if preProc is not None:
           try:
              rv = preProc['rvc'];
              rvcLoaded = 1
           except:
              pass
        if rvcLoaded == 0:
          rv = rbc.readRVC(curr_name, prog_curr);
        if 'LGN' in curr_name:
          rvcMod = 0; # this is the model we use for LGN RVCs
        else:
          rvcMod = 1; # otherwise, naka-rushton
        if sf_resp_ind is None:
          if rv['f1f0_rat'] == []:
            f1f0 = -1; # ensure it's complex...
          else:
            f1f0 = rv['f1f0_rat'];
          if f1f0 > 1: # simple
            resp_ind = 1;
          else:
            resp_ind = 0;
        else:
          resp_ind = sf_resp_ind

        mean, std = rv['counts_mean'][:, resp_ind], rv['counts_std'][:, resp_ind]
        conVals = rv['conVals'];
        # NOTE: Unlike other measures, we do NOT baseline subtract the RVC
        # - in fact, we will fit the responses including the 0% con responses (i.e. blank resp)
        if resp_ind == 0:
          bs_mean, bs_std = rv['blank']['mean'], rv['blank']['std'];
          mean = numpy.hstack((bs_mean, mean)); std = numpy.hstack((bs_std, std));
          conVals = numpy.hstack((0, conVals));

        c50, c50_emp, c50_eval, cg, params  = rvcTune(conVals, mean, std, rvcMod);
        rvc_dict = dict();
        rvc_dict['isSimple'] = resp_ind;
        rvc_dict['c50'] = c50; rvc_dict['c50_emp'] = c50_emp; rvc_dict['c50_eval'] = c50_eval; 
        rvc_dict['conGain'] = cg;
        rvc_dict['params'] = params;
        rvc_dict['rvcMod'] = rvcMod;
        rvc_dict['rvc_exp'] = rv;

        basic_outputs['rvc'] = rvc_dict;

      if 'tf' in prog:
        tfLoaded = 0;
        if preProc is not None:
           try:
              tf = preProc['tf'];
              tfLoaded = 1;
           except:
              pass
        if tfLoaded == 0:
          tf = rbc.readTf11(curr_name, prog_curr);
   
        if sf_resp_ind is None:
          if tf['f1f0_rat'] == []:
            f1f0 = -1; # ensure it's complex...
          else:
            f1f0 = tf['f1f0_rat'];
          if f1f0 > 1: # simple
            resp_ind = 1;
          else:
            resp_ind = 0;
        else:
          resp_ind = sf_resp_ind

        mean = tf['counts_mean'][:,0,resp_ind]
        if resp_ind == 0:
          baseline = tf['blank']['mean'];
          mean = mean - baseline;

        tfPref, tfBW, tfParams, tfParamsDoG = tfTune(tf['tfVals'], mean); # ignore other direction, if it's there...
        tf_dict = dict();
        tf_dict['isSimple'] = resp_ind;
        tf_dict['tfPref'] = tfPref;
        tf_dict['tfBW_oct'] = tfBW;
        tf_dict['tfParams'] = tfParams;
        tf_dict['tfParamsDoG'] = tfParamsDoG;
        tf_dict['charFreq'] = dog_charFreq(tfParamsDoG, DoGmodel=1); # 1 is Sach, that's what is used in tfTune
        tf_dict['tf_exp'] = tf;
        basic_outputs['tf'] = tf_dict;

      if 'rf' in prog:
        rfLoaded = 0;
        if preProc is not None:
           try:
              rf = preProc['rfsize'];
              rfLoaded = 1;
           except:
              pass
        if rfLoaded == 0:
          rf = rbc.readRFsize10(curr_name, prog_curr);
        if sf_resp_ind is None:
          if rf['f1f0_rat'] == []:
            f1f0 = -1; # ensure it's complex...
          else:
            f1f0 = rf['f1f0_rat'];
          if f1f0 > 1: # simple
            resp_ind = 1;
          else:
            resp_ind = 0;
        else:
          resp_ind = sf_resp_ind

        disk_mean, ann_mean = rf['counts_mean'][:,0,resp_ind], rf['counts_mean'][:,1,resp_ind]
        disk_std, ann_std = rf['counts_std'][:,0,resp_ind], rf['counts_std'][:,1,resp_ind];
        if resp_ind == 0: # i.e. complex
          baseline = rf['blank']['mean'];
          disk_mean = disk_mean - baseline;
          ann_mean = ann_mean - baseline;

        data, mod, to_plot, opt_params = sizeTune(rf['diskVals'], rf['annulusVals'], disk_mean, ann_mean, disk_std, ann_std);
        rf_dict = dict();
        rf_dict['isSimple'] = resp_ind;
        rf_dict['gsf_data'] = data['gsf']
        rf_dict['suprInd_data'] = data['sInd']
        rf_dict['gsf_model'] = mod['gsf']
        rf_dict['suprInd_model'] = mod['sInd']
        rf_dict['surrDiam_model'] = mod['surrDiam'];
        rf_dict['to_plot'] = to_plot;
        rf_dict['params'] = opt_params;
        rf_dict['rf_exp'] = rf;
        basic_outputs['rfsize'] = rf_dict;

      if 'or' in prog:
        oriLoaded = 0;
        if preProc is not None:
           try:
              ori = preProc['ori'];
              oriLoaded = 1;
           except:
              pass
        if oriLoaded == 0:
          ori = rbc.readOri16(curr_name, prog_curr);
        if sf_resp_ind is None:
          if ori['f1f0_rat'] == []:
            f1f0 = -1; # ensure it's complex...
          else:
            f1f0 = ori['f1f0_rat'];
          if f1f0 > 1: # simple
            resp_ind = 1;
          else:
            resp_ind = 0;
        else:
          resp_ind = sf_resp_ind;

        ori_vals, mean, std = ori['oriVals'], ori['counts_mean'][:, resp_ind], ori['counts_std'][:, resp_ind];
        if resp_ind == 0: # i.e. complex - subtract baseline
          baseline = ori['blank']['mean'];
          mean = mean - baseline;

        ori_dict = dict();
        ori_dict['isSimple'] = resp_ind;
        ori_dict['cv'] = oriCV(ori_vals, mean);
        #pref, bw, oriDS, params, mod, _ = oriTune(ori_vals, mean); # ensure the oriVals are in radians
        pref, bw, oriDS, params, mod, _ = oriTune(ori_vals, mean, std); # ensure the oriVals are in radians
        ori_dict['DS'] = oriDS;
        ori_dict['bw'] = bw;
        ori_dict['pref'] = pref;
        ori_dict['params'] = params;
        ori_dict['ori_exp'] = ori;
        basic_outputs['ori'] = ori_dict
      
    except:
      basic_outputs[prog] = None;

  return basic_outputs;
