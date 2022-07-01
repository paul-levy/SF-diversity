import math, numpy, random
from scipy.stats import norm, mode, poisson, nbinom, sem
from scipy.stats.mstats import gmean as geomean
from numpy.matlib import repmat
from helper_fcns_sfBB import get_resp_str
from re import findall as re_findall
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

# np_smart_load   - be smart about using numpy load
# nan_rm          - remove nan from array
# sigmoid         - return a sigmoid evaluated at the input values
# arcmin_to_deg   - go from arcmin to deg (or reverse)                   
# bw_lin_to_log
# bw_log_to_lin
# resample_array  - resample array if specified
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
# is_mod_DoG      - returns True if the model is a simple DoG, otherise False
# nParams_descrMod - how many parameters per descr. SF model?
# descrMod_name   - returns string for descriptive model fit
# descrLoss_name  - returns string for descriptive model loss type
# descrJoint_name - returns string for joint model type [DoG]
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
# get_descr_recovResponses - simulate model recovery responses given a descr. SF model type
# phase_advance - compute the phase advance (a la Movshon/Kiorpes/+ 2005)
# tf_to_ind - convert the given temporal frequency into an (integer) index into the fourier spectrum

### IV. descriptive fits to sf tuning/basic data analyses

# get_rvc_model - return the lambda function describing the rvc model
# naka_rushton - naka-rushton form of the response-versus-contrast, with flexibility to evaluate super-saturating RVCs (Peirce 2007)
# rvc_fit - fit response versus contrast with a model used in Movshon/Kiorpes/+ 2005

# DiffOfGauss - standard difference of gaussians (from Levitt et al, 2001)
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
# dog_total_volume - compute the total volume of all DoG in a given model

# dog_loss - compute the DoG loss, given responses and model parameters
# dog_init_params - given the responses, estimate initial parameters for a given DoG model
# dog_fit - used to fit the Diff of Gauss responses -- either separately for each con, or jointly for all cons within a given dispersion

# deriv_gauss - evaluate a derivative of a gaussian, specifying the derivative order and peak
# get_prefSF - Given a set of parameters for a flexible gaussian fit, return the preferred SF
# compute_SF_BW - returns the log bandwidth for height H given a fit with parameters and height H (e.g. half-height)
# fix_params - Intended for parameters of flexible Gaussian, makes all parameters non-negative
# DEPRECATED - flexible_Gauss - DEPRECATED -- replace with flexible_Gauss_np, as needed
# flexible_Gauss_np - As above, but written in numpy (rather than math)
# dog_to_four - For the parker-hawken model, convert the spatial DoG into the Fourier domain
# parker_hawken_transform - Transform the parameters into their more interpretable DoG-friendly form
# parker_hawken - Run the d-DoG-S model from Parker & Hawken 1987 and 1988
# parker_hawken_space_from_stim - given the stimulus specification, compute the response using spatial rep. of the d-DoG-S model
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
# nParamsLGN_joint - how many parameters in the joint LGN case?
# nParamsByType  - given the norm, exc, and LGN types for the tuned G.C. model, how many parameters??
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
        sleep_time = random_in_range([.3, 8])[0];
        sleep(sleep_time); # i.e. wait for 10 seconds
     except EOFError: # this happens, I believe, because of parallelization when running on the cluster; cannot properly open file, so let's wait and then try again
        sleep_time = random_in_range([.3, 8])[0];
        sleep(sleep_time); # i.e. wait for 10 seconds
     except: # pickling error???
        sleep_time = random_in_range([.3, 8])[0];
        sleep(sleep_time); # i.e. wait for 10 seconds
     nTry -= 1 #don't try indefinitely!

   return loaded;

def nan_rm(x):
   return x[~numpy.isnan(x)];

def sigmoid(x):
  if x<-100:
    return 0; # avoid overflow
  return 1.0 / (1 + numpy.exp(-x));

def arcmin_to_deg(x, reverse=False):
  if reverse: # i.e. go from deg to arcmin
    return x*60;
  else:
    return x/60;

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

def resample_array(resample, arr, holdout_frac=1, start_ind=None, return_inds=False):
  # --- if holdout_frac < 1, then we are doing holdout cross-validation
  # --- if start_ind is not None, we simply start at index 'i' and go to 'j',
  # ----- where 'j' is determined based on the holdout_frac (e.g. 60% of all data) and the start value 'i'
  # --- if start_ind is None, then we randomly sample WITHOUT replacement
  # --- if return_inds, we also return the train indices (to facilitate partitioning test/train data in descr_fits)
  if resample:
    non_nan = nan_rm(arr);
    if holdout_frac == 1:
      # note that np.random.choice is WITH replacement by default
      curr_resps = numpy.random.choice(non_nan, len(non_nan));
      if return_inds:
         return curr_resps, np.arange(len(curr_resps));
      else:
         return curr_resps;
    elif holdout_frac < 1: # then we are, for ex., doing cross-validation 
      # -- so, do NOT draw with replacement and instead partition into train, test
      n_train = numpy.floor(len(non_nan)*holdout_frac).astype('int');
      if start_ind is None: # then we permute, draw randomly
        permuted = numpy.random.permutation(numpy.arange(len(non_nan)));
        curr_inds = permuted[0:n_train];
      else: # otherwise, do it sequentially as below
        curr_inds = numpy.mod(start_ind+numpy.arange(n_train), len(non_nan));
      test_inds = numpy.setxor1d(numpy.arange(len(non_nan)), curr_inds);
      curr_resps = non_nan[curr_inds];
      test_resps = non_nan[test_inds];
      # as of 21.10.31, still only return the current resps (will work out the test resps separately)
      if return_inds:
         return curr_resps, curr_inds;
      else:
         return curr_resps#, test_resps;
  else:
    if return_inds:
       return arr, np.arange(len(arr));
    else:
       return arr;

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

def sf_com(resps, sfs, logSF=True):
  ''' model-free calculation of the tuning curve's center-of-mass
      input: resps, sfs (np arrays; sfs in linear cpd)
      output: center of mass of tuning curve (in linear cpd)
  '''
  np = numpy;
  if logSF:
    com = np.dot(np.log2(sfs), np.array(resps))/np.sum(resps);
  else:
    com = np.dot(sfs, np.array(resps))/np.sum(resps);
  try:
    if logSF:
      return np.power(2, com);
    else:
      return com;
  except:
    return np.nan

def sf_var(resps, sfs, sf_cm, logSF=True):
  ''' model-free calculation of the tuning curve's center-of-mass
      input: resps, sfs (np arrays), and center of mass (sfs, com in linear cpd)
      output: variance measure of tuning curve
  '''
  np = numpy;
  if logSF:
    sfVar = np.dot(resps, np.abs(np.log2(sfs)-np.log2(sf_cm)))/np.sum(resps);
  else:
    sfVar = np.dot(resps, np.abs(sfs-sf_cm))/np.sum(resps);
  try:
    return sfVar;
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
    #return 'dataList_210524.npy'
    return 'dataList_220222.npy';
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

def is_mod_DoG(DoGmodel):
   # returns True if the model is a single DoG, otherwise False
   if DoGmodel == 1 or DoGmodel == 2 or DoGmodel == 4:
      return True;
   else:
      return False;

def nParams_descrMod(DoGmodel):
   # how many parameters in the descriptive SF models?
   if DoGmodel == 0:
      nParam = 5;
   elif DoGmodel == 1 or DoGmodel == 2 or DoGmodel == 4: # and joint can stay as specified
      nParam = 4;
   elif DoGmodel == 3 or DoGmodel == 5: # d-DoG-S (Parker, Hawken)
      nParam = 10; # for now, since we do not enforce kc1-ks2 = kc2-ks2 (see Parker, Hawken, 1987, bottom of p.255)
   return nParam;

def descrMod_name(DoGmodel):
  ''' returns the string for a given SF descriptive model fit
  '''
  if DoGmodel == 0:
    modStr = 'flex';
  elif DoGmodel == 1:
    modStr = 'sach';
  elif DoGmodel == 2:
    modStr = 'tony';
  elif DoGmodel == 3:
    modStr = 'ddogs';
  elif DoGmodel == 4:
    modStr = 'sachVol';
  elif DoGmodel == 5:
    modStr = 'ddogsHawk'; # Mike Hawken's parameterization
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

def descrJoint_name(joint=0, modelName=None):
  # add the joint name part
  if joint==0:
     jStr = '';
  elif joint==1:
     if modelName is None or 'ddogs' not in modelName:
        jStr = '_JTsurr' # the surround is the same relative to the center for all contrasts
     else: # then this is a d-DoG-S fit, so the surround names/constraints are different
        jStr = '_JTflank'; # g, S
  elif joint==2:
     if modelName is None or 'ddogs' not in modelName:
        jStr = '_JTsurrShape' # the surround shape is fixed for all contrasts
     else: # then this is a d-DoG-S fit, so the surround names/constraints are different
        jStr = '_JTflankSurrShape'; # g, S AND the surround shape
  elif joint==3:
     if modelName is None or 'ddogs' not in modelName:
        jStr = '_JTvolRatio' # the volume ratio can vary (i.e. independent center, surround gains), but the radii are fixed
     else: # then this is a d-DoG-S fit, so the surround names/constraints are different
        jStr = '_JTsurr' # the surround gain, radius apply to both central and flanking DoG, and joint across all cons
  elif joint==4:
     jStr = '_JTcenterShape' # the center radius is fixed
  elif joint==5:
     jStr = '_JTsurrShapeAbs' # the surround radius is fixed (absolute, unlike relative, as in joint==2)
  elif joint==6:
     jStr = '_JTctrSurr' # the center and surround radii are free, as is the center gain (surround gain is fixed across contrast)
  elif joint==7:
     if modelName is None or 'ddogs' not in modelName:
        jStr = '_JTsurrShapeCtrRaSlope';
     else: # then this is a d-DoG-S fit, so the surround names/constraints are different
        jStr = '_JTflankSurrShapeCtrRaSlope';
  elif joint==8:
     if modelName is None or 'ddogs' not in modelName:
        jStr = '_JTsurrGainCtrRaSlope';
     else: # then this is a d-DoG-S fit, so the surround names/constraints are different
        jStr = '_JTflankCopyCtrRaSlope'; # the flank DoG will just be a copy of the central DoG, variable ratio by contrast
  elif joint==9:
     jStr = '_JTflankFixedCopyCtrRaSlope'; # the flank DoG will just be a copy of the central DoG, fixed strength across contrast

  return jStr;

def descrFit_name(lossType, descrBase=None, modelName = None, modRecov=False, joint=0, phAdj=None):
  ''' if modelName is none, then we assume we're fitting descriptive tuning curves to the data otherwise, pass in the fitlist name in that argument, and we fit descriptive curves to the model this simply returns the name
  '''
  # load descrFits
  floss_str = descrLoss_name(lossType);
  if descrBase is None:
    descrBase = 'descrFits';
  vecF1_str = ''; # default
  if phAdj==1: # from 22.05 on, we'll assume that all descrFits are on vecF1 responses, since that's the default for all V1 data; however, we put a flag if it's ph-amp adj. (e.g. like some LGN fits)
     phAdj_str = '_phAdj';
  else:
     phAdj_str = '';
  descrFitBase = '%s%s%s' % (descrBase, phAdj_str, floss_str);

  if modelName is None:
    descrName = '%s.npy' % descrFitBase;
  else:
    descrName = '%s_%s.npy' % (descrFitBase, modelName);
    
  jStr = descrJoint_name(joint, modelName);
  descrName = descrName.replace('.npy', jStr + '.npy'); # there will only be one '.' in the string...

  if modRecov:
    descrName = descrName.replace('.npy', '_modRecov.npy');

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
          print('condition d/con/sf || %02d/%02d/%02d has (some?) nan phi; r values below' % (d,con,sf));
          print(r_byTrial[val_trials, :]);
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
      try:
         prefSfEst[i] = dfits[cellNum-1]['prefSf'][0][hiCon]; # get high contrast, single grating prefSf
      except:
         pass
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
    folded = np.mod(spikeTimes+dir*ph0[0], np.multiply(n_cycles, stimPeriod[0])); # center the spikes relative to the 0 phase of the stim
    #folded = np.mod(spikeTimes-ph0[0], np.multiply(n_cycles, stimPeriod[0])); # center the spikes relative to the 0 phase of the stim
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
       RETURN: mean amplitude, mean ph, std. amp, var. ph (from vector averaging)
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

def get_all_fft(data, disp, expInd, cons=[], sfs=[], dir=-1, psth_binWidth=1e-3, all_trials=0, resample=False):
  ''' for a given cell and condition or set of conditions, compute the mean amplitude and phase
      also return the temporal frequencies which correspond to each condition
      if all_trials=1, then return the individual trial responses (i.e. not just avg over all repeats for a condition)
      - if resample==True, then we'll resample the trials to create a bootstrap set of responses/trials
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

      if resample: # COULD make this as part of resample_array function (or separate function)...but for now, it lives here (21.09.13)
        new_inds = numpy.random.randint(0, len(rel_amp), len(rel_amp)); # this means we resample
        rel_amp, ph_rel_stim = rel_amp[new_inds], ph_rel_stim[new_inds]
        # must handle tf separately, to keep it as a list (rather than np array)
        curr_tf = [curr_tf[i] for i in new_inds];

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
  ''' [For computational model; outdated as of 2021] Given a cell structure and normalization type, return the (simulated) spikes and model recovery set parameters
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

def get_descr_recovResponses(params, descrMod, sfVals, nTr):
  # Given a set of descriptive SF tuning parameters, return simulated spikes
  # NOTE: Noise model is Gaussian with variance equal to mean
  resps = numpy.nan * numpy.zeros((len(sfVals), nTr));

  # NOTE: Temporarily set std. (var?) of gaussian equal to 1% of mean

  for i,val in enumerate(sfVals):
    meanResp = get_descrResp(params, val, DoGmodel=descrMod);
    # meanResp shouldn't be negative, but in case it is, we take abs. value before sqrt
    resps[i] = numpy.random.normal(loc=meanResp, scale=0.01*numpy.abs(numpy.sqrt(numpy.abs(meanResp))), size=nTr);

  return resps;

def phase_advance(amps, phis, cons, tfs, n_repeats=50, ampSem=None, phiVar=None):
   ''' Given the mean amplitude/phase of responses over a range of contrasts, compute the linear model
       fit which describes the phase advance per unit contrast as described in Eq. 4 of
       Movshon, Kiorpes, Hawken, Cavanaugh; 2005
       - This fit applies PER unique spatial frequency
       RETURNS: phAdv_model (the model equation), the list of the optimal parameters, and the phase advance (in milliseconds)
       "Vectorized" - i.e. accepts arrays of amp/phi arrays
       Guide to inputs:
         all - lists of len(nSfs)
         allAmp/allPhi[i] - list of len(nCons)
         allAmp/allPhi[i][j] - [mean, variance] (for sf "i", contrast "j")
         allCons[i] - list of contrasts (typically ascending) presented for sf "i"
         allTf[i] - as in allCons
         n_repeats - how many attempts for fitting (with different initializations)
         ampSem - what's the S.E.M. for each amplitude (currently unused)
         phiVar - what's the variance 
         -- note: allTf[i][j] should always be an array (even if just one grating)
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
     if phiVar is not None:
        obj = lambda params: np.sum(np.square(abs_angle_diff(curr_phiMean, phAdv_model(params[0], params[1], curr_ampMean))) / flatten_list(np.sqrt(phiVar[i])));
     else:
        obj = lambda params: np.sum(np.square(abs_angle_diff(curr_phiMean, phAdv_model(params[0], params[1], curr_ampMean))));
     # just least squares...
     #obj = lambda params: np.sum(np.square(curr_phiMean - phAdv_model(params[0], params[1], curr_ampMean))); # just least squares...
     # phi0 (i.e. phase at zero response) --> just guess the phase at the lowest amplitude response
     # slope --> just compute the slope over the response range
     min_resp_ind = np.argmin(curr_ampMean);
     max_resp_ind = np.argmax(curr_ampMean);
     diff_sin = np.arcsin(np.sin(np.deg2rad(curr_phiMean[max_resp_ind]) - np.deg2rad(curr_phiMean[min_resp_ind])));
     init_slope = (np.rad2deg(diff_sin))/(curr_ampMean[max_resp_ind]-curr_ampMean[min_resp_ind]);
     init_slope = np.maximum(0,init_slope); # test, as of 21.10.28, on restricting phAdv slope
     best_loss = np.nan; best_params = [];

     for rpt in range(n_repeats):
        init_params = [random_in_range([0.5, 2])[0]*curr_phiMean[min_resp_ind], random_in_range([0.5, 2])[0]*init_slope];
        # 21.10.26 -- restrict slope to be positive?
        # 22.04.12 -- and no greater than 10
        to_opt = opt.minimize(obj, init_params, bounds=((None,None), (0,10)));
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
     curr_tfs = tfs[i][0] if tfs is not None else None;
     #curr_sfs = sfs[i]; # TODO: Think about using the spatial frequency in the phase_adv calculation - if [p] = s^2/cycles, then we have to multiply by cycles/deg?
     cycle_fraction = opt_params[1] * curr_ampMean[max_resp_ind] / 360; # slope*respAmpAtMaxCon --> phase shift (in degrees) from 0 to responseAtMaxCon
     # then, divide this by 360 to get fractions of a cycle
     #phase_adv = 1e3*opt_params[1]/curr_tfs[0]; # get just the first grating's TF...
     phase_adv = 1e3*cycle_fraction/curr_tfs[0] if curr_tfs is not None else np.nan; # get just the first grating's TF...
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

def rvc_fit(amps, cons, var = None, n_repeats = 100, mod=0, fix_baseline=False, prevFits=None, cond=None):
   ''' Given the mean amplitude of responses (by contrast value) over a range of contrasts, compute the model
       fit which describes the response amplitude as a function of contrast as described in Eq. 3 of
       Movshon, Kiorpes, Hawken, Cavanaugh; 2005
       -- Optionally, can include a measure of variability in each response to perform weighted least squares
       -- Optionally, can include mod = 0 (as above) or 1 (Naka-Rushton) or 2 (Peirce 2007 modification of Naka-Rushton)
       -- Optional: cond (will be [disp, sf] tuple, e.g. (0, 3)...), only used during fit_rvc_f0 calls
       RETURNS: rvc_model (the model equation), list of the optimal parameters, and the contrast gain measure
         OR
                the above, but packaged as a dictionary
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

     if prevFits is None or 'loss' not in prevFits: # if prevFits is not the right dictionary!
       best_loss = 1e6; # start with high value
       best_params = []; conGain = [];
     else: # load the previous best_loss/params/conGain
       if n_amps == 1: # then this is fit_rvc_f0, organized differently
         best_loss = prevFits['loss'][cond];
         best_params = prevFits['params'][cond];
         conGain = prevFits['conGain'][cond];
       else:
         best_loss = prevFits['loss'][i];
         best_params = prevFits['params'][i];
         conGain = prevFits['conGain'][i];
       # make sure that the value is not None, and replace with an arbitrarily large value, if so
       if np.isnan(best_loss):
         best_loss = 1e6; # We don't want a NaN loss

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

def DiffOfGauss(gain, f_c, gain_s, j_s, stim_sf, baseline=0):
  ''' Difference of gaussians - as formulated in Levitt et al, 2001
  gain      - overall gain term
  f_c       - characteristic frequency of the center, i.e. freq at which response is 1/e of maximum
  gain_s    - relative gain of surround (e.g. gain_s of 0.5 says peak surround response is half of peak center response
  j_s       - relative characteristic freq. of surround (i.e. char_surround = f_c * j_s)
  --- Note that if baseline is non-zero, we'll add this to the response but it is NOT optimized, as of 21.05.03
  '''
  np = numpy;
  tune = baseline + gain*(np.exp(-np.square(stim_sf/f_c)) - gain_s * np.exp(-np.square(stim_sf/(f_c*j_s))));
  #tune = baseline + np.maximum(0, gain*(np.exp(-np.square(stim_sf/f_c)) - gain_s * np.exp(-np.square(stim_sf/(f_c*j_s))))); 
  return tune, [];

def get_xc_from_slope(intercept, slope, con, base=10):
   # let's assume a power law for determining the center radius given the contrast
   # --- we get the the contrasts in arbitrary "base" by dividing the ln(con) by ln(base)

   return numpy.power(base, intercept + slope * numpy.log(con)/numpy.log(base));

def DoGsach(gain_c, r_c, gain_s, r_s, stim_sf, baseline=0, parker_hawken_equiv=True, ref_rc_val=None):
  ''' Difference of gaussians as described in Sach's thesis
  [0] gain_c    - gain of the center mechanism
  [1] r_c       - radius of the center
  [2] gain_s    - gain of surround mechanism [multiplier to make the term rel. to gain_c, if parker_hawken_equiv=True]
  [3] r_s       - radius of surround [multiplier to make the term rel. to r_c, if parker_hawken_equiv=True]
  --- Note that if baseline is non-zero, we'll add this to the response but it is NOT optimized, as of 21.05.03
  --- If parker_hawken_equiv, we change the multiplictive terms in front of the exp to be in parallel with Parker-Hawken model
  ------ DEFAULT IS TRUE, MEANING NOT BACKWARD COMPATIBLE (gain term will have multiplicative scalar offset)
  ------ in that case, we also re-parameterize the surround gain/radii to be relative to the center's
  --- If ref_rc_val is not None, then we compute the surround radius relative to ref_rc_val
  '''
  np = numpy;

  if parker_hawken_equiv:
     r_c_ref = ref_rc_val if ref_rc_val is not None else r_c; # do we have a reference center radius? (used for joint=5)
     tune = baseline + gain_c*(np.exp(-np.square(stim_sf*np.pi*r_c)) - gain_s*np.exp(-np.square(stim_sf*np.pi*r_s*r_c_ref)));
  else:
    tune = baseline + np.maximum(0, gain_c*np.pi*np.square(r_c)*np.exp(-np.square(stim_sf*np.pi*r_c)) - gain_s*np.pi*np.square(r_s)*np.exp(-np.square(stim_sf*np.pi*r_s)));
  return tune, [];

def DoGsachVol(gain_c, r_c, gain_s, r_s, stim_sf, baseline=0):
  ''' As in DoGsachVol, but parameterized such the mechanisms are normalized (and given in Ch 3, eq. 1 in Sokol thesis [2009])
      -- This is an attempt to reparameterize to better disentangle the influence of r_s, gain_s from one another (per Eero, 21.11.11)
  '''
  np = numpy;
  tune = baseline + gain_c*np.pi*(np.square(r_c)*np.exp(-np.square(stim_sf*np.pi*r_c)) - gain_s*np.square(r_c*r_s)*np.exp(-np.square(stim_sf*np.pi*r_s*r_c)));
  return tune, [];
 
def var_explained(data_resps, modParams, sfVals, dog_model = 2, baseline=0, ref_params=None, ref_rc_val=None):
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
    mod_resps = get_descrResp(modParams, stim_sf=sfVals, DoGmodel=dog_model, baseline=baseline, ref_params=ref_params, ref_rc_val=ref_rc_val);
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

def descr_prefSf(modParams, dog_model=2, all_sfs=numpy.logspace(-1, 1, 11), baseline=0, nSamps=500, ref_params=None, ref_rc_val=None):
  ''' Compute the preferred SF given a set of DoG [or in the case of dog_model==0, not DoG...) parameters
  '''
  np = numpy;
  sf_bound = (numpy.min(all_sfs), numpy.max(all_sfs));
  if dog_model == 0:
    return modParams[2]; # direct read out in this model!
  # if the solution is not analytical, then we compute
  sf_samps = np.geomspace(all_sfs[0], all_sfs[-1], nSamps);
  tuning = get_descrResp(modParams, sf_samps, dog_model, ref_params=ref_params, ref_rc_val=ref_rc_val);
  sf_evals = np.argmax(tuning);
  sf_peak = sf_samps[sf_evals];

  return sf_peak;

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
  if DoGmodel == 0 or DoGmodel == 5:
      f_c = numpy.nan; # Cannot compute charFreq without DoG model fit (see sandbox_careful.ipynb)
  elif DoGmodel == 1 or DoGmodel == 3 or DoGmodel == 4: # sach, d-DoG-S; sachVol
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

def dog_get_param(params, DoGmodel, metric, parker_hawken_equiv=True, con_val=None):
  ''' given a code for which tuning metric to get, and the model/parameters used, return that metric
      codes: 'gc', 'gs', 'rc', 'rs', 'vc', 'vs'
      -- parker_hawken_equiv=False means original sach formultion
      note: when comparing the two formulations for DoG (i.e. Sach and Tony), we use Sach values as the reference
        to this end, we make the following transformations of the Tony parameters
        - gain:   gain/(pi*r^2)
        - radius: 1/(pi*fc)
      -- NOTE: if con_val is not None, then we normalize the gain (and therefore the volume, too) by the contrast
  '''
  np = numpy;

  if DoGmodel == 0:
    return np.nan; # we cannot compute from that form of the model!
  #elif DoGmodel == 3 or DoGmodel == 5:
  #  return dog_get_param(
  #########
  ### Gain
  #########
  if metric == 'gc': # i.e. center gain
    if DoGmodel == 1: # sach
      gc = params[0]/(np.pi * np.square(dog_get_param(params, DoGmodel, 'rc', parker_hawken_equiv))) if parker_hawken_equiv else params[0];
    elif DoGmodel == 2: # tony
      fc = params[1];
      rc = np.divide(1, np.pi*fc);
      gc = np.divide(params[0], np.pi*np.square(rc));
    elif DoGmodel == 4: # sachVol
      # then it becomes gain*pi*radius^2
      gc = params[0]*np.pi*np.square(params[1]);
    if con_val is None:
      return gc;
    else:
      return gc / con_val;
  if metric == 'gs': # i.e. surround gain
    if DoGmodel == 1: # sach
      gs = params[0]*params[2]/(np.pi * np.square(dog_get_param(params, DoGmodel, 'rs', parker_hawken_equiv))) if parker_hawken_equiv else params[2];
    elif DoGmodel == 2: # tony
      fc = params[1];
      rs = np.divide(1, np.pi*fc*params[3]); # params[3] is the multiplier on fc to get fs
      gs = np.divide(params[0]*params[2], np.pi*np.square(rs));
    elif DoGmodel == 4: # sachVol
      # then it becomes gain*pi*radius^2
      # --- note that here, we do NOT pass in the con val, since we'll normalize that out below, if needed 
      gc, rs = dog_get_param(params, DoGmodel, 'gc', parker_hawken_equiv), dog_get_param(params, DoGmodel, 'rs', parker_hawken_equiv);
      gs = gc*params[2]; # params[2] is the relative surround gain
      gs = gs*np.pi*np.square(rs);
    if con_val is None:
      return gs;
    else:
      return gs / con_val;
  #########
  ### Radius
  #########
  if metric == 'rc': # i.e. center radius
    if DoGmodel == 1 or DoGmodel == 4: # sach, sachVol
      return params[1];
    elif DoGmodel == 2: # tony
      fc = params[1];
      return np.divide(1, np.pi*fc);
  if metric == 'rs': # i.e. surround radius
    if DoGmodel == 1 or DoGmodel == 4: # sach, sachVol
      return params[1]*params[3] if parker_hawken_equiv else params[3];
    elif DoGmodel == 2: # tony
      fc = params[1];
      rs = np.divide(1, np.pi*fc*params[3]); # params[3] is the multiplier on fc to get fs
      return rs;
  #########
  ### Volume (gain * radius^2)
  #########
  if metric == 'vc': # i.e. center vol.
      return dog_get_param(params, DoGmodel, 'gc', parker_hawken_equiv, con_val) * np.square(dog_get_param(params, DoGmodel, 'rc', parker_hawken_equiv));
  if metric == 'vs': # i.e. surr. vol.
      return dog_get_param(params, DoGmodel, 'gs', parker_hawken_equiv, con_val) * np.square(dog_get_param(params, DoGmodel, 'rs', parker_hawken_equiv));

def dog_total_volume(params, DoGmodel):
   # Given a set of parameters, compute the volume (will be if not a DoG-based model)
   if DoGmodel == 0:
      return 0;
   else: # TODO: Fix for d-DoG-S model
      vc = dog_get_param(params, DoGmodel, 'vc');
      vs = dog_get_param(params, DoGmodel, 'vs');
      return vc+vs;

## - for fitting DoG models

def DoG_loss(params, resps, sfs, loss_type = 3, DoGmodel=1, dir=-1, resps_std=None, var_to_mean=None, gain_reg=0, vol_lam=0, minThresh=0.1, joint=0, baseline=0, fracSig=1, enforceMaxPenalty=0, sach_equiv_p_h=True, n_fits=1, ref_params=None, ref_rc_val=None, conVals=None):
  '''Given the model params (i.e. sach or tony formulation)), the responses, sf values
  return the loss
  loss_type: 1 - lsq
             2 - sqrt
             3 - poiss
             4 - Sach sum{[(exp-obs)^2]/[k+o*rho/t]} where
                 k := 0.01*rho*max(obs); rho := variance to mean relationship for the call across ALL conditions; t is time, to account for variance being computed in counts
                 NOTE: Our var_to_mean parameter will already have the stimulus duration divided out
  DoGmodel: 0 - flexGauss (not DoG...)
            1 - sach (reparameterized to match dDoGs)
            2 - tony
            3 - ddogs (d-DoG-S, from Parker and Hawken, 1988)
            4 - sachVol (original, as in 2009 Sokol thesis)
            5 - ddogsHawk (as in #3, but reparameterized)

    - sach_equiv_p_h==True means that we use the version of the Sach DoG model in which the scalars are equivalent to those in the parker-hawken model

    - if joint>0, then resps, resps_std will be arrays in which to index
    --- then, we also must pass in n_fits
    - if enforceMaxPenalty, then we add a penalty if the maximum model response is more than 50% larger than the maximum data response
    - params will be 2*N+2, where N is the number of contrasts;
    --- "2" is for a shared (across all contrasts) ratio of gain/[radius/freq]
    --- then, there are two parameters fit uniquely to each contrast - center gain & [radius/freq]
  '''
  np = numpy;

  totalLoss = 0;

  for i in range(n_fits):
    if n_fits == 1: # i.e. joint is False!
      curr_params = params;
      curr_resps = resps;
      curr_std = resps_std;
      curr_sfs = sfs;

      if enforceMaxPenalty:
        max_data = np.max(curr_resps);

    else:
      curr_resps = resps[i];
      curr_std = resps_std[i];
      curr_sfs = sfs if len(sfs[0])==1 else sfs[i]; # i.e. if we've passed in a list of lists of SF values, then unpack per
      # --- the above is keeping backwards compatability if we pass just in one list of SF values
      # treatment of joint depends on whether it's a DoG or d-DoG-S model
      if is_mod_DoG(DoGmodel):
        if joint==1:
          curr_params = [params[2+i*2], params[3+i*2], params[0], params[1]];
        elif joint==2:
          curr_params = [params[1+i*3], params[2+i*3], params[3+i*3], params[0]];
        elif joint==3:
          curr_params = [params[2+i*2], params[0], params[3+i*2], params[1]];
        elif joint==4: # fixed center radius, surr. gain
          curr_params = [params[2+i*2], params[0], params[1], params[3+i*2]];
        elif joint==5: # fixed surr. radius, surr. gain
          curr_params = [params[2+i*2], params[3+i*2], params[0], params[1]];
          ref_rc_val = params[2]; # 2+(i=0)*3=2 --> center radius for the high contrast condition, which serves as reference for surround radius for all contrasts
        elif joint==6: # fixed center:surround gain ratio
          curr_params = [params[1+i*3], params[2+i*3], params[0], params[3+i*3]];
        elif joint==7 or joint==8: # surround radius [7] or gain [8] is fixed, center radius is determined from power law
          # we have two unique params per contrast, and three joint
          xc_curr = get_xc_from_slope(params[0], params[1], conVals[i]); # intercept, slope are first two args for get_xc_from_slope func
          curr_params = [params[3+i*2], xc_curr, params[4+i*2], params[2]] if joint==7 else [params[3+i*2], xc_curr, params[2], params[4+i*2]];
      else: # here, we can handle the no_surr case here, too
        # we know there are 10 params
        if joint==1:
           nParam = nParams_descrMod(DoGmodel)-2; # we subtract off two for the two joint parameters (which, conveniently, are at the end)
           start_ind=2+i*nParam;
           curr_params = [*params[start_ind:start_ind+nParam], params[0], params[1]];
           # also compute the high contrast parameters --> why? This will serve as reference for computing S at all contrasts
           ref_ind=2+(n_fits-1)*nParam;
           ref_params = [*params[ref_ind:ref_ind+nParam], params[0], params[1]];
        elif joint==2: # surr[1&2]_rad AND g, S are constant
           nParam = nParams_descrMod(DoGmodel)-4; # we sub. off 4 for the four joint parameters
           start_ind=4+i*nParam;
           curr_params = [*params[start_ind:start_ind+3], params[0],
                             *params[start_ind+3:start_ind+6], params[1], params[2], params[3]];
           # also compute the high contrast parameters --> why? This will serve as reference for computing S at all contrasts
           ref_ind=4+(n_fits-1)*nParam;
           ref_params = [*params[ref_ind:ref_ind+3], params[0], 
                         *params[ref_ind+3:ref_ind+6], params[1], params[2], params[3]];
        elif joint==3: # surr_gain/rad [same for both central and flank] AND g, S are constant
           nParam = nParams_descrMod(DoGmodel)-6; # we sub. off 6 for the 6 joint parameters (the surr gain/radius apply twice)
           start_ind=4+i*nParam; # but we're still only starting at +4
           curr_params = [*params[start_ind:start_ind+2], params[0], params[1],
                             *params[start_ind+2:start_ind+4], params[0], params[1], params[2], params[3]];
           # also compute the high contrast parameters --> why? This will serve as reference for computing S at all contrasts
           ref_ind=4+(n_fits-1)*nParam;
           ref_params = [*params[ref_ind:ref_ind+2], params[0], params[1],
                         *params[ref_ind+2:ref_ind+4], params[0], params[1], params[2], params[3]];
        elif joint==7 or joint==8 or joint==9: # center radii from slope; surr[1&2]_rad AND g, S are constant
           xc_curr = get_xc_from_slope(params[0], params[1], conVals[i]); # intercept, slope are first two args for get_xc_from_slope func
           xc_ref = get_xc_from_slope(params[0], params[1], conVals[-1]); # intercept, slope are first two args for get_xc_from_slope func
           if joint == 7:
              nParam = nParams_descrMod(DoGmodel)-6; # we sub. off 6 for parameters determined jointly
              start_ind=6+i*nParam;
              curr_params = [params[start_ind], xc_curr, params[start_ind+1], params[2],
                        params[start_ind+2], 1, params[start_ind+3], params[3], params[4], params[5]];
              # also compute the high contrast parameters --> why? This will serve as reference for computing S at all contrasts
              ref_ind=6+(n_fits-1)*nParam;
              ref_params = [params[ref_ind], xc_ref, params[ref_ind+1], params[2],
                        params[ref_ind+2], 1, params[ref_ind+3], params[3], params[4], params[5]]; # 22.06.01 --> note that we FIX xc2 = xc1
           elif joint == 8:
              nParam = nParams_descrMod(DoGmodel)-7; # as in joint==7, but no separate parameter for flanking surround
              start_ind=5+i*nParam; # as in joint==7, except surround radius 2 is same as surr radius 1
              curr_params = [params[start_ind], xc_curr, params[start_ind+1], params[2],
                        params[start_ind+2], 1, params[start_ind+1], params[2], params[3], params[4]];
              # also compute the high contrast parameters --> why? This will serve as reference for computing S at all contrasts
              ref_ind=5+(n_fits-1)*nParam;
              ref_params = [params[ref_ind], xc_ref, params[ref_ind+1], params[2],
                        params[ref_ind+2], 1, params[ref_ind+1], params[2], params[3], params[4]]; # 22.06.01 --> note that we FIX xc2 = xc1
           elif joint == 9:
              nParam = nParams_descrMod(DoGmodel)-8; # as in joint==8, but flanking DoG will always have the same relative strength across contrast
              start_ind=6+i*nParam; # as in joint==8, but 
              curr_params = [params[start_ind], xc_curr, params[start_ind+1], params[2],
                        params[3], 1, params[start_ind+1], params[2], params[4], params[5]];
              # also compute the high contrast parameters --> why? This will serve as reference for computing S at all contrasts
              ref_ind=6+(n_fits-1)*nParam;
              ref_params = [params[ref_ind], xc_ref, params[ref_ind+1], params[2],
                        params[3], 1, params[ref_ind+1], params[2], params[4], params[5]]; # 22.06.01 --> note that we FIX xc2 = xc1

      if enforceMaxPenalty:
        max_data = np.max(curr_resps);

    pred_spikes = get_descrResp(curr_params, curr_sfs, DoGmodel, minThresh, baseline, fracSig, sach_equiv_p_h=sach_equiv_p_h, ref_params=ref_params, ref_rc_val=ref_rc_val);
    if enforceMaxPenalty: # check if this gives separate max for each condition
      max_mod = get_descrResp(curr_params, np.array([descr_prefSf(curr_params, DoGmodel)]), DoGmodel, minThresh, baseline, fracSig, ref_params=ref_params, ref_rc_val=ref_rc_val)[0];
      applyPen = 1 if (max_mod-1.40*max_data)>0 else 0;
      # TODO: Make this penalty smooth/continuous rather than discrete...
      maxPen = applyPen*1*(max_mod-1.4*max_data); # scale factor of 1 chosen to be within the typical loss values (O(10),O(100), at least for loss_type=2) so this regularization does not overwhelm
      #maxPen = 0;
    else:
      maxPen = 0;

    if loss_type == 1: # lsq
      loss = np.sum(np.square(curr_resps - pred_spikes)) + maxPen;
      totalLoss = totalLoss + loss;
    elif loss_type == 2: # sqrt - now handles negative responses by first taking abs, sqrt, then re-apply the sign 
      loss = np.sum(np.square(np.sign(curr_resps)*np.sqrt(np.abs(curr_resps)) - np.sign(pred_spikes)*np.sqrt(np.abs(pred_spikes)))) + maxPen
      totalLoss = totalLoss + loss;
    elif loss_type == 3: # poisson model of spiking
      poiss = poisson.pmf(np.round(curr_resps), pred_spikes); # round since the values are nearly but not quite integer values (Sach artifact?)...
      ps = np.sum(poiss == 0);
      if ps > 0:
        poiss = np.maximum(poiss, 1e-6); # anything, just so we avoid log(0)
      totalLoss = totalLoss + sum(-np.log(poiss)) + maxPen;
    elif loss_type == 4: # Cavanaugh et al 2002a, & as similarly used in Sokol, 2009 (thesis)
      '''
      if np.ma.isMaskedArray(curr_std): # if it's a masked array
        nans = np.any(curr_std.mask); # if anything is masked out, then we cannot use sigma
      else:
        nans = np.any(np.isnan(curr_std));
      if resps_std is None or nans: # i.e. if any NaN, then we shouldn't use stderr
        sigma = np.ones_like(curr_resps);
      else:
        sigma = curr_std;
      '''
      k = 0.01*np.max(curr_resps)*var_to_mean; # 22.04.03 -- CORRECTED VERSION FROM CAVANAUGH ET AL 2002
      sq_err = np.square(curr_resps-pred_spikes);
      totalLoss = totalLoss + np.sum((sq_err/(k+curr_resps*var_to_mean))) + gain_reg*(params[0] + params[2])#; + maxPen;
      #totalLoss = totalLoss + np.sum((sq_err/(k+sigma))) + gain_reg*(params[0] + params[2])#; + maxPen;

  if vol_lam > 0: # i.e. we apply a penalty proportional to the sum of volumes of the DoG; only when model is DoG
    ### For reference, the distributions of tot_vol is 2^[-2,5] with a mean of 2^2 ~= 4
    # --- for details, see analysis_ch1_suppl.ipynb; but we'll work with log2 of volume
    # The below values are for sach, sqrt loss functions ONLY
    # -- for sach, sqrt loss functions, the np.log2(loss) ranges are [2,5], [1,5] respectively with means of 3.36, 2.14
    # -- vol_scalar is chosen to make the lambdas "make sense" given the typical tot_vol, and totalLoss values
    tot_vol = dog_total_volume(curr_params, DoGmodel);
    if loss_type == 2 or loss_type == 4: # this only works for these loss functions, as of 21.11.09
      vol_scalar = 1.5 if loss_type == 2 else 1; # why? well, if vol~2, loss~3.5, then vol_scalar~1.5 means vol,loss are comparable with lambda=1
      vol_penalty = vol_lam * vol_scalar * np.log2(tot_vol);
      totalLoss = np.log2(totalLoss) + vol_penalty;
    else:
      totalLoss = totalLoss + vol_penalty;

  return totalLoss;

def dog_init_params(resps_curr, base_rate, all_sfs, valSfVals, DoGmodel, bounds=None, fracSig=1, sach_equiv_p_h=True, no_surr=False, ref_params=None):
  ''' return the initial parameters for the DoG model, given the model choice and responses
      --- if bounds is not None, then we'll ensure that each parameter is within the specified bounds
      --- no_surr applies for d-DoG-S only (as of 21.12.06)
  TODO: Check why all_sfs AND valSfVals are passed in???
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
    if fracSig:
      range_sigmaHigh = (0.2, 0.75); # allow the fracSig value to go above the bound used for V1, since we adjust if bound is there

    init_base = random_in_range(range_baseline)[0]; # NOTE addition of [0] to "unwrap" random_in_range value
    init_amp = random_in_range(range_amp)[0];
    init_mu = random_in_range(range_mu)[0];
    init_sig_left = random_in_range(range_denom)[0];
    init_sig_right = random_in_range(range_sigmaHigh)[0] if fracSig else random_in_range(range_denom)[0];
    init_params = [init_base, init_amp, init_mu, init_sig_left, init_sig_right];

  #############
  ## Parker-Hawken (difference of DoG with separation, i.e. d-DoG-S; see Parker & Hawken 1987, 1988)
  #############
  elif DoGmodel == 3 or DoGmodel == 5:
    ### for reference: sigmoid([-/+]1.09)~= 0.25/0.75, sigmoid(0) = 0.5, sigmoid([-/+]2.94)~=0.05/0.95   
    maxResp -= base_rate # subtract the baseline/blank, since the get_descrResp() call adds the baseline/blank response to our model response

    # first, the values in common for both models
    sqrtMax = np.sqrt(maxResp); # sqrtMax is a useful way of sampling for a maximum response gain (per Hawk...)
    init_kc1 = maxResp + random_in_range((1, 6))[0] * sqrtMax;
    #init_xc1 = random_in_range((0.02, 0.2))[0];
    # note: a cursory analysis shows that the interquartile range of ratios between c. freq::pSf is VERY approximately (1.25, 2.75)
    # --- but we'll bias towards slightly lower charFreq/high XC values to avoid excessively short distance between sub-units
    max_sf_index = np.argmax(resps_curr); # what sf index gives peak response?
    mu_guess = np.maximum(0.1, valSfVals[max_sf_index]); # ensure that we don't guess zero sf!
    init_xc1 = 1./(np.pi*mu_guess*random_in_range((1.3, 2.2))[0]) # f_c = 1/(pi*xc) where f_c=pSf*ratio
    init_kc2 = random_in_range((-1.4, 0.8))[0]; # temporarily, as sigmoid rel. to kc1
    init_kS_rel1, init_kS_rel2 = random_in_range((-1.4, 0.8), size=2); # will be used as input to sigmoid
    init_gPr = random_in_range((-1.09, 1.09))[0]; # input to sigmoid (alone); 

    # then, those values that are specific to model type
    init_A = random_in_range((1.5, 3))[0] if DoGmodel==3 else random_in_range((0.6, 2))[0]; # xs1
    init_B = random_in_range((1.05, 3))[0] if DoGmodel==3 else random_in_range((0.05, 0.4))[0]; # xc2 --> 220120c
    init_C = random_in_range((2, 3.5))[0] if DoGmodel==3 else random_in_range((0.3, 2))[0]; # xs2
    init_sPr = random_in_range((-2.5, -1))[0] if DoGmodel == 3 else random_in_range((0.1, 0.3))[0];
    if no_surr:
       init_params = [init_kc1, init_xc1, init_kc2, init_B, init_gPr, init_sPr];
    else:
       init_params = [init_kc1, init_xc1, init_kS_rel1, init_A, 
                   init_kc2, init_B, init_kS_rel2, init_C,
                   init_gPr, init_sPr];

  ############
  ## SACH [and sachVol]
  ############
  elif DoGmodel == 1 or DoGmodel == 4:
    max_sf_index = np.argmax(resps_curr); # what sf index gives peak response?
    mu_guess = np.maximum(0.1, valSfVals[max_sf_index]); # ensure that we don't guess zero sf!
    init_radiusCent = 1./(np.pi*mu_guess*random_in_range((1.3, 2.2))[0]) # f_c = 1/(pi*xc) where f_c=pSf*ratio
    #init_radiusCent = random_in_range((0.02, 0.5))[0];
    if DoGmodel == 1:
      if sach_equiv_p_h: # then, to get maxResp, gainCent should be maxResp/{np.sqrt(pi)*radiusCent}
        init_gainCent = maxResp * random_in_range((0.85, 1.2))[0]; # 
        #init_gainCent = maxResp/(init_radiusCent*np.sqrt(np.pi)) * random_in_range((0.7, 1.3))[0]; # 
      else: # just keeping for backwards compatability
        init_gainCent = 5e2*random_in_range((1, 300))[0];
    elif DoGmodel == 4:
      # the term in front is gain*pi*r^2, so we divide out pi*r^2 to get the initial gain estimate
      init_gainCent = maxResp*random_in_range((0.7, 1.3))[0]/(np.pi*np.square(init_radiusCent));
    # -- surround parameters are relataive to center
    init_gainSurr = random_in_range((0.1, 0.8))[0]; #init_gainCent * random_in_range((0.5, 0.9))[0]; # start with a stronger surround?
    init_radiusSurr = random_in_range((2, 4))[0]; #init_radiusCent * random_in_range((0.9, 4))[0];
    #init_radiusSurr = random_in_range((1.1, 2))[0]; #init_radiusCent * random_in_range((0.9, 4))[0];
    init_params = [init_gainCent, init_radiusCent, init_gainSurr, init_radiusSurr];
  ############
  ## TONY
  ############
  elif DoGmodel == 2:
    init_gainCent = maxResp * random_in_range((0.9, 1.2))[0];
    init_freqCent = np.maximum(all_sfs[2], freqAtMaxResp * random_in_range((1.2, 1.5))[0]); # don't pick all_sfs[0] -- that could be zero (we're avoiding that)
    init_gainFracSurr = random_in_range((0.7, 1))[0];
    init_freqFracSurr = random_in_range((.25, .35))[0];
    init_params = [init_gainCent, init_freqCent, init_gainFracSurr, init_freqFracSurr];

  # For all -- try 
  if bounds is not None:
    try:
      for (ii,prm),bound in zip(enumerate(init_params), bounds):
         try:
            if prm < bound[0] or prm > bound[1]:
               init_params[ii] = bound[0] + (bound[1]-bound[0])*random_in_range([0.25, 0.75])[0] # some value in-between the two bounds
         except:
            pass; # perhaps that bound cannot be evaluated in the above way (e.g. None)
    except: # we end up here if bounds is somehow not equal in # of entries to init_params
      pass; # not ideal, but the parent function should handle failures of initialization by trying again, anyway

  return init_params

def dog_fit(resps, DoGmodel, loss_type, disp, expInd, stimVals, validByStimVal, valConByDisp, n_repeats=100, joint=0, gain_reg=0, ref_varExpl=None, veThresh=60, prevFits=None, baseline_DoG=True, fracSig=1, noDisp=0, debug=0, vol_lam=0, modRecov=False, ftol=2.220446049250313e-09, jointMinCons=2, no_surr=False, isolFits=None, flt32=True):
  ''' Helper function for fitting descriptive funtions to SF responses
      if joint>0, (and DoGmodel is not flexGauss), then we fit assuming
      --- joint==1: a fixed ratio for the center-surround gains and [freq/radius]
           - i.e. of the 4 DoG parameters, 2 are fit separately for each contrast, and 2 are fit 
             jointly across all contrasts!
      --- joint==2: a fixed center & surround radius for all contrasts, free gains
      --- joint==3: a surround radius for all contrasts; all other parameters (both gains, center radius) free at al contrasts
      - note that ref_varExpl (optional) will be of the same form that the output for varExpl will be
      - note that jointMinCons is the minimum # of contrasts that must be included for a joint fit to be run (e.g. 2)
      - if flt32, return param_list (for joint fits only) as float32 rather than float64 (the default)
      - As of 21.12.06, no_surr only applies with d-DoG-S model
      --- NOTE: We only use ftol if joint; if boot, we will artificially restrict ftol to avoid too many iteration steps (which yield minimal improvement)
      --- fracSig: if on, then the right-half (high SF) of the flex. gauss tuning curve is expressed as a fraction of the lower half
      --- modRecov: if True AND if prevFits is not None, we'll sample the fit params and fit to those resps, i.e. model recovery analysis
 
      inputs: self-explanatory, except for resps, which should be [resps_mean, resps_all, resps_sem, base_rate]
      outputs: bestNLL, currParams, varExpl, prefSf, charFreq, [overallNLL, paramList; if joint>0]
  '''
  np = numpy;

  nParam = nParams_descrMod(DoGmodel);
  if DoGmodel == 0:
    joint=0; # we cannot fit the flex gauss model jointly!

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
    stim_dur = get_exp_params(expInd).stimDur;
    max_resp = np.nanmax(resps_all);
    var_all = np.nanvar(resps_all*stim_dur); # we take the variance across COUNTS!
    var_to_mean = var_all/(stim_dur*np.nanmean(resps_all)); # but we divide out the stim_dur when calculating this...
    print('mean||var_to_mean: %.2f||%.2f' % (np.nanmean(resps_all), var_to_mean));
  else: # we don't really need to pass in resps_all
    max_resp = np.nanmax(resps_mean);
    var_to_mean = np.nan;

  # and set up initial arrays
  if prevFits is None or 'NLL' not in prevFits: # no existing fits, or not of correct format
    bestNLL = np.ones((nCons, )) * np.nan;
    currParams = np.ones((nCons, nParam)) * np.nan;
    varExpl = np.ones((nCons, )) * np.nan;
    prefSf = np.ones((nCons, )) * np.nan;
    charFreq = np.ones((nCons, )) * np.nan;
    success = np.zeros((nCons, ), dtype=np.bool_);
    if joint>0:
      overallNLL = np.nan;
      params = np.nan;
  else: # we DO have previous fits
    if noDisp:
      bestNLL, currParams, varExpl, prefSf, charFreq, success = prevFits['NLL'], prevFits['params'], prevFits['varExpl'], prevFits['prefSf'], prevFits['charFreq'], prevFits['success'];
    else:
      bestNLL, currParams, varExpl, prefSf, charFreq, success = prevFits['NLL'][disp,:], prevFits['params'][disp,:], prevFits['varExpl'][disp,:], prevFits['prefSf'][disp,:], prevFits['charFreq'][disp,:], prevFits['success'][disp];
    if joint>0:
       try:
          overallNLL = prevFits['totalNLL'][disp];
          params = prevFits['paramList'][disp];
       except:
          overallNLL = np.nan; params = np.nan;
    if modRecov:
      # ALSO, if it's model recovery, then we are overwriting the existing fits, so let's make the loss NaN
      # --- yes, this is hacky, but we want to easily pass in the fit parameters while also saving these fits
      bestNLL = np.nan * np.zeros_like(bestNLL);

  ############# 
  ### set bounds
  ############# 
  refBounds = None; # will replace (and use) if and only if joint fits AND d-DoG-S model
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
  elif DoGmodel == 1 or DoGmodel == 4: # SACH, sachVol
    bound_gainCent = (1, None);
    bound_radiusCent= (1e-2, 1.5);
    bound_gainSurr = (1e-2, 1); # multiplier on gainCent, thus the center must be weaker than the surround
    #bound_radiusSurr = (3, 3.0001); # multiplier on radiusCent, thus the surr. radius must be larger than the center
    bound_radiusSurr = (1, 10); # multiplier on radiusCent, thus the surr. radius must be larger than the center
    if joint>0: 
      if joint == 1: # original joint (fixed gain and radius ratios across all contrasts)
        bound_gainRatio = (1e-3, 1); # the surround gain will always be less than the center gain
        bound_radiusRatio= (1, 10); # the surround radius will always be greater than the ctr r
        # we'll add to allBounds later, reflecting joint gain/radius ratios common across all cons
        allBounds = (bound_gainRatio, bound_radiusRatio);
      elif joint == 2: # fixed surround radius for all contrasts
        allBounds = (bound_radiusSurr, );
      elif joint == 3: # fixed center AND surround radius for all contrasts
        allBounds = (bound_radiusCent, bound_radiusSurr);
      # In advance of the thesis/publishing the LGN data, we will replicate some of Sach's key results
      # In particular, his thesis covers 4 joint models:
      # -- volume ratio: center and surround radii are fixed, but gains can vary (already covered in joint == 3)
      # -- center radius: fixed center radius across contrast (joint=4) AND fixed volume (i.e. make surround gain constant across contrast)
      # -- surround radius: fixed surround radius across contrast (joint=5) AND fixed volume (i.e. make surround gain constant across contrast) // fixed not in proportion to center, but in absolute value
      # -- center-surround: center and surround radii can vary, but ratio of gains is fixed (joint == 6)
      # ---- NOTE: joints 3-5 have 2*nCons + 2 parms; joint==6 has 3*nCons + 1
      elif joint == 4: # fixed center radius
         allBounds = (bound_radiusCent, bound_gainSurr, ); # center radius AND bound_gainSurr are fixed across condition
      elif joint == 5: # fixed surround radius (again, in absolute terms here, not relative, as is usually specified)
         allBounds = (bound_gainSurr, bound_radiusSurr, ); # surround radius AND bound_gainSurr are fixed across condition
      elif joint == 6: # fixed center:surround gain ratio
         allBounds = (bound_gainSurr, ); # we can fix the ratio by allowing the center gain to vary and keeping the surround in fixed proportion
      elif joint == 7 or joint == 8: # center radius determined by slope! we'll also fixed surround radius; if joint == 8, fixed surround gain instead of radius
         bound_xc_slope = (-1, 1); # 220505 fits inbounded; 220519 fits bounded (-1,1)
         bound_xc_inter = (None, None); #bound_radiusCent; # intercept - shouldn't start outside the bounds we choose for radiusCent
         allBounds = (bound_xc_inter, bound_xc_slope, bound_radiusSurr, ) if joint == 7 else (bound_xc_slope, bound_xc_inter, bound_gainSurr, )
    else:
      allBounds = (bound_gainCent, bound_radiusCent, bound_gainSurr, bound_radiusSurr);
  elif DoGmodel == 2: # TONY
    bound_gainCent = (1e-3, None);
    bound_freqCent = (1e-1, 2e1); # let's set the charFreq upper bound at 20 cpd (is that ok?)
    bound_gainFracSurr = (1e-3, 2); # surround gain always less than center gain NOTE: SHOULD BE (1e-3, 1)
    bound_freqFracSurr = (5e-2, 1); # surround freq always less than ctr freq NOTE: SHOULD BE (1e-1, 1)
    if joint>0:
      if joint == 1: # original joint (fixed gain and radius ratios across all contrasts)
        bound_gainRatio = (1e-3, 3);
        bound_freqRatio = (1e-1, 1); 
        # we'll add to allBounds later, reflecting joint gain/radius ratios common across all cons
        allBounds = (bound_gainRatio, bound_freqRatio);
      elif joint == 2: # fixed surround radius for all contrasts
        allBounds = (bound_freqFracSurr,);
      elif joint == 3: # fixed center AND surround radius for all contrasts
        allBounds = (bound_freqCent, bound_freqFracSurr);
    elif joint==0:
      allBounds = (bound_gainCent, bound_freqCent, bound_gainFracSurr, bound_freqFracSurr);
  elif DoGmodel == 3 or DoGmodel == 5: # d-DoG-S
    # Since the bounds here are more restrictive than the above, particularly for the gain, we'll be more specific
    # ... with how we choose the bounds. Specifically, get the max and min of the "max response for a given disp X con condition"
    
    min_max_resp = np.nanmin(np.nanmax(resps_mean, axis=1)); # average across con to get max per d X con
    max_max_resp = np.nanmax(np.nanmax(resps_mean, axis=1)); # average across con to get max per d X con

    # -- first, some general bounds that we can apply as needed for different parameters
    gtMult_bound = (1, None); # (1, None) multiplier limit when we want param >= mult*orig_param [greaterThan-->gt] ||| was previously (1, 7) --> then (1,5) --> then (1, 15)
    mult_bound_xc = (1, 1.00001);
    sigmoid_bound = (None, None); # params that are input to sigmoids are unbounded - since sigmoid is bounded [0,1]
    noSurr_radBound = (0.01, 0.010001); # gain will be 0 if we are turning off surround; this will be just a small range of radius values to stop the optimization from exploring this parameter space
    noSurr_gainBound = (-9999.00001, -9999); # sigmoid of this value will be basically 0
    # -- then, the central gaussian
    kc1_bound = (0, None);
    xc1_bound = (0.01, 0.5); # values from Hawk; he also uses upper bound of 0.15
    surr1_gain_bound = sigmoid_bound; #(-1.73, 1.1); # through sigmoid translates to (0.15, 0.75), bounds per Hawk; previously was sigmoid_bound
    surr1_rad_bound =  gtMult_bound if DoGmodel==3 else (xc1_bound[1]+0.2, 4); # per Hawk; #gtMult_bound if multiplicative surround
    # -- next, the second (i.e. flanking) gaussian
    kc2_bound = sigmoid_bound;
    xc2_bound = mult_bound_xc if DoGmodel==3 else xc1_bound; # was previously gtMult_bound if ... else (0.015, 2*xc1_bound[1])
    surr2_gain_bound = surr1_gain_bound; # note that surr_gain_bound is LESS restrictive than Hawk
    surr2_rad_bound = gtMult_bound if DoGmodel==3 else (xc2_bound[1]+0.2, surr1_rad_bound[1]); # per Hawk
    # -- finally, the g & S parameters
    g_bound = (-9999.00001, -9999); # sigmoid of this value will be basically 0
    S_bound = sigmoid_bound if DoGmodel==3 else (0.1/8, 1.5*xc1_bound[1]); # was previously 0.1/6 (as compromise between {4,8} for {low,high} SF, per Hawk

    if joint>0: # for d-DoG-S model, the joint values mean the following
      if joint == 1: # g and S are constant across contrast, everything else is per-contrast condition
         allBounds = (g_bound, S_bound);
      elif joint == 2: # g, S, surround radii are constant across contrast, everything else is per-contrast condition
         if no_surr:
            allBounds = ((0.1, 0.1), (0.1, 0.1), g_bound, S_bound);
         else:
            allBounds = (surr1_rad_bound, surr2_rad_bound, g_bound, S_bound);
      elif joint == 3: # g, S constant; surround gain and radius are same for both DoG [no support for no_surr, as of 22.01.12]
         allBounds = (surr1_gain_bound, surr1_rad_bound, g_bound, S_bound);
      elif joint == 7 or joint == 8 or joint == 9: # xc from slope; and as in == 2
         bound_xc_slope = (-1, 1); # 220505 fits inbounded; 220519 fits bounded (-1,1)
         bound_xc_inter = (None, None); #bound_radiusCent; # intercept - shouldn't start outside the bounds we choose for radiusCent
         if joint == 7:
            allBounds = (bound_xc_inter, bound_xc_slope, surr1_rad_bound, surr2_rad_bound, g_bound, S_bound);
         elif joint == 8:
            allBounds = (bound_xc_inter, bound_xc_slope, surr1_rad_bound, g_bound, S_bound);
         elif joint == 9:
            allBounds = (bound_xc_inter, bound_xc_slope, surr1_rad_bound, kc2_bound, g_bound, S_bound);
      # continue to add more joint conditions later on
      # But, let's also get some reference bounds so that we are not out of the range
      if no_surr:
         refBounds = (kc1_bound, xc1_bound, noSurr_gainBound, noSurr_radBound,
                 kc2_bound, xc2_bound, noSurr_gainBound, noSurr_radBound, g_bound, S_bound);
      else:
         refBounds = (kc1_bound, xc1_bound, surr1_gain_bound, surr1_rad_bound,
                 kc2_bound, xc2_bound, surr2_gain_bound, surr2_rad_bound, g_bound, S_bound);
    else: # i.e. not joint
       # parameters are: center gain, center radius, surround gain, surround radius x2 [and, surrounds relative to center]
       #                 g (relative gain to left/right of central DoG) and S (spacing between two DoGs)
       if no_surr: # then no surrounds!
          allBounds = (kc1_bound, xc1_bound, kc2_bound, xc2_bound, g_bound, S_bound);
       else:
          allBounds = (kc1_bound, xc1_bound, surr1_gain_bound, surr1_rad_bound,
                 kc2_bound, xc2_bound, surr2_gain_bound, surr2_rad_bound, g_bound, S_bound);

  ### organize responses -- and fit, if joint=0
  allResps = []; allRespsSem = []; allRespsVar = []; allSfs = []; allRespsTr = []; allSfsTr = []; start_incl = 0; incl_inds = []; allCons = [];
  isolParams = []; # keep track of the previously saved parameters by contrast?

  for con in range(nCons):
    if con not in valConByDisp[disp]:
      continue;
    allCons.append(all_cons[con]);

    if validByStimVal is not None:
      valSfInds = np.array(get_valid_sfs(None, disp, con, expInd, stimVals, validByStimVal)); # we pass in None for data, since we're giving stimVals/validByStimVal, anyway
    else: # then we aren't skipping any
      valSfInds = np.arange(0,len(all_sfs));
    valSfVals = all_sfs[valSfInds];
    # ensure all are strictly GT 0
    valSfVals = valSfVals[valSfVals>0];

    respConInd = np.where(np.asarray(valConByDisp[disp]) == con)[0];
    if modRecov and prevFits is not None:
      # find out how many trials; get the parameters
      nResps = np.max(np.sum(~np.isnan(resps_all[disp,valSfInds, con]), axis=1))
      params = currParams[con];
      # then sample from that fit, and get the mean/sem
      resps_recov = get_descr_recovResponses(params, descrMod=DoGmodel, sfVals=valSfVals, nTr=nResps);
      resps_curr = np.mean(resps_recov, axis=1);
      sem_curr   = sem(resps_recov, axis=1);
      var_curr   = np.var(resps_recov, axis=1);
      resps_all_flat = []; valSfVals_tile = [];
    else:
      valSfInds_curr = np.where(~np.isnan(resps_mean[disp, valSfInds, con]))[0];
      resps_curr = resps_mean[disp, valSfInds[valSfInds_curr], con];
      sem_curr   = resps_sem[disp, valSfInds[valSfInds_curr], con];
      if resps_all is not None:
         var_curr = np.nanvar(resps_all[disp,valSfInds[valSfInds_curr],con], axis=1);
         resps_all_curr = resps_all[disp, valSfInds[valSfInds_curr], con];
         nn = ~np.isnan(resps_all_curr);
         resps_all_flat = resps_all_curr[nn];
         valSfVals_tile = np.repeat(valSfVals[valSfInds_curr], np.sum(nn,axis=1))
      else:
         var_curr = np.nan; resps_all_flat = []; valSfVals_tile = [];

    ### prepare for the joint fitting, if that's what we've specified!
    if joint>0:
      if resps_curr.size == 0:
         continue;
      if ref_varExpl is None:
        start_incl = 1; # this means if we don't have a reference varExpl, we just fit all conditions
      if start_incl == 0:
        if ref_varExpl[con] < veThresh:
          continue; # i.e. we're not adding this; yes we could move this up, but keep it here for now
        else:
          start_incl = 1; # now we're ready to start adding to our responses that we'll fit!

      try:
         if 'params' in isolFits:
            params_prev = isolFits['params'][disp,con];
         else: # we just passed in the params directly!
            params_prev = isolFits[disp,con];
         isolParams.append(params_prev);
      except:
         isolParams.append([]);
      incl_inds.append(con); # keep note of which contrast indices are included
      allResps.append(resps_curr);
      allRespsSem.append(sem_curr);
      allRespsVar.append(var_curr);
      allSfs.append(valSfVals[valSfInds_curr]);
      allRespsTr.append(resps_all_flat);
      allSfsTr.append(valSfVals_tile);
      # and add to the parameter list!
      if DoGmodel == 1: # SACH
        if joint == 1: # add the center gain and center radius for each contrast 
          allBounds = (*allBounds, bound_gainCent, bound_radiusCent);
        elif joint == 2: # add the center and surr. gain and center radius for each contrast 
          allBounds = (*allBounds, bound_gainCent, bound_radiusCent, bound_gainSurr);
        elif joint == 3:  # add the center and surround gain for each contrast 
          allBounds = (*allBounds, bound_gainCent, bound_gainSurr);
        elif joint == 4: # fixed center radius, so add all other parameters
          allBounds = (*allBounds, bound_gainCent, bound_radiusSurr);
        elif joint == 5: # add the center and surr. gain and center radius for each contrast 
          allBounds = (*allBounds, bound_gainCent, bound_radiusCent);
        elif joint == 6: # fixed center:surround gain ratio
          allBounds = (*allBounds, bound_gainCent, bound_radiusCent, bound_radiusSurr);
        elif joint == 7: # center radius det. by slope, surround radius fixed
          allBounds = (*allBounds, bound_gainCent, bound_gainSurr);
        elif joint == 8: # center radius det. by slope, surround gain fixed
          allBounds = (*allBounds, bound_gainCent, bound_radiusSurr);
      elif DoGmodel == 2: # TONY
        if joint == 1: # add the center gain and center radius for each contrast 
          allBounds = (*allBounds, bound_gainCent, bound_freqCent);
        elif joint == 2: # add the center and surr. gain and center radius for each contrast 
          allBounds = (*allBounds, bound_gainCent, bound_freqCent, bound_gainFracSurr);
        elif joint == 3:  # add the center and surround gain for each contrast 
          allBounds = (*allBounds, bound_gainCent, bound_gainFracSurr);
      elif DoGmodel == 3 or DoGmodel == 5: # d-DoG-S models
        if joint == 1: # add the center gain and center radius for each contrast 
           if no_surr:
              allBounds = (*allBounds, kc1_bound, xc1_bound, noSurr_gainBound, noSurr_radBound,
                           kc2_bound, xc2_bound, noSurr_gainBound, noSurr_radBound);
           else:
              allBounds = (*allBounds, kc1_bound, xc1_bound, surr1_gain_bound, surr1_rad_bound,
                 kc2_bound, xc2_bound, surr2_gain_bound, surr2_rad_bound);
        elif joint == 2: # add the center and surr. gain and center radius for each contrast 
           if no_surr:
              allBounds = (*allBounds, kc1_bound, xc1_bound, noSurr_gainBound,
                           kc2_bound, xc2_bound, noSurr_gainBound);
           else:
              allBounds = (*allBounds, kc1_bound, xc1_bound, surr1_gain_bound, 
                 kc2_bound, xc2_bound, surr2_gain_bound);
        elif joint == 3: # add the center gain and radius for each contrast
           allBounds = (*allBounds, kc1_bound, xc1_bound, kc2_bound, xc2_bound);
        elif joint == 7: # add the center and surr. gains ONLY
           allBounds = (*allBounds, kc1_bound, surr1_gain_bound, 
                                    kc2_bound, surr2_gain_bound);
        elif joint == 8: # akin to jt=7, but no separate surr2_gain_bound
           allBounds = (*allBounds, kc1_bound, surr1_gain_bound, 
                                    kc2_bound);
        elif joint == 9: # akin to jt=8, but no separate kc2_bound
           allBounds = (*allBounds, kc1_bound, surr1_gain_bound);

      continue;

    ### otherwise, we're really going to fit here! [i.e. if joint==0]
    # --- NOTE: We never reach here if it's a joint fitting!
    save_all = []; # only used if debug=0
    sfs_curr = valSfVals[valSfInds_curr];

    for n_try in range(n_repeats):
      ###########
      ### pick initial params
      ###########
      init_params = dog_init_params(resps_curr, base_rate, all_sfs, sfs_curr, DoGmodel, allBounds, no_surr=no_surr)

      # choose optimization method
      methodStr = 'L-BFGS-B'; # previously, alternated between L-BFGS-B and TNC (even/odd)
      
      if no_surr:
         # perhaps hacky (while we test it out), but important for passing in params
         obj = lambda params: DoG_loss([*params[0:2], -np.inf, 1,
                                        *params[2:4], -np.inf, 1, 
                                        *params[4:]], 
                                       resps_curr, sfs_curr, resps_std=sem_curr, var_to_mean=var_to_mean, loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, gain_reg=gain_reg, joint=joint, baseline=baseline, enforceMaxPenalty=1, vol_lam=vol_lam);
      else:
         obj = lambda params: DoG_loss(params, resps_curr, sfs_curr, resps_std=var_curr, var_to_mean=var_to_mean, loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, gain_reg=gain_reg, joint=joint, baseline=baseline, enforceMaxPenalty=1, vol_lam=vol_lam);
      try:
        maxfun = 145000 if not is_mod_DoG(DoGmodel) else 145000; # default is 15000; d-dog-s model often needs more iters to finish
        wax = opt.minimize(obj, init_params, method=methodStr, bounds=allBounds, options={'maxfun': maxfun});
      except:
        continue; # the fit has failed (bound issue, for example); so, go back to top of loop, try again

      # compare
      NLL = wax['fun'];
      params = wax['x'];
      #print('success? %d, %s' % (wax['success'], wax['message']));

      if no_surr:
         # then fill in the real params...
         params_full = np.zeros_like(currParams[con,:]);
         params_full[0:2] = params[0:2];
         params_full[2:4] = [-np.inf, 1]; # the surrounds
         params_full[4:6] = params[2:4];
         params_full[6:8] = [-np.inf, 1]; # the surrounds
         params_full[8:] = params[4:]
         params = params_full;

      if debug:
        save_all.append([wax, init_params]);
        
      if np.isnan(bestNLL[con]) or NLL < bestNLL[con]:
        bestNLL[con] = NLL;
        currParams[con, :] = params;
        varExpl[con] = var_explained(resps_curr, params, sfs_curr, DoGmodel, baseline=baseline);
        prefSf[con] = descr_prefSf(params, dog_model=DoGmodel, all_sfs=valSfVals, baseline=baseline);
        charFreq[con] = dog_charFreq(params, DoGmodel=DoGmodel);
        success[con] = wax['success'];

  if joint==0:
    if debug:
      return bestNLL, currParams, varExpl, prefSf, charFreq, None, None, save_all, success; # placeholding None for overallNLL, params [full list]
    else:
      return bestNLL, currParams, varExpl, prefSf, charFreq, None, None, success; # placeholding None for overallNLL, params [full list]

  ########### 
  ### NOW, we do the fitting if joint>0
  ### - note: outside of contrast loop, since we instead gathered info to be fit here
  ########### 
  def clean_sigmoid_params(prms, dogMod=3, sigmoid=[0,0,1,0,1,0,1,0,0,1], lower_bound=-2.5, upper_bound=2.5):
     ''' when using saved parameters for initialization, make sure that any sigmoided values are not at the extremes (makes opt. difficult)
      -- for ref: sigmoid([-/+]1.09)~= 0.25/0.75, sigmoid(0) = 0.5, sigmoid([-/+]2.94)~=0.05/0.95
      NOTE: we do not classify 'g' (2nd-last param.) as sigmoid by default, since that value is typically intentionally kept at an extreme value
     '''
     if dogMod != 3:
        return prms;
     else:
        out_prms = np.copy(prms);
        for (i,p),s in zip(enumerate(prms),sigmoid):
           if s: # i.e. if the param. is sigmoid
              if p>upper_bound:
                 out_prms[i]=upper_bound;
              if p<lower_bound:
                 out_prms[i]=lower_bound;
        return out_prms;
  ### --- end of internal function

  if joint>0: 
    if len(allResps)<jointMinCons: # need at least jointMinCons contrasts!
       # so, then we just return HERE
       if debug:
          return bestNLL, currParams, varExpl, prefSf, charFreq, overallNLL, params, None, success;
       else:
          return bestNLL, currParams, varExpl, prefSf, charFreq, overallNLL, params, success;
    ### now, we fit!
    for n_try in range(n_repeats):
      # first, estimate the joint parameters; then we'll add the per-contrast parameters after
      # --- we'll estimate the joint parameters based on the high contrast response
      ref_resps = allResps[-1]; ref_sfs = allSfs[-1];
      if isolParams[-1] == [] or n_try>0: # give one shot with the isolParams initialization, then move on
         ref_init = dog_init_params(ref_resps, base_rate, all_sfs, ref_sfs, DoGmodel, bounds=refBounds);
         if DoGmodel==3: # i.e. give one shot where we initialize straight from there, otherwise we know the typical range of the surr. radius joint params
            ### The following procedure/values for generating initial guesses of the joint parameters (surr ratio for DoGs 1 and 2, spacing constant) come from the analysis in ch1_suppl.ipynb::ddogs::"Smarter initialization"
            # -- in short, after taking the log2 of the final parameters and comparing the distr. to the initial guesses, we can do a lot better!
            # ---- in log2 space, the radius values are roughly exponential; we'll also use a transformed exponential for the spacing constant, though it's really bi-modal...
            # ---- also, we'll clip to avoid extreme values
            from scipy.stats import expon as expon_distr
            ref_init[3] = np.clip(np.power(2, expon_distr.rvs(loc=0,scale=0.84)), 1, 8); # simply draw from an exponential distr.
            ref_init[7] = np.clip(np.power(2, expon_distr.rvs(loc=0,scale=1.54)), 1, 8); # as above, but a longer tail
            ref_init[-1] = np.clip(np.sign(np.random.rand()-0.5)*np.power(2, 1.5-expon_distr.rvs(loc=0,scale=.8)), -3.5, 3.5); # here, we make a distribution the rises from -1 and beyond towards a peak at 2; then randomly choose a sign so that we have peaks at +/-2 (in log2, so 4 after np.power(x,2))
      else: # first attempt --> initialize from the isolated fits
         ref_init = clean_sigmoid_params(isolParams[-1], dogMod=DoGmodel); # initialize the joint parameters on the basis of the above
         print('initializing from isolated fits');
      if is_mod_DoG(DoGmodel):
         if joint == 1: # gain ratio (i.e. surround gain) [0] and shape ratio (i.e. surround radius) [1] are joint
            allInitParams = [ref_init[2], ref_init[3]];
         elif joint == 2: #  surround radius [0] (as ratio in 2; fixed in 5) is joint
            allInitParams = [ref_init[3]];
         elif joint == 3: # center radius [0] and surround radius [1] ratio are joint
            allInitParams = [ref_init[1], ref_init[3]];
         elif joint == 4: # center radius, surr. gain fixed
            allInitParams = [ref_init[1], ref_init[2]];
         elif joint == 5: #  surround gain AND radius [0] (as ratio in 2; fixed in 5) are joint
            allInitParams = [ref_init[2], ref_init[3]];
         elif joint == 6: # center:surround gain is fixed
            allInitParams = [ref_init[2]];
         elif joint == 7 or joint == 8: # center radius offset and slope fixed; surround radius fixed [7] or surr. gain fixed [8];
            # the slope will be calculated on log contrast, and will start from the lowest contrast
            # -- i.e. xc = np.power(10, init+slope*log10(con))
            init_slope = random_in_range([-0.15,0.15])[0]
            # now, work backwards to infer the initial intercept, given the slope and ref_init[1] (xc)
            # --- note: assumes base10 for slope model and 100% contrast as the reference...
            init_intercept = np.log10(ref_init[1]) - init_slope*np.log10(100);
            allInitParams = [init_intercept, init_slope, ref_init[3]] if joint == 7 else [init_intercept, init_slope, ref_init[2]];
      else: # d-DoG-S models
         if joint==1 or (no_surr and joint==2):
            allInitParams = [*ref_init[-2:]] # g,S (i.e. the final two parameters)
         elif joint==2: # already handled no_surr case in the above "if"
            allInitParams = [ref_init[3], ref_init[7], *ref_init[-2:]]; # surround radius, in addition to g, S
         elif joint==3: # need to initialize surround gain and surround radius (we'll take from the central DoG
            allInitParams = [ref_init[2], ref_init[3], *ref_init[-2:]] # surround gain and radius
         elif joint==7 or joint==8 or joint==9: # like DoG joint==7 on top of d-DoG-S joint == 2
            init_slope = random_in_range([-0.2,0.1])[0];
            # now, work backwards to infer the initial intercept, given the slope and ref_init[1] (xc)
            # --- note: assumes base10 for slope model and 100% contrast as the reference...
            init_intercept = np.log10(ref_init[1]) - init_slope*np.log10(100);
            if joint == 7: # besides slope, intercept: surround radius for both DoGs are joint, as are g,S
               allInitParams = [init_intercept, init_slope, ref_init[3], ref_init[7], *ref_init[-2:]]; 
            elif joint == 8: # only one surround ratio
               allInitParams = [init_intercept, init_slope, ref_init[3], *ref_init[-2:]];
            elif joint == 9: # as joint==8, but also with flank overall gain fixed across contrast
               allInitParams = [init_intercept, init_slope, ref_init[3], ref_init[4], *ref_init[-2:]]; 

      # now, we cycle through all responses and add the per-contrast parameters
      for resps_curr, sfs_curr, stds_curr, curr_init, resps_curr_tr, sfs_curr_tr, cons_curr in zip(allResps, allSfs, allRespsSem, isolParams, allRespsTr, allSfsTr, allCons):
        if resps_curr.size == 0:
           continue;
        if curr_init == [] or n_try>0:
           curr_init = dog_init_params(resps_curr, base_rate, all_sfs, sfs_curr, DoGmodel, bounds=refBounds);
           if DoGmodel==3: # we'll initialize by fitting d-DoG-S separately with all joint parameters fixed...
              if joint == 2:
                 obj_isol = lambda params: DoG_loss(np.array([*params[0:3], allInitParams[0], *params[3:], *allInitParams[1:4]]), resps_curr, sfs_curr, loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, resps_std=stds_curr, var_to_mean=var_to_mean, gain_reg=gain_reg, joint=joint, baseline=baseline, vol_lam=vol_lam);
                 bounds_isol = allBounds[4:10]; # skip the 4 joint parameters; same for all contrasts
                 wax_isol = opt.minimize(obj_isol, np.array([*curr_init[0:3], *curr_init[4:7]]), method='L-BFGS-B', bounds=bounds_isol);
                 ci = np.copy(ref_init);
                 ci[0:3] = wax_isol['x'][0:3];
                 ci[4:7] = wax_isol['x'][3:];
                 print('success in interim fit? %d [vExp=%.2f]' % (wax_isol['success'], var_explained(resps_curr, ci, sfs_curr, dog_model=DoGmodel, baseline=baseline)));
              if joint == 9:
                 init_xc = get_xc_from_slope(allInitParams[0], allInitParams[1], cons_curr);
                 obj_isol = lambda params: DoG_loss(np.array([params[0], init_xc, params[1], allInitParams[2], allInitParams[3], 1, params[1], allInitParams[2], allInitParams[4], allInitParams[5]]), resps_curr, sfs_curr, loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, resps_std=stds_curr, var_to_mean=var_to_mean, gain_reg=gain_reg, joint=0, baseline=baseline, vol_lam=vol_lam);
                 bounds_isol = (kc1_bound, surr1_gain_bound);
                 wax_isol = opt.minimize(obj_isol, np.array([curr_init[0], curr_init[2]]), method='L-BFGS-B', bounds=bounds_isol);
                 ci = np.copy(ref_init);
                 ci[0] = wax_isol['x'][0];
                 ci[2] = wax_isol['x'][1];
                 print('success in interim fit? %d [vExp=%.2f]' % (wax_isol['success'], var_explained(resps_curr, ci, sfs_curr, dog_model=DoGmodel, baseline=baseline)));
           curr_init = clean_sigmoid_params(ci, dogMod=DoGmodel);
        else: # first attempt --> initialize from the isolated fits
           vE = var_explained(resps_curr, curr_init, sfs_curr, dog_model=DoGmodel, baseline=baseline);
           print('...init from isolParams (varExpl=%.2f)' % vE);
           print(curr_init);
           curr_init = clean_sigmoid_params(curr_init, dogMod=DoGmodel);
        if is_mod_DoG(DoGmodel):
           if joint == 1:
              allInitParams = [*allInitParams, curr_init[0], curr_init[1]];
           elif joint == 2: # then we add center gain, center radius, surround gain (i.e. params 0:3
              allInitParams = [*allInitParams, curr_init[0], curr_init[1], curr_init[2]];
           elif joint == 3: # then we add center gain and surround gain (i.e. params 0, 2)
              allInitParams = [*allInitParams, curr_init[0], curr_init[2]];
           elif joint == 4: # then we add center gain, surround radius
              allInitParams = [*allInitParams, curr_init[0], curr_init[3]];
           elif joint == 5: # then we add center gain, center radius
              allInitParams = [*allInitParams, curr_init[0], curr_init[1]];
           elif joint == 6: # then we add center gain and both radii
              allInitParams = [*allInitParams, curr_init[0], curr_init[1], curr_init[3]];
           elif joint == 7: # then we add center and surround gains
              allInitParams = [*allInitParams, curr_init[0], curr_init[2]];
           elif joint == 8: # then we add center gain, surr. radius
              allInitParams = [*allInitParams, curr_init[0], curr_init[3]];

        else: # d-DoG-S models
           if joint==1:# or (no_surr and joint==2):
              allInitParams = [*allInitParams, *curr_init[0:-2]];
           elif joint==2: # already handled no_surr case in the above "if"
              allInitParams = [*allInitParams, *curr_init[0:3], *curr_init[4:-3]] # add in central/flank center gain, radius, and surround gain
           elif joint==3:
              allInitParams = [*allInitParams, *curr_init[0:2], *curr_init[4:6]] # add in central center gain, radius; flanking center gain, radius
           elif joint==7: # don't need to add center radii, since that's taken care of by slope model
              allInitParams = [*allInitParams, curr_init[0], curr_init[2], curr_init[4], curr_init[6]]; # we add gain parameters for each DoG mechanism (all radii are handled jointly)
           elif joint == 8: # don't need to add center radii, since that's taken care of by slope model
              allInitParams = [*allInitParams, curr_init[0], curr_init[2], curr_init[4]]; # no need to add separate flank surround gain (same as central surround gain)
           elif joint == 9: # don't need to add center radii, since that's taken care of by slope model
              allInitParams = [*allInitParams, curr_init[0], curr_init[2]]; # no need to add separate flank gains (flank surround is same as central surround; flank center gain is jointly fit)
              
      # previously, we choose optimization method (L-BFGS-B for even, TNC for odd) --- we now just choose the former
      methodStr = 'L-BFGS-B';
      #obj = lambda params: DoG_loss(params, allRespsTr, allSfsTr, resps_std=allRespsSem, loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, gain_reg=gain_reg, joint=joint, baseline=baseline, vol_lam=vol_lam, n_fits=len(allResps)); # trial-by-trial
      obj = lambda params: DoG_loss(params, allResps, allSfs, resps_std=allRespsSem, var_to_mean=var_to_mean, loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, gain_reg=gain_reg, joint=joint, baseline=baseline, vol_lam=vol_lam, n_fits=len(allResps), conVals=allCons, );
      # --- debugging ---
      try: # 95000; 35000; 975000
        maxfun = 1975000 if not is_mod_DoG(DoGmodel) else 155000; # default is 15000; d-dog-s model often needs more iters to finish
        wax = opt.minimize(obj, allInitParams, method=methodStr, bounds=allBounds, options={'ftol': ftol, 'maxfun': maxfun});
      except:
        continue; # if that particular fit fails, go back and try again

      print('%d: %s --> %s [loss: %.2e]' % (n_try, wax['success'], wax['message'], wax['fun']));
      # compare
      NLL = wax['fun'];
      params_curr = np.asarray(wax['x'], np.float32) if flt32 else wax['x']

      if np.isnan(overallNLL) or NLL < overallNLL or len(params_curr) != len(params): # the final check is if the # of parameters here is different from the exising # params --> then update, because our separate fits have updated and we have different # of conditions to fit
        overallNLL = NLL;
        params = params_curr;
        success = wax['success'];

    ### Done with multi-start fits; now, unpack the fits to fill in the "true" parameters for each contrast
    # --- first, get the global parameters
    ref_rc_val = None; # only used if is_mod_DoG and joint==5; otherwise, just pass in None
    if is_mod_DoG(DoGmodel):
      if joint == 1:
        gain_rat, shape_rat = params[0], params[1];
      elif joint == 2:
        surr_shape = params[0]; # radius or frequency, if Tony model
      elif joint == 3:
        center_shape, surr_shape = params[0], params[1]; # radius or frequency, if Tony model
      elif joint == 4: # center radius, surr. gain fixed
        center_shape, surr_gain = params[0], params[1];
      elif joint == 5: # surr. gain, surr. radius fixed
        surr_gain, surr_shape = params[0], params[1];
        ref_rc_val = params[2]; # center radius for high contrast
      elif joint == 6: # ctr:surr gain fixed
        surr_gain = params[0];
      elif joint == 7: # center gain det. from slope, surround radius fixed
        xc_inter, xc_slope, surr_shape = params[0:3];
      elif joint == 8: # center gain det. from slope, surround gain fixed
        xc_inter, xc_slope, surr_gain = params[0:3];
    else:
      if joint == 1:
        g, S = params[0], params[1];
      elif joint == 2:
        surr1_rad, surr2_rad = params[0], params[1];
        g, S = params[2:4];
      elif joint == 3:
        surr_gain, surr_rad = params[0], params[1];
        g, S = params[2:4];
      elif joint == 7 or joint == 8 or joint == 9:
        xc_inter, xc_slope = params[0:2];
        surr1_rad = params[2];
        if joint == 7 or joint == 9:
           if joint == 7:
              surr2_rad = params[3];
           elif joint == 9:
              flankGain = params[3];
           g, S = params[4:6];
        elif joint == 8:
           g, S = params[3:5];
        
    for con in reversed(range(len(allResps))):
       if allResps[con].size == 0:
          continue;
       # --- then, go through each contrast and get the "local", i.e. per-contrast, parameters
       if is_mod_DoG(DoGmodel):
          if joint == 1: # center gain, center shape
             center_gain = params[2+con*2]; 
             center_shape = params[3+con*2]; # shape, as in radius/freq, depending on DoGmodel
             curr_params = [center_gain, center_shape, gain_rat, shape_rat];
          elif joint == 2: # center gain, center radius, surround gain
             center_gain = params[1+con*3]; 
             center_shape = params[2+con*3];
             surr_gain = params[3+con*3];
             curr_params = [center_gain, center_shape, surr_gain, surr_shape];
          elif joint == 3: # center radius, surr radius fixed for all contrasts
             center_gain = params[2+con*2]; 
             surr_gain = params[3+con*2];
             curr_params = [center_gain, center_shape, surr_gain, surr_shape];
          elif joint == 4: # center radius, surr. gain fixed for all contrasts
             center_gain = params[2+con*2]; 
             surr_shape = params[3+con*2];
             curr_params = [center_gain, center_shape, surr_gain, surr_shape];
          elif joint == 5: # surround gain, radius fixed for all contrasts
             center_gain = params[2+con*2]; 
             center_shape = params[3+con*2];
             curr_params = [center_gain, center_shape, surr_gain, surr_shape];
          elif joint == 6: # ctr:surr gain fixed for all contrasts
             center_gain = params[1+con*3]; 
             center_shape = params[2+con*3];
             surr_shape = params[3+con*3];
             curr_params = [center_gain, center_shape, surr_gain, surr_shape];
          elif joint == 7 or joint == 8: # surr radius [7] or gain [8] fixed; need to determine center radius from slope
             center_gain = params[3+con*2]; 
             center_shape = get_xc_from_slope(params[0], params[1], allCons[con]);
             if joint == 7:
                surr_gain = params[4+con*2];
             elif joint == 8:
                surr_shape = params[4+con*2];
             curr_params = [center_gain, center_shape, surr_gain, surr_shape];
       else: ### unpack the d-DoG-S parameters [for joint fits]
          if joint == 1: # we've already grabbed g, S
             nParam_perCond = nParam-2; # the usual 10 parameters, minus 2 for those held joint
             start_ind = 2+con*nParam_perCond;
             curr_params = [*params[start_ind:(start_ind+nParam-2)], g, S]
          elif joint == 2: # we've added surround radius #1&2, g, S
             nParam_perCond = nParam-4; # the usual 10 parameters, minus 4 for those held joint
             start_ind = 4+con*nParam_perCond;
             curr_params = [*params[start_ind:start_ind+3], surr1_rad,
                            *params[start_ind+3:start_ind+6], surr2_rad, g, S];
          elif joint == 3:
             nParam_perCond = nParam-6; # the usual 10 parameters, minus 6 for those held joint
             start_ind = 4+con*nParam_perCond; # but we still start at +4 because the surround params are used twice
             curr_params = [*params[start_ind:start_ind+2], surr_gain, surr_rad,
                            *params[start_ind+2:start_ind+4], surr_gain, surr_rad, g, S];
          elif joint == 7 or joint == 8 or joint == 9: # need to determine center radii (both central and flank DoGs) from slope
             center_shape = get_xc_from_slope(params[0], params[1], allCons[con]);

             if joint == 7:
                nParam_perCond = nParam-6; # the usual 10 parameters, minus 6 for those held joint (only the gains are fit per contrast)
                start_ind = 6+con*nParam_perCond; # 6 joint parameters (two determining the center radius slope; g,S; surround rad 1 & 2)
                curr_params = [params[start_ind], center_shape, params[start_ind+1], surr1_rad,
                            params[start_ind+2], 1, params[start_ind+3], surr2_rad, g, S];
             elif joint == 8:
                nParam_perCond = nParam-7; # the usual 10 parameters, minus 7 for those held joint/fixed (only the central gains and flank relative gain)
                start_ind = 5+con*nParam_perCond; # 5 joint parameters (two determining the center radius slope; g,S; surround rad 1)
                curr_params = [params[start_ind], center_shape, params[start_ind+1], surr1_rad,
                            params[start_ind+2], 1, params[start_ind+1], surr1_rad, g, S];
             elif joint == 9: # as in joint == 8, but flank center gain is fixed across contrast
                nParam_perCond = nParam-8;
                start_ind = 6+con*nParam_perCond;
                curr_params = [params[start_ind], center_shape, params[start_ind+1], surr1_rad,
                            flankGain, 1, params[start_ind+1], surr1_rad, g, S];

       # -- then the responses, and overall contrast index
       resps_curr = allResps[con];
       sem_curr   = allRespsSem[con];
       respConInd = incl_inds[con];
       sfs_curr   = allSfs[con];
       #respConInd = valConByDisp[disp][con]; 
       
       if con==len(allResps)-1:
          ref_params = curr_params;

       # now, compute loss, explained variance, etc
       bestNLL[respConInd] = DoG_loss(curr_params, resps_curr, sfs_curr, resps_std=sem_curr, var_to_mean=var_to_mean, loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, gain_reg=gain_reg, joint=0, baseline=baseline, vol_lam=vol_lam, ref_params=ref_params, ref_rc_val=ref_rc_val); # not joint, now!
       currParams[respConInd, :] = curr_params;
       varExpl[respConInd] = var_explained(resps_curr, curr_params, sfs_curr, DoGmodel, baseline=baseline, ref_params=ref_params, ref_rc_val=ref_rc_val);
       prefSf[respConInd] = descr_prefSf(curr_params, dog_model=DoGmodel, all_sfs=valSfVals, baseline=baseline, ref_params=ref_params, ref_rc_val=ref_rc_val);
       charFreq[respConInd] = dog_charFreq(curr_params, DoGmodel=DoGmodel);
        

    # and NOW, we can return! (unpacked all values per contrast)
    if debug:
      return bestNLL, currParams, varExpl, prefSf, charFreq, overallNLL, params, None, success; # None should be save_all
    else:
      return bestNLL, currParams, varExpl, prefSf, charFreq, overallNLL, params, success;
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

    else: # we'll do this numerically rather than in closed form
      prefSf = descr_prefSf(fit, dog_model=sfMod, all_sfs=sf_range)
      peakResp = get_descrResp(fit, prefSf, sfMod);
      targetResp = peakResp*height;
      # todo: verify this is OK? 21.11.11
      obj = lambda sf: np.square(targetResp - get_descrResp(fit, stim_sf=sf, DoGmodel=sfMod));
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

    # Tuning function
    sf0   = numpy.divide(stim_sf, sfPref);

    sigma = numpy.full_like(sf0, sigmaLow);
    whereSigHigh = numpy.where(sf0>1);
    sigma[whereSigHigh] = sigmaHigh;

    shape = numpy.exp(-numpy.divide(numpy.square(numpy.log(sf0)), 2*numpy.square(sigma)))
                
    return numpy.maximum(minThresh, respFloor + respRelFloor*shape);

def dog_to_four(k, x, f):
   # simple relationship going from spatial domain to fourier domain for DoG
   # - as of 22.01.14, we do NOT include the spatial offset (handled within parker_hawken() call)

   return k * numpy.exp(-numpy.square(numpy.pi*f*x));

def parker_hawken_transform(params, twoDim=False, space_in_arcmin=False, isMult=False, ref_params=None):
    # Given the parameterization used for optimization, convert to "real", i.e. interpretable, parameters
    # -- non-default for easier interpretation, return spatial parameters in arcmin

    kc1, xc1, kS_rel1, A = params[0:4];
    kc2, B, kS_rel2, C = params[4:8];
    if twoDim:
      gPr, Spr, yh = params[8:]; # pr for prime
    else:
      gPr, Spr = params[8:]; # pr for prime (i.e. S')

    # Transform the parameters
    g = sigmoid(gPr);
    # --- 
    # - xc1 is first priority to specify (directly parameterized)
    # - then, we get xs1 relative to xc1 (xs1 >= xc1 and xs1 <= Z*xc1, e.g. Z=4) 
    xs1 = A*xc1 if isMult else A;
    # - in parallel, xc2 relative to xc1 (xc2>=xc1)
    xc2 = B*xc1 if isMult else B;
    # finally, xs2 relative to xc2 (xs2>=xc2)
    xs2 = C*xc2 if isMult else C;
    if isMult:
       if ref_params is None:
          S = numpy.minimum(xc1, xc2) + sigmoid(Spr) * 1 * numpy.maximum(xc1, xc2); # 220120 ;i.e. must be between [min(xc1, xc2), xc1+xc2]
       else:
          ref_xc1, ref_xc2 = ref_params[1], ref_params[1]*ref_params[5];
          S = numpy.minimum(ref_xc1, ref_xc2) + sigmoid(Spr) * 1 * numpy.maximum(ref_xc1, ref_xc2); # 220121a--; temporarily sigmoid()*2*(max)
          # ^^ i.e. must be between [min(xc1, xc2), xc1+xc2]
    else:
       S = Spr;
    kc2 = sigmoid(kc2)*kc1;
    ks1 = sigmoid(kS_rel1)*kc1;
    ks2 = sigmoid(kS_rel2)*kc2;

    params[2] = ks1;
    params[3] = xs1;
    params[4] = kc2;
    params[5] = xc2;
    params[6] = ks2;
    params[7] = xs2;
    params[8] = g;
    params[9] = S;

    if space_in_arcmin:
       sp_prms = [1, 3, 5, 7, 9];
       for i in sp_prms:
         params[i] = arcmin_to_deg(params[i], reverse=True);
  
    return params;

def parker_hawken(params, stim_sf=None, twoDim=False, inSpace=False, spaceRange=0.5, nSteps=60, debug=False, isMult=True, transform=True, ref_params=None, baseline=0):
    ''' space value (xc_i, xs_i) are specified in degrees of visual angle, i.e. 60 arcminuntes = 1 (deg)
        --- smart parameterization to enforce limits
        Limits (when isMult==True):
        - xc_1 <= xc_2
        - xc_i < xs_i
        - S < 2*(xc_1 + xc_2)
        - 0 <= g <= 1
        - ks_i <= kc_i
        THUS, (all apply when isMult==True; only mult. one when False is ks_i)
        - xc1 = A*xc2, where A will be used as input to sigmoid, thus bounded between [0, 1]
        - xc_i = B*xs_i, where B will be used as input to sigmoid, thus bounded between [0, 1)
        - S = C*2*(xc_1 + xc_2), where C will be used ...
        - g will be input to sigmoid
        - ks_i = D*kc_i where D will be used ...
        If noSurround, then the surround for each DoG has 0 amplitude
    
        - NOTE: If surrRadMult, then the surround radius is a multiple of the center; otherwise, it's independent
    '''

    np = numpy;

    if transform:
      # we don't want to modify the original params, so we pass in a copy
      params_corr = parker_hawken_transform(np.copy(params), twoDim=twoDim, isMult=isMult, ref_params=ref_params);
    else: # just used for debugging (e.g. comparing to published parameters in Parker/Hawken '87/'88
      params_corr = params; 

    kc1, xc1, ks1, xs1 = params_corr[0:4];
    kc2, xc2, ks2, xs2 = params_corr[4:8];
    if twoDim:
      g, S, yh = params_corr[8:]; # pr for prime
    else:
      g, S = params_corr[8:]; # pr for prime (i.e. S')

    if inSpace: # If we want the function in space

      xSamps = np.arange(-spaceRange, spaceRange+1.0/nSteps, 0.5/nSteps); # go in steps of half an arcminute
      # As described in the appendix of Hawken & Parker, 1987, it is necessary to divide out sqrt(pi)*radius from all gains in the spatial domain
      # -- this is necessary when we simply use a constant (independent of pi, radius, or any other factors) when fitting the frequency response
      dog1 = kc1*np.power(np.sqrt(np.pi)*xc1, -1)*np.exp(-np.square((xSamps+0)/xc1)) - ks1*np.power(np.sqrt(np.pi)*xs1, -1)*np.exp(-np.square((xSamps+0)/xs1))
      dog2 = kc2*np.power(np.sqrt(np.pi)*xc2, -1)*np.exp(-np.square((xSamps+S)/xc2)) - ks2*np.power(np.sqrt(np.pi)*xs2, -1)*np.exp(-np.square((xSamps+S)/xs2))
      dog3 = kc2*np.power(np.sqrt(np.pi)*xc2, -1)*np.exp(-np.square((xSamps-S)/xc2)) - ks2*np.power(np.sqrt(np.pi)*xs2, -1)*np.exp(-np.square((xSamps-S)/xs2))

      respSpace = dog1 - g*dog2 - (1-g)*dog3;   

      if twoDim:
        ySamps = np.linspace(-2*yh, 2*yh, 100);
        ## TODO -- finish

      if debug:
        return respSpace, xSamps, dog1, g*dog2, (1-g)*dog3;
      else:
        return respSpace, xSamps;

    else: # otherwise, if we want it in the Fourier domain
      stim_sf = np.geomspace(0.1, 10, 25) if stim_sf is None else stim_sf; # just so we have something, if needed

      dog1 = dog_to_four(kc1, xc1, stim_sf) - dog_to_four(ks1, xs1, stim_sf)
      dog2 = (dog_to_four(kc2, xc2, stim_sf) - dog_to_four(ks2, xs2, stim_sf)) * np.cos(2*np.pi*stim_sf*S); # even
      dog3 = (1-2*g) * (dog_to_four(kc2, xc2, stim_sf) - dog_to_four(ks2, xs2, stim_sf)) * np.sin(2*np.pi*stim_sf*S); # odd
       
      full = np.sqrt(np.square(dog1-dog2) + np.square(dog3)) + baseline;

      return full;

def parker_hawken_space_from_stim(stim_sfs, stim_cons, stim_tfs, stim_phi, stim_dur, curr_params, ref_params=None, spaceRange=0.5, nSteps=100, tSamp=400, debug=False):
   ''' Given a stimulus description and set of d-DoG-S fits:
       - compute the (1D) spatial profile of the receptive field
       - compute the (1D) stimulus at each time step and convolve with the RF
       - Threshold at 0, take the mean
       return: a scalar as response to the stimulus, [and the full time course, if debug==True]
   
       spaceRange -- full range is 2x (we go +&- of 0)
       nSteps     -- how many steps to make in sampling space
       tSamp      -- how many samples per second in time
   '''
   np = numpy;
   space_rf, xSamps = parker_hawken(curr_params, inSpace=True, spaceRange=spaceRange, nSteps=nSteps, ref_params=ref_params);

   ntSamps = stim_dur*tSamp;
   resp_time = np.zeros((ntSamps, ));
   for ts in range(ntSamps):
      wave_curr = 0;
      for (i,sf_curr), con, tf, phi in zip(enumerate(stim_sfs), stim_cons, stim_tfs, stim_phi):
         # --- now, progress the stimulus
         phi_step = 2*np.pi*tf*ts/tSamp;
         wave_curr += con*np.sin(2*np.pi*sf_curr*xSamps + phi + phi_step);
      scalar = 2*spaceRange/nSteps; # scalar that accounts for the n samples in space
      # ^ why 2*spaceRange? That's the full range; then divide by nSteps
      # -- this gives the step size in space between adjacent samples
      resp_time[ts] = np.multiply(scalar, np.dot(wave_curr, space_rf));

   resp_thresh = 0.5*np.pi*np.maximum(resp_time, 0); # why multiply by pi/2? that's the scale offset from half-wave rectification
   if debug:
      return np.mean(resp_thresh), resp_time;
   else:
      return np.mean(resp_thresh);

def parker_hawken_all_stim(trialInf, expInd, curr_params, spaceRange=0.5, nSteps=100, tSamp=400, debug=False, comm_s_calc=False):
   ''' wrapper to call parker_hawken_space_from_stim for all trials of a given cell
       we assume:
       - curr_params is [nDisp, nCons, params] --> we'll draw from single grating fits to predict all stimuli
       --- more specifically, we'll get the filter which corresponds to the highest contrast grating present, and compute with that
   '''
   np = numpy;
   try:
      nTrials = len(trialInf['num']);
   except:
      nTrials = len(trialInf['con'][0]);
   nStimComp = get_exp_params(expInd).nStimComp;
   stimDur = get_exp_params(expInd).stimDur;
   _, stimVals, val_con_by_disp, _, _ = tabulate_responses(trialInf, expInd);
   all_cons = stimVals[1]; # stimVals is [disp, con, sf]

   rsps = np.nan * np.zeros((nTrials, ));
   if debug:
      rsps_time = [];

   ref_params = curr_params[0, -1, :] if comm_s_calc else None;

   for tr in range(nTrials):
      #stimOr = numpy.empty((nStimComp,));
      stimTf = numpy.empty((nStimComp,));
      stimCo = numpy.empty((nStimComp,));
      stimPh = numpy.empty((nStimComp,));
      stimSf = numpy.empty((nStimComp,));
      
      for iC in range(nStimComp):
         #stimOr[iC] = trialInf['ori'][iC][tr] * numpy.pi/180; # in radians
         stimTf[iC] = trialInf['tf'][iC][tr];          # in cycles per second
         stimCo[iC] = trialInf['con'][iC][tr];         # in Michelson contrast
         stimPh[iC] = trialInf['ph'][iC][tr] * numpy.pi/180;  # in radians
         stimSf[iC] = trialInf['sf'][iC][tr];          # in cycles per degree
         
      conInd = np.argmin(np.square(all_cons[val_con_by_disp[0]] - np.max(stimCo)));
      curr_pms = curr_params[0, val_con_by_disp[0][conInd], :];
      # once we've chosen the right filter, however, we just want the contrasts relative to the highest present (i.e. normalized)
      stimCoNorm = np.divide(stimCo, np.max(stimCo));
      if debug:
         rsps[tr], rsp_time = parker_hawken_space_from_stim(stimSf, stimCoNorm, stimTf, stimPh, stimDur, curr_pms, ref_params, spaceRange, nSteps, tSamp, debug);
         rsps_time.append(rsp_time);
      else:
         rsps[tr] = parker_hawken_space_from_stim(stimSf, stimCoNorm, stimTf, stimPh, stimDur, curr_pms, ref_params, spaceRange, nSteps, tSamp, debug);

   if debug:
      return rsps, rsps_time;
   else:
      return rsps;

def get_descrResp(params, stim_sf, DoGmodel, minThresh=0.1, baseline=0, fracSig=1, sach_equiv_p_h=True, ref_params=None, ref_rc_val=None):
  ''' returns only pred_spikes; 0 is flexGauss.; 1 is DoG sach; 2 is DoG (tony)
      --- baseline is a non-optimized for additive constant that we can optionally use for diff. of gauss fits
      --- i.e. if we use it, we're simply adding the baseline response to the data, so the model fit is on top of that
      --- sach_equiv_p_h==True means that we use the version of the Sach DoG model in which the scalars are equivalent to those in the parker-hawken model
      --- ref_params --> used for d-DoG-S, only
      --- ref_rc, ref_rc_val --> used for Sach DoG only
  '''
  if DoGmodel == 0:
    pred_spikes = flexible_Gauss_np(params, stim_sf=stim_sf, minThresh=minThresh, fracSig=fracSig);
  elif DoGmodel == 1:
    pred_spikes, _ = DoGsach(*params, stim_sf=stim_sf, baseline=baseline, parker_hawken_equiv=sach_equiv_p_h, ref_rc_val=None);
  elif DoGmodel == 2:
    pred_spikes, _ = DiffOfGauss(*params, stim_sf=stim_sf, baseline=baseline);
  elif DoGmodel == 3:
    pred_spikes = parker_hawken(params, stim_sf, isMult=True, ref_params=ref_params, baseline=baseline);
  elif DoGmodel == 4:
    pred_spikes, _ = DoGsachVol(*params, stim_sf=stim_sf, baseline=baseline);
  elif DoGmodel == 5: # if isMult is False, then this is the Hawken parameterization of the d-DoG-S model
    pred_spikes = parker_hawken(params, stim_sf, isMult=False, baseline=baseline);
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

def jl_perCell(cell_ind, dataList, descrFits, dogFits, rvcFits, expDir, data_loc, dL_nm, fLW_nm, fLF_nm, dF_nm, dog_nm, rv_nm, superAnalysis=None, conDig=1, sf_range=[0.1, 10], rawInd=0, muLoc=2, varExplThresh=75, dog_varExplThresh=60, descrMod=0, dogMod=1, isSach=0, isBB=0, rvcMod=1, bootThresh=0.25, oldVersion=False, jointType=0, reducedSave=False):

   ''' - bootThresh (fraction of time for which a boot metric must be defined in order to be included in analysis)
   '''

   np = numpy;
   print('%s/%d' % (expDir, 1+cell_ind));

   arrtype = np.float32; # float32, rather than float64

   ###########
   ### meta parameters      
   ###########
   # get experiment name, load cell
   if isSach or isBB: # if it's Sach or Bauman+Bonds
     if isSach:
       expName = dataList[cell_ind]['cellName'];
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
       sys.path.append(os.path.dirname('LGN/sach/'))# fixed relative location to this file, always
       from LGN.sach.helper_fcns_sach import tabulateResponses as sachTabulate
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

   if expName[0] == 'm':
      mInd = int(re_findall('\d+', expName)[0]);
   else:
      mInd = '';
   meta = dict([('fullPath', data_loc),
               ('cellNum', cell_ind+1),
               ('dataList', dL_nm),
               ('fitListWght', fLW_nm),
               ('fitListFlat', fLF_nm),
               ('descrFits', dF_nm),
               ('descrMod', descrMod), 
               ('dogFits', dog_nm),
               ('dogMod', dogMod), 
               ('jointType', jointType), 
               ('rvcFits', rv_nm),
               ('expName', expName),
               ('mInd', mInd),
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
       basics_list = get_basic_tunings(basic_names, basic_order, reducedSave=reducedSave);
     except:
       try:
         # we've already put the basics in the data structure... (i.e. post-sorting 2021 data)
         basic_names = ['','','','',''];
         basic_order = ['rf', 'sf', 'tf', 'rvc', 'ori']; # order doesn't matter if they are already loaded
         basics_list = get_basic_tunings(basic_names, basic_order, preProc=cell, reducedSave=reducedSave)
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
   # set up the arrays (as dictionary entries) we need to store analyses
   ####
   # --- first, what are the metrics
   metrs = ['sfVar', 'sfCom', 'sfComCut', # model-free metrics
            'bw_sigma', 'lsfv', 'bwHalf', 'bw34', # bandwidth
            'pSf', 'sf70', 'dog_charFreq', 'sfE', # sf
            'arExpl', # variance explained (see below)
            'conGain', 'c50', 'c50_emp', 'c50_eval', 'c50_varExpl', # RVC-based
            'dog_mech',
            'bwHalf_split', 'bw34_split',
            # then the difference metrics
            'bwHalfDiffs', 'bw34Diffs', 'pSfRats', 'pSfModRat', 'sf70Rats', 'dog_charFreqRats', 'sfComRats', 'sfVarDiffs'];
   # ---then, which models (i.e. prefixes) apply to the above metrics
   mods_pfx = ['', 'dog_']; # either blank (i.e. just descr. fit, should be flex. Gaus) or DoG model
   no_pfx = ['']; # don't apply any other suffix (e.g. model-free)
   # --- corresponding to each metr, which models apply (the default is mods, as above, e.g flex gauss and DoG)
   metrs_which_mod = [no_pfx, no_pfx, no_pfx,
                      no_pfx, no_pfx, mods_pfx, mods_pfx,
                      mods_pfx, mods_pfx, no_pfx, mods_pfx,
                      ['sfV', 'dog_v'], # note the oddities
                      no_pfx, no_pfx, no_pfx, no_pfx, no_pfx,
                      no_pfx,
                      mods_pfx, mods_pfx,
                      mods_pfx, mods_pfx, mods_pfx, mods_pfx, mods_pfx, no_pfx, no_pfx, no_pfx];
   # --- if we do boot metrics for the above, what will they be?
   boot_metrs = ['_mn', '_md', '_std', '_stdLog'];
   # --- now, corresponding to each metr, ask whether we compute/save bootstrap metrics
   metrs_has_boot = [None, boot_metrs, None, 
                     None, None, boot_metrs, boot_metrs,
                     boot_metrs, boot_metrs, boot_metrs, None,
                     None,
                     None, None, None, None, None,
                     boot_metrs,
                     None, None,
                     None, None, None, None, None, None, None, None];
   # --- what sizes for the arrays?
   sz_typical = (nDisps, nCons); # what's the typical size for most metrics?
   sz_splits = (nDisps, nCons, 2); # e.g. split bandwidth for below/above sfPref
   sz_mech = (nDisps, nCons, 6); # mech for mechanism, for the DoG parameters (e.g. center radius, volumes)
   sz_diffs = (nDisps, nCons, nCons, 2); # for diffs/rats b/t vals @2cons; xtra dim len=2 is for raw/norm-to-con-change values
   sz_rvc = (nDisps, nSfs);
   metrs_size = [sz_typical, sz_typical, sz_typical,
                 sz_splits, sz_typical, sz_typical, sz_typical,
                 sz_typical, sz_typical, sz_typical, sz_typical,
                 sz_typical,
                 sz_rvc, sz_rvc, sz_rvc, sz_rvc, sz_rvc,
                 sz_mech,
                 sz_splits, sz_splits,
                 sz_diffs, sz_diffs, sz_diffs, sz_diffs, sz_diffs, sz_diffs, sz_diffs, sz_diffs];

   # --- set up the empty dictionary
   dataMetrics = dict();
   
   # --- then fill for the above in a programmatc fashion
   for name, mods, size, boots in zip(metrs, metrs_which_mod, metrs_size, metrs_has_boot):
      for mod in mods:
         #if reducedSave and mod=='':
         #   continue; # i.e. don't bother with non-DoG for reduced save
         curr_key = '%s%s' % (mod, name);
         dataMetrics[curr_key] = np.nan * np.zeros(size);
         if boots is not None:
            for boot in boots:
               boot_key = 'boot_%s%s' % (curr_key, boot);
               dataMetrics[boot_key] = np.nan * np.zeros(size, dtype=arrtype);

   # bwHalf, bw34, pSf, sfVar, sfCom, sf70, dog_sf70, dog_charFreq,  dog_bwHalfDiffs, dog_bw34Diffs, dog_pSfRats
   # -- evaluated from data at 1:.33 contrast (only for single gratings)
   diffsAtThirdCon = np.zeros((nDisps, 11, ), dtype=arrtype) * np.nan;

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
   # --- store the f1f0_ratio
   dataMetrics['f1f0_ratio'] = f1f0_ratio;
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
            if 'boot' in key:
               # then we expand on axis=1, since axis=0 is the boots
               currDict[key] = np.expand_dims(currDict[key], axis=1);
            else:
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
        rvcFits[cell_ind] = expand_sach(rvcFits[cell_ind][respKey]['mask']);
        descrFits[cell_ind] = expand_sach(descrFits[cell_ind][respKey]['mask']);
      except:
        print('****one of DoG, RVC, descr expand failed: sfBB, cell ind %d****' % cell_ind);

   # the last values to set-up -- done here so that we properly access sfBB
   try:
     nBoots_descr = descrFits[cell_ind]['boot_params'].shape[0]; # this is where we find out how many boot iters are present
   except:
     nBoots_descr = 1;
   try:
     nBoots_dog = dogFits[cell_ind]['boot_params'].shape[0]; # this is where we find out how many boot iters are present
   except:
     nBoots_dog = 1;
   if nBoots_descr == 1:
      try:
         nBoots_descr = descrFits[cell_ind]['boot_params'].shape[1]; # HORRIBLE HACK FOR SF_BB -- fix
      except:
         pass;
   if nBoots_dog == 1:
      try:
         nBoots_dog = dogFits[cell_ind]['boot_params'].shape[1]; # HORRIBLE HACK FOR SF_BB -- fix
      except:
         pass;
   # storing bootstrap values for computing, analysis later on [initialize here]
   boots_size_descr = (nDisps, nCons, nBoots_descr);
   boots_size_dog = (nDisps, nCons, nBoots_dog);
   for mod, sz in zip(['', 'dog_'], [boots_size_descr, boots_size_dog]):
      for key in ['sf70', 'bwHalf', 'bw34', 'pSf']:
         curr_key = 'boot_%s%s_values' % (mod, key);
         dataMetrics[curr_key] = np.zeros(sz, dtype=arrtype) * np.nan;
   # and the "one-off"s, since they are different
   dataMetrics['boot_dog_charFreq_values'] = np.zeros(boots_size_dog, dtype=arrtype) * np.nan; # gain/radius/volume for center, then surround
   dataMetrics['boot_dog_mech'] = np.zeros((nDisps, nCons, 6, nBoots_dog), dtype=arrtype) * np.nan; # gain/radius/volume for center, then surround
   dataMetrics['c50Rats'] = np.nan * np.zeros((nDisps, nSfs, nSfs), dtype=arrtype);

   for d in range(nDisps):
     #######
     ## spatial frequency stuff
     #######
     start_incl = False; # Once the lowest contrast value has passed the thresold, we start including
     start_incl_dog = False;
     for c in range(nCons):

       # zeroth...model-free metrics
       if isSach:
         curr_sfInd = np.arange(0, len(stimVals[2])); # all SFS are valid for Sach
       elif isBB:
         curr_sfInd = np.arange(0, len(stimVals[2])); # all SFS are valid for sfBB
       else:
         curr_sfInd = get_valid_sfs(tr, d, c, expInd=expInd, stimVals=stimVals, validByStimVal=validByStimVal)
       curr_sfs   = stimVals[2][curr_sfInd];
       curr_resps = sfTuning[d, curr_sfInd, c];
       sf_gt0 = np.where(curr_sfs>0)[0]; # if we include a zero-SF condition, then everything goes to zero!

       dataMetrics['sfCom'][d, c] = sf_com(curr_resps[sf_gt0], curr_sfs[sf_gt0])
       dataMetrics['sfVar'][d, c] = sf_var(curr_resps[sf_gt0], curr_sfs[sf_gt0], dataMetrics['sfCom'][d, c]);
       # get the c.o.m. based on the restricted set of SFs, only
       if cut_sf is not None:
          cut_sfs, cut_resps = np.array(stimVals[2])[cut_sf], sfTuning[d, cut_sf, c];
          dataMetrics['sfComCut'][d, c] = sf_com(cut_resps, cut_sfs)

       # first, DoG fit
       if cell_ind in dogFits:
         try:
           varExpl = dogFits[cell_ind]['varExpl'][d, c];
           # NOTE: We don't have real threshold for inclusion IF the fits are joint
           # --- why? As of 21.12.04, the joint fits use Sach's approach, which is to only include in the joint fits
           # ----- the contrasts starting with the lowest which has a varExpl greater than the threshold (typically 60%)
           #thresh_to_use = -np.inf if jointType > 0 else dog_varExplThresh;
           ### TEMP: Switch back to keeping threshold
           thresh_to_use = dog_varExplThresh;
           if not start_incl_dog: # check to see if we can start ubckydubg
              if varExpl > thresh_to_use:
                 start_incl_dog = True;
           if start_incl_dog:
             dataMetrics['dog_varExpl'][d,c] = varExpl;
             # on data
             dataMetrics['dog_pSf'][d, c] = dogFits[cell_ind]['prefSf'][d, c]
             dataMetrics['dog_charFreq'][d, c] = dogFits[cell_ind]['charFreq'][d, c]

             # get the params and do bandwidth, high-freq. cut-off measures
             dog_params_curr = dogFits[cell_ind]['params'][d, c];
             for ky,height in zip(['dog_bwHalf', 'dog_bw34'], [0.5, 0.75]):
                dataMetrics[ky][d, c] = compute_SF_BW(dog_params_curr, height, sf_range=sf_range, sfMod=dogMod, baseline=baseline_resp)[1];
             for ky,height in zip(['dog_bwHalf_split', 'dog_bw34_split'], [0.5, 0.75]):
                for splitInd,splitHalf in enumerate([-1,1]):
                   dataMetrics[ky][d, c, splitInd] = compute_SF_BW(dog_params_curr, height=height, sf_range=sf_range, which_half=splitHalf, sfMod=dogMod, baseline=baseline_resp)[1];
             # -- note that for sf_highCut with the Diff. of Gauss models, we do NOT need to subtract the baseline
             # -- why? becaue the descr. model is already fit on top of the baseline (i.e. the descr. fit response does not include baseline)
             for ky,frac in zip(['dog_sf70', 'dog_sfE'], [0.7, eFrac]):
                dataMetrics[ky][d, c] = sf_highCut(dog_params_curr, sfMod=dogMod, frac=frac, sfRange=(0.1, 15));
             # Also get spatial params, i.e. center gain, radius, volume; surround gain, radius, volume
             if not is_mod_DoG(dogMod): # i.e. it's a d-DoG-S
               dataMetrics['dog_mech'][d, c] = [dog_get_param(dog_params_curr[0:4], 1, x, con_val=cons[c]) for x in ['gc', 'rc', 'vc', 'gs', 'rs', 'vs']]; #dogMod 1, since they're param. like Sach
             else:
               dataMetrics['dog_mech'][d, c] = [dog_get_param(dog_params_curr, dogMod, x, con_val=cons[c]) for x in ['gc', 'rc', 'vc', 'gs', 'rs', 'vs']];

             # Now, bootstrap estimates
             try:
               # -- now, get the distribution of bootstrap estimates for sf70, pSf, bw_sigma, and bwHalf/bw34
               # temp. hack??!! fix code in descr_fits_sfBB [TODO]
               boot_prms = dogFits[cell_ind]['boot_params']
               if boot_prms.shape[0] == 1:
                 boot_prms = np.transpose(boot_prms, axes=(1,0,2,3)); # i.e. flip 1st and 2nd axes
               # ---- zeroth, mechanism
               if not is_mod_DoG(dogMod): # i.e. it's a d-DoG-S
                  try:
                     dataMetrics['boot_dog_mech'][d, c] = np.vstack([dog_get_param(np.transpose(boot_prms[:, d, c, 0:4]), 1, x, con_val=cons[c]) for x in ['gc', 'rc', 'vc', 'gs', 'rs', 'vs']]); #dogMod 1, since they're param. like Sach
                  except:
                     pass;
               else:
                  dataMetrics['boot_dog_mech'][d, c] = np.vstack([dog_get_param(np.transpose(boot_prms[:, d, c, :]), dogMod, x, con_val=cons[c]) for x in ['gc', 'rc', 'vc', 'gs', 'rs', 'vs']]);
               # then, in order: sf70, pSf, charFreq
               metrs = ['pSf', 'charFreq', 'sf70', 'bwHalf', 'bw34'];
               compute = ['np.nanmedian(', 'np.nanmean(', 'np.nanstd(', 'np.nanstd(np.log10(']
               for metr in metrs:
                  # get the key, values
                  curr_key = 'boot_dog_%s_values' % metr;
                  if 'sf70' in metr:
                     dataMetrics[curr_key][d,c] = [sf_highCut(boot_prms[boot_i, d, c, :], sfMod=dogMod, frac=0.7, sfRange=(0.1, 15), baseline_sub=baseline_resp) for boot_i in range(boot_prms.shape[0])];
                     #dataMetrics[curr_key][d,c] = np.array([sf_highCut(boot_prms[boot_i, d, c, :], sfMod=dogMod, frac=0.7, sfRange=(0.1, 15), baseline_sub=baseline_resp) for boot_i in range(boot_prms.shape[0])]);
                  elif 'pSf' in metr or 'charFreq' in metr:
                     # --- must manually specify for pSf, since we call it "pSf" but fits have "prefSf"
                     dataMetrics[curr_key][d,c] = dogFits[cell_ind]['boot_%s' % metr][:, d, c] if 'charFreq' in metr else dogFits[cell_ind]['boot_prefSf'][:, d, c]
                  elif 'bw' in metr:
                     height = 0.5 if 'Half' in metr else 0.75; # assumes it's either half or 3/4th height
                     dataMetrics[curr_key][d,c] = np.array([compute_SF_BW(boot_prms[boot_i, d, c, :], height=height, sf_range=sf_range, sfMod=dogMod)[1] for boot_i in range(boot_prms.shape[0])]);
                  # now compute the metrics
                  for boot_metr,comp in zip(boot_metrs, compute):
                     if 'bw' in metr and 'stdLog' in boot_metr:
                        continue; # don't compute log for bandwidth stuff
                     boot_key = 'boot_dog_%s%s' % (metr, boot_metr);
                     try: # need the if/else to handle the case where two parantheses need closing
                        dataMetrics[boot_key][d,c] = eval('%sdataMetrics[curr_key][d,c]))' % comp) if 'stdLog' in boot_metr else eval('%sdataMetrics[curr_key][d,c])' % comp);
                     except Exception as e:
                        print('----jl_perCell error: %s' % e);
                        pass
            
             except Exception as e:
               print('----jl_perCell error [%s/%02d]: %s' % (expDir, cell_ind+1, e));
               pass
                  
         except: # then this dispersion does not have that contrast value, but it's ok - we already have nan
           pass 

       # then, non-DoG descr fit
       if cell_ind in descrFits:
         try:
           varExpl = descrFits[cell_ind]['varExpl'][d, c];
           thresh_to_use = varExplThresh;
           if not start_incl: # check to see if we can start ubckydubg
              if varExpl > thresh_to_use:
                 start_incl = True;
           if start_incl:
             dataMetrics['sfVarExpl'][d, c] = varExpl;
             # on data
             dataMetrics['lsfv'][d, c] = compute_LSFV(descrFits[cell_ind]['params'][d, c, :]);
             params_curr = descrFits[cell_ind]['params'][d, c, :];
             for ky,height in zip(['bwHalf', 'bw34'], [0.5, 0.75]):
                dataMetrics[ky][d, c] = compute_SF_BW(params_curr, height, sf_range=sf_range, sfMod=descrMod, baseline=baseline_resp)[1];
             for splitInd,splitHalf in enumerate([-1,1]):
                dataMetrics['bw_sigma'][d, c, splitInd] = params_curr[3+splitInd]
                for ky,height in zip(['bwHalf_split', 'bw34_split'], [0.5, 0.75]):
                   dataMetrics[ky][d, c, splitInd] = compute_SF_BW(params_curr, height=height, sf_range=sf_range, which_half=splitHalf, sfMod=descrMod)[1];
             curr_params = descrFits[cell_ind]['params'][d, c, :];
             dataMetrics['pSf'][d, c] = curr_params[muLoc]
             for ky,frac in zip(['sf70', 'sfE'], [0.7, eFrac]):
                dataMetrics[ky][d, c] = sf_highCut(params_curr, sfMod=descrMod, frac=frac, sfRange=(0.1, 15));

             try:
               # -- now, get the distribution of bootstrap estimates for sf70, pSf, bw_sigma, and bwHalf/bw34
               # TEMPORARY HACK!!! fix code in descr_fits_sfBB [TODO]
               boot_prms = descrFits[cell_ind]['boot_params']
               if boot_prms.shape[0] == 1:
                 boot_prms = np.transpose(boot_prms, axes=(1,0,2,3)); # i.e. flip 1st and 2nd axes

               # then, in order: sf70, pSf, charFreq
               metrs = ['sf70', 'pSf', 'bwHalf', 'bw34'];
               compute = ['np.nanmedian(', 'np.nanmean(', 'np.nanstd(', 'np.nanstd(np.log10(']
               for metr in metrs:
                  # get the key, values
                  curr_key = 'boot_%s_values' % metr;
                  if 'sf70' in metr:
                     dataMetrics[curr_key][d,c] = [sf_highCut(boot_prms[boot_i, d, c, :], sfMod=descrMod, frac=0.7, sfRange=(0.1, 15), baseline_sub=baseline_resp) for boot_i in range(boot_prms.shape[0])];
                  elif 'pSf' in metr:
                     dataMetrics[curr_key][d,c] = descrFits[cell_ind]['boot_prefSf'][:, d, c]
                     # must manually specify, since the key is ...prefSf in the descrFits, but pSf here
                  elif 'bw' in metr:
                     height = 0.5 if 'Half' in metr else 0.75; # assumes it's either half or 3/4th height
                     dataMetrics[curr_key][d,c] = np.array([compute_SF_BW(boot_prms[boot_i, d, c, :], height=height, sf_range=sf_range, sfMod=descrMod)[1] for boot_i in range(boot_prms.shape[0])]);
                  # now compute the metrics, IF we have enough valid boot iters
                  n_nonNan = np.sum(~np.isnan(dataMetrics[curr_key][d,c]));
                  if n_nonNan < bootThresh*boot_prms.shape[0]:
                     continue; # i.e. don't do this if we have too few parameters

                  for boot_metr,comp in zip(boot_metrs, compute):
                     if 'bw' in metr and 'stdLog' in boot_metr:
                        continue; # don't compute log for bandwidth stuff
                     boot_key = 'boot_%s%s' % (metr, boot_metr);
                     try: # need the if/else to handle the case where two parantheses need closing
                        dataMetrics[boot_key][d,c] = eval('%sdataMetrics[curr_key][d,c]))' % comp) if 'stdLog' in boot_metr else eval('%sdataMetrics[curr_key][d,c])' % comp);
                     except:
                        pass
             except:
               pass


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
         dataMetrics['pSfModRat'][d] = [np.log2(evalRatio), np.log2(evalRatio)/logConRat];
     except: # then likely, no rvc/descr fits...
       pass
     # and likewise for DoG
     try:
       _, psf_model, opt_params = dog_prefSfMod(dogFits[cell_ind], allCons=cons, disp=d, varThresh=varExplThresh, dog_model=dogMod)
       valInds = np.where(dogFits[cell_ind]['varExpl'][d, :] > thresh_to_use)[0];
       if len(valInds) > 1:
         extrema = [cons[valInds[0]], cons[valInds[-1]]];
         logConRat = np.log2(extrema[1]/extrema[0]);
         evalPsf = psf_model(*opt_params, con=extrema);
         evalRatio = evalPsf[1]/evalPsf[0];
         dataMetrics['dog_pSfModRat'][d] = [np.log2(evalRatio), np.log2(evalRatio)/logConRat];
     except: # then likely, no rvc/descr fits...
       pass
     # NEW 22.06.02: get the slope if we have a slope model...
     if jointType>=7: # 7,8, and 9 joints (whether DoG or d-DoG-S) have the first two parameters specifying the slope of center radius
        prms = dogFits[cell_ind]['paramList'][0];
        dataMetrics['dog_mod_slope'] = prms[1]; # not negated here...
        dataMetrics['dog_mod_intercept'] = prms[0];
        prms = dogFits[cell_ind]['boot_paramList'];
        dataMetrics['boot_dog_mod_slope'] = np.array([x[0][1] for x in prms]); # not negated here...
        dataMetrics['boot_dog_mod_intercept'] = np.array([x[0][0] for x in prms]);
        
     #######
     ## RVC stuff
     #######
     for s in range(nSfs):
       if cell_ind in rvcFits:
         # on data
         try: # if from fit_RVC_F0
             dataMetrics['conGain'][d, s] = rvcFits[cell_ind]['conGain'][d,s];
             dataMetrics['c50'][d, s] = get_c50(rvcMod, rvcFits[cell_ind]['params'][d, s, :]);
             dataMetrics['c50_emp'][d, s], c50_eval[d, s] = c50_empirical(rvcMod, rvcFits[cell_ind]['params'][d, s, :]);
             dataMetrics['c50_varExpl'][d, s] = rvcFits[cell_ind]['varExpl'][d,s];
         except: # might just be arranged differently...(not fit_rvc_f0)
             try: # TODO: investigate why c50 param is saving for nan fits in hf.fit_rvc...
               if ~np.isnan(rvcFits[cell_ind][d]['loss'][s]): # only add it if it's a non-NaN loss value...
                 dataMetrics['conGain'][d, s] = rvcFits[cell_ind][d]['conGain'][s];
                 dataMetrics['c50'][d, s] = get_c50(rvcMod, rvcFits[cell_ind][d]['params'][s]);
                 dataMetrics['c50_emp'][d, s], c50_eval[d, s] = c50_empirical(rvcMod, rvcFits[cell_ind][d]['params'][s]);
                 dataMetrics['c50_varExpl'][d, s] = rvcFits[cell_ind][d]['varExpl'][s];
             except: # then this dispersion does not have that SF value, but it's ok - we already have nan
               pass;

     ## Now, after going through all cons/sfs, compute ratios/differences
     # first, with contrast
     #if oldVersion: # only do these things if we want the "old" jointList --- currently (21.09) don't use them and they take up extra space in the structure
       for comb in itertools.combinations(range(nCons), 2):
         # first, in raw values [0] and per log2 contrast change [1] (i.e. log2(highCon/lowCon))
         conChange = np.log2(cons[comb[1]]/cons[comb[0]]);

         # NOTE: For pSf, we will log2 the ratio, such that a ratio of 0 
         # reflects the prefSf remaining constant (i.e. log2(1/1)-->0)
         outsAll = ['bwHalfDiffs', 'bw34Diffs', 'pSfRats', 'sf70Rats', 'dog_charFreqRats', 'sfVarDiffs', 'sfComRats']
         modsAll = [mods_pfx, mods_pfx, mods_pfx, mods_pfx, no_pfx, no_pfx, no_pfx];
         insAll = ['bwHalf', 'bw34', 'pSf', 'sf70', 'dog_charFreq', 'sfVar', 'sfCom'];
         for outKey, currMods, inKey in zip(outsAll, modsAll, insAll):
            for currMod in currMods:
              curr_in = '%s%s' % (currMod, inKey);
              curr_out = '%s%s' % (currMod, outKey);
              if 'Rat' in outKey:
                 val = np.log2(dataMetrics[curr_in][d,comb[1]] / dataMetrics[curr_in][d,comb[0]]);
              elif 'Diff' in outKey:
                 val = dataMetrics[curr_in][d,comb[1]] - dataMetrics[curr_in][d,comb[0]];
              dataMetrics[curr_out][d,comb[0], comb[1]] = [val, val/conChange];

       # then, as function of SF
       for comb in itertools.permutations(range(nSfs), 2):
         dataMetrics['c50Rats'][d,comb[0],comb[1]] = dataMetrics['c50'][d,comb[1]] / dataMetrics['c50'][d,comb[0]]

   # Now, we make sure that everything we need is in dataMetrics
   # --- as of 21.10.18, mostly everything is added directly above; however, we add the following here
   dataMetrics['relDescr_inds'] = relDescr_inds;
   dataMetrics['mn_med_max'] = mn_med_max;
   dataMetrics['diffsAtThirdCon'] = diffsAtThirdCon;

   if oldVersion:
     dataMetrics['diffsAtThirdCon'] = diffsAtThirdCon;
     dataMetrics['bwHalfDiffs'] =bwHalfDiffs;
     #dataMetrics['bwHalfDiffs_split'] = bwHalfDiffs_split;
     dataMetrics['bw34Diffs'] = bw34Diffs;
     #dataMetrics['bw34Diffs_split'] = bw34Diffs_split;
     dataMetrics['lsfvRats'] = lsfvRats;
     dataMetrics['pSfRats'] = pSfRats;
     dataMetrics['pSfModRat'] = pSfModRat;
     dataMetrics['dog_pSfRats'] = dog_pSfRats;
     dataMetrics['dog_pSfModRat'] = dog_pSfModRat;
     dataMetrics['sf70Rats'] = sf70Rats;
     dataMetrics['sf70ModRat'] = sf70ModRat;
     dataMetrics['boot_sf70_mnRats'] = boot_sf70_mnRats;
     dataMetrics['boot_sf70_mdRats'] = boot_sf70_mdRats;
     dataMetrics['boot_sf70_stdRats'] = boot_sf70_stdRats;
     dataMetrics['dog_sf70Rats'] = dog_sf70Rats;
     dataMetrics['dog_sf70ModRat'] = dog_sf70ModRat;
     dataMetrics['sf75Rats'] = sf75Rats;
     dataMetrics['sf75ModRat'] = sf75ModRat;
     dataMetrics['dog_sf75Rats'] = dog_sf75Rats;
     dataMetrics['dog_sf75ModRat'] = dog_sf75ModRat;
     dataMetrics['sfERats'] = sfERats;
     dataMetrics['sfEModRat'] = sfEModRat;
     dataMetrics['dog_sfERats'] = dog_sfERats;
     dataMetrics['dog_sfEModRat'] = dog_sfEModRat;
     dataMetrics['sfVarDiffs'] = sfVarDiffs;
     dataMetrics['sfComRats'] = sfComRats;
     #dataMetrics['sfComLinRats'] = sfComLinRats;
     dataMetrics['c50Rats'] = c50Rats;

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
              conDig=1, sf_range=[0.1, 10], rawInd=0, muLoc=2, varExplThresh=75, dog_varExplThresh=60, descrMod=0, dogMod=1, toPar=1, jointType=0, reducedSave=False):
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

    #if expDir == 'LGN/':
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
      perCell_summary = partial(jl_perCell, dataList=dataList, descrFits=descrFits, dogFits=dogFits, rvcFits=rvcFits, expDir=expDir, data_loc=data_loc, dL_nm=dL_nm, fLW_nm=fLW_nm, fLF_nm=fLF_nm, dF_nm=dF_nm, dog_nm=dog_nm, rv_nm=rv_nm, superAnalysis=superAnalysis, conDig=conDig, sf_range=sf_range, rawInd=rawInd, muLoc=muLoc, varExplThresh=varExplThresh, dog_varExplThresh=dog_varExplThresh, descrMod=descrMod, dogMod=dogMod, isSach=isSach, rvcMod=rvcMod, isBB=isBB, jointType=jointType, reducedSave=reducedSave)

      #if isBB:
      #  oh = perCell_summary(30);
      #  pdb.set_trace();

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

def jl_get_metric_byCon(jointList, metric, conVal, disp, conTol=0.02, valIsFrac=False):
  ''' given a "jointList" structure, get the specified metric (as string) for a given conVal & dispersion
      returns: array of metric value for a given con X disp
      inputs:
        jointList - see above (jl_create)
        metric    - as a string, which metric are you querying (e.g. 'pSf', 'sfCom', etc)
        conVal    - what contrast (e.g. 33% or 99% or ...)
        disp      - which dispersion (0, 1, ...)
        [conTol]  - we consider the contrast to match conVal if within +/- 2% (given experiment, this is satisfactory to match con level across dispersions, versions)
        valIsFrac - If true, then the conVal is fraction of the maximum contrast value
  '''
  np = numpy;
  nCells = len(jointList);
  output = np.nan * np.zeros((nCells, ));
  matchCon = np.nan * np.zeros((nCells, ));

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
      max_con = np.max(curr_conVals);
      if valIsFrac:
        match_ind = np.where(np.abs(curr_conVals-(conVal*max_con))<=conTol)[0];
      else:
        match_ind = np.where(np.abs(curr_conVals-conVal)<=conTol)[0];

      if np.array_equal(match_ind, []):
        # e.g. if we want full contrast, but the program only goes up to 64% -- that's ok! (64% vs. 100% is minimal)
        if conVal >= 0.6: # i.e. we want large/full contrast
          if np.max(curr_conVals) > 0.35:
            match_ind = [np.argmax(curr_conVals)];
          else:
            continue; # i.e. didn't find the right contrast match

      try:
        full_con_ind = curr_byDisp[disp][match_ind[0]];
        output[ind] = curr_metr[disp][full_con_ind];
        matchCon[ind] = curr_conVals[match_ind];
      except:
        pass; # sometimes, there isn't a match...

  return output, matchCon;

def jl_get_metric_highComp(jointList, metric, whichMod, atLowest, disp=0, extraInds=None, returnBothCons=False):
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
    
    if returnBothCons:
      highCons = np.array([jointList[x]['metadata']['stimVals'][1][conInd] for x,conInd in zip(jointList.keys(), highInd)]);
      return highSf, compSf, highCons, compCons;
    else:
      return highSf, compSf, compCons

##################################################################
##################################################################
##################################################################
### IV. RETURN TO DESCRIPTIVE FITS/ANALYSES
##################################################################
##################################################################
##################################################################

def blankResp(cellStruct, expInd, spikes=None, spksAsRate=False, returnRates=False, resample=False):
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

    blank_tr = resample_array(resample, spikes[numpy.isnan(tr['con'][0])]);
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

def tabulate_responses(cellStruct, expInd, modResp = [], mask=None, overwriteSpikes=None, respsAsRates=False, modsAsRate=False, resample=False, cellNum=-1, cross_val=None):
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
        if resample; then for cross_val:
        -- if not None, then cross_val should be tuple with (fracInTest, startInd); i.e. what fraction of overall data is test, which index to start with?
        ---- Further note: if startInd is negative, then we'll just randomly sample WITHOUT replacement
        ----> why? for joint fits, better to randomly sample then take trials i through j for all conditions
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
                    
                if cross_val is None:
                   curr_resps = resample_array(resample, respToUse[valid_tr]); # will just be respToUse[valid_tr] if resample==0
                else:
                   if cross_val[1]>=0: # then take with start_ind
                      curr_resps = resample_array(resample, respToUse[valid_tr], holdout_frac=cross_val[0], start_ind=cross_val[1]);
                   else: # random resampling without replacement (just ignore start_ind argument)
                      curr_resps = resample_array(resample, respToUse[valid_tr], holdout_frac=cross_val[0]);

                respMean[d, sf, con] = np.mean(curr_resps/respDiv);
                respStd[d, sf, con] = np.std(curr_resps/respDiv);
                
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
                    
                    curr_resps = resample_array(resample, respToUse[val_tr]); # will just be the array if resample==0

                    curr_pred = curr_pred + np.mean(curr_resps/respDiv);
                    curr_var = curr_var + np.var(curr_resps/respDiv);
                    
                predMean[d, sf, con] = curr_pred;
                predStd[d, sf, con] = np.sqrt(curr_var);
                
                if mod: # if needed, convert spike counts in each trial to spike rate (spks/s)
                    nTrCurr = sum(valid_tr); # how many trials are we getting?
                    if modsAsRate == True: # i.e. we passed on the model repsonses as rates already!
                      divFactor = 1;
                    else: # default behavior
                      divFactor = stimDur;
 
                    try:
                      if cross_val is None:
                        curr_modResp = resample_array(resample, modResp[valid_tr]); # will just be the array if resample==0
                        nTrCurr = len(curr_modResp); # rewrite the nTrCurr, since it will be different if holdout_frac < 1
                        modRespOrg[d, sf, con, 0:nTrCurr] = np.divide(curr_modResp, divFactor);
                      else:
                        if cross_val[1]>=0: # then take with start_ind
                           curr_modResp = resample_array(resample, modResp[valid_tr], holdout_frac=cross_val[0], start_ind=cross_val[1]);
                           nTrCurr = len(curr_modResp); # rewrite the nTrCurr, since it will be different if holdout_frac < 1
                           modRespOrg[d, sf, con, 0:nTrCurr] = np.divide(curr_modResp, divFactor);
                        else:
                           curr_modResp, test_inds = resample_array(resample, modResp[valid_tr], holdout_frac=cross_val[0], return_inds=True);
                           modRespOrg[d, sf, con, test_inds] = np.divide(curr_modResp, divFactor);
                    except:
                      print('FAILED: uhoh [cell %d]: d/sf/con %d/%d/%d -- %d trial(s) out of %d/%d in the data/modResp' % (cellNum, d, sf, con, nTrCurr, len(val_tr), len(modResp)));
                      pass

            if np.any(~np.isnan(respMean[d, :, con])):
                if ~np.isnan(np.nanmean(respMean[d, :, con])):
                    val_con_by_disp[d].append(con);
                    
    return [respMean, respStd, predMean, predStd], [all_disps, all_cons, all_sfs], val_con_by_disp, [valid_disp, valid_con, valid_sf], modRespOrg;

def organize_adj_responses(data, rvcFits, expInd, vecF1=0):
  ''' Used as a wrapper to call the organize_adj_responses function for a given experiment
      BUT, also has organize_adj_responses for newer experiments (see "except" assoc. with main try block)
      We set the organize_adj_responses separately for each experiment since some versions don't have adjusted responses
        and for those that do, some aspects of the calculation may differ
  '''
  ### First, we'll see if there is a direct helper_fcns method for this
  dir = get_exp_params(expInd).dir;
  to_import = dir.replace('/', '.') + 'helper_fcns';
  if os.path.isfile(dir + 'helper_fcns'): # i.e. what if we don't have an associated helper_fcns? then do the (implicit) else below
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

def organize_phAdj_byMean(expStructure, expInd, all_opts, stimVals, val_con_by_disp, phAdv_model=None, resample=False, dir=1, redo_phAdv=True):
   ''' Organize the responses in the usual way (i.e. [d,sf,con])
         In the style of organize_resp, organize_adj_resp
   '''
   np = numpy;
   if phAdv_model is None:
      phAdv_model = get_phAdv_model();
      
   nDisps, nCons, nSfs = len(stimVals[0]), len(stimVals[1]), len(stimVals[2]);
   means = np.nan * np.zeros((nDisps, nSfs, nCons));

   for disp in range(nDisps):

      allAmp, allPhi, _, allCompCon, allCompSf = get_all_fft(expStructure, disp, expInd, dir=dir, all_trials=1, resample=resample);
      # get just the mean amp/phi and put into convenient lists
      allAmpMeans = [[x[0] for x in sf] for sf in allAmp]; # mean is in the first element; do that for each [mean, std] pair in each list (split by sf)
      allPhiMeans = [[x[0] for x in sf] for sf in allPhi]; # mean is in the first element; do that for each [mean, var] pair in each list (split by sf)

      if resample and redo_phAdv and disp==0:
         # then, we replicate (yes, bad...) a few lines from df.phaseAdvanceFit below
         # why? this means we will project based on fits done to this resampled data
         allCons = stimVals[1];
         conVals = allCons[val_con_by_disp[disp]];
         allCons = [conVals] * len(allAmp); # repeats list and nests
         # get list of mean amp, mean phi, std. mean, var phi
         # --- we can try to use this in fitting the phase-amplitude relationship...
         oyvey = [[polar_vec_mean([allAmp[i_sf][i_con][2]], [allPhi[i_sf][i_con][2]]) for i_con in range(len(allPhi[i_sf]))] for i_sf in range(len(allAmp))];
         phiVars = [[oyvey[x][y][3] for y in range(len(oyvey[x]))] for x in range(len(oyvey))];
         all_opts = phase_advance(allAmp, allPhi, allCons, tfs=None, ampSem=None, phiVar=phiVars, n_repeats=5)[1];

      adjMeans   = project_resp(allAmpMeans, allPhiMeans, phAdv_model, all_opts, disp, allCompSf, stimVals[2]); # list of lists [sf][con]
      # now, ready to add these adjMeans (again, phAmp corrected on the MEANS per condition)
      # --- but, if mixture, need to sum across conds
      if disp > 0: # then we need to sum component responses and get overall std measure (we'll fit to sum, not indiv. comp responses!)
        adjSumResp  = [np.sum(x, 1) if x else [] for x in adjMeans];
        # --- adjSemTr is [nSf x nValCon], i.e. s.e.m. per condition
        # TODO 22.06.04 --> WILL NEED TO ADD ADJBYTRIAL if uncommenting the below
        #adjSemTr    = [[sem(np.sum(hf.switch_inner_outer(x), 1)) for x in y] for y in adjByTrial];
        #adjSemCompTr  = [[sem(hf.switch_inner_outer(x)) for x in y] for y in adjByTrial];

      for sf,con in itertools.product(range(nSfs), range(len(val_con_by_disp[disp]))):
         con_ind = val_con_by_disp[disp][con];
         to_pass = adjMeans[sf][con].size >0 if disp==0 else len(allCompCon[sf])>0;
         if to_pass: # otherwise it's blank; already pre-populated with NaN
            means[disp,sf,con_ind] = adjMeans[sf][con] if disp==0 else adjSumResp[sf][con];

   return means;

def organize_resp(spikes, expStructure, expInd, mask=None, respsAsRate=False, resample=False, cellNum=-1, cross_val=None):
    ''' organizes the responses by condition given spikes, experiment structure, and expInd
        mask will be None OR list of trials to consider (i.e. trials not in mask/where mask is false are ignored)
        - respsAsRate: are "spikes" already in rates? if yes, pass in "True"; otherwise, we'll divide by stimDur to get rate
        - resample: if True, we'll resample the data to create a bootstrapped set of data
        -- NOTE: resample applies only to rateSfMix and allSfMix 
        -- if not None, then cross_val should be tuple with (fracInTest, startInd); i.e. what fraction of overall data is test, which index to start with?
        ---- Further note: if startInd is negative, then we'll just randomly sample WITHOUT replacement
        ------> why? for joint fits, better to randomly sample then take trials i through j for all conditions
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
      _, _, rateSfMix, allSfMix = v1_hf.organize_modResp(spikes, data, mask, resample=resample, cellNum=cellNum);
    else:
      # NOTE: we are getting the modRespOrg output of tabulate_responses, and ensuring the spikes are treated as rates (or raw counts) based on how they are passed in here
      allSfMix  = tabulate_responses(expStructure, expInd, spikes, mask, modsAsRate = respsAsRate, resample=resample, cellNum=cellNum, cross_val=cross_val)[4];
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
  # For tuned gain control model
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
  ''' Double Von Mises function as in Wang and Movshon, 2016
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
  modObj = lambda params: DoG_loss(params, tfResps, tfVals, loss_type=2, DoGmodel=1); # for sach
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

def get_basic_tunings(basicPaths, basicProgNames, forceSimple=None, preProc=None, reducedSave=False, forceOneChannel=True):
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
          sf = rbc.readSf11(curr_name, prog_curr, forceOneChannel=forceOneChannel);
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
        if not reducedSave:
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
          rv = rbc.readRVC(curr_name, prog_curr, forceOneChannel=forceOneChannel);
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
        if not reducedSave:
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
          tf = rbc.readTf11(curr_name, prog_curr, forceOneChannel=forceOneChannel);
   
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
        if not reducedSave:
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
          rf = rbc.readRFsize10(curr_name, prog_curr, forceOneChannel=forceOneChannel);
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
        rf_dict['params'] = opt_params;
        if not reducedSave:
           rf_dict['to_plot'] = to_plot; # this one takes up a lot of memory --> see sizeTune for details
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
          ori = rbc.readOri16(curr_name, prog_curr, forceOneChannel=forceOneChannel);
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
        if not reducedSave:
           ori_dict['ori_exp'] = ori;
        basic_outputs['ori'] = ori_dict
      
    except:
      basic_outputs[prog] = None;

  return basic_outputs;
