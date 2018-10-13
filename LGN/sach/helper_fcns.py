import numpy as np
import scipy.optimize as opt
import os
import random
from time import sleep
import pdb

# np_smart_load - loading that will account for parallelization issues - keep trying to load
# bw_lin_to_log 
# bw_log_to_lin

# DiffOfGauss - difference of gaussian models - formulation discussed with Tony
# DoGsach - difference of gaussian models - formulation discussed in Sach Sokol's NYU thesis
# var_explained - compute the variance explained given data/model fit
# dog_prefSf - compute the prefSF given a DoG model/fit
# dog_prefSfMod - smooth prefSf vs. contrast with a functional form/fit
# charFreq - given a model/parameter set, return the characteristic frequency of the tuning curve
# charFreqMod - smooth characteristic frequency vs. contrast with a functional form/fit

# deriv_gauss - evaluate a derivative of a gaussian, specifying the derivative order and peak
# compute_SF_BW - returns the log bandwidth for height H given a fit with parameters and height H (e.g. half-height)
# fix_params - Intended for parameters of flexible Gaussian, makes all parameters non-negative
# flexible_Gauss - Descriptive function used to describe/fit SF tuning
# blankResp - return mean/std of blank responses (i.e. baseline firing rate) for Sach's experiment
# tabulateResponses - Organizes measured and model responses for Sach's experiment
# random_in_range - random real number between a and b

# writeDataTxt - write [sf mean sem] for a given cell/contrast
# writeCellTxt - call writeDataTxt for all contrasts for a cell

def np_smart_load(file_path, encoding_str='latin1'):

   if not os.path.isfile(file_path):
     return [];
   loaded = [];
   while(True):
     try:
         loaded = np.load(file_path, encoding=encoding_str).item();
         print('loaded');
         break;
     except IOError: # this happens, I believe, because of parallelization when running on the cluster; cannot properly open file, so let's wait and then try again
         sleep(60); # i.e. wait for N seconds
         print('sleep');

   return loaded;

def bw_lin_to_log( lin_low, lin_high ):
    # Given the low/high sf in cpd, returns number of octaves separating the
    # two values

    return np.log2(lin_high/lin_low);

def bw_log_to_lin(log_bw, pref_sf):
    # given the preferred SF and octave bandwidth, returns the corresponding
    # (linear) bounds in cpd

    less_half = np.power(2, np.log2(pref_sf) - log_bw/2);
    more_half = np.power(2, log_bw/2 + np.log2(pref_sf));

    sf_range = [less_half, more_half];
    lin_bw = more_half - less_half;
    
    return lin_bw, sf_range

def DiffOfGauss(gain, f_c, gain_s, j_s, stim_sf):
  ''' Difference of gaussians 
  gain      - overall gain term
  f_c       - characteristic frequency of the center, i.e. freq at which response is 1/e of maximum
  gain_s    - relative gain of surround (e.g. gain_s of 0.5 says peak surround response is half of peak center response
  j_s       - relative characteristic freq. of surround (i.e. char_surround = f_c * j_s)
  '''
  dog = lambda f: np.maximum(0, gain*(np.exp(-np.square(f/f_c)) - gain_s * np.exp(-np.square(f/(f_c*j_s)))));

  norm = np.max(dog(stim_sf));
  if norm < 1e-5: # don't divide by zero, or anything close!
    dog_norm = [];
  else:
    dog_norm = lambda f: dog(f) / norm;
    dog_norm = dog_norm(stim_sf);

  return dog(stim_sf), dog_norm;

def DoGsach(gain_c, r_c, gain_s, r_s, stim_sf):
  ''' Difference of gaussians as described in Sach's thesis
  gain_c    - gain of the center mechanism
  r_c       - radius of the center
  gain_s    - gain of surround mechanism
  r_s       - radius of surround
  '''
  dog = lambda f: np.maximum(0, gain_c*np.pi*np.square(r_c)*np.exp(-np.square(f*np.pi*r_c)) - gain_s*np.pi*np.square(r_s)*np.exp(-np.square(f*np.pi*r_s)));

  norm = np.max(dog(stim_sf));
  if norm < 1e-5: # don't divide by zero, or anything close!
    dog_norm = [];
  else:
    dog_norm = lambda f: dog(f) / norm;
    dog_norm = dog_norm(stim_sf);

  return dog(stim_sf), dog_norm;

def var_explained(data, modParams, contrast, DoGmodel=1):
  ''' given a set of responses and model parameters, compute the variance explained by the model (DoGsach)
  '''
  resp_dist = lambda x, y: np.sum(np.square(x-y))/np.maximum(len(x), len(y))
  var_expl = lambda m, r, rr: 100 * (1 - resp_dist(m, r)/resp_dist(r, rr));

  respsSummary, stims, allResps = tabulateResponses(data); # Need to fit on f1 
  f1 = respsSummary[1];
  obs_mean = f1['mean'][contrast, :];

  NLL = 0;
  all_sfs = stims[1];

  if DoGmodel == 1:
    pred_mean, _ = DoGsach(*modParams, stim_sf=all_sfs);
  if DoGmodel == 2:
    pred_mean, _ = DiffOfGauss(*modParams, stim_sf=all_sfs);

  obs_grand_mean = np.mean(obs_mean) * np.ones_like(obs_mean); # make sure it's the same shape as obs_mean
    
  return var_expl(pred_mean, obs_mean, obs_grand_mean);

def dog_prefSf(modParams, all_sfs, dog_model=1):
  ''' Compute the preferred SF given a set of DoG parameters
  '''
  sf_bound = (np.min(all_sfs), np.max(all_sfs));
  if dog_model == 1:
    obj = lambda sf: -DoGsach(*modParams, stim_sf=sf)[0];
  elif dog_model == 2:
    obj = lambda sf: -DiffOfGauss(*modParams, stim_sf=sf)[0];
  init_sf = np.median(all_sfs);
  optz = opt.minimize(obj, init_sf, bounds=(sf_bound, ))
  return optz['x'];

def dog_prefSfMod(descrFit, allCons, varThresh=65):
  ''' Given a descrFit dict for a cell, compute a fit for the prefSf as a function of contrast
      Return ratio of prefSf at highest:lowest contrast, lambda of model, params
  '''
  # the model
  psf_model = lambda offset, slope, alpha, con: offset + slope*np.power(con-con[0], alpha);
  # gather the values
  #   only include prefSf values derived from a descrFit whose variance explained is gt the thresh
  validInds = np.where(descrFit['varExpl'][:] > varThresh)[0];
  if len(validInds) == 0: # i.e. no good fits...
    return np.nan, [], [];
  prefSfs = descrFit['prefSf'][disp, validInds];
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
  if DoGmodel == 1:
      r_c = prms[1];
      f_c = 1/(np.pi*r_c)
  elif DoGmodel == 2:
      f_c = prms[1];

  return f_c;

def dog_charFreqMod(descrFit, allCons, varThresh=65):
  ''' Given a descrFit dict for a cell, compute a fit for the prefSf as a function of contrast
      Return ratio of prefSf at highest:lowest contrast, lambda of model, params
  '''
  # the model
  fc_model = lambda offset, slope, alpha, con: offset + slope*np.power(con-con[0], alpha);
  # gather the values
  #   only include prefSf values derived from a descrFit whose variance explained is gt the thresh
  validInds = np.where(descrFit['varExpl'][:] > varThresh)[0];
  if len(validInds) == 0: # i.e. no good fits...
    return np.nan, [], [];
  prefSfs = descrFit['charFreq'][disp, validInds];
  conVals = allCons[validInds];
  weights = descrFit['varExpl'][disp, validInds];
  # set up the optimization
  obj = lambda params: np.sum(np.multiply(weights,
        np.square(fc_model(params[0], params[1], params[2], conVals) - prefSfs)))
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

  return fcRatio, fc_model, opt_params;


#####

def deriv_gauss(params, stimSf = np.logspace(np.log10(0.1), np.log10(10), 101)):

    prefSf = params[0];
    dOrdSp = params[1];

    sfRel = stimSf / prefSf;
    s     = pow(stimSf, dOrdSp) * np.exp(-dOrdSp/2 * pow(sfRel, 2));
    sMax  = pow(prefSf, dOrdSp) * np.exp(-dOrdSp/2);
    sNl   = s/sMax;
    selSf = sNl;

    return selSf, stimSf;

def compute_SF_BW(fit, height, sf_range):

    # 1/16/17 - This was corrected in the lead up to SfN (sometime 11/16). I had been computing
    # octaves not in log2 but rather in log10 - it is field convention to use
    # log2!

    # Height is defined RELATIVE to baseline
    # i.e. baseline = 10, peak = 50, then half height is NOT 25 but 30
    
    bw_log = np.nan;
    SF = np.empty((2, 1));
    SF[:] = np.nan;

    # left-half
    left_full_bw = 2 * (fit[3] * np.sqrt(2*np.log(1/height)));
    left_cpd = fit[2] * np.exp(-(fit[3] * np.sqrt(2*np.log(1/height))));

    # right-half
    right_full_bw = 2 * (fit[4] * np.sqrt(2*np.log(1/height)));
    right_cpd = fit[2] * np.exp((fit[4] * sqrt(2*np.log(1/height))));

    if left_cpd > sf_range[0] and right_cpd < sf_range[-1]:
        SF = [left_cpd, right_cpd];
        bw_log = np.log(right_cpd / left_cpd, 2);

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

    sigma = np.multiply(sigmaLow, [1]*len(sf0));

    sigma[[x for x in range(len(sf0)) if sf0[x] > 1]] = sigmaHigh;

    shape = [np.exp(-pow(np.log(x), 2) / (2*pow(y, 2))) for x, y in zip(sf0, sigma)];
                
    return [max(minThresh, respFloor + respRelFloor*x) for x in shape];

def blankResp(data):
  blanks = np.where(data['cont'] == 0);

  mu = np.mean(data['f0'][blanks]);
  std = np.std(data['f0'][blanks]);

  return mu, std;

def tabulateResponses(data):
  ''' Given the dictionary containing all of the data, organize the data into the proper responses
  Specifically, we know that Sach's experiments varied contrast and spatial frequency
  Thus, we will organize responses along these dimensions
  '''
  all_cons = np.unique(data['cont']);
  all_cons = all_cons[all_cons>0];
  all_sfs = np.unique(data['sf']);

  f0 = dict();
  f0mean= np.nan * np.zeros((len(all_cons), len(all_sfs))); 
  f0sem = np.nan * np.zeros((len(all_cons), len(all_sfs))); 
  f1 = dict();
  f1mean = np.nan * np.zeros((len(all_cons), len(all_sfs))); 
  f1sem = np.nan * np.zeros((len(all_cons), len(all_sfs))); 

  # rather than getting just the mean/s.e.m., we can also record/transfer the firing rate of each individual stimulus run
  f0arr = dict();
  f1arr = dict();
  
  for con in range(len(all_cons)):
    val_con = np.where(data['cont'] == all_cons[con]);
    f0arr[con] = dict();
    f1arr[con] = dict();
    for sf in range(len(all_sfs)):
      val_sf = np.where(data['sf'][val_con] == all_sfs[sf]);

      # take mean, since some conditions have repeats - just average them
      f0mean[con, sf] = np.mean(data['f0'][val_con][val_sf]);
      f0sem[con, sf] = np.mean(data['f0sem'][val_con][val_sf]);
      f1mean[con, sf] = np.mean(data['f1'][val_con][val_sf]);
      f1sem[con, sf] = np.mean(data['f1sem'][val_con][val_sf]);

      f0arr[con][sf] = data['f0arr'][val_con][val_sf];
      f1arr[con][sf] = data['f1arr'][val_con][val_sf];

  f0['mean'] = f0mean;
  f0['sem'] = f0sem;
  f1['mean'] = f1mean;
  f1['sem'] = f1sem;

  return [f0, f1], [all_cons, all_sfs], [f0arr, f1arr];

def random_in_range(lims, size = 1):

    return [random.uniform(lims[0], lims[1]) for i in range(size)]

def writeDataTxt(cellNum, f1, sfs, contrast, save_loc):
  
  obs_mean = f1['mean'][contrast, :];
  obs_sem = f1['sem'][contrast, :];
   
  write_name = 'cell%d_con%d.txt' % (cellNum, contrast);
  file = open(str(save_loc + write_name), 'w');

  for i in range(len(sfs)):
    file.write('%.3f %.3f %.3f\n' % (sfs[i], obs_mean[i], obs_sem[i]));

  file.close();

def writeCellTxt(cellNum, load_path, save_loc):

  dataList = np_smart_load(load_path + 'sachData.npy');
  data = dataList[cellNum-1]['data'];

  resps, conds, _ = tabulateResponses(data);
  f1 = resps[1];
  all_cons = conds[0];
  all_sfs = conds[1];
  
  for i in range(len(all_cons)):
    writeDataTxt(cellNum, f1, all_sfs, i, save_loc);
