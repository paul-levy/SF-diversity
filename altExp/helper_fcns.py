import math, numpy, random
from scipy.stats import norm, mode, poisson, nbinom
from scipy.stats.mstats import gmean as geomean
from numpy.matlib import repmat
import scipy.optimize as opt
import os
from time import sleep
sqrt = math.sqrt
log = math.log
exp = math.exp
import pdb

###  (kept)
## Functions:

# descrFit_name
# chiSq

# organize_modResp   - akin to o.modResp in /Analysis/Functions, used for preparing responses for chiSq calculation in optimization


### removed:

# np_smart_load - be smart about using numpy load
# bw_lin_to_log
# bw_log_to_lin

# deriv_gauss - evaluate a derivative of a gaussian, specifying the derivative order and peak
# get_prefSF - Given a set of parameters for a flexible gaussian fit, return the preferred SF
# compute_SF_BW - returns the log bandwidth for height H given a fit with parameters and height H (e.g. half-height)
# fix_params - Intended for parameters of flexible Gaussian, makes all parameters non-negative
# flexible_Gauss - Descriptive function used to describe/fit SF tuning
# blankResp - return mean/std of blank responses (i.e. baseline firing rate) for sfMixAlt experiment

# random_in_range - random real-valued number between A and B
# nbinpdf_log - was used with sfMix optimization to compute the negative binomial probability (likelihood) for a predicted rate given the measured spike count
# getSuppressiveSFtuning - returns the normalization pool response
# getNormParams - given model params and fit type, return relevant params for normalization
# genNormWeights - used to generate the weighting matrix for weighting normalization pool responses
# setSigmaFilter - create the filter we use for determining c50 with SF
# evalSigmaFilter - evaluate an arbitrary filter at a set of spatial frequencies to determine c50 (semisaturation contrast)
# setNormTypeArr - create the normTypeArr used in SFMGiveBof/Simulate to determine the type of normalization and corresponding parameters
# getConstraints - return list of constraints for optimization

# makeStimulus - was used last for sfMix experiment to generate arbitrary stimuli for use with evaluating model
# tabulate_responses - Organizes measured and model responses for sfMixAlt experiment

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
  descrFitBase = 'descrFits%s' % floss_str;

  if modelName is None:
    descrName = '%s.npy' % descrFitBase;
  else:
    descrName = '%s_%s' % (descrFitBase, modelName);
    
  return descrName;

def chiSq(data_resps, model_resps, stimDur=1):
  ''' given a set of measured and model responses, compute the chi-squared (see Cavanaugh et al '02a)
      assumes: resps are mean/variance for each stimulus condition (e.g. like a tuning curve)
        with each condition a tuple (or 2-array) with [mean, var]
  '''
  np = numpy;

  rats = np.divide(data_resps[1], data_resps[0]);
  nan_rm = lambda x: x[~np.isnan(x)]
  rho = geomean(nan_rm(rats));
  # now, only evaluate where both model and data are non-nan!
  fineTr = ~np.isnan(data_resps[0]) & ~np.isnan(model_resps[0]);
  k   = 0.10 * rho * np.nanmax(data_resps[0]) # default kMult from Cavanaugh is 0.01

  chi = np.sum(np.divide(np.square(data_resps[0][fineTr] - model_resps[0][fineTr]), k + data_resps[0][fineTr]*rho/stimDur));

  return chi;

######

def organize_modResp(respsByTr, cellStruct):
    ''' Given cell list of trial-by-trial responses and cell structure, returns the following:
          average per condition
          all responses per condition (i.e. response for each repeat) properly organized
        NOTE: For use in model_responses.py, for ex, to help with chiSquared calculation
    '''
    np = numpy;
    conDig = 3; # round contrast to the thousandth
    nReps  = 20; # n reps will always be <= 20 (really 10, but whatever)    

    data = cellStruct['sfm']['exp']['trial'];

    all_cons = np.unique(np.round(data['total_con'], conDig));
    all_cons = all_cons[~np.isnan(all_cons)];

    all_sfs = np.unique(data['cent_sf']);
    all_sfs = all_sfs[~np.isnan(all_sfs)];

    all_disps = np.unique(data['num_comps']);
    all_disps = all_disps[all_disps>0]; # ignore zero...

    nCons = len(all_cons);
    nSfs = len(all_sfs);
    nDisps = len(all_disps);
    
    respMean = np.nan * np.empty((nDisps, nSfs, nCons));
    respAll = np.nan * np.empty((nDisps, nSfs, nCons, nReps));

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

                if np.all(np.unique(valid_tr) == False):
                    continue;
                    
                respMean[d, sf, con] = np.mean(respsByTr[valid_tr]);
                respAll[d, sf, con, 0:sum(valid_tr)] = respsByTr[valid_tr]; # sum(valid_tr) is how many are True, i.e. actually being "grabbed"
                
            if np.any(~np.isnan(respMean[d, :, con])):
                if ~np.isnan(np.nanmean(respMean[d, :, con])):
                    val_con_by_disp[d].append(con);
                    
    return respMean, respAll;

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
