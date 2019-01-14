import math, numpy, random, os
from scipy.stats import norm, mode, poisson, nbinom
from scipy.stats.mstats import gmean as geomean
from numpy.matlib import repmat
from time import sleep
import scipy.optimize as opt
sqrt = math.sqrt
log = math.log
exp = math.exp
import pdb

## removed
# np_smart_load     - load a .npy file safely
# bw_lin_to_log
# bw_log_to_lin

# deriv_gauss       - evaluate a derivative of a gaussian, specifying the derivative order and peak
# compute_SF_BW     - returns the log bandwidth for height H given a fit with parameters and height H (e.g. half-height)
# fix_params        - Intended for parameters of flexible Gaussian, makes all parameters non-negative
# flexible_Gauss    - Descriptive function used to describe/fit SF tuning

# random_in_range    - random real-valued number between A and B
# nbinpdf_log        - was used with sfMix optimization to compute the negative binomial probability (likelihood) for a predicted rate given the measured spike count
# mod_poiss
# naka_rushton
# fit_CRF
# getSuppressiveSFtuning - returns the normalization pool response
# makeStimulus       - was used last for sfMix experiment to generate arbitrary stimuli for use with evaluating model
# getNormParams  - given the model params and fit type, return the relevant parameters for normalization
# genNormWeights     - used to generate the weighting matrix for weighting normalization pool responses
# setSigmaFilter     - create the filter we use for determining c50 with SF
# evalSigmaFilter    - evaluate an arbitrary filter at a set of spatial frequencies to determine c50 (semisaturation contrast)
# setNormTypeArr     - create the normTypeArr used in SFMGiveBof/Simulate to determine the type of normalization and corresponding parameters
# getConstraints     - get list of constraints for optimization

## (kept)
# Functions:
# descrFit_name
# chiSq

# get_center_con    - given a family and contrast index, returns the contrast level
# organize_modResp - Organizes measured and model responses 

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
  k   = 0.10 * rho * np.nanmax(data_resps[0]) # default kMult from Cavanaugh is 0.01

  chi = np.sum(np.divide(np.square(data_resps[0] - model_resps[0]), k + data_resps[0]*rho/stimDur));

  return chi;

######

def get_center_con(family, contrast):

    # hardcoded - based on sfMix as run in 2015/2016 (m657, m658, m660); given
    # the stimulus family and contrast level, returns the expected contrast of
    # the center frequency.
    
    # contrast = 1 means high contrast...otherwise, low contrast

    con = numpy.nan
    
    if family == 1:
        if contrast == 1:
            con = 1.0000;
        else:
            con = 0.3300;
    elif family == 2:
        if contrast == 1:
            con = 0.6717;
        else:
            con = 0.2217;
    elif family == 3:
        if contrast == 1:
            con = 0.3785;
        else:
            con = 0.1249;
    elif family == 4:
        if contrast == 1:
            con = 0.2161;
        else:
            con = 0.0713;
    elif family == 5:
        if contrast == 1:
            con = 0.1451;
        else:
            con = 0.0479;

    return con

def organize_modResp(modResp, expStructure):
    # the blockIDs are fixed...
    nFam = 5;
    nCon = 2;
    nCond = 11; # 11 sfCenters for sfMix
    nReps = 20; # never more than 20 reps per stim. condition
    
    # Analyze the stimulus-driven responses for the orientation tuning curve
    oriBlockIDs = numpy.hstack((numpy.arange(131, 155+1, 2), numpy.arange(132, 136+1, 2))); # +1 to include endpoint like Matlab

    rateOr = numpy.empty((0,));
    for iB in oriBlockIDs:
        indCond = numpy.where(expStructure['blockID'] == iB);
        if len(indCond[0]) > 0:
            rateOr = numpy.append(rateOr, numpy.mean(modResp[indCond]));
        else:
            rateOr = numpy.append(rateOr, numpy.nan);

    # Analyze the stimulus-driven responses for the contrast response function
    conBlockIDs = numpy.arange(138, 156+1, 2);
    iC = 0;

    rateCo = numpy.empty((0,));
    for iB in conBlockIDs:
        indCond = numpy.where(expStructure['blockID'] == iB);   
        if len(indCond[0]) > 0:
            rateCo = numpy.append(rateCo, numpy.mean(modResp[indCond]));
        else:
            rateCo = numpy.append(rateCo, numpy.nan);

    # Analyze the stimulus-driven responses for the spatial frequency mixtures

    # Initialize Variables        
    rateSfMix = numpy.ones((nFam, nCon, nCond)) * numpy.nan;
    allSfMix = numpy.ones((nFam, nCon, nCond, nReps)) * numpy.nan;
    for iE in range(nCon):
        for iW in range(nFam):

            StimBlockIDs  = numpy.arange(((iW)*(13*2)+1)+(iE), 1+((iW+1)*(13*2)-5)+(iE), 2);
            nStimBlockIDs = len(StimBlockIDs);
            #print('nStimBlockIDs = ' + str(nStimBlockIDs));
        
            iC = 0;

            for iB in StimBlockIDs:
                indCond = numpy.where(expStructure['blockID'] == iB);   
                if len(indCond[0]) > 0:
                    #print('setting up ' + str((iE, iW, iC)) + ' with ' + str(len(indCond[0])) + 'trials');
                    rateSfMix[iW, iE, iC] = numpy.nanmean(modResp[indCond]);
                    allSfMix[iW, iE, iC, 0:len(indCond[0])] = modResp[indCond];
                    iC         = iC+1;
                 
    return rateOr, rateCo, rateSfMix, allSfMix;
