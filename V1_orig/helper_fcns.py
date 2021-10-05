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

## (kept)
# Functions:

# nan_rm - remove NaN from array
# get_center_con    - given a family and contrast index, returns the contrast level
# get_num_comps
# organize_modResp - Organizes measured and model responses 

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

######

def nan_rm(x):
   return x[~numpy.isnan(x)];

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

def get_num_comps(con):
  ''' in effect, a reverse of the above (get_center_con). Given a contrast, determine what dispersion level it is
      we're mapping to numComponents, so we go x -> 2x+1 (e.g. family 0, single gratings, becomes 1; family 3 becomes 7)
  '''
  np = numpy;
  con_round = np.round(con, 4);
  num_comps = np.ones_like(con, dtype=np.int16);
  for fam in range(5): # 5 families
    num_comps[np.where(con_round == get_center_con(fam+1, 1))] = int(2*fam+1);
    num_comps[np.where(con_round == get_center_con(fam+1, 2))] = int(2*fam+1);
  # if we've made it here, then it's either not part of the sfMix series but still something (RVC or ori; defaulted to 1) or it's nothing (and we need to set to 0)
  num_comps[np.where(np.isnan(con_round))] = int(0); 

  return num_comps

def organize_modResp(modResp, expStructure, mask=None, resample=False, cellNum=-1):
    # 01.18.19 - changed order of SF & CON in arrays to match organize_resp from other experiments
    # the blockIDs are fixed...
    # - resample: bootstrap resampling -- applies only to rateSfMix/allSfMix
    nFam = 5;
    nCon = 2;
    nCond = 11; # 11 sfCenters for sfMix
    nReps = 20; # never more than 20 reps per stim. condition
 
    if mask is None:
      mask = numpy.ones((len(data['blockID']), ), dtype=bool); # i.e. look at all trials
   
    # Analyze the stimulus-driven responses for the orientation tuning curve
    oriBlockIDs = numpy.hstack((numpy.arange(131, 155+1, 2), numpy.arange(132, 136+1, 2))); # +1 to include endpoint like Matlab

    rateOr = numpy.empty((0,));
    for iB in oriBlockIDs:
        indCond = numpy.where(expStructure['blockID'][mask] == iB);
        if len(indCond[0]) > 0:
            rateOr = numpy.append(rateOr, numpy.mean(modResp[mask][indCond]));
        else:
            rateOr = numpy.append(rateOr, numpy.nan);

    # Analyze the stimulus-driven responses for the contrast response function
    conBlockIDs = numpy.arange(138, 156+1, 2);
    iC = 0;

    rateCo = numpy.empty((0,));
    for iB in conBlockIDs:
        indCond = numpy.where(expStructure['blockID'][mask] == iB);   
        if len(indCond[0]) > 0:
            rateCo = numpy.append(rateCo, numpy.mean(modResp[mask][indCond]));
        else:
            rateCo = numpy.append(rateCo, numpy.nan);

    # Analyze the stimulus-driven responses for the spatial frequency mixtures

    # Initialize Variables        
    rateSfMix = numpy.ones((nFam, nCond, nCon)) * numpy.nan;
    allSfMix = numpy.ones((nFam, nCond, nCon, nReps)) * numpy.nan;
    for iE in range(nCon):
        for iW in range(nFam):

            StimBlockIDs  = numpy.arange(((iW)*(13*2)+1)+(iE), 1+((iW+1)*(13*2)-5)+(iE), 2);
            nStimBlockIDs = len(StimBlockIDs);
            #print('nStimBlockIDs = ' + str(nStimBlockIDs));
        
            conInd = nCon-1-iE; # why? to match the way other experiments are organized (low to high contrast)
            iC = 0;

            for iB in StimBlockIDs:
                indCond = numpy.where(expStructure['blockID'][mask] == iB);   
                if len(indCond[0]) > 0:
                    #print('setting up ' + str((iE, iW, iC)) + ' with ' + str(len(indCond[0])) + 'trials');
                    if resample:
                      non_nan = nan_rm(modResp[mask][indCond]);
                      resps = numpy.random.choice(non_nan, len(non_nan));
                    else:
                      resps = modResp[mask][indCond];
                    rateSfMix[iW, iC, conInd] = numpy.nanmean(resps);
                    try:
                      nResps = len(resps);
                      allSfMix[iW, iC, conInd, 0:nResps] = resps;
                    except:
                      print('Could not put resps in allSfMix [cell %d]' % cellNum);
                    iC = iC+1;
                 
    return rateOr, rateCo, rateSfMix, allSfMix;
