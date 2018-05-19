import math, numpy, random
from scipy.stats import norm, mode, poisson, nbinom
from scipy.stats.mstats import gmean as geomean
from numpy.matlib import repmat
import scipy.optimize as opt
sqrt = math.sqrt
log = math.log
exp = math.exp
import pdb

# Functions:
# bw_lin_to_log
# bw_log_to_lin
# deriv_gauss - evaluate a derivative of a gaussian, specifying the derivative order and peak
# compute_SF_BW - returns the log bandwidth for height H given a fit with parameters and height H (e.g. half-height)
# fix_params - Intended for parameters of flexible Gaussian, makes all parameters non-negative
# flexible_Gauss - Descriptive function used to describe/fit SF tuning
# blankResp - return mean/std of blank responses (i.e. baseline firing rate) for sfMixAlt experiment
# tabulateResponses - Organizes measured and model responses for sfMixAlt experiment
# random_in_range - random real-valued number between A and B
# nbinpdf_log - was used with sfMix optimization to compute the negative binomial probability (likelihood) for a predicted rate given the measured spike count
# getSuppressiveSFtuning - returns the normalization pool response
# makeStimulus - was used last for sfMix experiment to generate arbitrary stimuli for use with evaluating model
# genNormWeights - used to generate the weighting matrix for weighting normalization pool responses
# evalSigmaFilter - evaluate an arbitrary filter at a set of spatial frequencies to determine c50 (semisaturation contrast)

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

def deriv_gauss(params, stimSf = numpy.logspace(numpy.log10(0.1), numpy.log10(10), 101)):

    prefSf = params[0];
    dOrdSp = params[1];

    sfRel = stimSf / prefSf;
    s     = pow(stimSf, dOrdSp) * numpy.exp(-dOrdSp/2 * pow(sfRel, 2));
    sMax  = pow(prefSf, dOrdSp) * numpy.exp(-dOrdSp/2);
    sNl   = s/sMax;
    selSf = sNl;

    return selSf, stimSf;

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
    tr = cellStruct['sfm']['exp']['trial'];
    blank_tr = tr['spikeCount'][numpy.isnan(tr['con'][0])];
    mu = numpy.mean(blank_tr);
    sig = numpy.std(blank_tr);
    
    return mu, sig, blank_tr;
    
def tabulate_responses(cellStruct, modResp = []):
    np = numpy;
    conDig = 3; # round contrast to the thousandth
    
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

                if np.all(np.unique(valid_tr) == False):
                    continue;

                respMean[d, sf, con] = np.mean(data['spikeCount'][valid_tr]);
                respStd[d, sf, con] = np.std((data['spikeCount'][valid_tr]));
                
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
                    
                    curr_pred = curr_pred + np.mean(data['spikeCount'][val_tr]);
                    curr_var = curr_var + np.var(data['spikeCount'][val_tr]);
                    
                predMean[d, sf, con] = curr_pred;
                predStd[d, sf, con] = np.sqrt(curr_var);
                
                if mod:
                    nTrCurr = sum(valid_tr); # how many trials are we getting?
                    modRespOrg[d, sf, con, 0:nTrCurr] = modResp[valid_tr];
        
            if np.any(~np.isnan(respMean[d, :, con])):
                if ~np.isnan(np.nanmean(respMean[d, :, con])):
                    val_con_by_disp[d].append(con);
                    
    return [respMean, respStd, predMean, predStd], [all_disps, all_cons, all_sfs], val_con_by_disp, [valid_disp, valid_con, valid_sf], modRespOrg;

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

def getSuppressiveSFtuning(): # written when still new to python. Probably to matlab-y...
    # Not updated for sfMixAlt - 1/31/18
    # normPool details are fixed, ya?
    # plot model details - exc/suppressive components
    omega = numpy.logspace(-2, 2, 1000);

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

def makeStimulus(stimFamily, conLevel, sf_c, template):

# returns [Or, Tf, Co, Ph, Sf, trial_used]

# 1/23/16 - This function is used to make arbitrary stimuli for use with
# the Robbe V1 model. Rather than fitting the model responses at the actual
# experimental stimuli, we instead can simulate from the model at any
# arbitrary spatial frequency in order to determine peak SF
# response/bandwidth/etc

# If argument 'template' is given, then orientation, phase, and tf_center will be
# taken from the template, which will be actual stimuli from that cell

    # Fixed parameters
    num_families = 4;
    num_gratings = 7;
    comps = [1, 3, 5, 7]; # number of components for each family

    spreadVec = numpy.logspace(math.log10(.125), math.log10(1.25), num_families);
    octSeries  = numpy.linspace(1.5, -1.5, num_gratings);

    # set contrast and spatial frequency
    if conLevel == 1:
        total_contrast = 1;
    elif conLevel == 2:
        total_contrast = 1/3;
    elif conLevel>=0 and conLevel<1:
        total_contrast = conLevel;
    else:
        #warning('Contrast should be given as 1 [full] or 2 [low/one-third]; setting contrast to 1 (full)');
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

def genNormWeights(cellStruct, nInhChan, gs_mean, gs_std, nTrials):
  np = numpy;
  # A: do the calculation here - more flexibility
  inhWeight = [];
  nFrames = 120;
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
  return evalC50;


