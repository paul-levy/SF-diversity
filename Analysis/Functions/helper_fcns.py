import math, numpy, random
sqrt = math.sqrt
log = math.log
exp = math.exp

def bw_lin_to_log( lin_low, lin_high ):
    # Given the low/high sf in cpd, returns number of octaves separating the
    # two values

    return math.log(lin_high/lin_low, 2);

def bw_log_to_lin(log_bw, pref_sf):
    # given the preferred SF and octave bandwidth, returns the corresponding
    # (linear) bounds in cpd

    less_half = math.pow(2, math.log(pref_sf, 2) - log_bw/2);
    more_half = math.pow(2, log_bw/2 + math.log(pref_sf, 2));

    sf_range = [less_half, more_half];
    lin_bw = more_half - less_half;
    
    return lin_bw, sf_range


def compute_SF_BW(fit, height, sf_range):

    # 1/16/17 - This was corrected in the lead up to SfN (sometime 11/16). I had been computing
    # octaves not in log2 but rather in log10 - it is field convention to use
    # log2!

    # Height is defined RELATIVE to baseline
    # i.e. baseline = 10, peak = 50, then half height is NOT 25 but 30
    
    bw_log = math.nan;
    SF = numpy.empty((2, 1));
    SF[:] = math.nan;

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

def flexible_Gauss(params, stim_sf):
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
                
    return [max(0.1, respFloor + respRelFloor*x) for x in shape];

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

    # Analyze the stimulus-driven responses for the contrast response function
    conBlockIDs = numpy.arange(138, 156+1, 2);
    iC = 0;

    rateCo = numpy.empty((0,));
    for iB in conBlockIDs:
        indCond = numpy.where(expStructure['blockID'] == iB);   
        if len(indCond[0]) > 0:
            rateCo = numpy.append(rateCo, numpy.mean(modResp[indCond]));

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

def getSuppressiveSFtuning(): # written when still new to python. Probably to matlab-y...
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
