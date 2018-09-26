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
import warnings
import pdb

# Functions:

### basics

# np_smart_load - be smart about using numpy load
# bw_lin_to_log
# bw_log_to_lin
# angle_xy - compute the angle of a vector given x, y coordinate
# fit_name - return the fit name iwth the proper direction flag (i.e. pos or neg)
# flatten - unpacks a list of lists into one list...

### fourier

# make_psth - create a psth for a given spike train
# spike_fft - compute the FFT for a given PSTH, extract the power at a given set of frequencies 

### phase/more psth

# project_resp - project the measured response onto the true/predicted phase and determine the "true" response amplitude
# first_ph0 - for a given stimulus start phase, compute how much of a cycle (and how much time) before the stimulus gets to the start of a cycle (i.e. ph=0)
# fold_psth - fold a psth for a given number of cycles (given spike times)
# get_true_phase - compute the response phase relative to the stimulus phase given a response phase (rel. to trial time window) and a stimulus phase (rel. to trial start)
# polar_vec_mean - compute the vector mean given a set of amplitude/phase pairs for responses on individual trials
# get_all_fft - extract the amp/phase for a condition or set of conditions
# get_rvc_model - return the lambda function describing the rvc model
# get_phAdv_model - return the lambda function describing the responsePhase-as-function-of-respAmplitude model
# rvc_fit - fit response versus contrast with a model used in Movshon/Kiorpes/+ 2005
# phase_advance - compute the phase advance (a la Movshon/Kiorpes/+ 2005)

### descriptive fits to sf tuning/basic data analyses

# DiffOfGauss - standard difference of gaussians
# DoGsach - difference of gaussians as implemented in sach's thesis
# var_explained - compute the variance explained for a given model fit/set of responses
# dog_prefSf - compute the prefSf for a given DoG model/parameter set
# dog_prefSfMod - fit a simple model of prefSf as f'n of contrast
# deriv_gauss - evaluate a derivative of a gaussian, specifying the derivative order and peak
# get_prefSF - Given a set of parameters for a flexible gaussian fit, return the preferred SF
# compute_SF_BW - returns the log bandwidth for height H given a fit with parameters and height H (e.g. half-height)
# fix_params - Intended for parameters of flexible Gaussian, makes all parameters non-negative
# flexible_Gauss - Descriptive function used to describe/fit SF tuning
# blankResp - return mean/std of blank responses (i.e. baseline firing rate) for sfMixAlt experiment
# tabulate_responses - Organizes measured and model responses for sfMixAlt experiment
# get_valid_trials - get the list of valid trials given a disp/con/sf combination - and return the list of all disps/cons/sfs
# random_in_range - random real-valued number between A and B
# nbinpdf_log - was used with sfMix optimization to compute the negative binomial probability (likelihood) for a predicted rate given the measured spike count
# getSuppressiveSFtuning - returns the normalization pool response
# makeStimulus - was used last for sfMix experiment to generate arbitrary stimuli for use with evaluating model
# genNormWeights - used to generate the weighting matrix for weighting normalization pool responses
# setSigmaFilter - create the filter we use for determining c50 with SF
# evalSigmaFilter - evaluate an arbitrary filter at a set of spatial frequencies to determine c50 (semisaturation contrast)
# setNormTypeArr - create the normTypeArr used in SFMGiveBof/Simulate to determine the type of normalization and corresponding parameters

def np_smart_load(file_path, encoding_str='latin1'):

   if not os.path.isfile(file_path):
     return [];
   loaded = [];
   while(True):
     try:
       loaded = numpy.load(file_path, encoding=encoding_str).item();
       break;
     except IOError: # this happens, I believe, because of parallelization when running on the cluster; cannot properly open file, so let's wait and then try again
       sleep(5); # i.e. wait for 5 seconds

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

def fit_name(base, dir):
  ''' Given the base name for a file, append the flag for the phase direction
  '''
  if dir == 1:
    base = base + '_pos.npy';
  if dir == -1:
    base = base + '_neg.npy';
  return base;

def flatten(l):
  flatten = lambda l: [item for sublist in l for item in sublist];
  return flatten(l);

### Basic Fourier analyses

def make_psth(spikeTimes, binWidth=1e-3, stimDur=1):
    # given an array of arrays of spike times, create the PSTH for a given bin width and stimulus duration
    # i.e. spikeTimes has N arrays, each of which is an array of spike times
    # TODO: Add a smoothing to the psth and return for plotting purposes only

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

### phase and amplitude analyses

def project_resp(amp, phi_resp, phAdv_model, phAdv_params):
  ''' Using our model fit of (expected) response phase as a function of response amplitude, we can
      determine the difference in angle between the expected and measured phase and then project the
      measured response vector (i.e. amp/phase in polar coordinates) onto the expected phase line
      eq: adjResp = measuredResp * cos(expPhi - measuredPhi)
      vectorized: expects/returns lists of amplitudes/phis
  '''
  all_proj = [];
  for i in range(len(amp)):
    phi_true = phAdv_model(phAdv_params[i][0], phAdv_params[i][1], amp[i]);
    proj = numpy.multiply(amp[i], numpy.cos(numpy.deg2rad(phi_resp[i])-numpy.deg2rad(phi_true)));
    all_proj.append(proj);
  
  return all_proj;
  
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

def get_true_phase(data, val_trials, dir=-1, psth_binWidth=1e-3, stimDur=1):
    ''' Returns resp-phase-rel-to-stim, stimulus phase, response phase, and stimulus tf
        Given the data and the set of valid trials, first compute the response phase
        and stimulus phase - then determine the response phase relative to the stimulus phase
    '''
    np = numpy;
    # prepare the TF information for each component - we know there are 5 components per stimulus
    all_tfs = np.vstack((data['tf'][0], data['tf'][1], data['tf'][2], data['tf'][3], data['tf'][4]));
    all_tfs = np.transpose(all_tfs)[val_trials];
    all_tfs = all_tfs.astype(int); # we know all TF are integers - convert to that so we can use as an index
    all_tf = [[x[0]] if x[0] == x[1] else x for x in all_tfs];

    # and the phase...
    all_phis = np.vstack((data['ph'][0], data['ph'][1], data['ph'][2], data['ph'][3], data['ph'][4]));
    all_phis = np.transpose(all_phis)[val_trials];
    all_phis = all_phis.astype(int);
    # only get PHI for the components we need - use the length of all_tf as a guide
    stim_phase = [all_phis[x][range(len(all_tf[x]))] for x in range(len(all_phis))]

    # perform the fourier analysis we need
    psth_val, _ = make_psth(data['spikeTimes'][val_trials], psth_binWidth, stimDur)
    _, rel_amp, full_fourier = spike_fft(psth_val, all_tf)
    # and finally get the stimulus-relative phase of each response
    resp_phase = [np.angle(full_fourier[x][all_tf[x]], True) for x in range(len(full_fourier))];

    phase_rel_stim = np.mod(np.multiply(dir, np.add(resp_phase, stim_phase)), 360);

    return phase_rel_stim, stim_phase, resp_phase, all_tf;

def polar_vec_mean(amps, phases):
   ''' Given a set of amplitudes ("r") and phases ("theta"; in degrees) for a given stimulus condition (or set of conditions)
       RETURN the mean amplitude and phase (in degrees) computed by vector summation/averaging
       Note: amps/phases must be passed in as arrays of arrays, so that we can compute the vec mean for multiple different
             stimulus conditions just by calling this function once
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
     # take the mean/std - TODO: HOW TO COMPUTE STD OF VECTOR MEAN IN POLAR COORD
     x_avg, y_avg = [np.mean(x_polar), np.mean(y_polar)]
     x_std, y_std = [np.std(x_polar), np.std(y_polar)]
     # now compute (and return) r and theta
     r = np.sqrt(np.square(x_avg) + np.square(y_avg));
     r_std = np.sqrt(np.square(x_std) + np.square(y_std));
     # now the angle
     theta = angle_xy([y_avg], [x_avg])[0]; # just get the one value (will be packed in list)
     theta_var = circ_var(curr_phis); # compute on the original phases

     all_r.append(r);
     all_phi.append(theta);
     all_r_std.append(r_std);
     all_phi_var.append(theta_var);

   return all_r, all_phi, all_r_std, all_phi_var;

def get_all_fft(cellStruct, disp, cons=[], sfs=[], dir=-1, psth_binWidth=1e-3, stimDur=1):
  ''' for a given cell and condition or set of conditions, compute the mean amplitude and phase
      also return the temporal frequencies which correspond to each condition
  '''

  resp, stimVals, val_con_by_disp, validByStimVal, mdRsp = tabulate_responses(cellStruct);
  data = cellStruct['sfm']['exp']['trial'];

  # gather the sf indices in case we need - this is a dictionary whose keys are the valid sf indices
  valSf = validByStimVal[2];

  if cons == []: # then get all valid cons for this dispersion
    cons = val_con_by_disp[disp];
  if sfs == []: # then get all valid sfs for this dispersion
    sfs = list(valSf.keys());

  all_r = []; all_ph = []; all_tf = []; 
  # all_r = dict(); all_ph = dict(); all_tf = dict();

  for s in sfs:
    #all_r[s] = dict(); all_ph[s] = dict(); all_tf[s] = dict();
    curr_r = []; curr_ph = []; curr_tf = [];
    for c in cons:
      val_trials, allDisps, allCons, allSfs = get_valid_trials(cellStruct, disp, c, s);

      if not numpy.any(val_trials[0]): # val_trials[0] will be the array of valid trial indices --> if it's empty, leave!
        warnings.warn('this condition is not valid');
        continue;

      # get the phase relative to the stimulus
      ph_rel_stim, stim_ph, resp_ph, curr_tf = get_true_phase(data, val_trials, dir, psth_binWidth, stimDur);
      # compute the fourier amplitudes
      psth_val, _ = make_psth(data['spikeTimes'][val_trials]);
      _, rel_amp, full_fourier = spike_fft(psth_val, curr_tf)

      avg_r, avg_ph, std_r, std_ph = polar_vec_mean([rel_amp], [ph_rel_stim]);
      curr_r.append([avg_r[0], std_r[0]]); # we can just grab 0 element, since we're going one value at a time, but it's packed in array
      curr_ph.append([avg_ph[0], std_ph[0]]); # same as above
      curr_tf.append(curr_tf);

      #all_r[s][c] = [avg_r, std_r];
      #all_ph[s][c] = [avg_ph, std_ph];
      #all_tf[s][c] = curr_tf;
    all_r.append(curr_r);
    all_ph.append(curr_ph);
    all_tf.append(curr_tf);

  return all_r, all_ph, all_tf;

def get_rvc_model():
  ''' simply return the rvc model used in the fits
  '''
  rvc_model = lambda b, k, c0, cons: b + k*numpy.log(1+numpy.divide(cons, c0));

  return rvc_model  

def get_phAdv_model():
  ''' simply return the phase advance model used in the fits
  '''
  phAdv_model = lambda phi0, slope, amp: numpy.mod(phi0 + numpy.multiply(slope, amp), 360);
  # must mod by 360! Otherwise, something like 340-355-005 will be fit poorly
  return phAdv_model;

def rvc_fit(amps, cons, var = None):
   ''' Given the mean amplitude of responses (by contrast value) over a range of contrasts, compute the model
       fit which describes the response amplitude as a function of contrast as described in Eq. 3 of
       Movshon, Kiorpes, Hawken, Cavanaugh; 2005
       Optionally, can include a measure of variability in each response to perform weighted least squares
       RETURNS: rvc_model (the model equation), list of the optimal parameters, and the contrast gain measure
       Vectorized - i.e. accepts arrays of amp/con arrays
   '''
   np = numpy;

   rvc_model = get_rvc_model();
   
   all_opts = []; all_loss = [];
   all_conGain = [];
   n_amps = len(amps);

   for i in range(n_amps):
     curr_amps = amps[i];
     curr_cons = cons[i];
     if var:
       loss_weights = np.divide(1, var[i]);
     else:
       loss_weights = np.ones_like(var[i]);
     obj = lambda params: np.sum(np.multiply(loss_weights, np.square(curr_amps - rvc_model(params[0], params[1], params[2], curr_cons))));
     init_params = [0, np.max(curr_amps), 0.5]; 
     # init_b = 0 --> per the paper, most b = 0 (b <= 0)
     # init_c0 = 0.5 --> halfway in the contrast range
     # init_k = np.max(curr_amps) --> with c0=0.5, k*log(1+maxCon/0.5) is approx. k (maxCon is 1);
     b_bounds = (0, 0); # 9.14.18 - per Tony, set to be just 0 for now
     k_bounds = (0, None);
     c0_bounds = (1e-3, 1);
     all_bounds = (b_bounds, k_bounds, c0_bounds); # set all bounds
     # now optimize
     to_opt = opt.minimize(obj, init_params, bounds=all_bounds);
     opt_params = to_opt['x'];
     opt_loss = to_opt['fun'];

     # now determine the contrast gain
     b = opt_params[0]; k = opt_params[1]; c0 = opt_params[2];
     if b < 0: 
       # find the contrast value at which the rvc_model crosses/reaches 0
       obj_whenR0 = lambda con: np.square(0 - rvc_model(b, k, c0, con));
       con_bound = (0, 1);
       init_r0cross = 0;
       r0_cross = opt.minimize(obj_whenR0, init_r0cross, bounds=(con_bound, ));
       con_r0 = r0_cross['x'];
       conGain = k/(c0*(1+con_r0/c0));
     else:
       conGain = k/c0;

     all_opts.append(opt_params);
     all_loss.append(opt_loss);
     all_conGain.append(conGain);

   return rvc_model, all_opts, all_conGain, all_loss;

def phase_advance(amps, phis, cons, tfs):
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
     print('#######%d#######\n' % i);
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
     init_params = [curr_phiMean[min_resp_ind], init_slope];
     print(init_params);
     to_opt = opt.minimize(obj, init_params);
     opt_params = to_opt['x'];
     opt_loss = to_opt['fun'];
     print(opt_params);
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

### Descriptive functions - fits to spatial frequency tuning, other related calculations

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

def var_explained(data_resps, modParams, sfVals, dog_model = 1):
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

def dog_prefSf(modParams, all_sfs, dog_model=1):
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

def dog_prefSfMod(descrFit, allCons, disp=0, varThresh=65):
  ''' Given a descrFit dict for a cell, compute a fit for the prefSf as a function of contrast
      Return ratio of prefSf at highest:lowest contrast, lambda of model, params
  '''
  np = numpy;
  # the model
  psf_model = lambda offset, slope, alpha, con: offset + slope*np.power(con-con[0], alpha) 
  # gather the values
  #   only include prefSf values derived from a descrFit whose variance explained is gt the thresh
  validInds = np.where(descrFit['varExpl'][disp, :])[0];
  prefSfs = descrFit['prefSf'][disp, validInds];
  conVals = allCons[validInds];
  weights = descrFit['varExpl'][disp, validInds];
  # set up the optimization
  obj = lambda params: np.sum(np.multiply(weights,
        np.square(psf_model(params[0], params[1], params[2], allCons[val_cons]) - prefSfs)))
  init_offset = prefSfs[0];
  conRange = conVals[-1] - conVals[0];
  init_slope = (prefSfs[-1] - prefSfs[0]) / conRange;
  init_alpha = 0.5; # most tend to be saturation (i.e. contrast exp < 1)
  # run
  optz = opt.minimize(obj, [init_offset, init_slope, init_alpha], bounds=((0, None), (None, None), (0.25, 4)))
  opt_params = optz['x'];
  # ratio:
  extrema = psf_model(*opt_params, con=(conVals[0], conVals[-1]))
  pSfRatio = extrema[-1] / extrema[0]

  return pSfRatio, psf_model, opt_params;

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
    tr = cellStruct['sfm']['exp']['trial'];
    blank_tr = tr['spikeCount'][numpy.isnan(tr['con'][0])];
    mu = numpy.mean(blank_tr);
    sig = numpy.std(blank_tr);
    
    return mu, sig, blank_tr;

def get_condition(data, n_comps, con, sf):
    ''' Returns the trial responses (f0 and f1) and correspondign trials for a given 
        dispersion level (note: # of components), contrast, and spatial frequency
    '''
    np = numpy;
    conDig = 3; # default...

    val_disp = data['num_comps'] == n_comps;
    val_con = np.round(data['total_con'], conDig) == con;
    val_sf = data['cent_sf'] == sf;

    #val_trials = val_disp & val_con & val_sf;
    val_trials = np.where(val_disp & val_con & val_sf)[0]; # get as int array of indices rather than boolean array
 
    f0 = data['spikeCount'][val_trials];
    f1 = data['power_f1'][val_trials];

    return f0, f1, val_trials;
    
def get_isolated_response(data, trials):
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

   # assumption is that #trials of mixture stimulus will be >= the number of repetitions of the isolated presentations of that stimulus component
   f0all = np.array(np.nan * np.zeros((n_comps, )), dtype='O'); # might have different number of responses for each component, so create object/flexible array
   f1all = np.array(np.nan * np.zeros((n_comps, )), dtype='O');
   f0summary = np.nan * np.zeros((n_comps, 2)); # mean/std in [:, 0 or 1], respectively
   f1summary = np.nan * np.zeros((n_comps, 2));

   for i in range(n_comps):

     # now go through for each component and get the response to that stimulus component when presented alone
     con = np.unique(data['con'][i][trials]);
     sf = np.unique(data['sf'][i][trials]);
     if len(con)>1 or len(sf)>1:
       warnings.warn('the trials requested must have only one sf/con for a given stimulus component');
       return [], [], [], [];
     
     f0curr, f1curr, _ = get_condition(data, 1, np.round(con, conDig), sf); # 1 is for #components - we're looking for single component trials/responses
     f0all[i] = f0curr;
     f1all[i] = f1curr;

     f0summary[i, :] = [np.nanmean(f0all[i]), np.nanstd(f0all[i])]; # nanmean/std in case fewer presentations of individual component than mixture
     f1summary[i, :] = [np.nanmean(f1all[i]), np.nanstd(f1all[i])];

   return f0summary, f1summary, f0all, f1all

def tabulate_responses(cellStruct, modResp = []):
    ''' Given cell structure (and opt model responses), returns the following:
        (i) respMean, respStd, predMean, predStd, organized by condition; pred is linear prediction
        (ii) all_disps, all_cons, all_sfs - i.e. the stimulus conditions of the experiment
        (iii) the valid contrasts for each dispersion level
        (iv) valid_disp, valid_con, valid_sf - which conditions are valid for this particular cell
        (v) modRespOrg - the model responses organized as in (i) - only if modResp argument passed in
    '''
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
    f1Mean = np.array(np.nan * np.empty((nDisps, nSfs, nCons)), dtype='O'); # create f1Mean/Std so that each entry can accomodate an array, rather than just one value
    f1Std = np.array(np.nan * np.empty((nDisps, nSfs, nCons)), dtype='O');
    predMeanF1 = np.nan * np.empty((nDisps, nSfs, nCons));
    predStdF1 = np.nan * np.empty((nDisps, nSfs, nCons));

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
                f1Mean[d, sf, con] = np.mean(data['power_f1'][valid_tr]); # default axis takes avg within components (and across trials)
                f1Std[d, sf, con] = np.std(data['power_f1'][valid_tr]); # that is the axis we want!
                curr_pred = 0;
                curr_var = 0; # variance (std^2) adds
                curr_pred_f1 = 0;
                curr_var_f1 = 0; # variance (std^2) adds

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
                    curr_pred_f1 = curr_pred_f1 + np.sum(np.mean(data['power_f1'][val_tr]));
                    curr_var_f1 = curr_var_f1 + np.sum(np.var(data['power_f1'][val_tr]));
                    
                predMean[d, sf, con] = curr_pred;
                predStd[d, sf, con] = np.sqrt(curr_var);
                predMeanF1[d, sf, con] = curr_pred_f1;
                predStdF1[d, sf, con] = np.sqrt(curr_var_f1);
                
                if mod:
                    nTrCurr = sum(valid_tr); # how many trials are we getting?
                    modRespOrg[d, sf, con, 0:nTrCurr] = modResp[valid_tr];
        
            if np.any(~np.isnan(respMean[d, :, con])):
                if ~np.isnan(np.nanmean(respMean[d, :, con])):
                    val_con_by_disp[d].append(con);
                    
    return [respMean, respStd, predMean, predStd, f1Mean, f1Std, predMeanF1, predStdF1], [all_disps, all_cons, all_sfs], val_con_by_disp, [valid_disp, valid_con, valid_sf], modRespOrg;

def get_valid_trials(cellStruct, disp, con, sf):
  ''' Given a cellStruct and the disp/con/sf indices (i.e. integers into the list of all disps/cons/sfs
      Determine which trials are valid (i.e. have those stimulus criteria)
      RETURN list of valid trials, lists for all dispersion values, all contrast values, all sf values
  '''
  _, stimVals, _, validByStimVal, _ = tabulate_responses(cellStruct);

  # gather the conditions we need so that we can index properly
  valDisp = validByStimVal[0];
  valCon = validByStimVal[1];
  valSf = validByStimVal[2];

  allDisps = stimVals[0];
  allCons = stimVals[1];
  allSfs = stimVals[2];

  val_trials = numpy.where(valDisp[disp] & valCon[con] & valSf[sf]);

  return val_trials, allDisps, allCons, allSfs;

def get_valid_sfs(cellStruct, disp, con):
  ''' 
  '''
  _, stimVals, _, validByStimVal, _ = tabulate_responses(cellStruct);

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
  c50_len = 11; # 11 parameters if we've optimized for the filter which sets c50 in a frequency-dependent way
  gauss_len = 10; # 10 parameters in the model if we've optimized for the gaussian which weights the normalization filters
  asym_len = 9; # 9 parameters in the model if we've used the old asymmetry calculation for norm weights

  inhAsym = 0; # set to 0 as default

  # now do the work
  if normTypeArr:
    norm_type = int(normTypeArr[0]); # typecast to int
    if norm_type == 2:
      if len(params) == c50_len:
        filt_offset = params[8];
        std_l = params[9];
        std_r = params[10];
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
      normTypeArr = [norm_type, filt_offset, std_l, std_r];

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
