import math, numpy, random
from scipy.stats import norm, mode, poisson, nbinom, sem
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

# fit_name - return the fit name with the proper direction flag (i.e. pos or neg)
# flatten - unpacks a list of lists into one list...
# switch_inner_outer - turn an NxM to an MxN, sort of
# sum_comps - sum the responses across components as determined in analyses for mixtures

# removed
# np_smart_load - be smart about using numpy load
# bw_lin_to_log
# bw_log_to_lin
# angle_xy - compute the angle of a vector given x, y coordinate

### fourier

# removed - temporally "reinstalled" on 2.13.19
# make_psth - create a psth for a given spike train
# spike_fft - compute the FFT for a given PSTH, extract the power at a given set of frequencies 

### phase/more psth

# (kept)
# project_resp - project the measured response onto the true/predicted phase and determine the "true" response amplitude
# project_resp_cond - project the individual responses for a given condition
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

# removed
# DiffOfGauss - standard difference of gaussians
# DoGsach - difference of gaussians as implemented in sach's thesis
# var_explained - compute the variance explained for a given model fit/set of responses
# deriv_gauss - evaluate a derivative of a gaussian, specifying the derivative order and peak
# get_prefSF - Given a set of parameters for a flexible gaussian fit, return the preferred SF
# compute_SF_BW - returns the log bandwidth for height H given a fit with parameters and height H (e.g. half-height)
# fix_params - Intended for parameters of flexible Gaussian, makes all parameters non-negative
# flexible_Gauss - Descriptive function used to describe/fit SF tuning
# blankResp - return mean/std of blank responses (i.e. baseline firing rate) for sfMixAlt experiment
# random_in_range - random real-valued number between A and B
# nbinpdf_log - was used with sfMix optimization to compute the negative binomial probability (likelihood) for a predicted rate given the measured spike count

# getSuppressiveSFtuning - returns the normalization pool response
# getNormParams  - given the model params and fit type, return the relevant parameters for normalization
# genNormWeights - used to generate the weighting matrix for weighting normalization pool responses
# setSigmaFilter - create the filter we use for determining c50 with SF
# evalSigmaFilter - evaluate an arbitrary filter at a set of spatial frequencies to determine c50 (semisaturation contrast)
# setNormTypeArr - create the normTypeArr used in SFMGiveBof/Simulate to determine the type of normalization and corresponding parameters
# makeStimulus - was used last for sfMix experiment to generate arbitrary stimuli for use with evaluating model

# (kept)

# dog_prefSf - compute the prefSf for a given DoG model/parameter set
# dog_prefSfMod - fit a simple model of prefSf as f'n of contrast
# dog_charFreq - given a model/parameter set, return the characteristic frequency of the tuning curve
# dog_charFreqMod - smooth characteristic frequency vs. contrast with a functional form/fit

# getCondition - returns trial-by-trial responses for specific disp/con/sf condition
# getConditionAdj - getCondition, but for adjusted/projected responses
# get_isolated_response - returns mean/std/trial-by-trial (f0 and f1), contrast/sf by component
# get_isolated_responseAdj - as get_isolated_response, but for adjusted/projected responses

# tabulate_responses - Organizes measured and model responses for sfMixAlt experiment
# organize_adj_responses - Organize the phase-adjusted responses into the format of tabulate_responses
# get_valid_trials - get the list of valid trials given a disp/con/sf combination - and return the list of all disps/cons/sfs
# get_valid_sfs - return indices of valid sfs for a given disp/con

# mod_poiss - evaluate the modulated poisson model
# naka_rushton - evaluate the naka-rushton model
# fit_CRF - fit a CRF/RVC using naka-rushton model

def fit_name(base, dir , byTrial=0):
  ''' Given the base name for a file, append the flag for the phase direction
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

def flatten(l):
  flatten = lambda l: [item for sublist in l for item in sublist];
  return flatten(l);

def switch_inner_outer(x):
  switch_inner_outer = lambda arr: [[x[i] for x in arr] for i in range(len(arr[0]))];
  return switch_inner_outer(x);

def sum_comps(l, stdFlag = 0):
  # assumes list - by sf - with each sf-specific list organized by contrast
  np = numpy;
  if stdFlag == 0:
    return [np.sum(x, 1) if x else [] for x in l];
  else:
    return [np.sqrt(np.sum(np.square(x), 1)) if x else [] for x in l];

### Basic Fourier analyses

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

### phase and amplitude analyses

def project_resp(amp, phi_resp, phAdv_model, phAdv_params, disp, allCompSf=None, allSfs=None):
  ''' Using our model fit of (expected) response phase as a function of response amplitude, we can
      determine the difference in angle between the expected and measured phase and then project the
      measured response vector (i.e. amp/phase in polar coordinates) onto the expected phase line
      eq: adjResp = measuredResp * cos(expPhi - measuredPhi)
      vectorized: expects/returns lists of amplitudes/phis
  '''
  np = numpy;

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
    elif disp == 1: # then we'll need to use the allCompSf to get the right phase advance fit for each component
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
          sf_ind = np.where(allSfs == allCompSf[i][con_ind][comp_ind])[0][0];
          phi_true = phAdv_model(phAdv_params[sf_ind][0], phAdv_params[sf_ind][1], curr_amp);
          if isinstance(phi_true, np.ndarray): # i.e. array
            if isinstance(phi_true[0], np.ndarray): # i.e. nested array
            # flatten into array of numbers rather than array of arrays (of one number) 
              phi_true = flatten(phi_true);
          # finally, project the response as usual
          proj = np.multiply(curr_amp, np.cos(np.deg2rad(curr_phi)-np.deg2rad(phi_true)));
          curr_proj_con.append(proj);
        all_proj[i].append(curr_proj_con);
  
  return all_proj;

def project_resp_cond(data, disp, con, sf, phAdv_model, phAdv_params, dir=-1):
  ''' NOTE: Not currently used, incomplete... 11.01.18
      Input: data structure, disp/con/sf (as indices, relative to the list of con/sf for that dispersion)
      Using our model fit of (expected) response phase as a function of response amplitude, we can
      determine the difference in angle between the expected and measured phase and then project the
      measured response vector (i.e. amp/phase in polar coordinates) onto the expected phase line
      eq: adjResp = measuredResp * cos(expPhi - measuredPhi)
  '''
  val_trials, allDisps, allCons, allSfs = get_valid_trials(data, disp, con, sf);

  if not numpy.any(val_trials[0]): # val_trials[0] will be the array of valid trial indices --> if it's empty, leave!
    warnings.warn('this condition is not valid');
    return [];

  allAmp, allPhi, _, allCompCon, allCompSf = get_all_fft(data, disp, dir=dir, all_trials=1);
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
    _, _, full_fourier = spike_fft(psth_val, all_tf)
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
     theta = angle_xy([x_avg], [y_avg])[0]; # just get the one value (will be packed in list)
     theta_var = circ_var(curr_phis); # compute on the original phases

     all_r.append(r);
     all_phi.append(theta);
     all_r_std.append(r_std);
     all_phi_var.append(theta_var);

   return all_r, all_phi, all_r_std, all_phi_var;

def get_all_fft(data, disp, cons=[], sfs=[], dir=-1, psth_binWidth=1e-3, stimDur=1, all_trials=0):
  ''' for a given cell and condition or set of conditions, compute the mean amplitude and phase
      also return the temporal frequencies which correspond to each condition
      if all_trials=1, then return the individual trial responses (i.e. not just avg over all repeats for a condition)
  '''

  _, _, val_con_by_disp, validByStimVal, _ = tabulate_responses(data);

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
    #all_r[s] = dict(); all_ph[s] = dict(); all_tf[s] = dict();
    curr_r = []; curr_ph = []; curr_tf = [];
    curr_conComp = []; curr_sfComp = [];
    for c in cons:
      val_trials, allDisps, allCons, allSfs = get_valid_trials(data, disp, c, s);

      if not numpy.any(val_trials[0]): # val_trials[0] will be the array of valid trial indices --> if it's empty, leave!
        warnings.warn('this condition is not valid');
        continue;

      # get the phase relative to the stimulus
      ph_rel_stim, stim_ph, resp_ph, curr_tf = get_true_phase(data, val_trials, dir, psth_binWidth, stimDur);
      # compute the fourier amplitudes
      psth_val, _ = make_psth(data['spikeTimes'][val_trials]);
      _, rel_amp, full_fourier = spike_fft(psth_val, curr_tf)
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
      elif disp == 1: # for mixtures
        # need to switch rel_amp to be lists of amplitudes by component (rather than list of amplitudes by trial)
        rel_amp = switch_inner_outer(rel_amp);
        rel_amp_sem = [sem(x) for x in rel_amp];
        # call get_isolated_response just to get contrast/sf per component
        _, _, _, _, conByComp, sfByComp = get_isolated_response(data, val_trials);
        # need to switch ph_rel_stim (and resp_phase) to be lists of phases by component (rather than list of phases by trial)
        ph_rel_stim = switch_inner_outer(ph_rel_stim);
        ph_rel_stim_sem = [sem(x) for x in ph_rel_stim];

        # compute vector mean, gather/organize
        avg_r, avg_ph, std_r, std_ph = polar_vec_mean(rel_amp, ph_rel_stim);
        if all_trials == 1:
          curr_r.append([avg_r, std_r, rel_amp, rel_amp_sem]);
          curr_ph.append([avg_ph, std_ph, ph_rel_stim, ph_rel_stim_sem]);
        elif all_trials == 0:
          curr_r.append([avg_r, std_r]);
          curr_ph.append([avg_ph, std_ph]);
        curr_tf.append(curr_tf);
        curr_conComp.append(conByComp);
        curr_sfComp.append(sfByComp);

    all_r.append(curr_r);
    all_ph.append(curr_ph);
    all_tf.append(curr_tf);
    if disp == 1:
      all_conComp.append(curr_conComp);
      all_sfComp.append(curr_sfComp);

  return all_r, all_ph, all_tf, all_conComp, all_sfComp;

def get_rvc_model():
  ''' simply return the rvc model used in the fits
  '''
  rvc_model = lambda b, k, c0, cons: b + k*numpy.log(1+numpy.divide(cons, c0));

  return rvc_model  

def get_phAdv_model():
  ''' simply return the phase advance model used in the fits
  '''
  # phAdv_model = [numpy.mod(phi0 + numpy.multiply(slope, x), 360) for x in amp] # because the sub-arrays of amp occasionally have
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
     
     if curr_amps == [] or curr_cons == []:
       # nothing to do - set to blank and move on
       all_opts.append([]);
       all_loss.append([]);
       all_conGain.append([]);
       continue;

     if var:
       loss_weights = np.divide(1, var[i]);
     else:
       loss_weights = np.ones_like(var[i]);
     obj = lambda params: np.sum(np.multiply(loss_weights, np.square(curr_amps - rvc_model(params[0], params[1], params[2], curr_cons))));
     init_params = [0, np.max(curr_amps), 0.5]; 
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
     print('\n#######%d#######\n' % i);
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

    #val_trials = val_disp & val_con & val_sf;
    val_trials = np.where(val_disp & val_con & val_sf)[0]; # get as int array of indices rather than boolean array
 
    f0 = data['spikeCount'][val_trials];
    f1 = data['power_f1'][val_trials];

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
  val_sf = data['cent_sf'] == sf;

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

def tabulate_responses(data, modResp = []):
    ''' Given cell structure (and opt model responses), returns the following:
        (i) respMean, respSEM, predMean, predStd, organized by condition; pred is linear prediction
        (ii) all_disps, all_cons, all_sfs - i.e. the stimulus conditions of the experiment
        (iii) the valid contrasts for each dispersion level
        (iv) valid_disp, valid_con, valid_sf - which conditions are valid for this particular cell
        (v) modRespOrg - the model responses organized as in (i) - only if modResp argument passed in
    '''

    # TODO: Problem here (with power_f1?) for new V1/ data
    np = numpy;
    conDig = 3; # round contrast to the thousandth
    
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
    respSEM = np.nan * np.empty((nDisps, nSfs, nCons));
    predMean = np.nan * np.empty((nDisps, nSfs, nCons));
    predStd = np.nan * np.empty((nDisps, nSfs, nCons));
    f1Mean = np.array(np.nan * np.empty((nDisps, nSfs, nCons)), dtype='O'); # create f1Mean/SEM so that each entry can accomodate an array, rather than just one value
    f1SEM = np.array(np.nan * np.empty((nDisps, nSfs, nCons)), dtype='O');
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
                respSEM[d, sf, con] = sem((data['spikeCount'][valid_tr]));
                try:
                  f1Mean[d, sf, con] = np.mean(data['power_f1'][valid_tr]); # default axis takes avg within components (and across trials)
                except:
                  f1Mean[d, sf, con] = np.nan;
                try:
                  if d > 0:
                    f1SEM[d, sf, con] = np.asarray([sem([x[i] for x in data['power_f1'][valid_tr]]) for i in range(all_disps[d])]); # need to be careful, since sem cannot handle numpy array well
                  else:
                    f1SEM[d, sf, con] = sem(data['power_f1'][valid_tr]);
                except:
                  f1SEM[d, sf, con] = np.nan;
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
                    try:
                      curr_pred_f1 = curr_pred_f1 + np.sum(np.mean(data['power_f1'][val_tr]));
                      curr_var_f1 = curr_var_f1 + np.sum(np.var(data['power_f1'][val_tr]));
                    except:
                      curr_pred_f1 = np.nan;
                      curr_var_f1 = np.nan;
                    
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
                    
    return [respMean, respSEM, predMean, predStd, f1Mean, f1SEM, predMeanF1, predStdF1], [all_disps, all_cons, all_sfs], val_con_by_disp, [valid_disp, valid_con, valid_sf], modRespOrg;

def organize_adj_responses(data, rvcFits):
  ''' Given the rvcFits, reorganize the responses into the format of tabulate_responses
      i.e. response in one array by disp X sf X con
      returns adjResps (as in tabulate_responses), adjByTrial (in trial order), 
        adjByComp (as in tabulate_responses, but each component separately), and 
        adjPred (predictions for mixtures based on adjusted single grating responses)
      NOTE: Checked validity of individual trial projections on 11.14.18
        The agreement between the value in adjResps and the mean of the responses in adjByTrial is high
        (We know from the nature of the projection/adjustment calculation that avg-->project will not be exactly the same
         as project-->avg)
  '''
  _, conds, val_con_by_disp, byTrial, _ = tabulate_responses(data);
  allDisps = conds[0];
  allCons = conds[1];
  allSfs = conds[2];  

  nDisps = len(allDisps);
  nCons = len(allCons);
  nSfs = len(allSfs);
  
  nTr = len(byTrial[0][0]); # [0][0] for dipsersion-single gratings, but doesn't matter!

  adjResps = numpy.nan * numpy.empty((nDisps, nSfs, nCons));
  adjByTrial = numpy.nan * numpy.empty((nTr, ));
  adjByComp = numpy.array(numpy.nan * numpy.empty((nDisps, nSfs, nCons)), dtype='O');
  adjPred = numpy.nan * numpy.empty((nDisps, nSfs, nCons));

  for d in range(nDisps):
    conInds = val_con_by_disp[d];
    sfInds = get_valid_sfs(data, d, conInds[0]); # it doesn't matter which contrast...
    for s in range(len(sfInds)):
      curr_resps = rvcFits[d]['adjMeans'][sfInds[s]];
      adjByComp[d, sfInds[s], conInds] = curr_resps;
      curr_resps_byTr = rvcFits[d]['adjByTr'][sfInds[s]];
      if d == 0:
        adjResps[d, sfInds[s], conInds] = curr_resps;
      elif d == 1: # sum over the components
        summed_resps = sum_comps([curr_resps]);
        adjResps[d, sfInds[s], conInds] = numpy.reshape(summed_resps, (len(conInds), ));

      for c in range(len(conInds)):
        val_trials, _, _, _ = get_valid_trials(data, d, conInds[c], sfInds[s]);

        # now, save response by trial
        if d == 0:
          curr_resp = flatten(curr_resps_byTr[c]);
        elif d == 1:
          curr_resp = sum_comps([switch_inner_outer(curr_resps_byTr[c])])[0];
        adjByTrial[val_trials[0]] = curr_resp;

        # and make prediction! (adjByTrial, even if incomplete, will have correct responses for these trials)
        isolResp, _, _, _ = get_isolated_responseAdj(data, val_trials, adjByTrial);
        # isolResp is organized as [mean, std] for each component - get the meanResp for each comp and sum
        adjPred[d, sfInds[s], conInds[c]] = numpy.sum(x[0] for x in isolResp);
    
  return adjResps, adjByTrial, adjByComp, adjPred;

def get_valid_trials(data, disp, con, sf):
  ''' Given a data and the disp/con/sf indices (i.e. integers into the list of all disps/cons/sfs
      Determine which trials are valid (i.e. have those stimulus criteria)
      RETURN list of valid trials, lists for all dispersion values, all contrast values, all sf values
  '''
  _, stimVals, _, validByStimVal, _ = tabulate_responses(data);

  # gather the conditions we need so that we can index properly
  valDisp = validByStimVal[0];
  valCon = validByStimVal[1];
  valSf = validByStimVal[2];

  allDisps = stimVals[0];
  allCons = stimVals[1];
  allSfs = stimVals[2];

  val_trials = numpy.where(valDisp[disp] & valCon[con] & valSf[sf]);

  return val_trials, allDisps, allCons, allSfs;

def get_valid_sfs(data, disp, con):
  ''' Self explanatory, innit? Returns the indices (into allSfs) of valid sfs for the given condition
  '''
  _, stimVals, _, validByStimVal, _ = tabulate_responses(data);

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

## 
