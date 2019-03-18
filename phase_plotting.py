# coding: utf-8

import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # to avoid GUI/cluster issues...
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import matplotlib.cm as cm
import seaborn as sns
sns.set(style='ticks')
import helper_fcns as hf
from scipy.stats import poisson, nbinom, norm
from scipy.stats.mstats import gmean
import warnings
warnings.filterwarnings('once');

import pdb

import sys # so that we can import model_responses (in different folder)
import model_responses

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/paul_plt_style.mplstyle');
from matplotlib import rcParams
rcParams['font.size'] = 20;
rcParams['pdf.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['ps.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['lines.linewidth'] = 2.5;
rcParams['axes.linewidth'] = 1.5;
rcParams['lines.markersize'] = 5;
rcParams['font.style'] = 'oblique';
rcParams['errorbar.capsize'] = 0;

# at CNS
loc_base = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/';
# personal mac
#loc_base = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/';
# prince cluster
#loc_base = '/home/pl1465/SF_diversity/';

expDir = sys.argv[3];

# temporary
if len(sys.argv) > 6:
  dir = int(sys.argv[6]);
  print('direction is %d' % dir);
  saveDir = hf.phase_fit_name('figuresTest', dir).replace('.npy', '/');
else:
  saveDir = 'figuresTest/';

dataPath = loc_base + expDir + 'structures/';
save_loc = loc_base + expDir + saveDir;

expName = 'dataList.npy'
phAdvName = 'phaseAdvanceFitsTest';
rvcName = 'rvcFitsTest';

def phase_by_cond(which_cell, data, expInd, disp, con, sf, sv_loc=save_loc, dir=-1, cycle_fold=2, n_bins_fold=8, dp=dataPath, expName=expName):
  ''' Given a cell and the disp/con/sf indices, plot the spike raster for each trial, a folded PSTH,
      and finally the response phase - first relative to trial onset, and finally relative to the stimulus phase 
      
      dir = -1 or +1 (i.e. stimulus moving left or right?)
      cycle_fold = over how many stimulus cycles to fold when creating folded psth?
      n_bins_fold = how many bins per stimulus period when folding?
  '''
  dataList = hf.np_smart_load(str(dp + expName));
  save_base = sv_loc + 'phasePlots/';

  val_trials, allDisps, allCons, allSfs = hf.get_valid_trials(data, disp, con, sf, expInd);

  if not np.any(val_trials[0]): # val_trials[0] will be the array of valid trial indices --> if it's empty, leave!
    warnings.warn('this condition is not valid');
    return;

  ### SINGLE GRATINGS
  if disp == 0: 

    # get the phase relative to the stimulus
    ph_rel_stim, stim_ph, resp_ph, all_tf = hf.get_true_phase(data, val_trials, expInd, dir);
    # compute the fourier amplitudes
    stimDur = hf.get_exp_params(expInd).stimDur;
    psth_val, _ = hf.make_psth(data['spikeTimes'][val_trials], stimDur=stimDur);
    _, rel_amp, full_fourier = hf.spike_fft(psth_val, all_tf, stimDur)

    # now plot!
    f, ax = plt.subplots(3, 2, figsize=(20, 30))

    relSpikes = data['spikeTimes'][val_trials];
    colors = cm.rainbow(np.linspace(0, 1, len(val_trials[0])))

    # plot spike raster - trial-by-trial
    # only works for SINGLE GRATINGS
    # draw the beginning of each cycle for each trial
    ax = plt.subplot(3, 1, 1)
    for i in range(len(relSpikes)):
        ax.scatter(relSpikes[i], i*np.ones_like(relSpikes[i]), s=45, color=colors[i]);
        stimPh = stim_ph[i];
        stimPeriod = np.divide(1.0, all_tf[i]);
        stimDur = hf.get_exp_params(expInd).stimDur;
        # i.e. at what point during the trial (in s) does the stimulus component first begin a cycle?
        firstPh0 = hf.first_ph0(stimPh, all_tf[i])[1];

        for j in range(len(all_tf[i])):
            allPh0 = [stimPeriod[j]*np.arange(-1, stimDur * all_tf[i][j]) + firstPh0[j]];
            allPh90 = allPh0 + stimPeriod[j]/4;
            allPh180 = allPh90 + stimPeriod[j]/4;
            allPh270 = allPh180 + stimPeriod[j]/4;
        ax.errorbar(allPh0[0], i*np.ones_like(allPh0[0]), 0.25, linestyle='none', color='k', linewidth=1)
        ax.errorbar(allPh90[0], i*np.ones_like(allPh0[0]), 0.05, linestyle='none', color='k', linewidth=1)
        ax.errorbar(allPh180[0], i*np.ones_like(allPh0[0]), 0.05, linestyle='none', color='k', linewidth=1)
        ax.errorbar(allPh270[0], i*np.ones_like(allPh0[0]), 0.05, linestyle='none', color='k', linewidth=1)
    ax.set_xlabel('time (s)');
    ax.set_ylabel('repetition #');
    ax.set_title('Spike rasters');

    # plot PSTH - per trial, but folded over N cycles
    # only works for SINGLE GRATINGS
    ax = plt.subplot(3, 1, 2)

    for i in range(len(relSpikes)):
        _, bin_edges, psth_norm = hf.fold_psth(relSpikes[i], all_tf[i], stim_ph[i], cycle_fold, n_bins_fold, dir=dir);
        plt.plot(bin_edges[0:-1], i-0.5+psth_norm, color=colors[i])
        stimPeriod = np.divide(1.0, all_tf[i]);
        for j in range(cycle_fold):
            cycStart = plt.axvline(j*stimPeriod[0]);
            cycHalf = plt.axvline((j+0.5)*stimPeriod[0], linestyle='--');
    ax.set_xlabel('time (s)');
    ax.set_ylabel('spike count (normalized by trial)');
    ax.set_title('PSTH folded');
    ax.set_xlim([-stimPeriod[0]/4.0, (cycle_fold+0.25)*stimPeriod[0]]);
    ax.legend((cycStart, cycHalf), ('ph = 0', 'ph = 180'));

    # plot response phase - without accounting for stimulus phase
    ax = plt.subplot(3, 2, 5, projection='polar')
    ax.scatter(np.radians(resp_ph), rel_amp, s=60, color=colors, clip_on=False);
    ax.set_title('Stimulus-blind')
    ax.set_ylim(auto='True')
    polar_ylim = ax.get_ylim();

    # now, compute the average amplitude/phase over all trials
    [avg_r, avg_ph, _, _] = hf.polar_vec_mean([rel_amp], [ph_rel_stim]);
    avg_r = avg_r[0]; # just get it out of the array!
    avg_ph = avg_ph[0]; # just get it out of the array!

    # plot response phase - relative to stimulus phase
    ax = plt.subplot(3, 2, 6, projection='polar')
    ax.scatter(np.radians(ph_rel_stim), rel_amp, s=60, color=colors, clip_on=False);
    ax.plot([0, np.radians(avg_ph)], [0, avg_r], color='k', linestyle='--', clip_on=False);
    ax.set_ylim(polar_ylim);
    ax.set_title('Stimulus-accounted');

  ### MIXTURE STIMULI
  elif disp>0:
    nComp = hf.get_exp_params(expInd).nStimComp;
    # a useful function for swapping inside/outside of nested list...
    switch_inner_outer = lambda arr: [[x[i] for x in arr] for i in range(len(arr[0]))];
  
    ### first, gather the isolated component responses from the mixture
    val_trials, allDisp, allCons, allSfs = hf.get_valid_trials(data, disp, con, sf, expInd)
    # outer-most list (for ph_rel_stim, stim_phase, resp_phase) is trial/repetition, inner lists are by component
    ph_rel_stim, stim_phase, resp_phase, all_tf = hf.get_true_phase(data, val_trials, expInd, dir=dir);
    # f1all is nComp lists, each with nReps/nTrials values
    _, _, _, f1all, conByComp, sfByComp = hf.get_isolated_response(data, val_trials);
    # need to switch ph_rel_stim (and resp_phase) to be lists of phases by component (rather than list of phases by trial)
    ph_rel_stim = switch_inner_outer(ph_rel_stim);
    resp_phase = switch_inner_outer(resp_phase);
    # compute vector means
    r_comp, th_comp, _, _ = hf.polar_vec_mean(f1all, ph_rel_stim)

    nrow = 1+nComp;
    ncol = 2;
    f, ax = plt.subplots(nrow, ncol, figsize=(ncol*10, nrow*10))

    # in-mixture response phase - NOT stimulus-aligned; just demonstrate for 1st component
    colors = cm.rainbow(np.linspace(0, 1, len(val_trials[0])))
    ax = plt.subplot(nrow, 1, 1, projection='polar') # pretend only one column, i.e. take up whole top row
    ax.scatter(np.radians(resp_phase[0]), hf.flatten(f1all[0]), s=45, color=colors, clip_on=False);
    ax.set_ylim([0, 1.1*np.max(hf.flatten(f1all[0]))]);
    ax.set_title('Stimulus-blind (compound; comp #1)');

    for i in range(nComp):

        # compute the isolated response
        # Then, pick one of the components and get the response (when presented in isolation) 
        # phases/amplitudes and align relative to the stimulus phase
        isolConInd = np.where(allCons == conByComp[i])[0][0]; # unwrap fully (will be only one value...)
        isolSfInd = np.where(allSfs == sfByComp[i])[0][0];
        val_trials_isol, _, _, _ = hf.get_valid_trials(data, disp=0, con=isolConInd, sf=isolSfInd, expInd=expInd)

        ph_rel_stim_isol, stim_phase_isol, resp_phase_isol, all_tf_isol = hf.get_true_phase(data, val_trials_isol, expInd, dir=dir);
        stimDur = hf.get_exp_params(expInd).stimDur;
        psth_val, _ = hf.make_psth(data['spikeTimes'][val_trials_isol], stimDur=stimDur)
        _, rel_amp_isol, _ = hf.spike_fft(psth_val, all_tf_isol, stimDur)
        # and compute vector mean
        r_isol, th_isol, _, _ = hf.polar_vec_mean([rel_amp_isol], [ph_rel_stim_isol]);

        # isolated response phase - relative to stimulus phase
        colors = cm.rainbow(np.linspace(0, 1, len(val_trials[0])))
        ax = plt.subplot(nrow, ncol, 2*(i+1) + 1, projection='polar')
        ax.scatter(np.radians(ph_rel_stim_isol), rel_amp_isol, s=45, color=colors, clip_on=False);
        ax.plot([0, np.radians(th_isol[0])], [0, r_isol[0]], ls='--', color='k');
        ax.set_ylim([0, 1.1*np.max(rel_amp_isol)]);
        ax.set_title('isolated (r, phi) = (%.2f, %.2f)' % (r_isol[0], th_isol[0]));

        # in-mixture response phase - relative to stimulus phase
        colors = cm.rainbow(np.linspace(0, 1, len(val_trials[0])))
        ax = plt.subplot(nrow, ncol, 2*(i+2), projection='polar')
        ax.scatter(np.radians(ph_rel_stim[i]), hf.flatten(f1all[i]), s=45, color=colors, clip_on=False);
        ax.plot([0, np.radians(th_comp[i])], [0, r_comp[i]], ls='--', color='k');
        ax.set_ylim([0, 1.1*np.max(rel_amp_isol)]);
        ax.set_title('compound (r, phi) = (%.2f, %.2f)' % (r_comp[i], th_comp[i]));

  ## Common to all
  f.subplots_adjust(wspace=0.2, hspace=0.25);
  try: # not always exists
    cell_label = dataList['unitType'][which_cell-1];
  except: # always works
    cell_label = dataList['unitArea'][which_cell-1];
  f.suptitle('%s #%d: disp %d, con %.2f, sf %.2f' % (cell_label, which_cell, allDisps[disp], allCons[con], allSfs[sf]));

  saveName = "/cell_%03d_d%dsf%dcon%d_phase.pdf" % (which_cell, disp, sf, con);
  save_loc = save_base + "cell_%03d/" % which_cell;
  full_save = os.path.dirname(str(save_loc));
  if not os.path.exists(full_save):
    os.makedirs(full_save)
  pdfSv = pltSave.PdfPages(full_save + saveName);
  pdfSv.savefig(f);
  plt.close(f);
  pdfSv.close();

def plot_phase_advance(which_cell, disp, sv_loc=save_loc, dir=-1, dp=dataPath, expName=expName, phAdvStr=phAdvName, rvcStr=rvcName):

  # basics
  dataList = hf.np_smart_load(str(dp + expName))
  cellName  = dataList['unitName'][which_cell-1];
  expInd = hf.get_exp_ind(dp, cellName)[0];
  cellStruct = hf.np_smart_load(str(dp + cellName + '_sfm.npy'));

  rvcFits = hf.np_smart_load(str(dp + hf.phase_fit_name(rvcStr, dir)));
  rvcFits = rvcFits[which_cell-1];
  rvc_model = hf.get_rvc_model();

  phAdvFits = hf.np_smart_load(str(dp + hf.phase_fit_name(phAdvStr, dir)));
  phAdvFits = phAdvFits[which_cell-1];
  phAdv_model = hf.get_phAdv_model();

  save_base = sv_loc + 'phasePlots/';

  # gather/compute everything we need
  data = cellStruct['sfm']['exp']['trial'];
  _, stimVals, val_con_by_disp, validByStimVal, _ = hf.tabulate_responses(data, expInd);
  
  valDisp = validByStimVal[0];
  valCon = validByStimVal[1];
  valSf = validByStimVal[2];

  allDisps = stimVals[0];
  allCons = stimVals[1];
  allSfs = stimVals[2];

  con_inds = val_con_by_disp[disp];

  # now get ready to plot
  fPhaseAdv = [];

  # we will summarize for all spatial frequencies for a given cell!
  for j in range(len(allSfs)): 
    # first, get the responses and phases that we need:
    amps = [];
    phis = [];
    sf = j;
    for i in con_inds:
        val_trials = np.where(valDisp[disp] & valCon[i] & valSf[sf])

        # get the phase of the response relative to the stimulus (ph_rel_stim)
        ph_rel_stim, stim_ph, resp_ph, all_tf = hf.get_true_phase(data, val_trials, expInd, dir=dir);
        phis.append(ph_rel_stim);
        # get the relevant amplitudes (i.e. the amplitudes at the stimulus TF)
        stimDur = hf.get_exp_params(expInd).stimDur;
        psth_val, _ = hf.make_psth(data['spikeTimes'][val_trials], stimDur=stimDur)
        _, rel_amp, _ = hf.spike_fft(psth_val, all_tf, stimDur)
        amps.append(rel_amp);

    r, th, _, _ = hf.polar_vec_mean(amps, phis); # mean amp/phase (outputs 1/2); std/var for amp/phase (outputs 3/4)
    # get the models/fits that we need:
    con_values = allCons[con_inds];
    ## phase advance
    opt_params_phAdv = phAdvFits['params'][sf];
    ph_adv = phAdvFits['phAdv'][sf];
    ## rvc
    opt_params_rvc = rvcFits[disp]['params'][sf];
    con_gain = rvcFits[disp]['conGain'][sf];
    adj_means = rvcFits[disp]['adjMeans'][sf]; 
    if disp == 1: # then sum adj_means (saved by component)
      adj_means  = [np.sum(x, 1) if x else [] for x in adj_means];
    # (Above) remember that we have to project the amp/phase vector onto the "correct" phase for estimate of noiseless response
    ## now get ready to plot!
    f, ax = plt.subplots(2, 2, figsize=(20, 10))
    fPhaseAdv.append(f);

    n_conds = len(r);
    colors = cm.viridis(np.linspace(0, 0.95, n_conds));

    ## now for plotting: first, response amplitude (with linear contrast)
    plot_cons = np.linspace(0, 1, 100);
    mod_fit = rvc_model(opt_params_rvc[0], opt_params_rvc[1], opt_params_rvc[2], plot_cons);

    ax = plt.subplot(2, 2, 1);
    plot_amp = adj_means;
    plt_measured = ax.scatter(allCons[con_inds], plot_amp, s=100, color=colors);
    plt_og = ax.plot(allCons[con_inds], r, linestyle='None', marker='o', markeredgecolor='k', markerfacecolor='None', alpha=0.5);
    plt_fit = ax.plot(plot_cons, mod_fit, linestyle='--', color='k');
    ax.set_xlabel('contrast');
    ax.set_ylabel('response (f1)');
    ax.set_title('response versus contrast')
    ax.legend((plt_measured, plt_fit[0]), ('data', 'model fit'), loc='upper left')

    ## also summarize the model fit on the plot
    ymax = np.maximum(np.max(r), np.max(mod_fit));
    plt.text(0.8, 0.30 * ymax, 'b: %.2f' % (opt_params_rvc[0]), fontsize=12, horizontalalignment='center', verticalalignment='center');
    plt.text(0.8, 0.20 * ymax, 'slope:%.2f' % (opt_params_rvc[1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
    plt.text(0.8, 0.10 * ymax, 'c0: %.2f' % (opt_params_rvc[2]), fontsize=12, horizontalalignment='center', verticalalignment='center');
    plt.text(0.8, 0.0 * ymax, 'con gain: %.2f' % (con_gain), fontsize=12, horizontalalignment='center', verticalalignment='center');

    ## 2. then the fit/plot of phase as a function of ampltude
    plot_amps = np.linspace(0, np.max(r), 100);
    mod_fit = phAdv_model(opt_params_phAdv[0], opt_params_phAdv[1], plot_amps);

    ax = plt.subplot(2, 1, 2);
    plt_measured = ax.scatter(r, th, s=100, color=colors);
    plt_fit = ax.plot(plot_amps, mod_fit, linestyle='--', color='k');
    ax.set_xlabel('response amplitude');
    ax.set_ylabel('response phase');
    ax.set_title('phase advance with amplitude')
    ax.legend((plt_measured, plt_fit[0]), ('data', 'model fit'), loc='upper left')

    ## and again, summarize the model fit on the plot
    xmax = np.maximum(np.max(r), np.max(plot_amps));
    ymin = np.minimum(np.min(th), np.min(mod_fit));
    ymax = np.maximum(np.max(th), np.max(mod_fit));
    yrange = ymax-ymin;
    plt.text(0.8*xmax, ymin + 0.25 * yrange, 'phi0: %.2f' % (opt_params_phAdv[0]), fontsize=12, horizontalalignment='center', verticalalignment='center');
    plt.text(0.8*xmax, ymin + 0.15 * yrange, 'slope:%.2f' % (opt_params_phAdv[1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
    plt.text(0.8*xmax, ymin + 0.05 * yrange, 'phase advance: %.2f ms' % (ph_adv), fontsize=12, horizontalalignment='center', verticalalignment='center');

    #center_phi = lambda ph1, ph2: np.arcsin(np.sin(np.deg2rad(ph1) - np.deg2rad(ph2)));

    ## now the polar plot of resp/phase together
    ax = plt.subplot(2, 2, 2, projection='polar')
    th_center = np.rad2deg(np.radians(-90)+np.radians(th[np.argmax(r)])); # "anchor" to the phase at the highest amplitude response
    #data_centered = center_phi(th, th_center);
    #model_centered = center_phi(mod_fit, th_center);
    #ax.scatter(data_centered, r, s=50, color=colors);
    #ax.plot(model_centered, plot_amps, linestyle='--', color='k');
    data_centered = np.mod(th-th_center, 360);
    model_centered = np.mod(mod_fit-th_center, 360);
    ax.scatter(np.deg2rad(data_centered), r, s=50, color=colors)
    ax.plot(np.deg2rad(model_centered), plot_amps, linestyle='--', color='k');
    #print('data|model');
    #print(data_centered);
    #print(model_centered);
    ax.set_ylim(0, 1.25*np.max(r))
    ax.set_title('phase advance')

    # overall title
    f.subplots_adjust(wspace=0.2, hspace=0.25);
    try:
      f.suptitle('%s #%d: disp %d, sf %.2f cpd' % (dataList['unitType'][which_cell-1], which_cell, allDisps[disp], allSfs[sf]));
    except:
      f.suptitle('%s #%d: disp %d, sf %.2f cpd' % (dataList['unitArea'][which_cell-1], which_cell, allDisps[disp], allSfs[sf]));

  saveName = "/cell_%03d_d%d_phaseAdv.pdf" % (which_cell, disp);
  save_loc = save_base + 'summary/';
  full_save = os.path.dirname(str(save_loc));
  print('saving at: %s' % str(full_save + saveName));
  if not os.path.exists(full_save):
    os.makedirs(full_save)
  pdfSv = pltSave.PdfPages(full_save + saveName);
  for f in fPhaseAdv:
    pdfSv.savefig(f)
    plt.close(f)
  pdfSv.close();

def batch_phase_by_cond(cell_num, disp, cons=[], sfs=[], dir=-1, dp=dataPath, expName=expName):
  ''' must specify dispersion (one value)
      if cons = [], then get/plot all valid contrasts for the given dispersion
      if sfs = [], then get/plot all valid contrasts for the given dispersion
  '''
  dataList = hf.np_smart_load(str(dp + expName));
  fileName = dataList['unitName'][cell_num-1];
  expInd   = hf.get_exp_ind(dp, fileName)[0];

  cellStruct = hf.np_smart_load(str(dp + fileName + '_sfm.npy'));
  data = cellStruct['sfm']['exp']['trial'];
  # prepare the valid stim parameters by condition in case needed
  resp, stimVals, val_con_by_disp, validByStimVal, mdRsp = hf.tabulate_responses(data, expInd);

  # gather the sf indices in case we need - this is a dictionary whose keys are the valid sf indices
  valSf = validByStimVal[2];

  if cons == []: # then get all valid cons for this dispersion
    cons = val_con_by_disp[disp];
  if sfs == []: # then get all valid sfs for this dispersion
    sfs = list(valSf.keys());

  for c in cons:
    for s in sfs:
      print('analyzing cell %d, dispersion %d, contrast %d, sf %d\n' % (cell_num, disp, c, s));
      phase_by_cond(cell_num, data, expInd, disp, c, s, dir=dir);  

if __name__ == '__main__':

    if len(sys.argv) < 3:
      print('uhoh...you need at least two arguments here');
      exit();

    cell_num = int(sys.argv[1]);
    disp = int(sys.argv[2]);
    expDir = sys.argv[3];
    ph_by_cond = int(sys.argv[4]);
    ph_adv_summary = int(sys.argv[5]);

    if len(sys.argv) > 6:
      dir = int(sys.argv[6]);
    else:
      dir = None;
    print('Running cell %d, dispersion %d' % (cell_num, disp));

    if ph_by_cond:
      if dir:
        batch_phase_by_cond(cell_num, disp, dir=dir);
      else:
        batch_phase_by_cond(cell_num, disp);
    if ph_adv_summary:
      if dir:
        plot_phase_advance(cell_num, disp, dir=dir);
      else:
        plot_phase_advance(cell_num, disp);
