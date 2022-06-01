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
import helper_fcns_sach as hf
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

### SET-UP
loc_base = os.getcwd() + '/';
date_suffix = '220411';
phAdvName = 'phAdv_%s' % date_suffix; 
rvcName = 'rvcFitsHPC_220219';

phAdv_set_ylim = 1; # if 1, then we make the ylim [0,360] for phAdv-mean plot; otherwise, we don't specify the limit

def plot_phase_advance(which_cell, sv_loc=loc_base + 'figures/', dir=1, dp=loc_base + 'structures/', expName='sachData.npy', phAdvStr=phAdvName, rvcStr=rvcName, date_suffix=date_suffix):
  ''' RVC, resp-X-phase, phase advance model split by SF within each cell
      1. response-versus-contrast; shows original and adjusted response
      2. polar plot of response amplitude and phase with phase advance model fit
      3. response amplitude (x) vs. phase (y) with the linear phase advance model fit
  '''

  # basics
  dataList = hf.np_smart_load(str(dp + expName))
  cellName  = dataList[which_cell-1]['cellName'];
  data = dataList[which_cell-1]['data'];
    
  rvcFits = hf.np_smart_load(str(dp + hf.phase_fit_name(rvcStr, dir)));
  rvcFits = rvcFits[which_cell-1];
  rvc_model = hf.get_rvc_model();

  phAdvFits = hf.np_smart_load(str(dp + hf.phase_fit_name(phAdvStr, dir)));
  phAdvFits = phAdvFits[which_cell-1];
  phAdv_model = hf.get_phAdv_model();

  save_base = sv_loc + 'phasePlots_%s/' % date_suffix;

  # gather/compute everything we need
  resps, stimVals, resps_all = hf.tabulateResponses(data, resample=False);
  
  allCons = stimVals[0];
  allSfs = stimVals[1];

  # now get ready to plot
  fPhaseAdv = [];

  # we will summarize for all spatial frequencies for a given cell!
  for sf in range(len(allSfs)): 
    # first, get the responses and phases that we need:
    amps = [];
    phis = [];
    val_sf = np.where(data['sf'] == allSfs[sf]);
    for con in allCons:
      val_con = np.where(data['cont'][val_sf] == con);
      # get the phase of the response relative to the stimulus (ph_rel_stim)
      ph_rel_stim = hf.nan_rm(data['f1pharr'][val_sf][val_con]);
      phis.append(ph_rel_stim);
      # get the relevant amplitudes (i.e. the amplitudes at the stimulus TF)
      curr_amp = hf.nan_rm(data['f1arr'][val_sf][val_con]);
      amps.append(curr_amp);

    r, th, _, _ = hf.polar_vec_mean(amps, phis); # mean amp/phase (outputs 1/2); std/var for amp/phase (outputs 3/4)
    # get the models/fits that we need:
    con_values = allCons;
    ## phase advance
    opt_params_phAdv = phAdvFits['params'][sf];
    ph_adv = phAdvFits['phAdv'][sf];
    ## rvc
    opt_params_rvc = rvcFits['params'][sf];
    con_gain = rvcFits['conGain'][sf];
    adj_means = [resps[1]['mean'][x][sf] for x in range(len(allCons))]; # for resps[1] --> f1 responses
    # (Above) remember that we have to project the amp/phase vector onto the "correct" phase for estimate of noiseless response
    ## now get ready to plot!
    f, ax = plt.subplots(2, 2, figsize=(20, 10))
    fPhaseAdv.append(f);

    n_conds = len(r);
    colors = cm.viridis(np.linspace(0, 0.95, n_conds));

    #####
    ## 1. now for plotting: first, response amplitude (with linear contrast)
    #####
    plot_cons = np.linspace(0, 1, 100);
    mod_fit = rvc_model(opt_params_rvc[0], opt_params_rvc[1], opt_params_rvc[2], plot_cons);

    ax = plt.subplot(2, 2, 1);
    plot_amp = adj_means;
    plt_measured = ax.scatter(allCons, plot_amp, s=100, color=colors, label='ph. corr');
    plt_og = ax.plot(allCons, r, linestyle='None', marker='o', markeredgecolor='k', markerfacecolor='None', alpha=0.5, label='vec. mean');
    plt_fit = ax.plot(plot_cons, mod_fit, linestyle='--', color='k', label='rvc fit');
    ax.set_xlabel('contrast');
    ax.set_ylabel('response (f1)');
    ax.set_title('response versus contrast')
    ax.legend(loc='upper left')
    #ax.legend((plt_measured, plt_fit[0]), ('data', 'model fit'), loc='upper left')

    # also summarize the model fit on this plot
    ymax = np.maximum(np.max(r), np.max(mod_fit));
    plt.text(0.8, 0.30 * ymax, 'b: %.2f' % (opt_params_rvc[0]), fontsize=12, horizontalalignment='center', verticalalignment='center');
    plt.text(0.8, 0.20 * ymax, 'slope:%.2f' % (opt_params_rvc[1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
    plt.text(0.8, 0.10 * ymax, 'c0: %.2f' % (opt_params_rvc[2]), fontsize=12, horizontalalignment='center', verticalalignment='center');
    plt.text(0.8, 0.0 * ymax, 'con gain: %.2f' % (con_gain), fontsize=12, horizontalalignment='center', verticalalignment='center');

    #####
    ## 3. then the fit/plot of phase as a function of ampltude
    #####
    plot_amps = np.linspace(0, np.max(r), 100);
    mod_fit = phAdv_model(opt_params_phAdv[0], opt_params_phAdv[1], plot_amps);

    ax = plt.subplot(2, 1, 2);
    plt_measured = ax.scatter(r, th, s=100, color=colors, clip_on=False, label='vec. mean');
    plt_fit = ax.plot(plot_amps, mod_fit, linestyle='--', color='k', clip_on=False, label='phAdv fit');
    ax.set_xlabel('response amplitude');
    if phAdv_set_ylim:
      ax.set_ylim([0, 360]);
    ax.set_ylabel('response phase');
    ax.set_title('phase advance with amplitude')
    ax.legend(loc='upper left')

    ## and again, summarize the model fit on the plot
    xmax = np.maximum(np.max(r), np.max(plot_amps));
    ymin = np.minimum(np.min(th), np.min(mod_fit));
    ymax = np.maximum(np.max(th), np.max(mod_fit));
    yrange = ymax-ymin;
    if phAdv_set_ylim:
      if mod_fit[-1]>260: # then start from ymin and go dwn
        start, sign = mod_fit[-1]-30, -1;
      else:
        start, sign = mod_fit[-1]+30, 1;
      plt.text(0.9*xmax, start + 1*30*sign, 'phi0: %.2f' % (opt_params_phAdv[0]), fontsize=12, horizontalalignment='center', verticalalignment='center');
      plt.text(0.9*xmax, start + 2*30*sign, 'slope:%.2f' % (opt_params_phAdv[1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
      plt.text(0.9*xmax, start + 3*30*sign, 'phase advance: %.2f ms' % (ph_adv), fontsize=12, horizontalalignment='center', verticalalignment='center');
    else:
      plt.text(0.8*xmax, ymin + 0.25 * yrange, 'phi0: %.2f' % (opt_params_phAdv[0]), fontsize=12, horizontalalignment='center', verticalalignment='center');
      plt.text(0.8*xmax, ymin + 0.15 * yrange, 'slope:%.2f' % (opt_params_phAdv[1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
      plt.text(0.8*xmax, ymin + 0.05 * yrange, 'phase advance: %.2f ms' % (ph_adv), fontsize=12, horizontalalignment='center', verticalalignment='center');

    #center_phi = lambda ph1, ph2: np.arcsin(np.sin(np.deg2rad(ph1) - np.deg2rad(ph2)));

    #####
    ## 2. now the polar plot of resp/phase together
    #####
    ax = plt.subplot(2, 2, 2, projection='polar')
    th_center = np.rad2deg(np.radians(-90)+np.radians(th[np.argmax(r)])); # "anchor" to the phase at the highest amplitude response
    #data_centered = center_phi(th, th_center);
    #model_centered = center_phi(mod_fit, th_center);
    #ax.scatter(data_centered, r, s=50, color=colors);
    #ax.plot(model_centered, plot_amps, linestyle='--', color='k');
    data_centered = np.mod(th-th_center, 360);
    model_centered = np.mod(mod_fit-th_center, 360);
    ax.scatter(np.deg2rad(data_centered), r, s=10, marker='o', edgecolors='k', c='None', alpha=0.5, label='vec. mean')
    ax.scatter(np.deg2rad(data_centered), plot_amp, s=50, color=colors, label='ph. corr')
    ax.plot(np.deg2rad(model_centered), plot_amps, linestyle='--', color='k');
    #print('data|model');
    #print(data_centered);
    #print(model_centered);
    ax.set_ylim(0, 1.25*np.max(r))
    ax.set_title('phase advance')

    # overall title
    f.subplots_adjust(wspace=0.2, hspace=0.25);
    f1f0_ratio = np.nan; #hf.compute_f1f0(data, which_cell, expInd, dp, descrFitName_f0=descrFit_f0)[0];
    f.suptitle('%s (%.2f) #%d: sf %.2f cpd' % (cellName, f1f0_ratio, which_cell, allSfs[sf]));

  saveName = "/cell_%03d_phaseAdv.pdf" % (which_cell);
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

if __name__ == '__main__':

    if len(sys.argv) < 3:
      print('uhoh...you need at least two arguments here');
      exit();

    cell_num = int(sys.argv[1]);
    dir      = int(sys.argv[2]);

    plot_phase_advance(cell_num, dir=dir);
