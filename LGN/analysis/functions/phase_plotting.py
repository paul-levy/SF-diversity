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
import warnings
from scipy.stats import poisson, nbinom, norm
from scipy.stats.mstats import gmean

import pdb

import sys # so that we can import model_responses (in different folder)
import model_responses

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/Analysis/Functions/paul_plt_cluster.mplstyle');
from matplotlib import rcParams
rcParams['font.size'] = 20;
rcParams['pdf.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['ps.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['lines.linewidth'] = 2.5;
rcParams['axes.linewidth'] = 1.5;
rcParams['lines.markersize'] = 5;
rcParams['font.style'] = 'oblique';
rcParams['errorbar.capsize'] = 0;

# personal mac
#dataPath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/analysis/structures/';
#save_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/analysis/figures/';
# prince cluster
dataPath = '/home/pl1465/SF_diversity/LGN/analysis/structures/';
save_loc = '/home/pl1465/SF_diversity/LGN/analysis/figures/';

expName = 'dataList.npy'

def phase_by_cond(which_cell, cellStruct, disp, con, sf, sv_loc=save_loc, dir=-1, cycle_fold=2, n_bins_fold=8, psth_binWidth=1e-3, stimDur=1):
  ''' Given a cell and the disp/con/sf indices, plot the spike raster for each trial, a folded PSTH,
      and finally the response phase - first relative to trial onset, and finally relative to the stimulus phase 
      
      dir = -1 or +1 (i.e. stimulus moving left or right?)
      cycle_fold = over how many stimulus cycles to fold when creating folded psth?
      n_bins_fold = how many bins per stimulus period when folding?
  '''
  save_base = sv_loc + 'phasePlots/';

  data = cellStruct['sfm']['exp']['trial'];

  resp, stimVals, val_con_by_disp, validByStimVal, mdRsp = hf.tabulate_responses(cellStruct);

  # gather the conditions we need so that we can index properly
  valDisp = validByStimVal[0];
  valCon = validByStimVal[1];
  valSf = validByStimVal[2];

  allDisps = stimVals[0];
  allCons = stimVals[1];
  allSfs = stimVals[2];

  val_trials = np.where(valDisp[disp] & valCon[con] & valSf[sf])

  if not np.any(val_trials[0]): # val_trials[0] will be the array of valid trial indices --> if it's empty, leave!
    warnings.warn('this condition is not valid');
    return;

  # get the phase relative to the stimulus
  ph_rel_stim, stim_ph, resp_ph, all_tf = hf.get_true_phase(data, val_trials, dir, psth_binWidth, stimDur);
  # compute the fourier amplitudes
  psth_val, _ = hf.make_psth(psth_binWidth, stimDur, data['spikeTimes'][val_trials])
  _, rel_amp, full_fourier = hf.spike_fft(psth_val, all_tf)

  # now plot!
  f, ax = plt.subplots(3, 2, figsize=(20, 30))

  relSpikes = data['spikeTimes'][val_trials];
  colors = cm.rainbow(np.linspace(0, 1, len(val_trials[0])))

  # spike raster - trial-by-trial
  # only works for SINGLE GRATINGS
  # draw the beginning of each cycle for each trial
  ax = plt.subplot(3, 1, 1)
  for i in range(len(relSpikes)):
      ax.scatter(relSpikes[i], i*np.ones_like(relSpikes[i]), color=colors[i]);
      stimPh = stim_ph[i];
      stimPeriod = np.divide(1.0, all_tf[i]);
      # i.e. at what point during the trial (in s) does the stimulus component first begin a cycle?
      firstPh0 = hf.first_ph0(stimPh, all_tf[i])[1];

      for j in range(len(all_tf[i])):
          allPh0 = [stimPeriod[j]*np.arange(-1, all_tf[i][j]) + firstPh0[j]];
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

  # PSTH - per trial, but folded over N cycles
  # only works for SINGLE GRATINGS
  ax = plt.subplot(3, 1, 2)

  for i in range(len(relSpikes)):
      _, bin_edges, psth_norm = hf.fold_psth(relSpikes[i], all_tf[i], stim_ph[i], cycle_fold, n_bins_fold);
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

  # response phase - without accounting for stimulus phase
  ax = plt.subplot(3, 2, 5, projection='polar')
  ax.scatter(np.radians(resp_ph), rel_amp, color=colors, clip_on=False);
  ax.set_title('Stimulus-blind')
  ax.set_ylim(auto='True')
  polar_ylim = ax.get_ylim();

  # response phase - relative to stimulus phase
  ax = plt.subplot(3, 2, 6, projection='polar')
  ax.scatter(np.radians(ph_rel_stim), rel_amp, color=colors, clip_on=False);
  ax.set_ylim(polar_ylim);
  ax.set_title('Stimulus-accounted');

  f.subplots_adjust(wspace=0.2, hspace=0.25);
  f.suptitle('Cell %d: disp %d, con %.2f, sf %.2f' % (which_cell, allDisps[disp], allCons[con], allSfs[sf]));

  saveName = "/cell_%03d_d%dsf%dcon%d_phase.pdf" % (which_cell, disp, sf, con);
  save_loc = save_base + "cell_%03d/" % which_cell;
  full_save = os.path.dirname(str(save_loc));
  if not os.path.exists(full_save):
    os.makedirs(full_save)
  pdfSv = pltSave.PdfPages(full_save + saveName);
  pdfSv.savefig(f);
  plt.close(f);
  pdfSv.close();

def batch_phase_by_cond(cell_num, disp, cons=[], sfs=[], dp=dataPath):
  ''' must specify dispersion (one value)
      if cons = [], then get/plot all valid contrasts for the given dispersion
      if sfs = [], then get/plot all valid contrasts for the given dispersion
  '''
  dataList = hf.np_smart_load(str(dp + expName));
  fileName = dataList['unitName'][cell_num-1];
  
  cellStruct = hf.np_smart_load(str(dp + fileName + '_sfm.npy'));

  # prepare the valid stim parameters by condition in case needed
  resp, stimVals, val_con_by_disp, validByStimVal, mdRsp = hf.tabulate_responses(cellStruct);

  # gather the conditions we need so that we can index properly
  valDisp = validByStimVal[0];
  valCon = validByStimVal[1];
  valSf = validByStimVal[2];

  allDisps = stimVals[0];
  allCons = stimVals[1];
  allSfs = stimVals[2];

  if cons == []: # then get all valid cons for this dispersion
    cons = val_con_by_disp[disp];
  if sfs == []: # then get all valid sfs for this dispersion
    sfs = valSf; # just take all sfs - if a condition isn't valid, phase_by_cond will just return!

  for c in cons:
    for s in sfs:
      print('analyzing cell %d, dispersion %d, contrast %d, sf %d\n' % (cell_num, disp, c, s));
      phase_by_cond(cell_num, cellStruct, disp, c, s);    

if __name__ == '__main__':

    if len(sys.argv) < 3:
      print('uhoh...you need at least two arguments here');
      exit();

    cell_num = int(sys.argv[1]);
    disp = int(sys.argv[2]);
    print('Running cell %d, dispersion %d' % (cell_num, disp+1));

    batch_phase_by_cond(cell_num, disp);
