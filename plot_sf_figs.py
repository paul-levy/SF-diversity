import os
import sys
import numpy as np
import matplotlib
import matplotlib.cm as cm
matplotlib.use('Agg') # to avoid GUI/cluster issues...
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
from matplotlib.ticker import FuncFormatter
import seaborn as sns
sns.set(style='ticks')
from scipy.stats import poisson, nbinom
from scipy.stats.mstats import gmean
import time

import helper_fcns as hf
from helper_fcns_sfBB import get_resp_str
import model_responses_pytorch as mrpt
sys.path.insert(0, 'LGN/sach/');
import helper_fcns_sach as hfs

import warnings
warnings.filterwarnings('once');

import pdb

f1_expCutoff = 2; # if 1, then all but V1_orig/ are allowed to have F1; if 2, then altExp/ is also excluded

# See plot_sf_figs_breakAxisAttempt.py to continue attempt at making a proper break in the x-axis for zero frequency, rather than just pretending it's a slightly lower freq.

def prepare_sfs_plot_sach(data_loc, expDir, cellNum, rvcAdj, rvcAdjSigned, rvcMod, fLname, rvcBase, phBase, phAmpByMean, respVar, joint, old_refprmm):
  # helper function called in plot_sfs

  descrFits = hf.np_smart_load(data_loc + fLname);
  descrFits = descrFits[cellNum-1]; # just get this cell
  descrParams = descrFits['params'];

  if rvcAdj == 1:
    vecF1 = 1 if rvcAdjSigned==-1 else 0
    dir = 1 if rvcAdjSigned==1 else None # we dont' have pos/neg phase if vecF1
    phAdj = 1;
  elif rvcAdj == 0: # we'll never reach here...Sach is LGN data, always do rvcAdj/phAdj
    vecF1 = None
    dir = None;
    phAdj = 0;

  rvcFits = hf.np_smart_load(data_loc + hf.rvc_fit_name(rvcBase, rvcMod, vecF1=vecF1, dir=dir));
  rvcFits = rvcFits[cellNum-1];

  allData = hf.np_smart_load(data_loc + 'sachData.npy');
  cellStruct = allData[cellNum-1];
  data = cellStruct['data'];

  resps, stimVals, _ = hfs.tabulateResponses(data, phAdjusted=phAdj);
  # all responses on log ordinate (y axis) should be baseline subtracted

  all_cons = stimVals[0];
  all_sfs = stimVals[1];
  n_v_cons = len(all_cons);
  v_cons = all_cons; # for sach, all cons are valid cons (no mixtures means no different total contrast values)

  # Unpack responses
  f1 = resps[1]; # power at fundamental freq. of stimulus
  respMean, respSem = f1['mean'], f1['sem'];
  baseline_resp = 0; # these are F1 responses!

  ref_params = None; # only applies for d-DoG-S fits (don't use on Sach data)
  ref_rc_val = None; # only applied for DoG, joint=5 (not used as of 22.08.08)

  return respMean, respSem, baseline_resp, n_v_cons, v_cons, all_cons, all_sfs, descrParams, ref_params, ref_rc_val;

def prepare_sfs_plot(data_loc, expDir, cellNum, rvcAdj, rvcAdjSigned, rvcMod, fLname, rvcBase, phBase, phAmpByMean, respVar, joint, disp, old_refprm, mod_fits=None, isBB=False, normType=1, lgnFrontEnd=0, excType=1, lossType=1, dgNormFunc=True, f1_expCutoff=f1_expCutoff):
  # helper function called in plot_sfs
  # --- if mod_resps/mod_dw are not None, then we also organize/return the model responses

  expName = hf.get_datalist(expDir, force_full=1, new_v1=True);
  dataList = hf.np_smart_load(data_loc + expName);

  cellName = dataList['unitName'][cellNum-1];
  try:
    overwriteExpName = dataList['expType'][cellNum-1];
  except:
    overwriteExpName = None;
  expInd   = hf.get_exp_ind(data_loc, cellName, overwriteExpName)[0];
  if expInd <= f1_expCutoff: # if expInd <= 2, then there cannot be rvcAdj, anyway!
    rvcAdj = 0; # then we'll load just below!
  stimDur = hf.get_exp_params(expInd).stimDur;

  descrFits = hf.np_smart_load(data_loc + fLname);

  if rvcAdj == 1:
    vecF1 = 1 if rvcAdjSigned==-1 else 0
    dir = 1 if rvcAdjSigned==1 else None # we dont' have pos/neg phase if vecF1
    rvcFits = hf.np_smart_load(data_loc + hf.rvc_fit_name(rvcBase, modNum=rvcMod, dir=dir, vecF1=vecF1)); # i.e. positive
    force_baseline = False; # plotting baseline will depend on F1/F0 designation
  elif rvcAdj == 0:
    vecF1 = None
    rvcFits = hf.np_smart_load(data_loc + rvcBase + '_f0_NR.npy');
    force_baseline = True;
  rvcFits = rvcFits[cellNum-1];
  expData  = hf.np_smart_load(str(data_loc + cellName + '_sfm.npy'));
  trialInf = expData['sfm']['exp']['trial'];
  descrParams = descrFits[cellNum-1]['params'];
  f1f0rat = hf.compute_f1f0(trialInf, cellNum, expInd, data_loc, descrFitName_f0=fLname)[0];

  # more tabulation - stim vals, organize measured responses
  overwriteSpikes = None;
  _, stimVals, val_con_by_disp, validByStimVal, _ = hf.tabulate_responses(expData, expInd);
  rvcModel = hf.get_rvc_model();
  if rvcAdj == 0:
    rvcFlag = '_f0';
    force_dc = True;
  else:
    rvcFlag = '' if phAmpByMean==0 else '_phAmpMean';
    force_dc = False;
  if expDir == 'LGN/':
    force_f1 = True;
  else:
    force_f1 = False;
  rvcSuff = hf.rvc_mod_suff(rvcMod);
  rvcBase = '%s%s' % (rvcBase, rvcFlag);

  if rvcAdjSigned==1 and phAmpByMean: # i.e. phAdv correction
      #phBase = 'phaseAdvanceFits%s_220531' % (hpc_str) if expDir=='LGN/' else 'phaseAdvanceFits%s_220609' % (hpc_str)
      print(phBase);
      phAdvFits = hf.np_smart_load(data_loc + hf.phase_fit_name(phBase, dir=1));
      all_opts = phAdvFits[cellNum-1]['params'];

  # NOTE: We pass in the rvcFits where rvcBase[name] goes, and use -1 in rvcMod to indicate that we've already loaded the fits
  spikes_rate, which_measure = hf.get_adjusted_spikerate(trialInf, cellNum, expInd, data_loc, rvcFits, rvcMod=-1, descrFitName_f0 = fLname, baseline_sub=False, force_dc=force_dc, force_f1=force_f1, return_measure=True, vecF1=vecF1);
  # let's also get the baseline
  if force_baseline or (f1f0rat < 1 and expDir != 'LGN/'): # i.e. if we're in LGN, DON'T get baseline, even if f1f0 < 1 (shouldn't happen)
    baseline_resp = hf.blankResp(trialInf, expInd, spikes=spikes_rate, spksAsRate=True)[0];
  else:
    baseline_resp = int(0);

  # now get the measured responses
  divFactor = 1;
  #_, _, respOrg, respAll = hf.organize_resp(spikes_rate, trialInf, expInd);
  _, _, respOrg, respAll = hf.organize_resp(spikes_rate, trialInf, expInd, respsAsRate=True);
  if rvcAdjSigned==1 and phAmpByMean and (force_f1 or f1f0rat>1):
    try:
      respMean = hf.organize_phAdj_byMean(trialInf, expInd, all_opts, stimVals, val_con_by_disp);
      # TEMP/TODO/HORRIBLE 
      divFactor = 1/stimDur;
    except: # why would it fail? Only when there isn't a trial for each condition - in which case, these are disregarded cells...
      respMean = respOrg;
  else:
      respMean = respOrg;
  respStd = np.nanstd(respAll, -1); # take std of all responses for a given condition
  # compute SEM, too
  findNaN = np.isnan(respAll);
  nonNaN  = np.sum(findNaN == False, axis=-1);
  respSem = np.nanstd(respAll, -1) / np.sqrt(nonNaN);
  # pick which measure of response variance
  if respVar == 1:
    respVar = respSem;
  else:
    respVar = respStd;

  all_disps = stimVals[0];
  all_cons = stimVals[1];
  all_sfs = stimVals[2];

  nCons = len(all_cons);
  nSfs = len(all_sfs);
  nDisps = len(all_disps);

  v_cons = val_con_by_disp[disp];
  n_v_cons = len(v_cons);

  if old_refprm:
    ref_params = descrParams[disp, v_cons[-1]] if joint>0 else None; # the reference parameter is the highest contrast for that dispersion
    ref_rc_val = ref_params[2] if joint>0 else None; # will be used iff joint==5 (center radius at highest con)
  else:
    if joint<10:
      ref_params = np.array([np.nanmin(descrParams[disp, v_cons, 1]), 1]) if joint>0 else None;
    else: # if joint==10, then we allow xc2 NEQ xc1
      ref_params = np.array([np.nanmin(descrParams[disp, v_cons, 1]), descrFits[cellNum-1]['paramList'][disp][4]]); # param[4] is the xc2 ratio, rel. xc1

    ref_rc_val = None;

  # IF applicable, organize model responses
  # -- default to modByCond = None
  modByCond = None; mod_to_sub = None;
  if mod_fits is not None:
    # define the responses to organize the datawrapper with
    # get the correct, adjusted F1 response
    if expInd > f1_expCutoff and which_measure == 1:
      respOverwrite = hf.adjust_f1_byTrial(trialInf, expInd);
    else:
      respOverwrite = None;
    
    #divFactor = stimDur if which_measure == 0 else 1;
    #divFactor = 1/divFactor
    #divFactor = 1;
    mod_prms = mod_fits[cellNum-1][get_resp_str(which_measure)]['params']
    use_mod = mrpt.sfNormMod(mod_prms, expInd=expInd, excType=excType, normType=normType, lossType=lossType, lgnFrontEnd=lgnFrontEnd, toFit=False, dgNormFunc=dgNormFunc);
    mod_to_sub = np.maximum(0, use_mod.noiseLate.detach().numpy()/divFactor); # baseline 
    mod_dw = mrpt.dataWrapper(trialInf, respMeasure=which_measure, expInd=expInd, respOverwrite=respOverwrite);
    mod_resps = use_mod.forward(mod_dw.trInf, respMeasure=which_measure).detach().numpy();
    if not isBB:
      if which_measure == 1: # make sure the blank components have a zero response (we'll do the same with the measured responses)
        blanks = np.where(mod_dw.trInf['con']==0);
        mod_resps[blanks] = 0;
        # next, sum up across components
        mod_resps = np.sum(mod_resps, axis=1);
      # finally, make sure this fills out a vector of all responses (just have nan for non-modelled trials)
      nTrialsFull = len(trialInf['num']);
      modResps_full = np.nan * np.zeros((nTrialsFull, ));
      modResps_full[mod_dw.trInf['num']] = mod_resps;
      # organize responses so that we can package them for evaluating varExpl...
      # modByCond is [nDisp, nSf, nCon]
      _, _, modByCond, _ = hf.organize_resp(np.divide(modResps_full, divFactor), trialInf, expInd); 
    else: # isBB
      print('FIX THIS --> or just use the other one and only do this for the other ones (i.e. altExp & V1 only')


  return respMean, respSem, baseline_resp, n_v_cons, v_cons, all_cons, all_sfs, descrParams, ref_params, ref_rc_val, (modByCond, mod_to_sub);

def plot_sfs(ax, i, j, cellNum, expDir, rvcBase, descrBase, descrMod, joint, rvcAdj, phBase=None, descrLoss=2, rvcMod=1, phAmpByMean=1, respVar=1, plot_sem_on_log=1, disp=0, forceLog=1, subplot_title=False, specify_ticks=True, old_refprm=False, fracSig=1, incl_legend=False, nrow=2, subset_cons=None, minToPlot = 1, despine_offset=2, incl_zfreq=True, use_tex=True, mod_fits=None, normType=1, lgnFrontEnd=0, excType=1, lossType=1, dgNormFunc=True, incl_data=True):
  # --- IF mod_fits is not None, then we've passed in the model responses (as generated from mrpt.sfNormMod.forward()

  # Set up, load some files
  x_lblpad=6; y_lblpad=8;

  loc_base = os.getcwd() + '/';

  data_loc = loc_base + expDir + 'structures/';
  save_loc = loc_base + expDir + 'figures/';

  isSach = True if 'sach' in expDir else False;
  isBB = True if 'BB' in expDir else False;

  rvcAdjSigned = rvcAdj;
  rvcAdj = np.abs(rvcAdj);

  if not forceLog:
    minToPlot = -2.5; # it's ok to have negative..

  modStr  = hf.descrMod_name(descrMod)
  fLname  = hf.descrFit_name(descrLoss, descrBase=descrBase, modelName=modStr, joint=joint, phAdj=1 if rvcAdjSigned==1 else None);
  # set the save directory to save_loc, then create the save directory if needed
  subDir = fLname.replace('Fits', '').replace('.npy', '');
  save_loc = str(save_loc + subDir + '/');

  if not os.path.exists(save_loc):
    os.makedirs(save_loc);

  # call the necessary function
  if not isSach and not isBB: # this works for altExp, V1, V1_orig, LGN
    respMean, respSem, baseline_resp, n_v_cons, v_cons, all_cons, all_sfs, descrParams, ref_params, ref_rc_val, mod_resps = prepare_sfs_plot(data_loc, expDir, cellNum, rvcAdj, rvcAdjSigned, rvcMod, fLname, rvcBase, phBase, phAmpByMean, respVar, joint, disp, old_refprm, mod_fits=mod_fits, normType=normType, lgnFrontEnd=lgnFrontEnd, excType=excType, lossType=lossType, dgNormFunc=dgNormFunc);
  else:
    if isSach:
      respMean, respSem, baseline_resp, n_v_cons, v_cons, all_cons, all_sfs, descrParams, ref_params, ref_rc_val, mod_resps = prepare_sfs_plot_sach(data_loc, expDir, cellNum, rvcAdj, rvcAdjSigned, rvcMod, fLname, rvcBase, phBase, phAmpByMean, respVar, joint, old_refprm, mod_fits=mod_fits, normType=normType, lgnFrontEnd=lgnFrontEnd, excType=excType, lossType=lossType, dgNormFunc=dgNormFunc);
    else: # only here if isBB
      # TODO: not written as of 22.08.08
      pass;

  # Getting ready to plot
  if all_sfs[0]==0:
    hasZfreq = True;
    #sfs_plot = np.hstack((np.linspace(0,all_sfs[1],10), np.logspace(np.log10(all_sfs[1]), np.log10(all_sfs[-1]), 100)));
    sfs_plot = np.logspace(np.log10(all_sfs[1]), np.log10(all_sfs[-1]), 100);
  else:
    hasZfreq = False;
    sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);
  # this was maxResp across all conditions --> but this is only single grats
  #maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)])));
  #minResp = np.min(np.min(np.min(respMean[~np.isnan(respMean)]))); 
  # --- disp=0
  maxResp = np.nanmax(respMean[0])
  minResp = np.nanmin(respMean[0])
  # -- decide which contrasts we'll plot...
  if subset_cons is not None:
    if len(subset_cons)==2: # then it's start index, how many to skip
      to_plot = np.arange(subset_cons[0], n_v_cons, subset_cons[1]);
    else: # then we've passed in the list to plot
      to_plot = subset_cons;
  else:
    to_plot = range(n_v_cons);

  # Plot!
  for c in reversed(range(n_v_cons)):

      if not np.in1d(c, to_plot):
        continue;
      # also make sure that we have parameters for this data
      prms_curr = descrParams[c] if isSach else descrParams[disp, v_cons[c]];
      if np.any(np.isnan(prms_curr)):
        continue;

      if isSach:
        #plot_resp = respMean[c];
        v_sfs = np.where(all_sfs>0)[0];
        plot_resp = respMean[c, v_sfs];
      else:
        v_sfs = ~np.isnan(respMean[disp, :, v_cons[c]]);
        plot_resp = respMean[disp, v_sfs, v_cons[c]];

      col = [(n_v_cons-c-1)/float(n_v_cons), (n_v_cons-c-1)/float(n_v_cons), (n_v_cons-c-1)/float(n_v_cons)];
      col = np.sqrt(col);
      if forceLog == 1:
        if baseline_resp > 0: #is not None:
          to_sub = baseline_resp;
        else:
          to_sub = np.array(0);
        plot_resp = plot_resp - to_sub;
        baseline_resp_curr = 0; # don't plot the baseline, since we've subtracted off; and don't add to model resp
      else:
        to_sub = np.array(0);
        baseline_resp_curr = baseline_resp;

      curr_con = v_cons[c] if isSach else all_cons[v_cons[c]];

      if plot_sem_on_log:
        sem_curr = respSem[c, v_sfs] if isSach else respSem[disp, v_sfs, v_cons[c]];
        #sem_curr = respSem[c] if isSach else respSem[disp, v_sfs, v_cons[c]];
        # errbars should be (2,n_sfs)
        high_err = sem_curr; # no problem with going to higher values
        low_err = np.minimum(sem_curr, plot_resp-minToPlot-1e-2); # i.e. don't allow us to make the err any lower than where the plot will cut-off (incl. negatives)
        errs = np.vstack((low_err, high_err));
        curr_plot_sfs = all_sfs[v_sfs] if isSach else all_sfs[v_sfs];
        #curr_plot_sfs = all_sfs if isSach else all_sfs[v_sfs];
        if mod_fits is None or incl_data: # let's optionally exclude the data
          ax[i,j].errorbar(curr_plot_sfs[plot_resp>minToPlot], plot_resp[plot_resp>minToPlot], errs[:, plot_resp>minToPlot], fmt='o', clip_on=True, color=col);

        if hasZfreq and isSach and incl_zfreq: # then also plot the zero frequency data/model, but isolated...
          fake_sf = all_sfs[1]/10; # .../2, or something to make it even smaller
          ax[i,j].errorbar(fake_sf, respMean[c, 0], np.vstack((np.minimum(respSem[c,0], respMean[c,0]-minToPlot-1e-2), respSem[c,0])), fmt='o', clip_on=True, color=col);
          fake_sfs = np.geomspace(fake_sf/np.sqrt(1.5), fake_sf*np.sqrt(1.5), 25);
          descrResp = hf.get_descrResp(prms_curr, fake_sfs, descrMod, baseline=baseline_resp, fracSig=fracSig, ref_params=ref_params, ref_rc_val=ref_rc_val);
          plt_resp = descrResp-to_sub;
          ax[i,j].plot(fake_sfs[plt_resp>minToPlot], plt_resp[plt_resp>minToPlot], color=col, clip_on=True);

      if mod_fits is None:
        descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod, baseline=baseline_resp, fracSig=fracSig, ref_params=ref_params, ref_rc_val=ref_rc_val);
        plt_resp = descrResp-to_sub;
        sfs_plot_curr = sfs_plot
      else:
        modResp = mod_resps[0][disp, v_sfs, v_cons[c]];
        mod_to_sub = mod_resps[1] if to_sub!=0 else 0;
        plt_resp = modResp-mod_to_sub;
        sfs_plot_curr = all_sfs[v_sfs] # only evaluating at tested locs for now...
      if use_tex: # need to use \ for escape
        ax[i,j].plot(sfs_plot_curr[plt_resp>minToPlot], plt_resp[plt_resp>minToPlot], color=col, clip_on=True, label='%s\%%' % (str(int(100*np.round(curr_con, 2)))));
      else:
        ax[i,j].plot(sfs_plot_curr[plt_resp>minToPlot], plt_resp[plt_resp>minToPlot], color=col, clip_on=True, label='%s%%' % (str(int(100*np.round(curr_con, 2)))));

  ax[i,j].set_xlim((0.5*min(all_sfs), 1.2*max(all_sfs)));

  ax[i,j].set_xscale('log');
  if expDir == 'LGN/' or forceLog == 1: # we want double-log if it's the LGN!
    ax[i,j].set_yscale('log');
    #ax[i,j].set_ylim((minToPlot, 1.5*maxResp));
    if not specify_ticks or maxResp>90:
        ax[i,j].set_ylim((minToPlot, 300)); # common y axis for ALL plots
    else:
        ax[i,j].set_ylim((minToPlot, 1.2*maxResp));
    logSuffix = 'log_';
    #ax[i,j].set_aspect('equal'); # if both axes are log, must make equal scales!
    ax[i,j].axis('scaled'); # this works better for minimizing white space (as compared to set_aspect('equal'))
  else:
    ax[i,j].set_ylim((np.minimum(-5, minResp-5), 1.25*maxResp));
    logSuffix = '';

  pltd_sfs = all_sfs if isSach else all_sfs[v_sfs];
  for jj, axis in enumerate([ax[i,j].xaxis, ax[i,j].yaxis]):
      if forceLog:
        axis.set_major_formatter(FuncFormatter(lambda x,y: '%d' % x if x>=1 else '%.1f' % x)) # this will make everything in non-scientific notation!
      else:
        axis.set_major_formatter(FuncFormatter(lambda x,y: '%d' % x)) # this will make everything in non-scientific notation!
      if jj == 0 and specify_ticks: # i.e. x-axis
        core_ticks = np.array([1]);
        if np.min(pltd_sfs)<=0.2: # was 0.3
            core_ticks = np.hstack((0.1, core_ticks));
        if np.max(pltd_sfs)>=7:
            core_ticks = np.hstack((core_ticks, 10));
        axis.set_ticks(core_ticks)
        # really hacky, but allows us to put labels at 0.3/3 cpd, format them properly, and not add any extra labels
        inter_val = 3;
        axis.set_minor_formatter(FuncFormatter(lambda x,y: '%d' % x if np.square(x-inter_val)<1e-3 else '%.1f' % x if np.square(x-inter_val/10)<1e-3 else '')) # this will make everything in non-scientific notation!
        axis.set_tick_params(which='minor', pad=5.5); # Determined by trial and error: make the minor/major align??
      else:
        #axis.set_tick_params(labelleft=True); 
        if jj == 1 and specify_ticks and (forceLog or expDir=='LGN/'): # y axis
          core_ticks = np.array([1, 10, ]);
          if maxResp>=90:
              core_ticks = np.hstack((core_ticks, 100));
          axis.set_ticks(core_ticks)
          inter_val = 3; # put labels (minor) at 3/30 spks/s
          axis.set_minor_formatter(FuncFormatter(lambda x,y: '%d' % x if np.square(x-inter_val*10)<1e-5 else '%d' % x if np.square(x-inter_val)<1e-5 else '')) # this will make everything in non-scientific notation!
          axis.set_tick_params(which='minor', pad=5.5); # Determined by trial and error: make the minor/major align??
 
  lbl_str = '';
  if j==0:
    ax[i,j].set_ylabel('Response %s(spikes/s)' % lbl_str, labelpad=y_lblpad);
  if i==nrow-1: # 
    ax[i,j].set_xlabel('Spatial frequency (c/deg)', labelpad=x_lblpad); 

  if subplot_title:
    if mod_fits is None:
      ax[i,j].set_title('%s%02d j%d' % (expDir, cellNum, joint));
    else:
      ax[i,j].set_title('%s' % hf.fitList_name('', fitType=normType, excType=excType, lossType=lossType, lgnType=lgnFrontEnd, dgNormFunc=dgNormFunc).replace('.npy', '').replace('_d', 'd'));
  if incl_legend:
    ax[i,j].legend(fontsize='x-small'); 
  
  # Set ticks out, remove top/right axis, put ticks only on bottom/left
  sns.despine(ax=ax[i,j], offset=despine_offset, trim=False); 

  return ax[i,j];

def plot_rvcs(ax, i, j, cellNum, expDir, rvcBase, descrBase, descrMod, joint, rvcAdj, phBase=None, descrLoss=2, rvcMod=1, phAmpByMean=1, respVar=1, plot_sem_on_log=1, disp=0, forceLog=1, subplot_title=False, specify_ticks=True, old_refprm=False, fracSig=1, incl_legend=False, nrow=2, subset_sfs=None, minToPlot = 1, despine_offset=2, incl_zfreq=True, conMult=1, alph=1, color='k'):

  # Set up, load some files
  x_lblpad=6; y_lblpad=8;

  loc_base = os.getcwd() + '/';

  data_loc = loc_base + expDir + 'structures/';
  save_loc = loc_base + expDir + 'figures/';

  isSach = True if 'sach' in expDir else False;
  isBB = True if 'BB' in expDir else False;

  whichArea = 'LGN' if 'LGN' in expDir else 'V1';

  rvcAdjSigned = rvcAdj;
  rvcAdj = np.abs(rvcAdj);

  modStr  = hf.descrMod_name(descrMod)
  fLname  = hf.descrFit_name(descrLoss, descrBase=descrBase, modelName=modStr, joint=joint, phAdj=1 if rvcAdjSigned==1 else None);
  # set the save directory to save_loc, then create the save directory if needed
  subDir = fLname.replace('Fits', '').replace('.npy', '');
  save_loc = str(save_loc + subDir + '/');

  if not os.path.exists(save_loc):
    os.makedirs(save_loc);

  # call the necessary function
  if not isSach and not isBB: # this works for altExp, V1, V1_orig, LGN
    respMean, respSem, baseline_resp, n_v_cons, v_cons, all_cons, all_sfs, descrParams, ref_params, ref_rc_val = prepare_sfs_plot(data_loc, expDir, cellNum, rvcAdj, rvcAdjSigned, rvcMod, fLname, rvcBase, phBase, phAmpByMean, respVar, joint, disp, old_refprm);
  else:
    if isSach:
      respMean, respSem, baseline_resp, n_v_cons, v_cons, all_cons, all_sfs, descrParams, ref_params, ref_rc_val = prepare_sfs_plot_sach(data_loc, expDir, cellNum, rvcAdj, rvcAdjSigned, rvcMod, fLname, rvcBase, phBase, phAmpByMean, respVar, joint, old_refprm);
    else: # only here if isBB
      # TODO: not written as of 22.08.08
      pass;

  # Getting ready to plot
  maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)]))); 
  # -- decide which SFs we'll plot...
  n_v_sfs = len(all_sfs);
  # --- assumes that all SFs are used at all contrasts
  v_sf_inds = np.arange(n_v_sfs);
  if subset_sfs is not None:
    if len(subset_sfs)==2: # then it's start index, how many to skip
      to_plot = np.arange(subset_sfs[0], n_v_cons, subset_sfs[1]);
    else: # then we've passed in the list to plot
      to_plot = subset_sfs;
  else:
    to_plot = range(n_v_sfs);

  if np.isnan(i) or np.isnan(j):
    curr_ax = ax[i] if np.isnan(j) else ax[j];
  else:
    curr_ax = ax[i,j]

  # Plot!
  curr_sf_pltd = 0;
  
  wids = np.linspace(2,0.5,len(to_plot))
  sizes = np.linspace(5,3,len(to_plot));
  
  for sf in zip(range(n_v_sfs)):

      if not np.in1d(sf, to_plot):
        continue;

      wid, size = wids[curr_sf_pltd], sizes[curr_sf_pltd];

      sf_ind = v_sf_inds[sf];
      v_cons = ~np.isnan(respMean[disp, sf_ind, :]);
      n_cons = sum(v_cons);

      sf_str = str(np.round(all_sfs[sf_ind], 2));
      # plot data
      plot_resp = respMean[disp, sf_ind, v_cons];
      if forceLog == 1:
        if baseline_resp > 0: #is not None:
          to_sub = baseline_resp;
        else:
          to_sub = np.array(0);
        plot_resp = plot_resp - to_sub;
      # make sure we have error bars, too
      sem_curr = respSem[disp, sf_ind, v_cons];
      # errbars should be (2,n_cons)
      high_err = sem_curr; # no problem with going to higher values
      low_err = np.minimum(sem_curr, plot_resp-minToPlot-1e-2); # i.e. don't allow us to make the err any lower than where the plot will cut-off (incl. negatives)
      errs = np.vstack((low_err, high_err));

      curr_ax.errorbar(conMult*all_cons[v_cons][plot_resp>minToPlot], plot_resp[plot_resp>minToPlot], errs[:, plot_resp>minToPlot], fmt='o', color=color, elinewidth=wid, \
                                    clip_on=False, markersize=size, label='%s c/deg' % sf_str, alpha=alph, markerfacecolor='w', markeredgecolor=color, mew=1);
      if subplot_title:
          curr_ax.set_title('D%d: RVC data' % (disp+1));

      cons = all_cons[v_cons];
      resps_curr = np.array([hf.get_descrResp(descrParams[disp, vc], all_sfs[sf_ind], descrMod, baseline=baseline_resp, fracSig=fracSig, ref_params=ref_params, ref_rc_val=ref_rc_val) for vc in np.where(v_cons)[0]]) - to_sub;
      curr_ax.plot(conMult*all_cons[v_cons][resps_curr>minToPlot], resps_curr[resps_curr>minToPlot], color=color, \
                   clip_on=False, linestyle='-', marker=None, linewidth=wid);
      if subplot_title:
        curr_ax.set_title('D%d: RVC from SF fit' % (disp+1));

      curr_sf_pltd += 1; # add one to how many curves we've plotted...

  # Set ticks out, remove top/right axis, put ticks only on bottom/left
  sns.despine(ax=curr_ax, offset=despine_offset, trim=False); 
  curr_ax.set_xscale('log');
  curr_ax.set_yscale('log');

  for jj, axis in enumerate([curr_ax.xaxis, curr_ax.yaxis]):
    if jj==0: # i.e. x-axis
      core_ticks = np.array([0.1, 1])
      axis.set_ticks(core_ticks)
      inter_val_curr = 3;
      # really hacky, but allows us to put labels at 3/30% contrast, format them properly, and not add any extra labels
      axis.set_minor_formatter(FuncFormatter(lambda x,y: '%.1f' % x if np.square(x-inter_val_curr/10)<1e-5 else '%.2f' % x if np.square(x-inter_val_curr/100)<1e-5 else '')) # this will make everything in non-scientific notation!
    if jj==1 and specify_ticks: # i.e. y-axis
      core_ticks = np.array([1, 10]);
      if maxResp>=90:
          core_ticks = np.hstack((core_ticks, 100));
      axis.set_ticks(core_ticks)
      inter_val = 3;
      # really hacky, but allows us to put labels at 3/30% contrast, format them properly, and not add any extra labels
      axis.set_minor_formatter(FuncFormatter(lambda x,y: '%d' % x if np.square(x-inter_val*10)<1e-5 else '%d' % x if np.square(x-inter_val)<1e-5 else '')) # this will make everything in non-scientific notation!

    if conMult == 100:
      axis.set_major_formatter(FuncFormatter(lambda x,y: '%d' % x if x>=1 else '%.1f' % x)) # this will make everything in non-scientific notation!
    else:
      axis.set_major_formatter(FuncFormatter(lambda x,y: '%d' % x if x>=1 else '%.1f' % x if x>=0.1 else '%.2f' % x)) # this will make
    axis.set_tick_params(which='minor', pad=5.5); # Determined by trial and error: make the minor/major align??

  lbl_str = '';
  if j==0:
    curr_ax.set_ylabel('Response %s(spikes/s)' % lbl_str, labelpad=y_lblpad);
  if i==nrow-1 or nrow==1:
    con_str = ' (\%)' if conMult==100 else ''
    curr_ax.set_xlabel('Contrast%s' % con_str, labelpad=x_lblpad); 

  if subplot_title:
      curr_ax.set_title('%s%02d j%d' % (expDir, cellNum, joint));
  if incl_legend:
    curr_ax.legend(fontsize='x-small'); 
  
  return curr_ax;
