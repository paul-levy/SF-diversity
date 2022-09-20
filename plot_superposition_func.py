import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # to avoid GUI/cluster issues...
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import matplotlib.animation as anim
import matplotlib.cm as cm
import seaborn as sns
import itertools
import helper_fcns as hf
import helper_fcns_sfBB as hf_sf
import model_responses as mod_resp
import model_responses_pytorch as mrpt
import scipy.optimize as opt
from scipy.stats.mstats import gmean as geomean
import time

import pdb

import sys # so that we can import model_responses (in different folder)

####### TO USE ########
### akin to other parallelized functions, call 'python3.6 plot_superposition_func.py -181 LGN/' for example

import warnings
warnings.filterwarnings('once');

def plot_save_superposition(which_cell, expDir, use_mod_resp=0, fitType=2, excType=1, useHPCfit=1, conType=None, lgnFrontEnd=None, force_full=1, f1_expCutoff=2, to_save=1):

  if use_mod_resp == 2:
    rvcAdj   = -1; # this means vec corrected F1, not phase adjustment F1...
    _applyLGNtoNorm = 0; # don't apply the LGN front-end to the gain control weights
    recenter_norm = 1;
    newMethod = 1; # yes, use the "new" method for mrpt (not that new anymore, as of 21.03)
    lossType = 1; # sqrt
    _sigmoidSigma = 5;

  basePath = os.getcwd() + '/'
  if 'pl1465' in basePath or useHPCfit:
    loc_str = 'HPC';
  else:
    loc_str = '';

  rvcName = 'rvcFits%s_220531' % loc_str if expDir=='LGN/' else 'rvcFits%s_220718' % loc_str
  rvcFits = None; # pre-define this as None; will be overwritten if available/needed
  if expDir == 'altExp/': # we don't adjust responses there...
    rvcName = None;
  phAdvName = 'phaseAdvanceFits%s_220531' % loc_str if expDir=='LGN/' else 'phaseAdvanceFits%s_220718' % loc_str;
  dFits_base = 'descrFits%s_220609' % loc_str if expDir=='LGN/' else 'descrFits%s_220721' % loc_str
  if use_mod_resp == 1:
    rvcName = None; # Use NONE if getting model responses, only
    if excType == 1:
      fitBase = 'fitList_200417';
    elif excType == 2:
      fitBase = 'fitList_200507';
    lossType = 1; # sqrt
    fitList_nm = hf.fitList_name(fitBase, fitType, lossType=lossType);
  elif use_mod_resp == 2:
    rvcName = None; # Use NONE if getting model responses, only
    if excType == 1:
      fitBase = 'fitList%s_210308_dG' % loc_str
      if recenter_norm:
        #fitBase = 'fitList%s_pyt_210312_dG' % loc_str
        fitBase = 'fitList%s_pyt_210331_dG' % loc_str
    elif excType == 2:
      fitBase = 'fitList%s_pyt_210310' % loc_str
      if recenter_norm:
        #fitBase = 'fitList%s_pyt_210312' % loc_str
        fitBase = 'fitList%s_pyt_210331' % loc_str
    fitList_nm = hf.fitList_name(fitBase, fitType, lossType=lossType, lgnType=lgnFrontEnd, lgnConType=conType, vecCorrected=-rvcAdj);

  # ^^^ EDIT rvc/descrFits/fitList names here; 

  ############
  # Before any plotting, fix plotting paramaters
  ############
  plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/paul_plt_style.mplstyle');
  from matplotlib import rcParams
  rcParams['font.size'] = 20;
  rcParams['pdf.fonttype'] = 42 # should be 42, but there are kerning issues
  rcParams['ps.fonttype'] = 42 # should be 42, but there are kerning issues
  rcParams['lines.linewidth'] = 2.5;
  rcParams['axes.linewidth'] = 1.5;
  rcParams['lines.markersize'] = 8; # this is in style sheet, just being explicit
  rcParams['lines.markeredgewidth'] = 0; # no edge, since weird tings happen then

  rcParams['xtick.major.size'] = 15
  rcParams['xtick.minor.size'] = 5; # no minor ticks
  rcParams['ytick.major.size'] = 15
  rcParams['ytick.minor.size'] = 0; # no minor ticks

  rcParams['xtick.major.width'] = 2
  rcParams['xtick.minor.width'] = 2;
  rcParams['ytick.major.width'] = 2
  rcParams['ytick.minor.width'] = 0

  rcParams['font.style'] = 'oblique';
  rcParams['font.size'] = 20;

  ############
  # load everything
  ############
  dataListNm = hf.get_datalist(expDir, force_full=force_full);
  descrFits_f0 = None;
  dLoss_num = 2; # see hf.descrFit_name/descrMod_name/etc for details
  if expDir == 'LGN/':
    rvcMod = 0; 
    dMod_num = 1;
    rvcDir = 1;
    vecF1 = 0;
  else:
    rvcMod = 1; # i.e. Naka-rushton (1)
    dMod_num = 3; # d-dog-s
    rvcDir = 1;
    if expDir == 'altExp/':
      vecF1 = 0 # was None?
    else:
      vecF1 = 0; # was previously 1, but now we do phAmp, not just vecF1

  dFits_mod = hf.descrMod_name(dMod_num)
  descrFits_name = hf.descrFit_name(lossType=dLoss_num, descrBase=dFits_base, modelName=dFits_mod, phAdj=1 if vecF1==0 else None);

  ## now, let it run
  dataPath = basePath + expDir + 'structures/'
  save_loc = basePath + expDir + 'figures/'
  save_locSuper = save_loc + 'superposition_220918/'
  if use_mod_resp == 1:
    save_locSuper = save_locSuper + '%s/' % fitBase

  dataList = hf.np_smart_load(dataPath + dataListNm);
  print('Trying to load descrFits at: %s' % (dataPath + descrFits_name));
  descrFits = hf.np_smart_load(dataPath + descrFits_name);
  if use_mod_resp == 1 or use_mod_resp == 2:
    fitList = hf.np_smart_load(dataPath + fitList_nm);
  else:
    fitList = None;

  if not os.path.exists(save_locSuper):
    os.makedirs(save_locSuper)

  cells = np.arange(1, 1+len(dataList['unitName']))

  zr_rm = lambda x: x[x>0];
  # more flexible - only get values where x AND z are greater than some value "gt" (e.g. 0, 1, 0.4, ...)
  zr_rm_pair = lambda x, z, gt: [x[np.logical_and(x>gt, z>gt)], z[np.logical_and(x>gt, z>gt)]];
  # zr_rm_pair = lambda x, z: [x[np.logical_and(x>0, z>0)], z[np.logical_and(x>0, z>0)]] if np.logical_and(x!=[], z!=[])==True else [], [];

  # here, we'll save measures we are going use for analysis purpose - e.g. supperssion index, c50
  curr_suppr = dict();

  ############
  ### Establish the plot, load cell-specific measures
  ############
  nRows, nCols = 6, 2;
  cellName = dataList['unitName'][which_cell-1];
  expInd = hf.get_exp_ind(dataPath, cellName)[0]
  S = hf.np_smart_load(dataPath + cellName + '_sfm.npy')
  expData = S['sfm']['exp']['trial'];

  # 0th, let's load the basic tuning characterizations AND the descriptive fit
  try:
    dfit_curr = descrFits[which_cell-1]['params'][0,-1,:]; # single grating, highest contrast
  except:
    dfit_curr = None;
  # - then the basics
  try:
    basic_names, basic_order = dataList['basicProgName'][which_cell-1], dataList['basicProgOrder']
    basics = hf.get_basic_tunings(basic_names, basic_order);
  except:
    try:
      # we've already put the basics in the data structure... (i.e. post-sorting 2021 data)
      basic_names = ['','','','',''];
      basic_order = ['rf', 'sf', 'tf', 'rvc', 'ori']; # order doesn't matter if they are already loaded
      basics = hf.get_basic_tunings(basic_names, basic_order, preProc=S, reducedSave=True)
    except:
      basics = None;

  ### TEMPORARY: save the "basics" in curr_suppr; should live on its own, though; TODO
  curr_suppr['basics'] = basics;

  try:
    oriBW, oriCV = basics['ori']['bw'], basics['ori']['cv'];
  except:
    oriBW, oriCV = np.nan, np.nan;
  try:
    tfBW = basics['tf']['tfBW_oct'];
  except:
    tfBW = np.nan;
  try:
    suprMod = basics['rfsize']['suprInd_model'];
  except:
    suprMod = np.nan;
  try:
    suprDat = basics['rfsize']['suprInd_data'];
  except:
    suprDat = np.nan;

  try:
    cellType = dataList['unitType'][which_cell-1];
  except:
    # TODO: note, this is dangerous; thus far, only V1 cells don't have 'unitType' field in dataList, so we can safely do this
    cellType = 'V1';


  ############
  ### compute f1f0 ratio, and load the corresponding F0 or F1 responses
  ############
  f1f0_rat = hf.compute_f1f0(expData, which_cell, expInd, dataPath, descrFitName_f0=descrFits_f0)[0];
  curr_suppr['f1f0'] = f1f0_rat;
  respMeasure = 1 if f1f0_rat > 1 else 0;

  if vecF1 == 1:
    # get the correct, adjusted F1 response
    if expInd > f1_expCutoff and respMeasure == 1:
      respOverwrite = hf.adjust_f1_byTrial(expData, expInd);
    else:
      respOverwrite = None;

  _, stimVals, val_con_by_disp, _, _ = hf.tabulate_responses(expData, expInd); # call just to get these values (not spikes/activity)
  respsPhAdv_mean_ref = None; # defaulting to None here -- if not None, then we've collected mean responses based on phAmp corr. on means (not trial by trial)

  if (respMeasure == 1 or expDir == 'LGN/') and expDir != 'altExp/' : # i.e. if we're looking at a simple cell, then let's get F1
    if vecF1 == 1:
      spikes_byComp = respOverwrite
      # then, sum up the valid components per stimulus component
      allCons = np.vstack(expData['con']).transpose();
      blanks = np.where(allCons==0);
      spikes_byComp[blanks] = 0; # just set it to 0 if that component was blank during the trial
      spikes = np.array([np.sum(x) for x in spikes_byComp]);
    else: # then we're doing phAdj -- however, we'll still get trial-by-trial estimates for variability estimates
    # ---- BUT [TODO] those trial-by-trial estimates are not adjusted on means (instead adj. by trial)
      rvcFits = hf.get_rvc_fits(dataPath, expInd, which_cell, rvcName=rvcName, rvcMod=rvcMod, direc=rvcDir, vecF1=vecF1);
      spikes, which_measure = hf.get_adjusted_spikerate(expData, which_cell, expInd, dataPath, rvcName=rvcFits, rvcMod=-1, baseline_sub=False, return_measure=1, vecF1=vecF1);
      # now, get the real mean responses we'll use (again, the above is just for variability)
      phAdvFits = hf.np_smart_load(dataPath + hf.phase_fit_name(phAdvName, dir=rvcDir));
      all_opts = phAdvFits[which_cell-1]['params'];
      try:
        respsPhAdv_mean_ref = hf.organize_phAdj_byMean(expData, expInd, all_opts, stimVals, val_con_by_disp);
      except:
        print('\n*******\nFailed!!!\n*******\n');
        pass; # this will fail IFF there isn't a trial for each condition - these cells are ignored in the analysis, anyway; in those cases, just give non-phAdj responses so that we don't fail
    rates = True if vecF1 == 0 else False; # when we get the spikes from rvcFits, they've already been converted into rates (in hf.get_all_fft)
    baseline = None; # f1 has no "DC", yadig?
  else: # otherwise, if it's complex, just get F0
    respMeasure = 0;
    spikes = hf.get_spikes(expData, get_f0=1, rvcFits=None, expInd=expInd);
    rates = False; # get_spikes without rvcFits is directly from spikeCount, which is counts, not rates!
    baseline = hf.blankResp(expData, expInd)[0]; # we'll plot the spontaneous rate
    # why mult by stimDur? well, spikes are not rates but baseline is, so we convert baseline to count (i.e. not rate, too)
    spikes = spikes - baseline*hf.get_exp_params(expInd).stimDur; 

  #print('###\nGetting spikes (data): rates? %d\n###' % rates);
  _, _, _, respAll = hf.organize_resp(spikes, expData, expInd, respsAsRate=rates); # only using respAll to get variance measures
  resps_data, _, _, _, _ = hf.tabulate_responses(expData, expInd, overwriteSpikes=spikes, respsAsRates=rates, modsAsRate=rates);

  if fitList is None:
    resps = resps_data; # otherwise, we'll still keep resps_data for reference
  elif fitList is not None: # OVERWRITE the data with the model spikes!
    if use_mod_resp == 1:
      curr_fit = fitList[which_cell-1]['params'];
      modResp = mod_resp.SFMGiveBof(curr_fit, S, normType=fitType, lossType=lossType, expInd=expInd, cellNum=which_cell, excType=excType)[1];
      if f1f0_rat < 1: # then subtract baseline..
        modResp = modResp - baseline*hf.get_exp_params(expInd).stimDur; 
      # now organize the responses
      resps, stimVals, val_con_by_disp, _, _ = hf.tabulate_responses(expData, expInd, overwriteSpikes=modResp, respsAsRates=False, modsAsRate=False);
    elif use_mod_resp == 2: # then pytorch model!
      resp_str = hf_sf.get_resp_str(respMeasure)
      curr_fit = fitList[which_cell-1][resp_str]['params'];
      model = mrpt.sfNormMod(curr_fit, expInd=expInd, excType=excType, normType=fitType, lossType=lossType, lgnFrontEnd=lgnFrontEnd, newMethod=newMethod, lgnConType=conType, applyLGNtoNorm=_applyLGNtoNorm)
      ### get the vec-corrected responses, if applicable
      if expInd > f1_expCutoff and respMeasure == 1:
        respOverwrite = hf.adjust_f1_byTrial(expData, expInd);
      else:
        respOverwrite = None;

      dw = mrpt.dataWrapper(expData, respMeasure=respMeasure, expInd=expInd, respOverwrite=respOverwrite); # respOverwrite defined above (None if DC or if expInd=-1)
      modResp = model.forward(dw.trInf, respMeasure=respMeasure, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm).detach().numpy();

      if respMeasure == 1: # make sure the blank components have a zero response (we'll do the same with the measured responses)
        blanks = np.where(dw.trInf['con']==0);
        modResp[blanks] = 0;
        # next, sum up across components
        modResp = np.sum(modResp, axis=1);
      # finally, make sure this fills out a vector of all responses (just have nan for non-modelled trials)
      nTrialsFull = len(expData['num']);
      modResp_full = np.nan * np.zeros((nTrialsFull, ));
      modResp_full[dw.trInf['num']] = modResp;

      if respMeasure == 0: # if DC, then subtract baseline..., as determined from data (why not model? we aren't yet calc. response to no stim, though it can be done)
        modResp_full = modResp_full - baseline*hf.get_exp_params(expInd).stimDur;

      # TODO: This is a work around for which measures are in rates vs. counts (DC vs F1, model vs data...)
      stimDur = hf.get_exp_params(expInd).stimDur;
      asRates = False;
      #divFactor = stimDur if asRates == 0 else 1;
      #modResp_full = np.divide(modResp_full, divFactor);
      # now organize the responses
      resps, stimVals, val_con_by_disp, _, _ = hf.tabulate_responses(expData, expInd, overwriteSpikes=modResp_full, respsAsRates=asRates, modsAsRate=asRates);

  predResps = resps[2];

  respMean = resps[0] if respsPhAdv_mean_ref is None else respsPhAdv_mean_ref; # equivalent to resps[0];
  respStd = np.nanstd(respAll, -1); # take std of all responses for a given condition
  # compute SEM, too
  findNaN = np.isnan(respAll);
  nonNaN  = np.sum(findNaN == False, axis=-1);
  respSem = np.nanstd(respAll, -1) / np.sqrt(nonNaN);

  ############
  ### first, fit a smooth function to the overall pred V measured responses
  ### --- from this, we can measure how each example superposition deviates from a central tendency
  ### --- i.e. the residual relative to the "standard" input:output relationship
  ############
  all_resps = respMean[1:, :, :].flatten() # all disp>0
  all_preds = predResps[1:, :, :].flatten() # all disp>0
  # a model which allows negative fits
  #         myFit = lambda x, t0, t1, t2: t0 + t1*x + t2*x*x;
  #         non_nan = np.where(~np.isnan(all_preds)); # cannot fit negative values with naka-rushton...
  #         fitz, _ = opt.curve_fit(myFit, all_preds[non_nan], all_resps[non_nan], p0=[-5, 10, 5], maxfev=5000)
  # naka rushton
  myFit = lambda x, g, expon, c50: hf.naka_rushton(x, [0, g, expon, c50]) 
  non_neg = np.where(all_preds>0) # cannot fit negative values with naka-rushton...
  try:
    if use_mod_resp == 1: # the reference will ALWAYS be the data -- redo the above analysis for data
      predResps_data = resps_data[2];
      respMean_data = resps_data[0];
      all_resps_data = respMean_data[1:, :, :].flatten() # all disp>0
      all_preds_data = predResps_data[1:, :, :].flatten() # all disp>0
      non_neg_data = np.where(all_preds_data>0) # cannot fit negative values with naka-rushton...
      fitz, _ = opt.curve_fit(myFit, all_preds_data[non_neg_data], all_resps_data[non_neg_data], p0=[100, 2, 25], maxfev=5000)
    else:
      # Add bounds, smarter initialization (22.09.18)
      max_resp, max_pred = np.nanmax(all_resps[non_neg]), np.nanmax(all_preds[non_neg]);
      bounds = np.vstack(((0,1.5*max_resp), (0.5, 4), (0,1.5*max_pred))).transpose(); # gain, c50 must be pos; exp. between 0.5-4
      init_gain = np.nanmax(all_resps[non_neg]);
      init_exp = hf.random_in_range([0.75, 2.5])[0];
      init_c50 = np.nanmedian(all_preds[non_neg]);
      fitz, _ = opt.curve_fit(myFit, all_preds[non_neg], all_resps[non_neg], p0=[init_gain, init_exp, init_c50], bounds=bounds, maxfev=5000)
    rel_c50 = np.divide(fitz[-1], np.max(all_preds[non_neg]));
  except:
    fitz = None;
    rel_c50 = -99;

  ############
  ### organize stimulus information
  ############
  all_disps = stimVals[0];
  all_cons = stimVals[1];
  all_sfs = stimVals[2];

  nCons = len(all_cons);
  nSfs = len(all_sfs);
  nDisps = len(all_disps);

  maxResp = np.maximum(np.nanmax(respMean), np.nanmax(predResps));
  # by disp
  clrs_d = cm.viridis(np.linspace(0,0.75,nDisps-1));
  lbls_d = ['disp: %s' % str(x) for x in range(nDisps)];
  # by sf
  val_sfs = hf.get_valid_sfs(S, disp=1, con=val_con_by_disp[1][0], expInd=expInd) # pick 
  clrs_sf = cm.viridis(np.linspace(0,.75,len(val_sfs)));
  lbls_sf = ['sf: %.2f' % all_sfs[x] for x in val_sfs];
  # by con
  val_con = all_cons;
  clrs_con = cm.viridis(np.linspace(0,.75,len(val_con)));
  lbls_con = ['con: %.2f' % x for x in val_con];

  ############
  ### create the figure
  ############
  fSuper, ax = plt.subplots(nRows, nCols, figsize=(10*nCols, 8*nRows))
  sns.despine(fig=fSuper, offset=10)

  allMix = [];
  allSum = [];

  ### plot reference tuning [row 1 (i.e. 2nd row)]
  ## on the right, SF tuning (high contrast)
  sfRef = hf.nan_rm(respMean[0, :, -1]); # high contrast tuning
  ax[1, 1].plot(all_sfs, sfRef, 'k-', marker='o', label='ref. tuning (d0, high con)', clip_on=False)
  ax[1, 1].set_xscale('log')
  ax[1, 1].set_xlim((0.1, 10));
  ax[1, 1].set_xlabel('sf (c/deg)')
  ax[1, 1].set_ylabel('response (spikes/s)')
  ax[1, 1].set_ylim((-5, 1.1*np.nanmax(sfRef)));
  ax[1, 1].legend(fontsize='x-small');

  #####
  ## then on the left, RVC (peak SF)
  #####
  sfPeak = np.argmax(sfRef); # stupid/simple, but just get the rvc for the max response
  v_cons_single = val_con_by_disp[0]
  rvcRef = hf.nan_rm(respMean[0, sfPeak, v_cons_single]);
  # now, if possible, let's also plot the RVC fit
  if rvcFits is not None:
    rvcFits = hf.get_rvc_fits(dataPath, expInd, which_cell, rvcName=rvcName, rvcMod=rvcMod);
    rel_rvc = rvcFits[0]['params'][sfPeak]; # we get 0 dispersion, peak SF
    plt_cons = np.geomspace(all_cons[0], all_cons[-1], 50);
    c50, pk = hf.get_c50(rvcMod, rel_rvc), rvcFits[0]['conGain'][sfPeak];
    c50_emp, c50_eval = hf.c50_empirical(rvcMod, rel_rvc); # determine c50 by optimization, numerical approx.
    if rvcMod == 0:
      rvc_mod = hf.get_rvc_model();
      rvcmodResp = rvc_mod(*rel_rvc, plt_cons);
    else: # i.e. mod=1 or mod=2
      rvcmodResp = hf.naka_rushton(plt_cons, rel_rvc);
    if baseline is not None:
      rvcmodResp = rvcmodResp - baseline; 
    ax[1, 0].plot(plt_cons, rvcmodResp, 'k--', label='rvc fit (c50=%.2f, gain=%0f)' %(c50, pk))
    # and save it
    curr_suppr['c50'] = c50; curr_suppr['conGain'] = pk;
    curr_suppr['c50_emp'] = c50_emp; curr_suppr['c50_emp_eval'] = c50_eval
  else:
    curr_suppr['c50'] = np.nan; curr_suppr['conGain'] = np.nan;
    curr_suppr['c50_emp'] = np.nan; curr_suppr['c50_emp_eval'] = np.nan;

  ax[1, 0].plot(all_cons[v_cons_single], rvcRef, 'k-', marker='o', label='ref. tuning (d0, peak SF)', clip_on=False)
  #         ax[1, 0].set_xscale('log')
  ax[1, 0].set_xlabel('contrast (%)');
  ax[1, 0].set_ylabel('response (spikes/s)')
  ax[1, 0].set_ylim((-5, 1.1*np.nanmax(rvcRef)));
  ax[1, 0].legend(fontsize='x-small');

  # plot the fitted model on each axis
  pred_plt = np.linspace(0, np.nanmax(all_preds), 100);
  if fitz is not None:
    ax[0, 0].plot(pred_plt, myFit(pred_plt, *fitz), 'r--', label='fit')
    ax[0, 1].plot(pred_plt, myFit(pred_plt, *fitz), 'r--', label='fit')

  for d in range(nDisps):
    if d == 0: # we don't care about single gratings!
      dispRats = [];
      continue; 
    v_cons = np.array(val_con_by_disp[d]);
    n_v_cons = len(v_cons);

    # plot split out by each contrast [0,1]
    for c in reversed(range(n_v_cons)):
      v_sfs = hf.get_valid_sfs(S, d, v_cons[c], expInd)
      for s in v_sfs:
        mixResp = respMean[d, s, v_cons[c]];
        allMix.append(mixResp);
        sumResp = predResps[d, s, v_cons[c]];
        allSum.append(sumResp);
  #      print('condition: d(%d), c(%d), sf(%d):: pred(%.2f)|real(%.2f)' % (d, v_cons[c], s, sumResp, mixResp))
        # PLOT in by-disp panel
        if c == 0 and s == v_sfs[0]:
          ax[0, 0].plot(sumResp, mixResp, 'o', color=clrs_d[d-1], label=lbls_d[d], clip_on=False)
        else:
          ax[0, 0].plot(sumResp, mixResp, 'o', color=clrs_d[d-1], clip_on=False)
        # PLOT in by-sf panel
        sfInd = np.where(np.array(v_sfs) == s)[0][0]; # will only be one entry, so just "unpack"
        try:
          if d == 1 and c == 0:
            ax[0, 1].plot(sumResp, mixResp, 'o', color=clrs_sf[sfInd], label=lbls_sf[sfInd], clip_on=False);
          else:
            ax[0, 1].plot(sumResp, mixResp, 'o', color=clrs_sf[sfInd], clip_on=False);
        except:
          pass;
          #pdb.set_trace();
        # plot baseline, if f0...
  #       if baseline is not None:
  #         [ax[0, i].axhline(baseline, linestyle='--', color='k', label='spon. rate') for i in range(2)];


    # plot averaged across all cons/sfs (i.e. average for the whole dispersion) [1,0]
    mixDisp = respMean[d, :, :].flatten();
    sumDisp = predResps[d, :, :].flatten();
    mixDisp, sumDisp = zr_rm_pair(mixDisp, sumDisp, 0.5);
    curr_rats = np.divide(mixDisp, sumDisp)
    curr_mn = geomean(curr_rats); curr_std = np.std(np.log10(curr_rats));
  #  curr_rat = geomean(np.divide(mixDisp, sumDisp));
    ax[2, 0].bar(d, curr_mn, yerr=curr_std, color=clrs_d[d-1]);
    ax[2, 0].set_yscale('log')
    ax[2, 0].set_ylim(0.1, 10);
  #  ax[2, 0].yaxis.set_ticks(minorticks)
    dispRats.append(curr_mn);
  #  ax[2, 0].bar(d, np.mean(np.divide(mixDisp, sumDisp)), color=clrs_d[d-1]);

    # also, let's plot the (signed) error relative to the fit
    if fitz is not None:
      errs = mixDisp - myFit(sumDisp, *fitz);
      ax[3, 0].bar(d, np.mean(errs), yerr=np.std(errs), color=clrs_d[d-1])
      # -- and normalized by the prediction output response
      errs_norm = np.divide(mixDisp - myFit(sumDisp, *fitz), myFit(sumDisp, *fitz));
      ax[4, 0].bar(d, np.mean(errs_norm), yerr=np.std(errs_norm), color=clrs_d[d-1])

    # and set some labels/lines, as needed
    if d == 1:
        ax[2, 0].set_xlabel('dispersion');
        ax[2, 0].set_ylabel('suppression ratio (linear)')
        ax[2, 0].axhline(1, ls='--', color='k')
        ax[3, 0].set_xlabel('dispersion');
        ax[3, 0].set_ylabel('mean (signed) error')
        ax[3, 0].axhline(0, ls='--', color='k')
        ax[4, 0].set_xlabel('dispersion');
        ax[4, 0].set_ylabel('mean (signed) error -- as frac. of fit prediction')
        ax[4, 0].axhline(0, ls='--', color='k')

    curr_suppr['supr_disp'] = dispRats;

  ### plot averaged across all cons/disps
  sfInds = []; sfRats = []; sfRatStd = []; 
  sfErrs = []; sfErrsStd = []; sfErrsInd = []; sfErrsIndStd = []; sfErrsRat = []; sfErrsRatStd = [];
  curr_errNormFactor = [];
  for s in range(len(val_sfs)):
    try: # not all sfs will have legitmate values;
      # only get mixtures (i.e. ignore single gratings)
      mixSf = respMean[1:, val_sfs[s], :].flatten();
      sumSf = predResps[1:, val_sfs[s], :].flatten();
      mixSf, sumSf = zr_rm_pair(mixSf, sumSf, 0.5);
      rats_curr = np.divide(mixSf, sumSf); 
      sfInds.append(s); sfRats.append(geomean(rats_curr)); sfRatStd.append(np.std(np.log10(rats_curr)));

      if fitz is not None:
        #curr_NR = myFit(sumSf, *fitz); # unvarnished
        curr_NR = np.maximum(myFit(sumSf, *fitz), 0.5); # thresholded at 0.5...

        curr_err = mixSf - curr_NR;
        sfErrs.append(np.mean(curr_err));
        sfErrsStd.append(np.std(curr_err))

        curr_errNorm = np.divide(mixSf - curr_NR, mixSf + curr_NR);
        sfErrsInd.append(np.mean(curr_errNorm));
        sfErrsIndStd.append(np.std(curr_errNorm))

        curr_errRat = np.divide(mixSf, curr_NR);
        sfErrsRat.append(np.mean(curr_errRat));
        sfErrsRatStd.append(np.std(curr_errRat));

        curr_normFactors = np.array(curr_NR)
        curr_errNormFactor.append(geomean(curr_normFactors[curr_normFactors>0]));
      else:
        sfErrs.append([]);
        sfErrsStd.append([]);
        sfErrsInd.append([]);
        sfErrsIndStd.append([]);
        sfErrsRat.append([]);
        sfErrsRatStd.append([]);
        curr_errNormFactor.append([]);
    except:
      pass

  # get the offset/scale of the ratio so that we can plot a rescaled/flipped version of
  # the high con/single grat tuning for reference...does the suppression match the response?
  offset, scale = np.nanmax(sfRats), np.nanmax(sfRats) - np.nanmin(sfRats);
  sfRef = hf.nan_rm(respMean[0, val_sfs, -1]); # high contrast tuning
  sfRefShift = offset - scale * (sfRef/np.nanmax(sfRef))
  ax[2,1].scatter(all_sfs[val_sfs][sfInds], sfRats, color=clrs_sf[sfInds], clip_on=False)
  ax[2,1].errorbar(all_sfs[val_sfs][sfInds], sfRats, sfRatStd, color='k', linestyle='-', clip_on=False, label='suppression tuning')
  #         ax[2,1].plot(all_sfs[val_sfs][sfInds], sfRats, 'k-', clip_on=False, label='suppression tuning')
  ax[2,1].plot(all_sfs[val_sfs], sfRefShift, 'k--', label='ref. tuning', clip_on=False)
  ax[2,1].axhline(1, ls='--', color='k')
  ax[2,1].set_xlabel('sf (cpd)')
  ax[2,1].set_xscale('log')
  ax[2,1].set_xlim((0.1, 10));
  #ax[2,1].set_xlim((np.min(all_sfs), np.max(all_sfs)));
  ax[2,1].set_ylabel('suppression ratio');
  ax[2,1].set_yscale('log')
  #ax[2,1].yaxis.set_ticks(minorticks)
  ax[2,1].set_ylim(0.1, 10);        
  ax[2,1].legend(fontsize='x-small');
  curr_suppr['supr_sf'] = sfRats;

  ### residuals from fit of suppression
  if fitz is not None:
    # mean signed error: and labels/plots for the error as f'n of SF
    ax[3,1].axhline(0, ls='--', color='k')
    ax[3,1].set_xlabel('sf (cpd)')
    ax[3,1].set_xscale('log')
    ax[3,1].set_xlim((0.1, 10));
    #ax[3,1].set_xlim((np.min(all_sfs), np.max(all_sfs)));
    ax[3,1].set_ylabel('mean (signed) error');
    ax[3,1].errorbar(all_sfs[val_sfs][sfInds], sfErrs, sfErrsStd, color='k', marker='o', linestyle='-', clip_on=False)
    # -- and normalized by the prediction output response + output respeonse
    val_errs = np.logical_and(~np.isnan(sfErrsRat), np.logical_and(np.array(sfErrsIndStd)>0, np.array(sfErrsIndStd) < 2));
    norm_subset = np.array(sfErrsInd)[val_errs];
    normStd_subset = np.array(sfErrsIndStd)[val_errs];
    ax[4,1].axhline(0, ls='--', color='k')
    ax[4,1].set_xlabel('sf (cpd)')
    ax[4,1].set_xscale('log')
    ax[4,1].set_xlim((0.1, 10));
    #ax[4,1].set_xlim((np.min(all_sfs), np.max(all_sfs)));
    ax[4,1].set_ylim((-1, 1));
    ax[4,1].set_ylabel('error index');
    ax[4,1].errorbar(all_sfs[val_sfs][sfInds][val_errs], norm_subset, normStd_subset, color='k', marker='o', linestyle='-', clip_on=False)
    # -- AND simply the ratio between the mixture response and the mean expected mix response (i.e. Naka-Rushton)
    # --- equivalent to the suppression ratio, but relative to the NR fit rather than perfect linear summation
    val_errs = np.logical_and(~np.isnan(sfErrsRat), np.logical_and(np.array(sfErrsRatStd)>0, np.array(sfErrsRatStd) < 2));
    rat_subset = np.array(sfErrsRat)[val_errs];
    ratStd_subset = np.array(sfErrsRatStd)[val_errs];
    #ratStd_subset = (1/np.log(2))*np.divide(np.array(sfErrsRatStd)[val_errs], rat_subset);
    ax[5,1].scatter(all_sfs[val_sfs][sfInds][val_errs], rat_subset, color=clrs_sf[sfInds][val_errs], clip_on=False)
    ax[5,1].errorbar(all_sfs[val_sfs][sfInds][val_errs], rat_subset, ratStd_subset, color='k', linestyle='-', clip_on=False, label='suppression tuning')
    ax[5,1].axhline(1, ls='--', color='k')
    ax[5,1].set_xlabel('sf (cpd)')
    ax[5,1].set_xscale('log')
    ax[5,1].set_xlim((0.1, 10));
    ax[5,1].set_ylabel('suppression ratio (wrt NR)');
    ax[5,1].set_yscale('log', basey=2)
  #         ax[2,1].yaxis.set_ticks(minorticks)
    ax[5,1].set_ylim(np.power(2.0, -2), np.power(2.0, 2));
    ax[5,1].legend(fontsize='x-small');
    # - compute the variance - and put that value on the plot
    errsRatVar = np.var(np.log2(sfErrsRat)[val_errs]);
    curr_suppr['sfRat_VAR'] = errsRatVar;
    ax[5,1].text(0.1, 2, 'var=%.2f' % errsRatVar);

    # compute the unsigned "area under curve" for the sfErrsInd, and normalize by the octave span of SF values considered
    val_errs = np.logical_and(~np.isnan(sfErrsRat), np.logical_and(np.array(sfErrsIndStd)>0, np.array(sfErrsIndStd) < 2));
    val_x = all_sfs[val_sfs][sfInds][val_errs];
    ind_var = np.var(np.array(sfErrsInd)[val_errs]);
    curr_suppr['sfErrsInd_VAR'] = ind_var;
    # - and put that value on the plot
    ax[4,1].text(0.1, -0.25, 'var=%.3f' % ind_var);
  else:
    curr_suppr['sfErrsInd_VAR'] = np.nan
    curr_suppr['sfRat_VAR'] = np.nan

  #########
  ### NOW, let's evaluate the derivative of the SF tuning curve and get the correlation with the errors
  #########
  mod_sfs = np.geomspace(all_sfs[0], all_sfs[-1], 1000);
  mod_resp = hf.get_descrResp(dfit_curr, mod_sfs, DoGmodel=dMod_num);
  deriv = np.divide(np.diff(mod_resp), np.diff(np.log10(mod_sfs)))
  deriv_norm = np.divide(deriv, np.maximum(np.nanmax(deriv), np.abs(np.nanmin(deriv)))); # make the maximum response 1 (or -1)
  # - then, what indices to evaluate for comparing with sfErr?
  errSfs = all_sfs[val_sfs][sfInds];
  mod_inds = [np.argmin(np.square(mod_sfs-x)) for x in errSfs];
  deriv_norm_eval = deriv_norm[mod_inds];
  # -- plot on [1, 1] (i.e. where the data is)
  ax[1,1].plot(mod_sfs, mod_resp, 'k--', label='fit (g)')
  ax[1,1].legend();
  # Duplicate "twin" the axis to create a second y-axis
  ax2 = ax[1,1].twinx();
  ax2.set_xscale('log'); # have to re-inforce log-scale?
  ax2.set_ylim([-1, 1]); # since the g' is normalized
  # make a plot with different y-axis using second axis object
  ax2.plot(mod_sfs[1:], deriv_norm, '--', color="red", label='g\'');
  ax2.set_ylabel("deriv. (normalized)",color="red")
  ax2.legend();
  sns.despine(ax=ax2, offset=10, right=False);
  # -- and let's plot rescaled and shifted version in [2,1]
  offset, scale = np.nanmax(sfRats), np.nanmax(sfRats) - np.nanmin(sfRats);
  derivShift = offset - scale * (deriv_norm/np.nanmax(deriv_norm));
  ax[2,1].plot(mod_sfs[1:], derivShift, 'r--', label='deriv(ref. tuning)', clip_on=False)
  ax[2,1].legend(fontsize='x-small');
  # - then, normalize the sfErrs/sfErrsInd and compute the correlation coefficient
  if fitz is not None:
    norm_sfErr = np.divide(sfErrs, np.nanmax(np.abs(sfErrs)));
    norm_sfErrInd = np.divide(sfErrsInd, np.nanmax(np.abs(sfErrsInd))); # remember, sfErrsInd is normalized per condition; this is overall
    non_nan = np.logical_and(~np.isnan(norm_sfErr), ~np.isnan(deriv_norm_eval))
    corr_nsf, corr_nsfN = np.corrcoef(deriv_norm_eval[non_nan], norm_sfErr[non_nan])[0,1], np.corrcoef(deriv_norm_eval[non_nan], norm_sfErrInd[non_nan])[0,1]
    curr_suppr['corr_derivWithErr'] = corr_nsf;
    curr_suppr['corr_derivWithErrsInd'] = corr_nsfN;
    ax[3,1].text(0.1, 0.25*np.nanmax(sfErrs), 'corr w/g\' = %.2f' % corr_nsf)
    ax[4,1].text(0.1, 0.25, 'corr w/g\' = %.2f' % corr_nsfN)
  else:
    curr_suppr['corr_derivWithErr'] = np.nan;
    curr_suppr['corr_derivWithErrsInd'] = np.nan;

  # make a polynomial fit
  try:
    hmm = np.polyfit(allSum, allMix, deg=1) # returns [a, b] in ax + b 
  except:
    hmm = [np.nan];
  curr_suppr['supr_index'] = hmm[0];

  for j in range(1):
    for jj in range(nCols):
      ax[j, jj].axis('square')
      ax[j, jj].set_xlabel('prediction: sum(components) (imp/s)');
      ax[j, jj].set_ylabel('mixture response (imp/s)');
      ax[j, jj].plot([0, 1*maxResp], [0, 1*maxResp], 'k--')
      ax[j, jj].set_xlim((-5, maxResp));
      ax[j, jj].set_ylim((-5, 1.1*maxResp));
      ax[j, jj].set_title('Suppression index: %.2f|%.2f' % (hmm[0], rel_c50))
      ax[j, jj].legend(fontsize='x-small');

  fSuper.suptitle('Superposition: %s #%d [%s; f1f0 %.2f; szSupr[dt/md] %.2f/%.2f; oriBW|CV %.2f|%.2f; tfBW %.2f]' % (cellType, which_cell, cellName, f1f0_rat, suprDat, suprMod, oriBW, oriCV, tfBW))

  if fitList is None:
    save_name = 'cell_%03d.pdf' % which_cell
  else:
    save_name = 'cell_%03d_mod%s.pdf' % (which_cell, hf.fitType_suffix(fitType))
  pdfSv = pltSave.PdfPages(str(save_locSuper + save_name));
  pdfSv.savefig(fSuper)
  pdfSv.close();

  #########
  ### Finally, add this "superposition" to the newest 
  #########

  if to_save:

    if fitList is None:
      from datetime import datetime
      suffix = datetime.today().strftime('%y%m%d')
      super_name = 'superposition_analysis_%s.npy' % suffix;
    else:
      super_name = 'superposition_analysis_mod%s.npy' % hf.fitType_suffix(fitType);

    pause_tm = 5*np.random.rand();
    print('sleeping for %d secs (#%d)' % (pause_tm, which_cell));
    time.sleep(pause_tm);

    if os.path.exists(dataPath + super_name):
      suppr_all = hf.np_smart_load(dataPath + super_name);
    else:
      suppr_all = dict();
    suppr_all[which_cell-1] = curr_suppr;
    np.save(dataPath + super_name, suppr_all);
  
  return curr_suppr;

 
if __name__ == '__main__':

    cell_num   = int(sys.argv[1]);
    if cell_num < -99: 
      # i.e. 3 digits AND negative, then we'll treat the first two digits as where to start, and the second two as when to stop
      # -- in that case, we'll do this as multiprocessing
      asMulti = 1;
      end_cell = int(np.mod(-cell_num, 100));
      start_cell = int(np.floor(-cell_num/100));
    else:
      asMulti = 0;
    expDir     = sys.argv[2];

    fitList=None; # TEMPORARY

    if asMulti:
      from functools import partial
      import multiprocessing as mp
      nCpu = mp.cpu_count()-1; # heuristics say you should reqeuest at least one fewer processes than their are CPU
      print('***cpu count: %02d***' % nCpu);

      with mp.Pool(processes = nCpu) as pool:
        sup_perCell = partial(plot_save_superposition, expDir=expDir, use_mod_resp=0, fitType=2, excType=1, useHPCfit=1, conType=None, lgnFrontEnd=None, to_save=0);
        supFits = pool.map(sup_perCell, range(start_cell, end_cell+1));
        pool.close();

      ### do the saving HERE!
      dataPath = os.getcwd() + '/' + expDir + 'structures/'
      if fitList is None:
        from datetime import datetime
        suffix = datetime.today().strftime('%y%m%d')
        super_name = 'superposition_analysis_%s.npy' % suffix;
      else:
        super_name = 'superposition_analysis_mod%s.npy' % hf.fitType_suffix(fitType);

      if os.path.exists(dataPath + super_name):
        suppr_all = hf.np_smart_load(dataPath + super_name);
      else:
        suppr_all = dict();
      for iii, sup_fit in enumerate(supFits):
        suppr_all[iii] = sup_fit;
      np.save(dataPath + super_name, suppr_all);
    else:
      plot_save_superposition(cell_num, expDir, to_save=1);

