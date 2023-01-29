import os
import numpy as np
import pandas as pd
import matplotlib as mpl
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
from matplotlib.ticker import FuncFormatter

import pdb

import sys # so that we can import model_responses (in different folder)

####### TO USE ########
### akin to other parallelized functions, call 'python3.6 plot_superposition_func.py -181 LGN/' for example
### 'python3.6 plot_superposition_func.py 4 altExp/ 1 1 2 1 0'
### - cellNum, expDir, plot suppression index, prince corr. to suprInd, modResp (2 for pyt; 0 for data rather than model); normType (1/2 - flat/gain), lgnOn (0/1)

import warnings
#warnings.filterwarnings('once');

def get_responses(expData, which_cell, expInd, expDir, dataPath, respMeasure, stimVals, val_con_by_disp, rvcFits=None, phAdvName=None, vecF1=0, f1_expCutoff=2, rvcDir=1, val_by_stim_val=None, sum_power=1, resample=False):
  # Get the correct DC or F1 responses
  if vecF1 == 1:
    # get the correct, adjusted F1 response
    if expInd > f1_expCutoff and respMeasure == 1:
      respOverwrite = hf.adjust_f1_byTrial(expData, expInd);
    else:
      respOverwrite = None;

  respsPhAdv_mean_ref = None; # defaulting to None here -- if not None, then we've collected mean responses based on phAmp corr. on means (not trial by trial)
  respsPhAdv_mean_pred = None;
  adjMeansByComp = None; val_tr_by_cond = None;

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
      spikes, which_measure = hf.get_adjusted_spikerate(expData, which_cell, expInd, dataPath, rvcName=rvcFits, rvcMod=-1, baseline_sub=False, return_measure=1, vecF1=vecF1);
      # now, get the real mean responses we'll use (again, the above is just for variability)
      phAdvFits = hf.np_smart_load(dataPath + hf.phase_fit_name(phAdvName, dir=rvcDir));
      all_opts = phAdvFits[which_cell-1]['params'];
      try:
        respsPhAdv_mean_ref, respsPhAdv_mean_pred, adjMeansByComp, val_tr_by_cond = hf.organize_phAdj_byMean(expData, expInd, all_opts, stimVals, val_con_by_disp, incl_preds=True, val_by_stim_val=val_by_stim_val, return_comps=True, sum_power=sum_power);
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
  resps_data, _, _, _, _ = hf.tabulate_responses(expData, expInd, overwriteSpikes=spikes, respsAsRates=rates, modsAsRate=rates, sum_power=sum_power, verbose=True, resample=resample);

  return resps_data, respAll, respsPhAdv_mean_ref, respsPhAdv_mean_pred, baseline, adjMeansByComp, val_tr_by_cond;

def get_model_responses(expData, fitList, expInd, which_cell, excType, fitType, f1f0_rat, respMeasure, baseline, lossType=1, lgnFrontEnd=0, newMethod=1, lgnConType=1, _applyLGNtoNorm=0, _sigmoidSigma=5, recenter_norm=0, normToOne=1, debug=False, use_mod_resp=2, sum_power=1, dgNormFunc=0, resample=False):
  # This is ONLY for getting model responses
  if use_mod_resp == 1: # deprecated...
    curr_fit = fitList[which_cell-1]['params'];
    modResp = mod_resp.SFMGiveBof(curr_fit, expData, normType=fitType, lossType=lossType, expInd=expInd, cellNum=which_cell, excType=excType)[1];
    if f1f0_rat < 1: # then subtract baseline..
      modResp = modResp - baseline*hf.get_exp_params(expInd).stimDur; 
    # now organize the responses
    resps = hf.tabulate_responses(expData, expInd, overwriteSpikes=modResp, respsAsRates=False, modsAsRate=False, sum_power=sum_power)[0];

  elif use_mod_resp == 2: # then pytorch model!
    resp_str = hf_sf.get_resp_str(respMeasure)
    if (which_cell-1) in fitList:
      try:
        curr_fit = fitList[which_cell-1][resp_str]['params'];
      except: # failed...
        return None, None; # code so that we quit
    else:
      curr_fit = fitList; # we already passed in parameters
    model = mrpt.sfNormMod(curr_fit, expInd=expInd, excType=excType, normType=fitType, lossType=lossType, lgnFrontEnd=lgnFrontEnd, newMethod=newMethod, lgnConType=lgnConType, applyLGNtoNorm=_applyLGNtoNorm, normToOne=normToOne, normFiltersToOne=False, toFit=False, dgNormFunc=dgNormFunc)
    ### get the vec-corrected responses, if applicable
    # NOTE: NEED TO FIX THIS, esp. for 
    if expInd > 2 and respMeasure == 1: # only can get F1 if expInd>=2
      respOverwrite = hf.adjust_f1_byTrial(expData, expInd);
    else:
      respOverwrite = None;

    dw = mrpt.dataWrapper(expData, respMeasure=respMeasure, expInd=expInd, respOverwrite=respOverwrite);#, shufflePh=True, shuffleTf=True);
    # ^^^ respOverwrite defined above (None if DC or if expInd=-1)
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

    model_baseline = model.noiseLate.detach().numpy()
    if respMeasure == 0: # if DC, then subtract baseline..., as determined from data (why not model? we aren't yet calc. response to no stim, though it can be done)
      stimDur = hf.get_exp_params(expInd).stimDur
      if normToOne==1 and newMethod==1: # then noiseLate is exactly the noiseLate
        modResp_full -= model_baseline;  # Model is counts --> no need to factor in stimDur
      else: # sub the data baseline, since our model should've found that anyway...
        modResp_full = modResp_full - baseline*stimDur;

    # TODO: This is a work around for which measures are in rates vs. counts (DC vs F1, model vs data...)
    stimDur = hf.get_exp_params(expInd).stimDur;
    asRates = False if respMeasure==0 else True; # NOT YET VERIFIED/VALIDATED
    divFactor = stimDur if asRates == 0 else 1;
    modResp_full = np.divide(modResp_full, divFactor);
    # now organize the responses
    #resps = hf.organize_resp(modResp_full, expData, expInd);
    resps = hf.tabulate_responses(expData, expInd, overwriteSpikes=modResp_full, respsAsRates=True, modsAsRate=asRates, verbose=True, sum_power=sum_power, resample=resample)[0];

  if debug:
    return model.respPerCell(dw.trInf, debug=True, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm);

  return resps, model_baseline/divFactor;

def fit_overall_suppression(all_resps, all_preds, expFixed=True):
  ''' Fit the overall suppression (i.e. superposition failures) with a Naka-Rushton fit
      - assumes resps/preds are nConds x 1 (i.e. flattened) and only includes mixtures, ie not single gratings
  '''
  # Setting the function we'll use
  myFit = lambda x, g, expon, c50: hf.naka_rushton(x, [0, g, expon, c50]) 

  # Setting bounds
  non_neg = all_preds>0; # cannot fit negative values with naka-rushton...
  non_nan = np.logical_and(~np.isnan(all_resps), ~np.isnan(all_preds));
  val_inds = np.logical_and(non_neg, non_nan);
  max_resp, max_pred = np.nanmax(all_resps[val_inds]), np.nanmax(all_preds[val_inds]);
  if expFixed: # fixed at 1...
    bounds = np.vstack(((0,1.5*max_resp), (1, 1.000001), (0,1.5*max_pred))).transpose(); # gain, c50 must be pos; exp. fixed at 1
  else:
    bounds = np.vstack(((0,1.5*max_resp), (0.5, 4), (0,1.5*max_pred))).transpose(); # gain, c50 must be pos; exp. between 0.5-4

  # Setting initial params
  init_gain = np.nanmax(all_resps[val_inds]);
  init_exp = 1 if expFixed else hf.random_in_range([0.75, 2.5])[0];
  init_c50 = np.nanmedian(all_preds[val_inds]);
  
  # Running the optimization
  fit, _ = opt.curve_fit(myFit, all_preds[val_inds], all_resps[val_inds], p0=[init_gain, init_exp, init_c50], bounds=bounds, maxfev=5000)
  max_input_resp = np.max(all_preds[val_inds]);
  rel_c50 = np.divide(fit[-1], max_input_resp);
  # now, let's also return an emperical c50, i.e. relative to the max response, what is the input response that gives 50% of max pred?
  half_val = myFit(max_input_resp, *fit)/2; # half of max resp. over pred. range
  obj = lambda resp_in: np.square(half_val - myFit(resp_in, *fit));
  try:
    rel_c50_emp = opt.minimize(obj, x0=max_input_resp/2, bounds=((0,max_input_resp), ))['x'][0]/max_input_resp;
  except:
    rel_c50_emp = np.nan;
 
  return fit, rel_c50, myFit, rel_c50_emp

def selected_supr_metrics(df):
  ''' Add selected metrics here, which you can then call in the primary function --> this will make it easier to add into jointList
  '''

  # i. Suppression index averaged by dispersion, only for conditions which appear at all dispersion levels
  which_grat_as_ref = df['disp'].unique()[-1]; # use the largest # of grats as the reference (most restrictive disp.)
  valid_sf_cons = df[df['disp']==which_grat_as_ref][['sf', 'total_con']].values

  oy = [];
  which_metr = 'rel_supr'
  for comb in valid_sf_cons:
      oy.append(to_use[np.logical_and(np.isclose(to_use['sf'], comb[0]), np.isclose(to_use['total_con'], comb[1], atol=0.02))].groupby('disp')[which_metr].mean())
  vey = pd.DataFrame(oy)

  return None;

def plot_save_superposition(which_cell, expDir, use_mod_resp=0, fitType=1, excType=1, useHPCfit=1, lgnConType=None, lgnFrontEnd=None, force_full=1, f1_expCutoff=2, to_save=1, plt_f1_plots=False, useTex=False, simple_plot=True, altHollow=True, ltThresh=0.5, ref_line_alpha=0.5, ref_all_sfs=False, plt_supr_ind=False, supr_ind_prince=False, sum_power=1, spec_disp = None, spec_con = None, fixRespExp=2, scheduler=False, singleGratsOnly=False, dataAsRef=False, tuningOverlay=True, incl_baseline=False, dgNormFunc=0, verbose=False, resample=False, make_plots=True, reducedSave=False, dataList=None, descrFits=None, fitList=None):

  # if ref_all_sfs, then colors for superposition plots are referenced across all SFS (not just those that appear for dispersion=1)
  if isinstance(which_cell, tuple):
    # we can pacakge f1f0_rat as part of which_cell to save us one timely func. call!
    which_cell, f1f0_rat = which_cell;
  else:
    f1f0_rat = np.nan

  if use_mod_resp == 2:
    rvcAdj   = 0; # phAmp corr.
    #rvcAdj   = -1; # this means vec corrected F1, not phase adjustment F1...
    _applyLGNtoNorm = 1; # apply the LGN front-end to the gain control weights??
    #recenter_norm = 2;
    recenter_norm = 0;
    newMethod = 1; # yes, use the "new" method for mrpt (not that new anymore, as of 21.03)
    lossType = 1; # sqrt
    _sigmoidSigma = 5;

  basePath = os.getcwd() + '/'
  if 'pl1465' in basePath or useHPCfit:
    loc_str = 'HPC';
  else:
    loc_str = '';

  if expDir == 'V1/':
    dt_rvc = '230111';
    dt_ph = dt_rvc;
    dt_sf = '%svEs' % dt_rvc;
  elif expDir == 'LGN/':
    dt_rvc = '220928'
    dt_ph = dt_rvc;
    dt_sf = '220810vEs'
  else: # altExp
    dt_rvc = '221126'
    dt_sf = '%svEs' % dt_rvc;
    dt_ph = None;
  rvcName = 'rvcFits%s_%s' % (loc_str, dt_rvc);
  phAdvName = 'phaseAdvanceFits%s_%s' % (loc_str, dt_ph);
  rvcFits = None; # pre-define this as None; will be overwritten if available/needed
  if expDir == 'altExp/': # we don't adjust responses there...
    rvcName = '%s_f0' % rvcName;
  dFits_base = 'descrFits%s_%s' % (loc_str, dt_sf);
  #dFits_base = 'descrFits%s_220609' % 'HPC' if expDir=='LGN/' else 'descrFits%s_220721' % 'HPC'
  #dFits_base = 'descrFits%s_220609' % loc_str if expDir=='LGN/' else 'descrFits%s_220721' % loc_str
  if use_mod_resp == 1:
    rvcName = None; # Use NONE if getting model responses, only
    if excType == 1:
      fitBase = 'fitList_200417';
    elif excType == 2:
      fitBase = 'fitList_200507';
    lossType = 1; # sqrt
    fitList_nm = hf.fitList_name(fitBase, fitType, lossType=lossType);
  elif use_mod_resp == 2:
    fitBase='fitList%s_pyt_nr230118a%s%s%s' % ('HPC', '_noRE' if fixRespExp is not None else '', '_noSched' if scheduler==False else '', '_sg' if singleGratsOnly else '');
    fitList_nm = hf.fitList_name(fitBase, fitType, lossType=lossType, lgnType=lgnFrontEnd, lgnConType=lgnConType, vecCorrected=-rvcAdj, excType=excType, dgNormFunc=dgNormFunc);
  # ^^^ EDIT rvc/descrFits/fitList names here;

  if use_mod_resp>0 and verbose:
    print('\n***Fitlist name:[%s]***\n' % fitList_nm);

  ############
  # Before any plotting, fix plotting paramaters
  ############
  from matplotlib import rcParams
  tex_width = 469; # per \layout in Overleaf on document
  sns_offset = 2; 
  hist_width = 0.9;
  hist_ytitle = 0.94; # moves the overall title a bit further down on histogram plots0

  rcParams.update(mpl.rcParamsDefault)

  fontsz = 10;
  tick_scalar = 1.5;

  rcParams['pdf.fonttype'] = 42
  rcParams['ps.fonttype'] = 42

  if useTex:
    rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
    params = {'text.usetex' : True,
#              'font.size' : fontsz,
              'font.family': 'lmodern',
               'font.style': 'italic'}
    plt.rcParams.update(params)
  else:
    rcParams['font.style'] = 'oblique';

  # rcParams['lines.linewidth'] = 2.5;
  rcParams['lines.markeredgewidth'] = 0; # no edge, since weird tings happen then
  # rcParams['axes.linewidth'] = 2; # was 1.5
  # rcParams['lines.markersize'] = 5;

  tick_adj = ['xtick.major.size', 'xtick.minor.size', 'ytick.major.size', 'ytick.minor.size']
  for adj in tick_adj:
      rcParams[adj] = rcParams[adj] * tick_scalar;

  ############
  # load everything
  ############
  dataListNm = hf.get_datalist(expDir, force_full=force_full, new_v1=True);
  descrFits_f0 = None;
  dLoss_num = 2; # see hf.descrFit_name/descrMod_name/etc for details
  if expDir == 'LGN/':
    rvcMod = 1; 
    dMod_num = 1;
    rvcDir = 1;
    vecF1 = 0;
  else: 
    rvcMod = 1; # i.e. Naka-rushton (1)
    dMod_num = 3; # d-dog-s
    if expDir == 'altExp/':
      rvcDir = None;
      #vecF1 = 0 # was None?
      vecF1 = None # was None?
    else:
      rvcDir = 1;
      vecF1 = 0; # was previously 1, but now we do phAmp, not just vecF1

  dFits_mod = hf.descrMod_name(dMod_num)
  descrFits_name = hf.descrFit_name(lossType=dLoss_num, descrBase=dFits_base, modelName=dFits_mod, phAdj=1 if vecF1==0 else None);

  ## now, let it run
  dataPath = basePath + expDir + 'structures/'
  save_loc = basePath + expDir + 'figures/'
  from datetime import datetime
  save_locSuper = save_loc + 'superposition_%s%s%s%s%s/vertical/' % (datetime.today().strftime('%y%m%d'), '_simple' if simple_plot else '', '' if plt_supr_ind else '_mse', '_prince' if supr_ind_prince else '', '_p%d' % sum_power if sum_power!=1 else '')
  if use_mod_resp == 1:
    save_locSuper = save_locSuper + '%s/' % fitBase

  if dataList is None: # otherwise, we've passed it in
    dataList = hf.np_smart_load(dataPath + dataListNm);
  if descrFits is None:
    descrFits = hf.np_smart_load(dataPath + descrFits_name);
  if fitList is None:
    if use_mod_resp == 1 or use_mod_resp == 2:
      fitList = hf.np_smart_load(dataPath + fitList_nm);
    else:
      fitList = None;

  if not os.path.exists(save_locSuper) and make_plots:
    os.makedirs(save_locSuper);

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
  nRows, nCols = 3, 1;
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
  basics = None;
  if not reducedSave and not resample:
    try:
      basic_names, basic_order = dataList['basicProgName'][which_cell-1], dataList['basicProgOrder']
      basics = hf.get_basic_tunings(basic_names, basic_order, reducedSave=True);
    except:
      try:
        # we've already put the basics in the data structure... (i.e. post-sorting 2021 data)
        basic_names = ['','','','',''];
        basic_order = ['rf', 'sf', 'tf', 'rvc', 'ori']; # order doesn't matter if they are already loaded
        basics = hf.get_basic_tunings(basic_names, basic_order, preProc=S, reducedSave=True)
      except:
        pass; # basics already declared as None

  try:
    cellType = dataList['unitType'][which_cell-1];
  except:
    # TODO: note, this is dangerous; thus far, only V1 cells don't have 'unitType' field in dataList, so we can safely do this
    cellType = 'V1';

  ############
  ### compute f1f0 ratio, and load the corresponding F0 or F1 responses
  ############
  if np.isnan(f1f0_rat):
    f1f0_rat = hf.compute_f1f0(expData, which_cell, expInd, dataPath, descrFitName_f0=descrFits_f0)[0];
  respMeasure = 1 if (f1f0_rat > 1 and expInd>2) else 0;

  # load rvcFits in case needed
  try:
    rvcFits = hf.get_rvc_fits(dataPath, expInd, which_cell, rvcName=rvcName, rvcMod=rvcMod, direc=rvcDir, vecF1=vecF1);
  except:
    rvcFits = None;

  _, stimVals, val_con_by_disp, val_by_stim_val, _ = hf.tabulate_responses(expData, expInd); # call just to get these values (not spikes/activity)
  resps_data, respAll, respsPhAdv_mean_ref, respsPhAdv_mean_preds, baseline, comp_resp_org, val_tr_org = get_responses(expData, which_cell, expInd, expDir, dataPath, respMeasure, stimVals, 
                                                                                                                       val_con_by_disp, rvcFits, phAdvName, vecF1, f1_expCutoff=f1_expCutoff, rvcDir=rvcDir, val_by_stim_val=val_by_stim_val, sum_power=sum_power, resample=resample);

  if fitList is None:
    resps = resps_data; # otherwise, we'll still keep resps_data for reference
  elif fitList is not None: # OVERWRITE the data with the model spikes!
    resps, model_baseline = get_model_responses(expData, fitList, expInd, which_cell, excType, fitType, f1f0_rat, respMeasure, baseline, lossType=lossType, lgnFrontEnd=lgnFrontEnd, lgnConType=lgnConType, _applyLGNtoNorm=_applyLGNtoNorm, recenter_norm=recenter_norm, sum_power=sum_power, dgNormFunc=dgNormFunc, resample=resample);

    if resps is None: # then the model didn't load...
      return None;

  if use_mod_resp == 2: # then get model resp
    predResps = resps[2]
    respMean = resps[0]
  else:
    predResps = resps[2] if respsPhAdv_mean_preds is None else respsPhAdv_mean_preds;
    respMean = resps[0] if respsPhAdv_mean_ref is None else respsPhAdv_mean_ref; # equivalent to resps[0;]
  # note: respAll always refers to the data! (i.e. never the model)
  respStd = np.nanstd(respAll, -1); # take std of all responses for a given condition
  predStd = resps[3]; # WARNING/todo: if fitList is not None, this might be a problem?
  # compute SEM, too
  findNaN = np.isnan(respAll);
  nonNaN  = np.sum(findNaN == False, axis=-1);
  respSem = np.nanstd(respAll, -1) / np.sqrt(nonNaN);

  ############
  ### zeroth...just organize into pandas for some potential/future processing
  ###
  ############
  # broadcast disps/cons/sfs for use with pandas
  disps_expand = np.broadcast_to(np.expand_dims(np.expand_dims(stimVals[0], axis=-1), axis=-1), respMean.shape)
  cons_expand = np.broadcast_to(np.expand_dims(np.expand_dims(stimVals[1], axis=0), axis=0), respMean.shape)
  sfs_expand = np.broadcast_to(np.expand_dims(np.expand_dims(stimVals[2], axis=0), axis=-1), respMean.shape)
  mixture_exp = pd.DataFrame(data=np.column_stack((disps_expand.flatten(), sfs_expand.flatten(), cons_expand.flatten(), respMean.flatten(), respStd.flatten(), predResps.flatten(), predStd.flatten())),
                        columns=['disp', 'sf', 'total_con', 'respMean', 'respStd', 'predMean', 'predStd'], dtype=np.float32)
  # drop the NaN conditions (i.e. no respMean and/or no predMean for a given dispXconXsf)
  mixture_exp = mixture_exp.dropna(subset=['respMean', 'predMean'], )

  ############
  ### first, fit a smooth function to the overall pred V measured responses
  ### --- from this, we can measure how each example superposition deviates from a central tendency
  ### --- i.e. the residual relative to the "standard" input:output relationship
  ############
  all_resps = respMean[1:, :, :].flatten() # all disp>0
  all_preds = predResps[1:, :, :].flatten() # all disp>0
  myFit = lambda x, g, expon, c50: hf.naka_rushton(x, [0, g, expon, c50]) 
  try:
    if use_mod_resp == 1: # the reference will ALWAYS be the data -- redo the above analysis for data
      # TODO? Make sure we can use fit_overall_suppression for this too?
      predResps_data = resps_data[2];
      respMean_data = resps_data[0];
      all_resps_data = respMean_data[1:, :, :].flatten() # all disp>0
      all_preds_data = predResps_data[1:, :, :].flatten() # all disp>0
      non_neg_data = np.where(all_preds_data>0) # cannot fit negative values with naka-rushton...
      fitz, _ = opt.curve_fit(myFit, all_preds_data[non_neg_data], all_resps_data[non_neg_data], p0=[100, 2, 25], maxfev=5000)
      rel_c50 = np.divide(fitz[-1], np.max(all_preds[non_neg_data]));
    else:
      fitz, rel_c50, _, rel_c50_emp = fit_overall_suppression(all_resps, all_preds)
  except:
    fitz = None;
    rel_c50 = -99;
  if dataAsRef: # i.e. if we want the data to set the N-R fit,
    predResps_data = resps_data[2];
    respMean_data = resps_data[0];
    all_resps_data = respMean_data[1:, :, :].flatten() # all disp>0
    all_preds_data = predResps_data[1:, :, :].flatten() # all disp>0
    fitz, rel_c50, _, rel_c50_emp = fit_overall_suppression(all_resps_data, all_preds_data)
  curr_suppr['rel_c50'] = np.float32(rel_c50);
  curr_suppr['rel_c50_emp'] = np.float32(rel_c50_emp);
  #### Now, recapitulate the key measures for the dataframe
  if fitz is not None:
    # and add the model prediction given the input drive (i.e. predMean)
    mixture_exp['mod_pred'] = np.maximum(myFit(mixture_exp['predMean'], *fitz), ltThresh)
    # --- only keep mixtures in this dataframe
    mixute_exp_mixs = mixture_exp[mixture_exp['disp']>1];
    to_use = mixute_exp_mixs;
    # 1. Relative suppression (expected/measured)
    to_use['rel_err'] = to_use['respMean'] - to_use['mod_pred']
    # 2. Suppression index  [r-p]/[r+p]
    to_use['supr_ind'] = (to_use['respMean']-to_use['mod_pred'])/(to_use['respMean']+to_use['mod_pred'])
    err_offset = 2*np.std(to_use['rel_err'])
    to_use['supr_ind_wOffset'] = (to_use['respMean']-to_use['mod_pred'])/(to_use['respMean']+to_use['mod_pred']+err_offset)
    supr_ind_str = 'supr_ind_wOffset' if supr_ind_prince else 'supr_ind'
    # 3. rel. supr
    to_use['rel_supr'] = to_use['respMean']/to_use['mod_pred']
    only_pos = np.logical_and(to_use['respMean']>0, to_use['mod_pred']>0);
    curr_suppr['var_expl'] = np.float32(hf.var_explained(to_use['respMean'][only_pos], to_use['mod_pred'][only_pos], sfVals=None));

  ############
  ### organize stimulus information
  ############
  all_disps = stimVals[0];
  all_cons = stimVals[1];
  all_sfs = stimVals[2];
  sf_xmin, sf_xmax = 0.9*all_sfs[0], 1.1*all_sfs[-1]

  nCons = len(all_cons);
  nSfs = len(all_sfs);
  nDisps = len(all_disps);

  # max resp AMONG non-single grats only
  maxResp_y = np.nanmax(respMean[1:])
  maxResp_x = np.nanmax(predResps[1:])
  maxResp = np.maximum(maxResp_x, maxResp_y);
  # by disp
  clrs_d = cm.viridis(np.linspace(0,0.75,nDisps-1));
  lbls_d = ['disp: %s' % str(x) for x in range(nDisps)];
  # by sf
  # --- the 2 commented out lines out below would give us a more common set of colors, even for experiments with fewer dispersions...
  val_sfs = None;
  if ref_all_sfs:
    val_sfs = hf.get_valid_sfs(S, disp=0, con=val_con_by_disp[0][0], expInd=expInd) # pick
  else:
    val_sfs = hf.get_valid_sfs(S, disp=1, con=val_con_by_disp[1][0], expInd=expInd) # pick
    # -- annoying (only applies to V1/27-28 [incomplete expts]), but make sure we check for valid trials at all mixture cons
    # ---- why? because it was an incomplete expt, some contrast levels for disp=1 didn't have all sfs
    for vcbd in val_con_by_disp[1]:
      v_sfs_curr = hf.get_valid_sfs(S, disp=1, con=vcbd, expInd=expInd) # pick
      if val_sfs is None:
        val_sfs = v_sfs_curr;
      else:
        val_sfs = np.union1d(val_sfs, v_sfs_curr);
  clrs_sf = cm.viridis(np.linspace(0,.95,len(val_sfs))); # was 0,0.75
  lbls_sf = ['sf: %.2f' % all_sfs[x] for x in val_sfs];
  # by con
  val_con = all_cons;
  clrs_con = cm.viridis(np.linspace(0,.75,len(val_con)));
  lbls_con = ['con: %.2f' % x for x in val_con];

  if make_plots:
    ############
    ### create the key figure (i.e. Abramov-Levine '75)
    ############
    width_frac = 0.475
    extra_height = 1.5/width_frac;
    fSuper, ax = plt.subplots(nRows, nCols, figsize=hf.set_size(width_frac*tex_width, extra_height=extra_height), 
                              gridspec_kw={'height_ratios': [5, 3, 2]});
    mrkrsz = mpl.rcParams['lines.markersize']*0.75; # when we make the size adjustment above, the points are too large for the scatter
    mew = 0.2 * mrkrsz; # just very faint

  allMix = [];
  allSum = [];

  ### plot reference tuning [row 1 (i.e. 2nd row)]
  ## on the right, SF tuning (high contrast)
  #sf_ref_row = 3

  if use_mod_resp>0: # i.e. we are using model responses, then will need to get these values specifically
    sfRef = hf.nan_rm(resps_data[0][0,:,-1])
  else:
    sfRef = hf.nan_rm(respMean[0, :, -1]); # high contrast tuning
  ref_std = hf.nan_rm(respStd[0,:,-1]);
  if make_plots:
    ax[nRows-1].errorbar(all_sfs, sfRef, yerr=ref_std, color='k', marker='o', label='ref. tuning', clip_on=False, linestyle='None')
    ax[nRows-1].set_xscale('log')
    ax[nRows-1].set_xlim((sf_xmin, sf_xmax));
    ax[nRows-1].set_xlabel('Spatial frequency (c/deg)')
    ax[nRows-1].set_ylabel('Response (spikes/s)')

  #####
  ## then on the left, RVC (peak SF) --> in same position regardless of full or simplified plot
  #####
  sfPeak = np.argmax(sfRef); # stupid/simple, but just get the rvc for the max response
  v_cons_single = val_con_by_disp[0]
  rvcRef = hf.nan_rm(respMean[0, sfPeak, v_cons_single]) if use_mod_resp==0 else hf.nan_rm(resps_data[0][0, sfPeak, v_cons_single]);
  # now, if possible, let's also plot the RVC fit
  rel_rvc = None # default to this...
  if make_plots: # only load the rvc if plotting
    if rvcFits is not None:
      rvcFits = hf.get_rvc_fits(dataPath, expInd, which_cell, rvcName=rvcName, rvcMod=rvcMod);
      rel_rvc = rvcFits[0]['params'][sfPeak]; # we get 0 dispersion, peak SF
      c50, pk = hf.get_c50(rvcMod, rel_rvc), rvcFits[0]['conGain'][sfPeak];
    else:
      try:
        rvcFits_temp = hf.np_smart_load('%s%s' % (dataPath, hf.rvc_fit_name(rvcName, rvcMod, dir=rvcDir)));
        rvcFits_temp = rvcFits_temp[which_cell-1];
        rel_rvc = rvcFits_temp['params'][0, sfPeak]; # we get 0 dispersion, peak SF
        try: # assume that it's f0 type (altExp, since that's the only time we should here)
          c50, pk = hf.get_c50(rvcMod, rel_rvc), rvcFits_temp['conGain'][0, sfPeak]; # zero dispersion
        except:
          c50, pk = hf.get_c50(rvcMod, rel_rvc), rvcFits_temp[0]['conGain'][sfPeak];
      except:
        c50, pk = np.nan, np.nan
   
  if rel_rvc is not None and make_plots:
    plt_cons = np.geomspace(all_cons[0], all_cons[-1], 50);
    c50_emp, c50_eval = hf.c50_empirical(rvcMod, rel_rvc); # determine c50 by optimization, numerical approx.
    # and save it
    curr_suppr['c50'] = np.float32(c50); curr_suppr['conGain'] = np.float32(pk);
    curr_suppr['c50_emp'] = np.float32(c50_emp); curr_suppr['c50_emp_eval'] = np.float32(c50_eval[0]);
  else:
    curr_suppr['c50'] = np.nan; curr_suppr['conGain'] = np.nan;
    curr_suppr['c50_emp'] = np.nan; curr_suppr['c50_emp_eval'] = np.nan;
    
  dispRats = [];
  disps_plt = range(0,nDisps) if spec_disp is None else range(spec_disp,spec_disp+1);

  if respMeasure == 0 and incl_baseline: # i.e. DC
    #stimDur = hf.get_exp_params(expInd).stimDur;
    if use_mod_resp>0:
      baseline_add = model_baseline # 
    else:
      baseline_add = hf.blankResp(expData, expInd)[0] # rate
  else:
    baseline_add = 0;

  for d in disps_plt: #range(0,nDisps):#nDisps):
    if d == 0: # we don't care about single gratings!
      dispRats = [];
      continue; 
    v_cons = np.array(val_con_by_disp[d]);
    n_v_cons = len(v_cons);

    # plot split out by each contrast [0,1]
    for c in reversed(range(n_v_cons)):
      if spec_con is not None:
        if c!=spec_con: # use for contrast-specific plots
          continue;
      
      v_sfs = hf.get_valid_sfs(S, d, v_cons[c], expInd)
      for iii, s in enumerate(v_sfs):
        mixResp = respMean[d, s, v_cons[c]] + baseline_add;
        allMix.append(mixResp);
        sumResp = predResps[d, s, v_cons[c]] + baseline_add
        allSum.append(sumResp);
        if make_plots:
          try:
            sfInd = np.where(np.array(val_sfs) == s)[0][0]; # will only be one entry, so just "unpack"
            if d == 1 and c == 0: # just make the label once...
              if altHollow:
                ax[0].plot(sumResp, mixResp, 'o', label=lbls_sf[sfInd], clip_on=False, markersize=mrkrsz - mew*np.mod(sfInd,2), markeredgecolor='w' if np.mod(sfInd,2)==0 else clrs_sf[sfInd], markeredgewidth=mew, markerfacecolor='None' if np.mod(sfInd,2)==1 else clrs_sf[sfInd]);
              else:
                ax[0].plot(sumResp, mixResp, 'o', color=clrs_sf[sfInd], label=lbls_sf[sfInd], clip_on=False, markersize=mrkrsz, markeredgecolor='w', markeredgewidth=mew);
            else:
              if altHollow:
                ax[0].plot(sumResp, mixResp, 'o', color=clrs_sf[sfInd], clip_on=False, markersize=mrkrsz - mew*np.mod(sfInd,2), markeredgecolor='w' if np.mod(sfInd,2)==0 else clrs_sf[sfInd], markeredgewidth=mew, markerfacecolor='None' if np.mod(sfInd,2)==1 else clrs_sf[sfInd]);
              else:
                ax[0].plot(sumResp, mixResp, 'o', color=clrs_sf[sfInd], clip_on=False, markersize=mrkrsz, markeredgecolor='w', markeredgewidth=mew);
          except:
            pass;
    #if make_plots:
    #  ax[0].axis('scaled');
    # plot averaged across all cons/sfs (i.e. average for the whole dispersion) [1,0]
    mixDisp = respMean[d, :, :].flatten();
    sumDisp = predResps[d, :, :].flatten();
    mixDisp, sumDisp = zr_rm_pair(mixDisp, sumDisp, 0.5);
    curr_rats = np.divide(mixDisp, sumDisp)
    curr_mn = geomean(curr_rats); curr_std = np.std(np.log10(curr_rats));
    dispRats.append(curr_mn);

  ### plot averaged across all cons/disps
  sfInds = []; sfRats = []; sfRatStd = []; 
  sfErrs = []; sfErrsStd = []; sfErrsInd = []; sfErrsIndStd = []; sfErrsRat = []; sfErrsRatStd = [];
  curr_errNormFactor = [];
  ### Organize all of the metrics by SF
  for s in range(len(val_sfs)):
    try: # not all sfs will have legitmate values;
      # only get mixtures (i.e. ignore single gratings)
      mixSf = respMean[1:, val_sfs[s], :].flatten();
      sumSf = predResps[1:, val_sfs[s], :].flatten();
      mixSf, sumSf = zr_rm_pair(mixSf, sumSf, ltThresh);
      rats_curr = np.divide(mixSf, sumSf); 
      sfInds.append(s); sfRats.append(geomean(rats_curr)); sfRatStd.append(np.std(np.log10(rats_curr)));

      if fitz is not None:
        #curr_NR = myFit(sumSf, *fitz); # unvarnished
        curr_NR = np.maximum(myFit(sumSf, *fitz), ltThresh); # thresholded at 0.5...

        curr_err = mixSf - curr_NR;
        sfErrs.append(np.mean(curr_err));
        sfErrsStd.append(np.std(curr_err))

        denom_offset = err_offset if supr_ind_prince else 0;
        curr_errNorm = np.divide(mixSf - curr_NR, mixSf + curr_NR + denom_offset);
        sfErrsInd.append(np.nanmean(curr_errNorm));
        sfErrsIndStd.append(np.nanstd(curr_errNorm))

        curr_errRat = np.divide(mixSf, curr_NR);
        sfErrsRat.append(np.nanmean(curr_errRat));
        sfErrsRatStd.append(np.nanstd(curr_errRat));

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
  if not reducedSave:
    curr_suppr['supr_sf'] = np.float32(sfRats);

  ### residuals from fit of suppression
  add_log_jitter = lambda x, frac, log_step: np.exp(np.log(x) + (np.random.rand()-0.5)*frac*log_step)
  sfs = np.unique(to_use['sf']);
  # --- ok_inds used to index into to_use (we exclude conditions that have either pred or resp < ltThresh)
  ok_inds = np.logical_and(to_use['respMean']>ltThresh, to_use['predMean']>ltThresh)
  # --- e.g. to_use[ok_inds].groupby('sf')...

  if fitz is not None:
    # -- and normalized by the prediction output response + output respeonse
    val_errs = np.logical_and(~np.isnan(sfErrsRat), np.logical_and(np.array(sfErrsIndStd)>0, np.array(sfErrsIndStd) < 2));

    norm_subset = np.array(sfErrsInd)[val_errs];
    normStd_subset = np.array(sfErrsIndStd)[val_errs];
    row_ind_offset = -3 if simple_plot else 0;

    # compute the unsigned "area under curve" for the sfErrsInd, and normalize by the octave span of SF values considered
    val_x = all_sfs[val_sfs][sfInds][val_errs];
    try:
      # first, figure out which of the overall sfs are in the ok_inds subset
      ok_sfs = to_use[ok_inds]['sf'].unique();
      ok_curr = np.in1d(sfs, ok_sfs);
      ind_var = np.var(to_use[ok_inds].groupby('sf')['supr_ind'].mean()[val_errs[ok_curr]]);
      ind_var_offset = np.var(to_use[ok_inds].groupby('sf')['supr_ind_wOffset'].mean()[val_errs[ok_curr]]);
      curr_suppr['sfErrsInd_VAR'] = np.float32(ind_var);
      curr_suppr['sfErrsInd_VAR_prince'] = np.float32(ind_var_offset);
      # take weighted variance?
      # variance is sqrdErr/(N-1) --> here, we just apply weights and divide out the avg. weight
      mnz = to_use[ok_inds].groupby('sf')['supr_ind_wOffset'].mean()[val_errs[ok_curr]];
      inverrs = 1./to_use[ok_inds].groupby('sf')['supr_ind_wOffset'].sem()[val_errs[ok_curr]];
      wtd_var_numer = inverrs * np.square(mnz - np.mean(mnz));
      wtd_var_denom = (len(ok_curr)-1) * np.mean(inverrs); # mean weight * (n-1)
      wtd_var = np.sum(wtd_var_numer)/wtd_var_denom;
      curr_suppr['sfErrsInd_wtdVar_prince'] = np.float32(wtd_var);
    except:
      curr_suppr['sfErrsInd_VAR'] = np.nan;
      curr_suppr['sfErrsInd_VAR_prince'] = np.nan;
      ind_var_offset = np.nan;
      curr_suppr['sfErrsInd_wtdVar_prince'] = np.nan;
  else: # if we don't have a fit...
    curr_suppr['sfErrsInd_VAR'] = np.nan
    curr_suppr['sfRat_VAR'] = np.nan

  ax[nRows-2].axhline(0, ls='--', color='k', alpha=ref_line_alpha)
  #ax[nRows-2].set_xlabel('Spatial Frequency (c/deg)')
  ax[nRows-2].set_xscale('log')
  ax[nRows-2].set_xlim((sf_xmin, sf_xmax));
  #ax[nRows-2].set_xlim((0.1, 10));
  #ax[nRows-2].set_xlim((np.min(all_sfs), np.max(all_sfs)));
  ax[nRows-2].set_ylim((-0.5, 0.5));
  ax[nRows-2].set_ylabel(r'Suppression index, $\delta$');
  #ax[nRows-2].xaxis.label.set_visible(False)
  ax[nRows-2].tick_params('x', labelbottom=False)
  #ax[nRows-2].errorbar(all_sfs[val_sfs][sfInds][val_errs], norm_subset, normStd_subset, color='k', marker='o', linestyle='-', alpha=0.2)
  try:
    ax[nRows-2].plot(all_sfs[val_sfs][sfInds][val_errs], norm_subset, color='k', marker=None, linestyle='-', alpha=0.5)
  except:
    pdb.set_trace();
    pass; # UHOH!!!!
  #ax[nRows-2].plot(all_sfs[val_sfs][sfInds][val_errs], to_use.groupby('sf')['supr_ind'].mean()[val_errs], color='k', marker=None, linestyle='--', alpha=0.5)
  mrkrsz = np.square(mpl.rcParams['lines.markersize']*0.5); # when we make the size adjustment above, the points are too large for the scatter --> and in scatter plots, s=mrks^2
  sf_inds_curr = np.searchsorted(sfs, to_use['sf']);
  try:
    sct_clrs = clrs_sf[sf_inds_curr];
  except:
    sct_clrs = 'k'
  ok_inds = np.logical_and(to_use['respMean']>ltThresh, to_use['predMean']>ltThresh)
  try:
    ax[nRows-2].scatter([add_log_jitter(x, 0.3 if simple_plot else 0.7, np.nanmean(np.log(sfs[1:]/sfs[0:-1]))) for x in to_use['sf'][ok_inds]], to_use[supr_ind_str][ok_inds], alpha=0.5, s=mrkrsz, color=sct_clrs[ok_inds])
  except: # for the 1-2 V1 cells with oddities, the above won't work --> just skip it
    pass;
  # - and put that value on the plot
  ax[nRows-2].text(0.3, 0.35, 'var=%.4f [%.1e]' % (ind_var_offset if supr_ind_prince else ind_var, wtd_var), fontsize='xx-small');

  #########
  ### NOW, let's evaluate the (derivative of the) SF tuning curve and get the correlation with the errors
  #########
  mod_sfs = np.geomspace(all_sfs[0], all_sfs[-1], 1000);
  mod_resp = hf.get_descrResp(dfit_curr, mod_sfs, DoGmodel=dMod_num);
  deriv = np.divide(np.diff(mod_resp), np.diff(np.log10(mod_sfs)))
  deriv_norm = np.divide(deriv, np.maximum(np.nanmax(deriv), np.abs(np.nanmin(deriv)))); # make the maximum response 1 (or -1)
  # - then, what indices to evaluate for comparing with sfErr?
  errSfs = all_sfs[val_sfs][sfInds];
  mod_inds = [np.argmin(np.square(mod_sfs-x)) for x in errSfs];
  deriv_norm_eval = deriv_norm[mod_inds];
  # -- plot ref. tuning - WILL BE DIFFERENT depending on data vs. comp. model responses!
  if use_mod_resp>0: # then let's plot the model tuning here
    ok_inds = ~np.isnan(respMean[0,:,-1])
    sfs_to_plt, resps_to_plt = all_sfs[ok_inds], respMean[0,ok_inds,-1]; # high con, single grating resps
  else:
    sfs_to_plt, resps_to_plt = mod_sfs, mod_resp
  if make_plots:
    ax[nRows-1].plot(sfs_to_plt, resps_to_plt, 'k--', label='fit (g)')
    for jj, axis in enumerate([ax[nRows-1].xaxis, ax[nRows-1].yaxis]):
      if jj == 0:
        axis.set_major_formatter(FuncFormatter(lambda x,y: '%d' % x if x>=1 else '%.1f' % x)) # this will make everything in non-scientific notation!
        core_ticks = np.array([1,3]);
        pltd_sfs = all_sfs[val_sfs];
        if np.min(pltd_sfs)<=0.4:
            core_ticks = np.hstack((0.3, core_ticks));
        if np.max(pltd_sfs)>=7:
            core_ticks = np.hstack((core_ticks, 10));
        axis.set_ticks(core_ticks)

  if fitz is not None:
    norm_sfErr = np.divide(sfErrs, np.nanmax(np.abs(sfErrs)));
    norm_sfErrInd = np.divide(sfErrsInd, np.nanmax(np.abs(sfErrsInd))); # remember, sfErrsInd is normalized per condition; this is overall
    # CORRELATION WITH TUNING
    non_nan = np.logical_and(~np.isnan(norm_sfErr), ~np.isnan(sfRefShift))
    corr_sf, corr_sfN = np.corrcoef(sfRefShift[non_nan], norm_sfErr[non_nan])[0,1], np.corrcoef(sfRefShift[non_nan], norm_sfErrInd[non_nan])[0,1]
    curr_suppr['corr_tuneWithErr'] = np.float32(corr_sf);
    curr_suppr['corr_tuneWithErrsInd'] = np.float32(corr_sfN);
  else:
    curr_suppr['corr_derivWithErr'] = np.nan;
    curr_suppr['corr_derivWithErrsInd'] = np.nan;

  # make a polynomial fit - DEPRECATE 
  try:
    hmm = np.polyfit(allSum, allMix, deg=1) # returns [a, b] in ax + b 
  except:
    hmm = [np.nan];
  curr_suppr['supr_index'] = np.float32(hmm[0]);
  # compute area under the curve --- both for the linear expectation and for the Naka-Rushton
  xvals = np.linspace(0, np.nanmax(all_preds), 100);
  lin_area = np.trapz(xvals, x=xvals);
  nr_area = np.trapz(myFit(xvals, *fitz), x=xvals);
  curr_suppr['supr_area'] = np.float32(nr_area/lin_area);

  if make_plots:
    # plot the fitted model on each axis
    pred_plt = np.linspace(0, np.nanmax(all_preds), 100);
    if fitz is not None:
      ax[0].plot(pred_plt, myFit(pred_plt, *fitz), 'r--', label='fit')
    ax[0].axis('scaled') # scaled?
    ax[0].set_xlabel('Grating prediction (spikes/s)');
    ax[0].set_ylabel('Mixture response (spikes/s)');
    ax[0].plot([0, 1*maxResp_y], [0, 1*maxResp_y], 'k--', alpha=ref_line_alpha)
    ax[0].set_xlim((-5, maxResp_x));
    ax[0].set_ylim((-5, 1.1*maxResp_y));
    #ax[j].set_title('rAUC|c50: %.2f|%.2f [%.1f%s%%]' % (curr_suppr['supr_area'], curr_suppr['rel_c50'], curr_suppr['var_expl'], '\\' if useTex else ''))
    #ax[j].set_title('Suppression index: %.2f|%.2f [%.1f\%%]' % (curr_suppr['supr_index'], curr_suppr['rel_c50'], curr_suppr['var_expl']));
    #ax[j].legend(fontsize='xx-small', ncol=2); # want two legend columns for SF

    cell_name_use = cellName.replace('_','\_') if useTex else cellName
    fSuper.suptitle('%s %s#%d [%s]; f1f0 %.2f]' % (cellType, '\\' if useTex else '', which_cell, cell_name_use, f1f0_rat), y=1-0.03*simple_plot, fontsize='x-small')

    #fSuper.tight_layout();
  fSuper.subplots_adjust(hspace=0.15, top=1.03, left=0.25);
  sns.despine(fig=fSuper, offset=sns_offset)
  if spec_disp is None:
    disp_str = '';
  else:
    disp_str = '_disp%d' % spec_disp;
  if spec_con is None:
    con_str = '';
  else:
    con_str = '_con%d' % spec_con;

  if make_plots:
    if fitList is None:
      save_name = 'cell_%03d%s%s.pdf' % (which_cell, disp_str, con_str);
    else:
      save_name = 'cell_%03d_mod%s%s%s%s%s.pdf' % (which_cell, hf.fitType_suffix(fitType, dgNormFunc=dgNormFunc), hf.lgnType_suffix(lgnFrontEnd, lgnConType=1), disp_str, con_str, '_dtRef' if dataAsRef else '')
    pdfSv = pltSave.PdfPages(str(save_locSuper + save_name));
    pdfSv.savefig(fSuper)
    pdfSv.close();

  
  return curr_suppr;
  ####
  ## END of plot_superposition_func
  ####

 
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
    if len(sys.argv)>3:
      plt_supr_ind = bool(sys.argv[3]); # if 1, then we plot
    else:
      plt_supr_ind = True;
    if len(sys.argv)>4:
      supr_ind_prince = int(sys.argv[4])==1;
    else:
      supr_ind_prince = True;
    if len(sys.argv)>5:
      use_mod_resp = int(sys.argv[5]);
    else:
      use_mod_resp = 0;
    if len(sys.argv)>6:
      nBoots = int(sys.argv[6]);
      if nBoots<=1:
        nBoots = 1;
    else:
      nBoots = 1;
    if len(sys.argv)>7:
      normType = int(sys.argv[7]);
    else:
      normType = 1; # default to flat
    if len(sys.argv)>8:
      lgnOn = int(sys.argv[8]);
    else:
      lgnOn = 0; # default to no LGN
    if len(sys.argv)>9:
      excType = int(sys.argv[9]);
    else:
      excType = 1; # default to dG (not flex. Gauss, which is 2)
    if len(sys.argv)>10:
      dgNormFunc = int(sys.argv[10])
    else:
      dgNormFunc = 0; # default to old method (log Gauss)
    if len(sys.argv)>11:
      fixRespExp = int(sys.argv[11]);
      if fixRespExp <= 0:
        fixRespExp = None
    else:
      fixRespExp = 2; # default to fixing at 2
    if len(sys.argv)>12:
      spec_disp = int(sys.argv[12]);
    else:
      spec_disp = None;
    if len(sys.argv)>13:
      spec_con = int(sys.argv[13]);
    else:
      spec_con = None;
      
    #dataAsRef = False;
    dataAsRef = True;

    if nBoots<=1:
      nBoots = 1;
      isBoot = False
      resample = False;
    else:
      isBoot = True;
      resample = True;
      kys_to_skip = ['conGain', 'c50_emp', 'c50_emp_eval', 'c50', 'f1f0']
    # make plots only if not resample...
    make_plots = False if resample else True;
    reducedSave = True if resample else False;
    # unlikely to change ever...
    useHPCfit = True;
    scheduler=False
    singleGratsOnly=False

    if asMulti:
      # set up the name
      dataPath = os.getcwd() + '/' + expDir + 'structures/'
      from datetime import datetime
      suffix = datetime.today().strftime('%y%m%d')
      if use_mod_resp==0:
        super_name = 'superposition_analysis_%s.npy' % suffix;
      else:
        super_name = 'superposition_analysis_%s_mod%s%s.npy' % (suffix, hf.fitType_suffix(normType, dgNormFunc=dgNormFunc), hf.lgnType_suffix(lgnOn, lgnConType=1));

      # pre-load all large files that we would otherwise load in each function call, declare them as globals so not copied each time
      global dataList, descrFits, fitList
      basePath = os.getcwd() + '/'
      if 'pl1465' in basePath or useHPCfit:
        loc_str = 'HPC';
      else:
        loc_str = '';
      # --- datalist
      dataListNm = hf.get_datalist(expDir, force_full=True, new_v1=True);
      dataList = hf.np_smart_load(dataPath + dataListNm);
      # --- descr. fits
      dFits_mod = hf.descrMod_name(1 if expDir=='LGN/' else 3);
      dt_sf = '230111vEs' if expDir == 'V1/' else '221126vEs' if expDir == 'altExp/' else '220810vEs';
      vecF1 = None if expDir=='altExp/' else 0;
      dFits_base = 'descrFits%s_%s' % (loc_str, dt_sf);
      descrFits_name = hf.descrFit_name(lossType=2, descrBase=dFits_base, modelName=dFits_mod, phAdj=1 if vecF1==0 else None);
      descrFits = hf.np_smart_load(dataPath + descrFits_name);
      # --- comp. model fits
      if use_mod_resp == 2:
        fitBase='fitList%s_pyt_nr230118a%s%s%s' % (loc_str, '_noRE' if fixRespExp is not None else '', '_noSched' if scheduler==False else '', '_sg' if singleGratsOnly else '');
        fitList_nm = hf.fitList_name(fitBase, fitType, lossType=lossType, lgnType=lgnFrontEnd, lgnConType=lgnConType, vecCorrected=-rvcAdj, excType=excType, dgNormFunc=dgNormFunc);
        fitList = hf.np_smart_load(dataPath + fitList_nm);
      else:
        fitList = None;

      from functools import partial
      import cProfile
      import multiprocessing as mp
      nCpu = mp.cpu_count()-5; # heuristics say you should reqeuest at least one fewer processes than their are CPU
      print('***cpu count: %02d***' % nCpu);
      f1f0_rats = np.nan * np.zeros((end_cell-start_cell+1, ));
      with mp.Pool(processes = nCpu) as pool:
        import time
        strt = time.time();
        print('%d boots!' % nBoots)
        for nb in range(nBoots):
          if np.mod(nb, int(nBoots/5))==0: # announce every 20%
            elpsd = time.time() - strt;
            print('boot #%d of %d [t=%.1f minutes]' % (nb, nBoots, elpsd/60));
          sup_perCell = partial(plot_save_superposition, expDir=expDir, use_mod_resp=use_mod_resp, fitType=normType, useHPCfit=useHPCfit, lgnConType=1, lgnFrontEnd=lgnOn, to_save=0, plt_supr_ind=plt_supr_ind, supr_ind_prince=supr_ind_prince, spec_disp=spec_disp, spec_con=spec_con, fixRespExp=fixRespExp, excType=excType, dataAsRef=dataAsRef, dgNormFunc=dgNormFunc, resample=resample, make_plots=make_plots, reducedSave=reducedSave, dataList=dataList, descrFits=descrFits, fitList=fitList);
          #oy = cProfile.runctx('sup_perCell(3)', globals(), locals(), sort='cumtime')
          #pdb.set_trace();
          supFits = pool.map(sup_perCell, zip(range(start_cell, end_cell+1), f1f0_rats));
          # if we're not doing boots, then we're done! otherwise...
          if isBoot:
            if nb==0: # i.e. first time around...
              # load the existing file, if applicable
              if os.path.exists(dataPath + super_name):
                suppr_all = hf.np_smart_load(dataPath + super_name);
              else:
                suppr_all = dict();
              for iii, sup_fit in enumerate(supFits):
                if iii not in suppr_all:
                  suppr_all[iii] = dict();
                for ky in sup_fit.keys():
                  if ky=='f1f0':
                    f1f0_rats[iii] = sup_fit[ky];
                  if ky in kys_to_skip:
                    continue; # don't want to add these
                  suppr_all[iii]['%s_boot' % ky] = np.nan * np.zeros((nBoots, ));
                  # if this is the 1st time around, let's save f1f0_rats to avoid the costly call within sup_perCell
            # then, regardless of boot #, add the value!
            for iii, sup_fit in enumerate(supFits):
              for ky in sup_fit.keys():
                if ky in kys_to_skip:
                  continue; # don't want to add these
                suppr_all[iii]['%s_boot' % ky][nb] = sup_fit[ky];
        # end of boots...
        pool.close();

      if not isBoot:
        ### do the load/setting HERE if not boot!
        if os.path.exists(dataPath + super_name):
          suppr_all = hf.np_smart_load(dataPath + super_name);
        else:
          suppr_all = dict();
        for iii, sup_fit in enumerate(supFits):
          if iii not in suppr_all:
            suppr_all[iii] = dict();
          for ky in sup_fit.keys():
            # why add keys like this? so that we don't overwrite existing keys
            suppr_all[iii][ky] = sup_fit[ky];
      # but save, regardless of whether boot or not
      np.save(dataPath + super_name, suppr_all);

    else: # i.e. not multi
      resample = False;
      make_plots = False if resample else True;
      plot_save_superposition(cell_num, expDir, to_save=1, plt_supr_ind=plt_supr_ind, use_mod_resp=use_mod_resp, supr_ind_prince=supr_ind_prince, lgnConType=1, lgnFrontEnd=lgnOn, fitType=normType, spec_disp=spec_disp, spec_con=spec_con, fixRespExp=fixRespExp, excType=excType, dataAsRef=dataAsRef, dgNormFunc=dgNormFunc, verbose=True, resample=resample, make_plots=make_plots, reducedSave=reducedSave);

