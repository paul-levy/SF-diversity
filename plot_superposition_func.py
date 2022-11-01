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
warnings.filterwarnings('once');

def get_responses(expData, which_cell, expInd, expDir, dataPath, respMeasure, stimVals, val_con_by_disp, rvcFits=None, phAdvName=None, vecF1=0, f1_expCutoff=2, rvcDir=1, val_by_stim_val=None, sum_power=1):
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
  resps_data, _, _, _, _ = hf.tabulate_responses(expData, expInd, overwriteSpikes=spikes, respsAsRates=rates, modsAsRate=rates, sum_power=sum_power);

  return resps_data, respAll, respsPhAdv_mean_ref, respsPhAdv_mean_pred, baseline, adjMeansByComp, val_tr_by_cond;

def get_model_responses(expData, fitList, expInd, which_cell, excType, fitType, f1f0_rat, respMeasure, baseline, lossType=1, lgnFrontEnd=0, newMethod=1, lgnConType=1, _applyLGNtoNorm=0, _sigmoidSigma=5, recenter_norm=2, normToOne=1, debug=False, use_mod_resp=2):
  # This is ONLY for getting model responses
  if use_mod_resp == 1:
    curr_fit = fitList[which_cell-1]['params'];
    modResp = mod_resp.SFMGiveBof(curr_fit, expData, normType=fitType, lossType=lossType, expInd=expInd, cellNum=which_cell, excType=excType)[1];
    if f1f0_rat < 1: # then subtract baseline..
      modResp = modResp - baseline*hf.get_exp_params(expInd).stimDur; 
    # now organize the responses
    resps = hf.tabulate_responses(expData, expInd, overwriteSpikes=modResp, respsAsRates=False, modsAsRate=False)[0];

  elif use_mod_resp == 2: # then pytorch model!
    resp_str = hf_sf.get_resp_str(respMeasure)
    if (which_cell-1) in fitList:
      curr_fit = fitList[which_cell-1][resp_str]['params'];
    else:
      curr_fit = fitList; # we already passed in parameters
    # TEMP:
    #curr_fit[5] = 0; # turn off early noise
    # END TEMP:
    model = mrpt.sfNormMod(curr_fit, expInd=expInd, excType=excType, normType=fitType, lossType=lossType, lgnFrontEnd=lgnFrontEnd, newMethod=newMethod, lgnConType=lgnConType, applyLGNtoNorm=_applyLGNtoNorm, normToOne=normToOne)
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

    if respMeasure == 0: # if DC, then subtract baseline..., as determined from data (why not model? we aren't yet calc. response to no stim, though it can be done)
      stimDur = hf.get_exp_params(expInd).stimDur
      if normToOne==1 and newMethod==1: # then noiseLate is exactly the noiseLate
        modResp_full -= model.noiseLate.detach().numpy() # Model is counts --> no need to factor in stimDur
      else: # sub the data baseline, since our model should've found that anyway...
        modResp_full = modResp_full - baseline*stimDur;

    # TODO: This is a work around for which measures are in rates vs. counts (DC vs F1, model vs data...)
    stimDur = hf.get_exp_params(expInd).stimDur;
    asRates = False;
    divFactor = stimDur if asRates == 0 else 1;
    modResp_full = np.divide(modResp_full, divFactor);
    # now organize the responses
    #resps = hf.organize_resp(modResp_full, expData, expInd);
    resps = hf.tabulate_responses(expData, expInd, overwriteSpikes=modResp_full, respsAsRates=asRates, modsAsRate=asRates)[0];

  if debug:
    return model.respPerCell(dw.trInf, debug=True, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm);

  return resps;

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
  rel_c50 = np.divide(fit[-1], np.max(all_preds[val_inds]));
  
  return fit, rel_c50, myFit

def make_f1_comp_plots(expData, which_cell, mixture_df, respMean, resp_std, predResps, comp_resp_org, val_tr_org, save_loc, stimVals, val_con_by_disp, isol_pred=True, specify_ticks=True, tex_width=469, sns_offset = 5):
  ''' inputs: expData, which_cell: cell structure, cell num
              mixture_df: dataframe organizing all of the conditions, responses, etc
              respMean, resp_std, predResps: mean/std of mixture responses, mixture predictions based on isol. comp responses
              comp_resp_org, val_tr_org: organized by condition (disp/sf/con), the component responses in mixtures and the trial numbers assoc. w/each mixture
  '''
  # here are the single grating sfs/cons to consider

  disp_mixes = []; # save all the dispersion plots to save in one PDF

  if comp_resp_org is not None:

      marker_by_disp = ['', '<', '>', '*']

      save_loc_curr = save_loc + 'byComp/'
      saveName = 'cell_%02d%s.pdf' % (which_cell, '_wPred' if isol_pred else '')
      save_loc_disp = save_loc + 'byDisp/'
      saveName_disps = 'cell_%02d.pdf' % (which_cell)

      con_inds = val_con_by_disp[0];
      disps, sfs, cons, all_cons = stimVals[0], stimVals[2], stimVals[1][val_con_by_disp[0]], stimVals[1];
      nsfs, ncons = len(sfs), len(cons);

      clrs = cm.viridis(np.linspace(0, 1, nsfs))

      # make the figure for plots organized by isolated component
      #f, ax = plt.subplots(nrows=nsfs, ncols=ncons, figsize=hf.set_size(tex_width, subplots=(nsfs,ncons), extra_height=2), sharey='row');
      f, ax = plt.subplots(nrows=nsfs, ncols=ncons, figsize=(ncons*2.75,nsfs*2), sharey='row')

      # plot isol. component responses
      for d in range(1,len(disps)):

          curr_subset = mixture_df[mixture_df['disp']==disps[d]]
          curr_mix_sfs, curr_mix_cons = mixture_df[mixture_df['disp']==disps[d]]['sf'].unique(), mixture_df[mixture_df['disp']==disps[d]]['total_con'].unique()
          # set up the figure for looking at specific mixtures (new figure for each dispersion)
          curr_disp_nrow, curr_disp_ncol = len(curr_mix_sfs), len(curr_mix_cons);
          g, disp_ax = plt.subplots(nrows=curr_disp_nrow, ncols=curr_disp_ncol, 
                                    #figsize=hf.set_size(tex_width, extra_height=2, subplots=(curr_disp_nrow, curr_disp_ncol)), 
                                    #sharex=True, sharey=True)
                                    figsize=(len(curr_mix_cons)*3,len(curr_mix_sfs)*2.5), sharex=True, sharey=True)

          for sf_ind_mix, con_ind_mix in itertools.product(range(val_tr_org.shape[1]), range(val_tr_org.shape[2])):

              if np.all(np.isnan(comp_resp_org[d, sf_ind_mix, con_ind_mix])):
                  continue;

              disp_con_ind = np.where(all_cons[con_ind_mix] == curr_mix_cons)[0][0]
              disp_sf_ind = np.where(sfs[sf_ind_mix] == curr_mix_sfs)[0][0]

              # How much was the mixture response as a fraction of the prediction, i.e. overall mixture response suppressed?
              overall_red = respMean[d, sf_ind_mix, con_ind_mix]/predResps[d, sf_ind_mix, con_ind_mix]

              # Otherwise, we have a valid reponse!               
              curr_trials = hf.nan_rm(val_tr_org[d, sf_ind_mix, con_ind_mix])
              curr_sfs_plt = [];
              for comp_i, resp_i in enumerate(comp_resp_org[d, sf_ind_mix, con_ind_mix]):
                  curr_sf, curr_con = np.unique(expData['sf'][comp_i, curr_trials.astype('int')]), np.unique(expData['con'][comp_i, curr_trials.astype('int')])
                  sf_ind = np.where(np.isclose(curr_sf, sfs, atol=0.001))[0][0]
                  con_ind = np.where(np.isclose(curr_con, cons, atol=0.01))[0][0]

                  ### plot the isolated component response - first, organized by component
                  rand_xoffset = hf.random_in_range([0,0.3])[0];
                  if curr_sf==sfs[sf_ind_mix]:
                      rand_sgn = np.sign(np.random.rand()-0.5); # just to avoid values showing up exactly on top of the specific response
                      rand_xloc = rand_sgn*(1.01 + hf.random_in_range([0, 0.04])[0]);
                  else: # scale the distance according to how far the mixture SF was from the current SF component
                      rel_step = np.log2(sfs[sf_ind_mix]/curr_sf)
                      rand_xloc = 1 + rel_step*hf.random_in_range([0.05, 0.07])[0];
                  ax[sf_ind, con_ind].plot(rand_xloc, resp_i, color=clrs[sf_ind_mix], alpha=all_cons[con_ind_mix], 
                                           marker=marker_by_disp[d], label='%.2f,%.2f[%d]' % (all_cons[con_ind_mix], sfs[sf_ind_mix], disps[d]))
                  if isol_pred:
                      pred_i = respMean[0, sf_ind, con_ind] * overall_red
                      ax[sf_ind, con_ind].plot(rand_xloc, pred_i, color=clrs[sf_ind_mix], alpha=all_cons[con_ind_mix],
                                   marker=marker_by_disp[d], fillstyle='none')
                      # and plot connecting line
                      ax[sf_ind, con_ind].plot([rand_xloc, rand_xloc], [resp_i, pred_i], color=clrs[sf_ind_mix], alpha=all_cons[con_ind_mix],marker=None, linestyle='--')

                  ### - then, organized by mixture: get the current SF (to plot all components together), and plot the isol. response
                  curr_sfs_plt.append(sfs[sf_ind]);
                  disp_ax[disp_sf_ind, disp_con_ind].errorbar(sfs[sf_ind], respMean[0, sf_ind, con_inds[con_ind]],
                                           resp_std[0, sf_ind, con_inds[con_ind]], marker='o', color=clrs[sf_ind])
              ### For byDisp organization, plot it together (with lines connecing the isolated mixture responses)
              sf_order = np.argsort(curr_sfs_plt)
              disp_ax[disp_sf_ind, disp_con_ind].semilogx(np.array(curr_sfs_plt)[sf_order], 
                                                      np.array(comp_resp_org[d, sf_ind_mix, con_ind_mix])[sf_order], '-o');
              if disp_con_ind==0: # i.e. left-most column
                  disp_ax[disp_sf_ind, disp_con_ind].set_ylabel('Resp (spks/s)')
              if disp_sf_ind==(len(curr_mix_sfs)-1): # bottom row
                  disp_ax[disp_sf_ind, disp_con_ind].set_xlabel('Spatial frequency (c/deg)');
              log_supr = np.log2(overall_red) if overall_red>0 else np.nan
              disp_ax[disp_sf_ind, disp_con_ind].set_title('%.2f cpd, %.2f [log(supr)=%.2f]' % (sfs[sf_ind_mix], all_cons[con_ind_mix], log_supr), fontsize='medium')

              # format to have integers (not scientific notation)
              for jj, axis in enumerate([disp_ax[disp_sf_ind, disp_con_ind].xaxis, disp_ax[disp_sf_ind, disp_con_ind].yaxis]):
                  axis.set_major_formatter(FuncFormatter(lambda x,y: '%d' % x if x>=1 else '%.1f' % x)) # this will make everything in non-scientific notation!
                  if jj == 0 and specify_ticks: # i.e. x-axis
                      core_ticks = np.array([1]);
                      if np.min(sfs)<=0.2:
                          core_ticks = np.hstack((0.1, core_ticks));
                      if np.max(sfs)>=7:
                          core_ticks = np.hstack((core_ticks, 10));
                      axis.set_ticks(core_ticks)
                      # really hacky, but allows us to put labels at 0.3/3 cpd, format them properly, and not add any extra labels
                      inter_val = 3;
                      axis.set_minor_formatter(FuncFormatter(lambda x,y: '%d' % x if np.square(x-inter_val)<1e-3 else '%.1f' % x if np.square(x-inter_val/10)<1e-3 else '')) # this will make everything in non-scientific notation!
                      axis.set_tick_params(which='minor', pad=5.5); # Determined by trial and error: make the minor/major align??

              g.tight_layout();
              sns.despine(offset=sns_offset, ax=disp_ax[disp_sf_ind, disp_con_ind]);
          # DONE WITH LOOP OVER ALL CONDITIONS
          disp_mixes.append(g);

      # then plot isolated responses
      for sf_ind,con_ind in itertools.product(range(len(sfs)), range(len(cons))):
          ax[sf_ind, con_ind].errorbar(1, respMean[0, sf_ind, con_inds[con_ind]], resp_std[0, sf_ind, con_inds[con_ind]], marker='o', color=clrs[sf_ind]);
          ax[sf_ind, con_ind].legend(fontsize='xx-small', framealpha=0.3, ncol=2);
          ax[sf_ind, con_ind].set_title('%.2f cpd, %.2f' % (sfs[sf_ind], cons[con_ind]), fontsize='small');
          if con_ind==0:
              ax[sf_ind, con_ind].set_ylabel('Resp (spks/s)')
          ax[sf_ind, con_ind].xaxis.set_visible(False)

      # now save
      if not os.path.exists(save_loc_curr):
          os.makedirs(save_loc_curr)
      pdfSv = pltSave.PdfPages(save_loc_curr + saveName);
      pdfSv.savefig(f, bbox_inches='tight') # only one figure here...
      pdfSv.close()

      # and save
      if not os.path.exists(save_loc_disp):
          os.makedirs(save_loc_disp)
      pdfSv = pltSave.PdfPages(save_loc_disp + saveName_disps);
      for gg in disp_mixes:
          pdfSv.savefig(gg)
          plt.close(gg)
      pdfSv.close();

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

def plot_save_superposition(which_cell, expDir, use_mod_resp=0, fitType=1, excType=2, useHPCfit=1, lgnConType=None, lgnFrontEnd=None, force_full=1, f1_expCutoff=2, to_save=1, plt_f1_plots=False, useTex=False, simple_plot=True, altHollow=True, ltThresh=0.5, ref_line_alpha=0.5, ref_all_sfs=False, plt_supr_ind=False, supr_ind_prince=False, sum_power=1, spec_disp = None, spec_con = None):

  # if ref_all_sfs, then colors for superposition plots are referenced across all SFS (not just those that appear for dispersion=1)

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

  rvcName = 'rvcFits%s_220928' % ''# % loc_str
  phAdvName = 'phaseAdvanceFits%s_220928' % ''# % loc_str
  #rvcName = 'rvcFits_220926' if expDir=='LGN/' else 'rvcFits%s_220718' % loc_str
  #phAdvName = 'phaseAdvanceFits%s_220531' % loc_str if expDir=='LGN/' else 'phaseAdvanceFits%s_220718' % loc_str;
  rvcFits = None; # pre-define this as None; will be overwritten if available/needed
  if expDir == 'altExp/': # we don't adjust responses there...
    rvcName = '%s_f0' % rvcName;
  dFits_base = 'descrFits%s_220609' % 'HPC' if expDir=='LGN/' else 'descrFits%s_220721' % 'HPC'
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
    rvcName = None; # Use NONE if getting model responses, only
    fitBase = 'fitList%s_pyt_nr221029a_noRE' % '' # loc_str
    #fitBase = 'fitList%s_pyt_Fnr221021' % loc_str
    #fitBase = 'fitList%s_pyt_221013_noRE_noSched' % loc_str
    fitList_nm = hf.fitList_name(fitBase, fitType, lossType=lossType, lgnType=lgnFrontEnd, lgnConType=lgnConType, vecCorrected=-rvcAdj);
  # ^^^ EDIT rvc/descrFits/fitList names here;

  if use_mod_resp>0:
    print('\n***Fitlist name:[%s]***\n' % fitList_nm);

  ############
  # Before any plotting, fix plotting paramaters
  ############
  plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/paul_plt_style.mplstyle');
  from matplotlib import rcParams
  tex_width = 469; # per \layout in Overleaf on document
  sns_offset = 2; 
  hist_width = 0.9;
  hist_ytitle = 0.94; # moves the overall title a bit further down on histogram plots0

  rcParams.update(mpl.rcParamsDefault)

  fontsz = 12;
  tick_scalar = 1.5;

  rcParams['pdf.fonttype'] = 42
  rcParams['ps.fonttype'] = 42

  if useTex:
    rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
    params = {'text.usetex' : True,
              'font.size' : fontsz,
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
  dataListNm = hf.get_datalist(expDir, force_full=force_full);
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
      vecF1 = 0 # was None?
    else:
      rvcDir = 1;
      vecF1 = 0; # was previously 1, but now we do phAmp, not just vecF1

  dFits_mod = hf.descrMod_name(dMod_num)
  descrFits_name = hf.descrFit_name(lossType=dLoss_num, descrBase=dFits_base, modelName=dFits_mod, phAdj=1 if vecF1==0 else None);

  ## now, let it run
  dataPath = basePath + expDir + 'structures/'
  save_loc = basePath + expDir + 'figures/'
  from datetime import datetime
  save_locSuper = save_loc + 'superposition_%s%s%s%s%s/' % (datetime.today().strftime('%y%m%d'), '_simple' if simple_plot else '', '' if plt_supr_ind else '_mse', '_prince' if supr_ind_prince else '', '_p%d' % sum_power if sum_power!=1 else '')
  if use_mod_resp == 1:
    save_locSuper = save_locSuper + '%s/' % fitBase
  #print('saving %s' % save_locSuper);

  dataList = hf.np_smart_load(dataPath + dataListNm);
  #print('Trying to load descrFits at: %s' % (dataPath + descrFits_name));
  descrFits = hf.np_smart_load(dataPath + descrFits_name);
  if use_mod_resp == 1 or use_mod_resp == 2:
    fitList = hf.np_smart_load(dataPath + fitList_nm);
  else:
    fitList = None;

  if not os.path.exists(save_locSuper):
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
  if simple_plot: # then just 2x2
    nRows, nCols = 2, 2;
  else:
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
  respMeasure = 1 if (f1f0_rat > 1 and expInd>2) else 0;

  # load rvcFits in case needed
  try:
    rvcFits = hf.get_rvc_fits(dataPath, expInd, which_cell, rvcName=rvcName, rvcMod=rvcMod, direc=rvcDir, vecF1=vecF1);
  except:
    rvcFits = None;

  _, stimVals, val_con_by_disp, val_by_stim_val, _ = hf.tabulate_responses(expData, expInd); # call just to get these values (not spikes/activity)
  resps_data, respAll, respsPhAdv_mean_ref, respsPhAdv_mean_preds, baseline, comp_resp_org, val_tr_org = get_responses(expData, which_cell, expInd, expDir, dataPath, respMeasure, stimVals, 
                                                                                                                       val_con_by_disp, rvcFits, phAdvName, vecF1, f1_expCutoff=f1_expCutoff, rvcDir=rvcDir, val_by_stim_val=val_by_stim_val, sum_power=sum_power);

  if fitList is None:
    resps = resps_data; # otherwise, we'll still keep resps_data for reference
  elif fitList is not None: # OVERWRITE the data with the model spikes!
    resps = get_model_responses(expData, fitList, expInd, which_cell, excType, fitType, f1f0_rat, respMeasure, baseline, lossType=lossType, lgnFrontEnd=lgnFrontEnd, lgnConType=lgnConType, _applyLGNtoNorm=_applyLGNtoNorm, recenter_norm=recenter_norm);

  predResps = resps[2] if respsPhAdv_mean_preds is None else respsPhAdv_mean_preds;
  respMean = resps[0] if respsPhAdv_mean_ref is None else respsPhAdv_mean_ref; # equivalent to resps[0];
  respStd = np.nanstd(respAll, -1); # take std of all responses for a given condition
  predStd = resps[3]; # WARNING/todo: if fitList is not None, this might be a problem?
  # compute SEM, too
  findNaN = np.isnan(respAll);
  nonNaN  = np.sum(findNaN == False, axis=-1);
  respSem = np.nanstd(respAll, -1) / np.sqrt(nonNaN);
  #pdb.set_trace();

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
  ###
  ### Intermission - make some plots that allows us to compare specific component responses in mixtures against isol. resp
  ### --- this only works for simple/LGN cells (i.e. need F1)
  ###
  ############
  if plt_f1_plots and comp_resp_org is not None and val_tr_org is not None: # i.e. this was a set of F1 responses...
    make_f1_comp_plots(expData, which_cell, mixture_exp, respMean, respStd, predResps, comp_resp_org, val_tr_org, save_locSuper, stimVals, val_con_by_disp, isol_pred=True, specify_ticks=True, tex_width=tex_width)

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
      fitz, rel_c50, nr_mod = fit_overall_suppression(all_resps, all_preds)
  except:
    fitz = None;
    rel_c50 = -99;
  curr_suppr['rel_c50'] = rel_c50;

  #### Now, recapitulate the key measures for the dataframe
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
  curr_suppr['var_expl'] = hf.var_explained(to_use['respMean'][only_pos], to_use['mod_pred'][only_pos], sfVals=None);
  
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

  ############
  ### create the key figure (i.e. Abramov-Levine '75)
  ############
  #fSuper, ax = plt.subplots(nRows, nCols, figsize=hf.set_size(tex_width, subplots=(nRows,nCols), extra_height=nRows));
  fSuper, ax = plt.subplots(nRows, nCols, figsize=(4.5*nCols, 3*nRows))
  mrkrsz = mpl.rcParams['lines.markersize']*0.75; # when we make the size adjustment above, the points are too large for the scatter
  mew = 0.2 * mrkrsz; # just very faint
  #print('%.2f, %.2f' % (mrkrsz, mew))
  sns.despine(fig=fSuper, offset=sns_offset)

  allMix = [];
  allSum = [];

  ### plot reference tuning [row 1 (i.e. 2nd row)]
  ## on the right, SF tuning (high contrast)
  sfRef = hf.nan_rm(respMean[0, :, -1]); # high contrast tuning
  sf_ref_row = 0 if simple_plot else 1 
  sf_ref_col = 0 if simple_plot else 1
  ax[sf_ref_row, sf_ref_col].errorbar(all_sfs, sfRef, yerr=hf.nan_rm(respStd[0,:,-1]), color='k', marker='o', label='ref. tuning', clip_on=False, linestyle='None')
  #ax[sf_ref_row, sf_ref_col].plot(all_sfs, sfRef, 'k-', marker='o', label='ref. tuning', clip_on=False)
  ax[sf_ref_row, sf_ref_col].set_xscale('log')
  ax[sf_ref_row, sf_ref_col].set_xlim((0.1, 10));
  ax[sf_ref_row, sf_ref_col].set_xlabel('Spatial frequency (c/deg)')
  ax[sf_ref_row, sf_ref_col].set_ylabel('Response (spikes/s)')
  #ax[sf_ref_row, sf_ref_col].set_ylim((-5, 1.25*np.nanmax(sfRef)));

  #####
  ## then on the left, RVC (peak SF) --> in same position regardless of full or simplified plot
  #####
  sfPeak = np.argmax(sfRef); # stupid/simple, but just get the rvc for the max response
  v_cons_single = val_con_by_disp[0]
  rvcRef = hf.nan_rm(respMean[0, sfPeak, v_cons_single]);
  # now, if possible, let's also plot the RVC fit
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
      rel_rvc = None
      c50, pk = np.nan, np.nan
   
  if rel_rvc is not None:
    plt_cons = np.geomspace(all_cons[0], all_cons[-1], 50);
    c50_emp, c50_eval = hf.c50_empirical(rvcMod, rel_rvc); # determine c50 by optimization, numerical approx.
    if rvcMod == 0:
      rvc_mod = hf.get_rvc_model();
      rvcmodResp = rvc_mod(*rel_rvc, plt_cons);
    else: # i.e. mod=1 or mod=2
      rvcmodResp = hf.naka_rushton(plt_cons, rel_rvc);
    if baseline is not None:
      rvcmodResp = rvcmodResp - baseline; 
    ax[1, 0].plot(plt_cons, rvcmodResp, 'k--', label='rvc fit (c50=%.2f%s)' %(c50, 'gain=%0f' % pk if rvcMod==0 else ''))
    # and save it
    curr_suppr['c50'] = c50; curr_suppr['conGain'] = pk;
    curr_suppr['c50_emp'] = c50_emp; curr_suppr['c50_emp_eval'] = c50_eval
  else:
    curr_suppr['c50'] = np.nan; curr_suppr['conGain'] = np.nan;
    curr_suppr['c50_emp'] = np.nan; curr_suppr['c50_emp_eval'] = np.nan;

  ax[1, 0].errorbar(all_cons[v_cons_single], rvcRef, yerr=hf.nan_rm(respStd[0,sfPeak,v_cons_single]), color='k', marker='o', label='ref. tuning (d0, peak SF)', linestyle='None')
  #ax[1, 0].plot(all_cons[v_cons_single], rvcRef, 'k-', marker='o', label='ref. tuning (d0, peak SF)')
  ax[1, 0].set_xlabel('Contrast');
  ax[1, 0].set_ylabel('Response (spikes/s)')
  ax[1, 0].set_ylim((-5, 1.25*np.nanmax(rvcRef)));
  ax[1, 0].legend(fontsize='x-small', loc='lower right');

  # plot the fitted model on each axis
  pred_plt = np.linspace(0, np.nanmax(all_preds), 100);
  if fitz is not None:
    if not simple_plot: # there's only summation plot here if the full plot
      ax[0, 0].plot(pred_plt, myFit(pred_plt, *fitz), 'r--', label='fit')
    ax[0, 1].plot(pred_plt, myFit(pred_plt, *fitz), 'r--', label='fit')
    
  dispRats = [];
  disps_plt = range(0,nDisps) if spec_disp is None else range(spec_disp,spec_disp+1);
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
        mixResp = respMean[d, s, v_cons[c]];
        allMix.append(mixResp);
        sumResp = predResps[d, s, v_cons[c]];
        allSum.append(sumResp);
  #      print('condition: d(%d), c(%d), sf(%d):: pred(%.2f)|real(%.2f)' % (d, v_cons[c], s, sumResp, mixResp))
        # PLOT in by-disp panel
        if not simple_plot: # only do this for the full plot
          if c == 0 and s == v_sfs[0]:
            ax[0, 0].plot(sumResp, mixResp, 'o', color=clrs_d[d-1], label=lbls_d[d], clip_on=False, markersize=mrkrsz, markeredgecolor='w', markeredgewidth=mew)
          else:
            ax[0, 0].plot(sumResp, mixResp, 'o', color=clrs_d[d-1], clip_on=False, markersize=mrkrsz, markeredgecolor='w', markeredgewidth=mew)
        # PLOT in by-sf panel --> same regardless of full or simplified plot
        try:
          sfInd = np.where(np.array(val_sfs) == s)[0][0]; # will only be one entry, so just "unpack"
          if d == 1 and c == 0: # just make the label once...
            if altHollow:
              ax[0, 1].plot(sumResp, mixResp, 'o', label=lbls_sf[sfInd], clip_on=False, markersize=mrkrsz - mew*np.mod(sfInd,2), markeredgecolor='w' if np.mod(sfInd,2)==0 else clrs_sf[sfInd], markeredgewidth=mew, markerfacecolor='None' if np.mod(sfInd,2)==1 else clrs_sf[sfInd]);
            else:
              ax[0, 1].plot(sumResp, mixResp, 'o', color=clrs_sf[sfInd], label=lbls_sf[sfInd], clip_on=False, markersize=mrkrsz, markeredgecolor='w', markeredgewidth=mew);
          else:
            if altHollow:
              ax[0, 1].plot(sumResp, mixResp, 'o', color=clrs_sf[sfInd], clip_on=False, markersize=mrkrsz - mew*np.mod(sfInd,2), markeredgecolor='w' if np.mod(sfInd,2)==0 else clrs_sf[sfInd], markeredgewidth=mew, markerfacecolor='None' if np.mod(sfInd,2)==1 else clrs_sf[sfInd]);
            else:
              ax[0, 1].plot(sumResp, mixResp, 'o', color=clrs_sf[sfInd], clip_on=False, markersize=mrkrsz, markeredgecolor='w', markeredgewidth=mew);
        except:
          pass;

    ax[0,1].set_xlim([0,400]);
    ax[0,1].axis('scaled');
    # plot averaged across all cons/sfs (i.e. average for the whole dispersion) [1,0]
    mixDisp = respMean[d, :, :].flatten();
    sumDisp = predResps[d, :, :].flatten();
    mixDisp, sumDisp = zr_rm_pair(mixDisp, sumDisp, 0.5);
    curr_rats = np.divide(mixDisp, sumDisp)
    curr_mn = geomean(curr_rats); curr_std = np.std(np.log10(curr_rats));
    if not simple_plot:
      ax[2, 0].bar(d, curr_mn, yerr=curr_std, color=clrs_d[d-1]);
      ax[2, 0].set_yscale('log')
      ax[2, 0].set_ylim(0.1, 10);
    dispRats.append(curr_mn);

    # also, let's plot the (signed) error relative to the fit
    if fitz is not None and not simple_plot:
      errs = mixDisp - myFit(sumDisp, *fitz);
      ax[3, 0].bar(d, np.mean(errs), yerr=np.std(errs), color=clrs_d[d-1])
      # -- and normalized by the prediction output response
      errs_norm = np.divide(mixDisp - myFit(sumDisp, *fitz), myFit(sumDisp, *fitz));
      ax[4, 0].bar(d, np.mean(errs_norm), yerr=np.std(errs_norm), color=clrs_d[d-1])

    # and set some labels/lines, as needed
    if d == 1 and not simple_plot:
        ax[2, 0].set_xlabel('dispersion');
        ax[2, 0].set_ylabel('suppression ratio (linear)')
        ax[2, 0].axhline(1, ls='--', color='k', alpha=ref_line_alpha)
        ax[3, 0].set_xlabel('dispersion');
        ax[3, 0].set_ylabel('mean (signed) error')
        ax[3, 0].axhline(0, ls='--', color='k', alpha=ref_line_alpha)
        ax[4, 0].set_xlabel('dispersion');
        ax[4, 0].set_ylabel('mean (signed) error -- as frac. of fit prediction')
        ax[4, 0].axhline(0, ls='--', color='k', alpha=ref_line_alpha)

    curr_suppr['supr_disp'] = dispRats;

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
  curr_suppr['supr_sf'] = sfRats;
  if not simple_plot:
    ax[2,1].scatter(all_sfs[val_sfs][sfInds], sfRats, color=clrs_sf[sfInds], clip_on=False)
    ax[2,1].errorbar(all_sfs[val_sfs][sfInds], sfRats, sfRatStd, color='k', linestyle='-', label='suppression tuning')
    ax[2,1].plot(all_sfs[val_sfs], sfRefShift, 'k--', label='ref. tuning')
    ax[2,1].axhline(1, ls='--', color='k', alpha=ref_line_alpha)
    ax[2,1].set_xlabel('Spatial Frequency (c/deg)')
    ax[2,1].set_xscale('log')
    ax[2,1].set_xlim((0.1, 10));
    ax[2,1].set_ylabel('Suppression ratio');
    ax[2,1].set_yscale('log')
    ax[2,1].set_ylim(0.1, 10);        
    ax[2,1].legend(fontsize='x-small');

  ### residuals from fit of suppression
  add_log_jitter = lambda x, frac, log_step: np.exp(np.log(x) + (np.random.rand()-0.5)*frac*log_step)
  sfs = np.unique(to_use['sf']);
  # --- ok_inds used to index into to_use (we exclude conditions that have either pred or resp < ltThresh)
  ok_inds = np.logical_and(to_use['respMean']>ltThresh, to_use['predMean']>ltThresh)
  # --- e.g. to_use[ok_inds].groupby('sf')...

  row_ind_offset = -2 if simple_plot else 0;
  if fitz is not None:
    if not simple_plot or (simple_plot and not plt_supr_ind):
      # mean signed error: and labels/plots for the error as f'n of SF
      ax[3+row_ind_offset,1].axhline(0, ls='--', color='k', alpha=ref_line_alpha)
      ax[3+row_ind_offset,1].set_xlabel('Spatial Frequency (c/deg)')
      ax[3+row_ind_offset,1].set_xscale('log')
      ax[3+row_ind_offset,1].set_xlim((0.1, 10));
      ax[3+row_ind_offset,1].set_ylabel('mean (signed) error');
      #ax[3+row_ind_offset,1].errorbar(all_sfs[val_sfs][sfInds], sfErrs, sfErrsStd, color='k', marker='o', linestyle='-')
      ax[3+row_ind_offset,1].plot(all_sfs[val_sfs][sfInds], sfErrs, color='k', marker=None, linestyle='-', alpha=0.5)
      mrkrsz = np.square(mpl.rcParams['lines.markersize']*0.5); # when we make the size adjustment above, the points are too large for the scatter --> and in scatter plots, s=mrks^2
      sf_inds_curr = np.searchsorted(sfs, to_use['sf']);
      try:
        sct_clrs = clrs_sf[sf_inds_curr];
      except:
        sct_clrs = 'k'
      try:
        ax[3+row_ind_offset,1].scatter([add_log_jitter(x, 0.6 if simple_plot else 0.7, np.nanmean(np.log(sfs[1:]/sfs[0:-1]))) for x in to_use['sf'][ok_inds]], to_use['rel_err'][ok_inds], alpha=0.5, s=mrkrsz, color=sct_clrs[ok_inds])
      except: # for the 1-2 V1 cells with oddities, the above won't work --> just skip it
        pass;
      #val_errs = np.logical_and(~np.isnan(sfErrs), np.array(sfErrsStd)>0);
      # NOTE: 22.09.29 --> var of sfErrs is a completely arbitrary metric, since the errs are not bounded/normalized in anyway
      #ax[3+row_ind_offset,1].text(0.1, 3, 'var=%.3f' % np.var(np.array(sfErrs)[val_errs]));

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
      curr_suppr['sfErrsInd_VAR'] = ind_var;
      curr_suppr['sfErrsInd_VAR_prince'] = ind_var_offset;
    except:
      curr_suppr['sfErrsInd_VAR'] = np.nan;
      curr_suppr['sfErrsInd_VAR_prince'] = np.nan;
      ind_var_offset = np.nan;

    if not simple_plot or (simple_plot and plt_supr_ind):
      ax[4+row_ind_offset,1].axhline(0, ls='--', color='k', alpha=ref_line_alpha)
      ax[4+row_ind_offset,1].set_xlabel('Spatial Frequency (c/deg)')
      ax[4+row_ind_offset,1].set_xscale('log')
      ax[4+row_ind_offset,1].set_xlim((0.1, 10));
      #ax[4+row_ind_offset,1].set_xlim((np.min(all_sfs), np.max(all_sfs)));
      ax[4+row_ind_offset,1].set_ylim((-1, 1));
      ax[4+row_ind_offset,1].set_ylabel('Saturation index');
      #ax[4+row_ind_offset,1].errorbar(all_sfs[val_sfs][sfInds][val_errs], norm_subset, normStd_subset, color='k', marker='o', linestyle='-', alpha=0.2)
      ax[4+row_ind_offset,1].plot(all_sfs[val_sfs][sfInds][val_errs], norm_subset, color='k', marker=None, linestyle='-', alpha=0.5)
      #ax[4+row_ind_offset,1].plot(all_sfs[val_sfs][sfInds][val_errs], to_use.groupby('sf')['supr_ind'].mean()[val_errs], color='k', marker=None, linestyle='--', alpha=0.5)
      mrkrsz = np.square(mpl.rcParams['lines.markersize']*0.5); # when we make the size adjustment above, the points are too large for the scatter --> and in scatter plots, s=mrks^2
      sf_inds_curr = np.searchsorted(sfs, to_use['sf']);
      try:
        sct_clrs = clrs_sf[sf_inds_curr];
      except:
        sct_clrs = 'k'
      ok_inds = np.logical_and(to_use['respMean']>ltThresh, to_use['predMean']>ltThresh)
      try:
        ax[4+row_ind_offset,1].scatter([add_log_jitter(x, 0.6 if simple_plot else 0.7, np.nanmean(np.log(sfs[1:]/sfs[0:-1]))) for x in to_use['sf'][ok_inds]], to_use[supr_ind_str][ok_inds], alpha=0.5, s=mrkrsz, color=sct_clrs[ok_inds])
      except: # for the 1-2 V1 cells with oddities, the above won't work --> just skip it
        pass;
      # - and put that value on the plot
      ax[4+row_ind_offset,1].text(0.1, -0.25, 'var=%.3f' % (ind_var_offset if supr_ind_prince else ind_var));
      # --- check if the pandas grouping gives the same results as what we have above?
      #mns, sems = to_use.groupby('sf')['supr_ind'].mean(), to_use.groupby('sf')['supr_ind'].std()/to_use.groupby('sf')['supr_ind'].count()
      #ax[4,1].errorbar(sfs, mns, yerr=sems, marker='*', linestyle='--');


    # -- AND simply the ratio between the mixture response and the mean expected mix response (i.e. Naka-Rushton)
    # --- equivalent to the suppression ratio, but relative to the NR fit rather than perfect linear summation
    val_errs = np.logical_and(~np.isnan(sfErrsRat), np.logical_and(np.array(sfErrsRatStd)>0, np.array(sfErrsRatStd) < 2));
    rat_subset = np.array(sfErrsRat)[val_errs];
    ratStd_subset = np.array(sfErrsRatStd)[val_errs];
    #ratStd_subset = (1/np.log(2))*np.divide(np.array(sfErrsRatStd)[val_errs], rat_subset);
    errsRatVar = np.var(np.log2(sfErrsRat)[val_errs]);
    curr_suppr['sfRat_VAR'] = errsRatVar;
    if not simple_plot:
      ax[5,1].scatter(all_sfs[val_sfs][sfInds][val_errs], rat_subset, color=clrs_sf[sfInds][val_errs], clip_on=False)
      ax[5,1].errorbar(all_sfs[val_sfs][sfInds][val_errs], rat_subset, ratStd_subset, color='k', linestyle='-', label='suppression tuning')
      ax[5,1].axhline(1, ls='--', color='k', alpha=ref_line_alpha)
      ax[5,1].set_xlabel('Spatial Frequency (c/deg)')
      ax[5,1].set_xscale('log')
      ax[5,1].set_xlim((0.1, 10));
      ax[5,1].set_ylabel('suppression ratio (wrt NR)');
      ax[5,1].set_yscale('log', basey=2)
    #         ax[2,1].yaxis.set_ticks(minorticks)
      ax[5,1].set_ylim(np.power(2.0, -2), np.power(2.0, 2));
      ax[5,1].legend(fontsize='x-small');
      # - compute the variance - and put that value on the plot
      ax[5,1].text(0.1, 2, 'var=%.2f' % errsRatVar);

  else: # if we don't have a fit...
    curr_suppr['sfErrsInd_VAR'] = np.nan
    curr_suppr['sfRat_VAR'] = np.nan

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
  # -- plot ref. tuning
  ax[sf_ref_row, sf_ref_col].plot(mod_sfs, mod_resp, 'k--', label='fit (g)')
  ax[sf_ref_row, sf_ref_col].legend(fontsize='x-small');
  if not simple_plot: # DEPRECATE? NOT NEEDED WITHOUT DERIV.
    # Duplicate "twin" the axis to create a second y-axis
    ax2 = ax[sf_ref_row, sf_ref_col].twinx();
    ax2.set_xscale('log'); # have to re-inforce log-scale?
    ax2.set_ylim([-1, 1]); # since the g' is normalized
    sns.despine(ax=ax2, offset=sns_offset, right=False);
    # - then, normalize the sfErrs/sfErrsInd and compute the correlation coefficient
  if fitz is not None:
    norm_sfErr = np.divide(sfErrs, np.nanmax(np.abs(sfErrs)));
    norm_sfErrInd = np.divide(sfErrsInd, np.nanmax(np.abs(sfErrsInd))); # remember, sfErrsInd is normalized per condition; this is overall
    # CORRELATION WITH TUNING
    non_nan = np.logical_and(~np.isnan(norm_sfErr), ~np.isnan(sfRefShift))
    corr_sf, corr_sfN = np.corrcoef(sfRefShift[non_nan], norm_sfErr[non_nan])[0,1], np.corrcoef(sfRefShift[non_nan], norm_sfErrInd[non_nan])[0,1]
    curr_suppr['corr_tuneWithErr'] = corr_sf;
    curr_suppr['corr_tuneWithErrsInd'] = corr_sfN;
    if not simple_plot:
      ax[3,1].text(0.1, 0.25*np.nanmax(sfErrs), 'corr w/g = %.2f' % corr_sf)
      ax[4,1].text(0.1, 0.25, 'corr w/g = %.2f' % corr_sfN)
    # CORRELATION WITH DERIV. (deprecated)
    non_nan = np.logical_and(~np.isnan(norm_sfErr), ~np.isnan(deriv_norm_eval))
    corr_nsf, corr_nsfN = np.corrcoef(deriv_norm_eval[non_nan], norm_sfErr[non_nan])[0,1], np.corrcoef(deriv_norm_eval[non_nan], norm_sfErrInd[non_nan])[0,1]
    curr_suppr['corr_derivWithErr'] = corr_nsf;
    curr_suppr['corr_derivWithErrsInd'] = corr_nsfN;
    #ax[3,1].text(0.1, 0.25*np.nanmax(sfErrs), 'corr w/g\' = %.2f' % corr_nsf)
    #ax[4,1].text(0.1, 0.25, 'corr w/g\' = %.2f' % corr_nsfN)
  else:
    curr_suppr['corr_derivWithErr'] = np.nan;
    curr_suppr['corr_derivWithErrsInd'] = np.nan;

  # make a polynomial fit - DEPRECATE 
  try:
    hmm = np.polyfit(allSum, allMix, deg=1) # returns [a, b] in ax + b 
  except:
    hmm = [np.nan];
  curr_suppr['supr_index'] = hmm[0];
  # compute area under the curve --- both for the linear expectation and for the Naka-Rushton
  xvals = np.linspace(0, np.nanmax(all_preds), 100);
  lin_area = np.trapz(xvals, x=xvals);
  nr_area = np.trapz(myFit(xvals, *fitz), x=xvals);
  curr_suppr['supr_area'] = nr_area/lin_area;

  for j in range(1):
    for jj in range(simple_plot, nCols): # i.e. just do it for col. 1 if simple_plot, otherwise do it for cols 0 and 1
      ax[j, jj].axis('square')
      ax[j, jj].set_xlabel('sum(components) (spikes/s)');
      ax[j, jj].set_ylabel('Mixture response (spikes/s)');
      ax[j, jj].plot([0, 1*maxResp], [0, 1*maxResp], 'k--', alpha=ref_line_alpha)
      ax[j, jj].set_xlim((-5, maxResp));
      ax[j, jj].set_ylim((-5, 1.1*maxResp));
      ax[j, jj].set_title('rAUC|c50: %.2f|%.2f [%.1f%s%%]' % (curr_suppr['supr_area'], curr_suppr['rel_c50'], curr_suppr['var_expl'], '\\' if useTex else ''))
      #ax[j, jj].set_title('Suppression index: %.2f|%.2f [%.1f\%%]' % (curr_suppr['supr_index'], curr_suppr['rel_c50'], curr_suppr['var_expl']));
      ax[j, jj].legend(fontsize='xx-small', ncol=1+jj); # want two legend columns for SF

  cell_name_use = cellName.replace('_','\_') if useTex else cellName
  fnt_sz = 'medium' if simple_plot else 'small'
  fSuper.suptitle('%s %s#%d [%s]; f1f0 %.2f; szSupr[dt/md] %.2f/%.2f; oriBW|CV %.2f|%.2f; tfBW %.2f]' % (cellType, '\\' if useTex else '', which_cell, cell_name_use, f1f0_rat, suprDat, suprMod, oriBW, oriCV, tfBW), fontsize=fnt_sz, y=1-0.03*simple_plot)

  #if not simple_plot:
  fSuper.tight_layout();

  if spec_disp is None:
    disp_str = '';
  else:
    disp_str = '_disp%d' % spec_disp;
  if spec_con is None:
    con_str = '';
  else:
    con_str = '_con%d' % spec_con;

  if fitList is None:
    save_name = 'cell_%03d%s%s.pdf' % (which_cell, disp_str, con_str);
  else:
    save_name = 'cell_%03d_mod%s%s%s%s.pdf' % (which_cell, hf.fitType_suffix(fitType), hf.lgnType_suffix(lgnFrontEnd, lgnConType=1), disp_str, con_str)
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
      super_name = 'superposition_analysis_%s%s.npy' % (suffix, '_p%d' % sum_power if sum_power!=1 else '');
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
      supr_ind_prince = False;
    if len(sys.argv)>5:
      use_mod_resp = int(sys.argv[5]);
    else:
      use_mod_resp = 0;
    if len(sys.argv)>6:
      normType = int(sys.argv[6]);
    else:
      normType = 1; # default to flat
    if len(sys.argv)>7:
      lgnOn = int(sys.argv[7]);
    else:
      lgnOn = 0; # default to no LGN
    if len(sys.argv)>8:
      spec_disp = int(sys.argv[8]);
    else:
      spec_disp = None;
    if len(sys.argv)>9:
      spec_con = int(sys.argv[9]);
    else:
      spec_con = None;

    fitList='fitList%s_pyt_nr221021' % 'HPC'; # TEMPORARY

    if asMulti:
      from functools import partial
      import multiprocessing as mp
      nCpu = mp.cpu_count()-1; # heuristics say you should reqeuest at least one fewer processes than their are CPU
      print('***cpu count: %02d***' % nCpu);

      with mp.Pool(processes = nCpu) as pool:
        sup_perCell = partial(plot_save_superposition, expDir=expDir, use_mod_resp=use_mod_resp, fitType=normType, useHPCfit=1, lgnConType=1, lgnFrontEnd=lgnOn, to_save=0, plt_supr_ind=plt_supr_ind, supr_ind_prince=supr_ind_prince, spec_disp=spec_disp, spec_con=spec_con);
        supFits = pool.map(sup_perCell, range(start_cell, end_cell+1));
        pool.close();

      ### do the saving HERE!
      dataPath = os.getcwd() + '/' + expDir + 'structures/'
      if fitList is None:
        from datetime import datetime
        suffix = datetime.today().strftime('%y%m%d')
        super_name = 'superposition_analysis_%s.npy' % suffix;
      else:
        super_name = 'superposition_analysis_mod%s%s.npy' % (hf.fitType_suffix(normType), hf.lgnType_suffix(lgnOn, lgnConType=1));

      if os.path.exists(dataPath + super_name):
        suppr_all = hf.np_smart_load(dataPath + super_name);
      else:
        suppr_all = dict();
      for iii, sup_fit in enumerate(supFits):
        suppr_all[iii] = sup_fit;
      np.save(dataPath + super_name, suppr_all);

    else: # i.e. not multi
      plot_save_superposition(cell_num, expDir, to_save=1, plt_supr_ind=plt_supr_ind, use_mod_resp=use_mod_resp, supr_ind_prince=supr_ind_prince, lgnConType=1, lgnFrontEnd=lgnOn, fitType=normType, spec_disp=spec_disp, spec_con=spec_con);

