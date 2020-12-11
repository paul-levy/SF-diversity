import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # to avoid GUI/cluster issues...
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import seaborn as sns
sns.set(style='ticks')
from scipy.stats import poisson, nbinom
from scipy.stats.mstats import gmean

import helper_fcns as hf
import helper_fcns_sfBB as hf_sf
import model_responses_pytorch as mrpt

import warnings
warnings.filterwarnings('once');

import pdb

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/paul_plt_style.mplstyle');
from matplotlib import rcParams

### Some globals
maskInd, baseInd = hf_sf.get_mask_base_inds();
expInd = -1;
newMethod = 1;
modStrs = ['V1', 'LGN'];
nMods = len(modStrs);
typeClrs = ['k', 'r']; # for data, model, respectively
onsetDur = 200; # in mS, how much of the initial PSTH to we model as part of the onset transient
halfWidth = 15; # in mS, how wide (in one direction) is the smooth/sliding PSTH window (usually 15 or 25)

###
for i in range(2):
  # must run twice for changes to take effect?
  from matplotlib import rcParams, cm
  rcParams['font.family'] = 'sans-serif'
  rcParams['font.style'] = 'oblique'
  rcParams['font.size'] = 30;
  rcParams['pdf.fonttype'] = 3 # should be 42, but there are kerning issues
  rcParams['ps.fonttype'] = 3 # should be 42, but there are kerning issues
  rcParams['lines.linewidth'] = 3;
  rcParams['lines.markeredgewidth'] = 0; # remove edge??
        
  rcParams['axes.linewidth'] = 3;
  rcParams['lines.markersize'] = 12; # 8 is the default
        
  rcParams['font.style'] = 'oblique';

  rcParams['xtick.major.size'] = 25
  rcParams['xtick.minor.size'] = 12
  rcParams['ytick.major.size'] = 25
  rcParams['ytick.minor.size'] = 0; # i.e. don't have minor ticks on y...

  rcParams['xtick.major.width'] = 2
  rcParams['xtick.minor.width'] = 2
  rcParams['ytick.major.width'] = 2
  rcParams['ytick.minor.width'] = 0

### Load the inputs to specify which model, cell, etc
cellNum  = int(sys.argv[1]);
excType  = int(sys.argv[2]);
lossType = int(sys.argv[3]);
expDir   = sys.argv[4]; 
lgnFrontEnd = int(sys.argv[5]);
kMult  = float(sys.argv[6]);
whichPlots = int(sys.argv[7]); # which plots to make

if len(sys.argv) > 8:
  fixRespExp = float(sys.argv[8]);
  if fixRespExp <= 0: # this is the code to not fix the respExp
    fixRespExp = None;
else:
  fixRespExp = None; # default (see modCompare.ipynb for details)

if len(sys.argv) > 9:
  respVar = int(sys.argv[9]);
else:
  respVar = 1;

### Then start to load the data, set directories, etc
loc_base = os.getcwd() + '/';
data_loc = loc_base + expDir + 'structures/';
save_loc = loc_base + expDir + 'figures/';

if 'pl1465' in loc_base:
  loc_str = 'HPC';
else:
  loc_str = '';

### DATALIST
dl_name = hf.get_datalist(expDir);

### LOAD CELL
dataList = hf.np_smart_load(data_loc + dl_name)
expName = 'sfBB_core';
unitNm = dataList['unitName'][cellNum-1];
cell = hf.np_smart_load('%s%s_sfBB.npy' % (data_loc, unitNm));
expInfo = cell[expName]
byTrial = expInfo['trial'];

### FITLIST
if excType == 1:
  fitBase = 'fitList_pyt_200417'; # excType 1
  #fitBase = 'fitList_pyt_201017'; # excType 1
elif excType == 2:
  #fitBase = 'fitList_pyt_200507'; # excType 2
  fitBase = 'fitList_pyt_201107'; # excType 2
else:
  fitBase = None;

if fitBase is not None:
  if lossType == 4: # chiSq...
    fitBase = '%s%s' % (fitBase, hf.chiSq_suffix(kMult));

  if fixRespExp is not None:
    fitBase = '%s_re%d' % (fitBase, np.round(fixRespExp*10)); # suffix to indicate that the response exponent is fixed...

  # now, LGN-specific naming
  if fixRespExp is not None:
    fitBase_lgn = '%s_re%d' % (fitBase, np.round(fixRespExp*10)); # suffix to indicate that the response exponent is fixed...
  else:
    fitBase_lgn = fitBase;

  if lgnFrontEnd == 1:
    fitBase_lgn = '%s_LGN' % fitBase_lgn
  elif lgnFrontEnd == 2:
    fitBase_lgn = '%s_LGNb' % fitBase_lgn
  elif lgnFrontEnd == 99:
    fitBase_lgn = '%s_jLGN' % fitBase_lgn

  # first the fit type
  fitSuf_fl = '_flat';
  fitSuf_wg = '_wght';
  # then the loss type
  if lossType == 1:
    lossSuf = '_sqrt.npy';
    loss = lambda resp, pred: np.sum(np.square(np.sqrt(resp) - np.sqrt(pred)));
  elif lossType == 2:
    lossSuf = '_poiss.npy';
    loss = lambda resp, pred: poisson.logpmf(resp, pred);
  elif lossType == 3:
    lossSuf = '_modPoiss.npy';
    loss = lambda resp, r, p: np.log(nbinom.pmf(resp, r, p));
  elif lossType == 4:
    lossSuf = '_chiSq.npy';
    # LOSS HERE IS TEMPORARY
    loss = lambda resp, pred: np.sum(np.square(np.sqrt(resp) - np.sqrt(pred)));

  # NOTE: We choose weighted gain control for non-LGN model, flat gain control for the model with the LGN front end
  fitName = str(fitBase + fitSuf_wg + lossSuf);
  fitName_lgn = str(fitBase_lgn + fitSuf_fl + lossSuf);

  fitList = hf.np_smart_load(data_loc + fitName); # V1 only
  fitList_lgn = hf.np_smart_load(data_loc + fitName_lgn); # with LGN, no tuned gain control

  # -- update the save_loc
  compDir  = str(fitBase + '_diag' + lossSuf);
  subDir   = compDir.replace('fitList', 'fits').replace('.npy', '');
  save_loc = str(save_loc + subDir + '/');

  # -- back to where we were
  dc_str = hf_sf.get_resp_str(respMeasure=0);
  f1_str = hf_sf.get_resp_str(respMeasure=1);

  modFit_V1_dc = fitList[cellNum-1][dc_str]['params']; # 
  modFit_lgn_dc = fitList_lgn[cellNum-1][dc_str]['params']; # 
  modFit_V1_f1 = fitList[cellNum-1][f1_str]['params']; # 
  modFit_lgn_f1 = fitList_lgn[cellNum-1][f1_str]['params']; # 

  normTypes = [2, 1]; # weighted, then flat
  lgnTypes = [0, lgnFrontEnd];

  # set the models up
  mod_V1_dc  = mrpt.sfNormMod(modFit_V1_dc, expInd=expInd, excType=excType, normType=2, lossType=lossType, lgnFrontEnd=0, newMethod=newMethod)
  mod_LGN_dc = mrpt.sfNormMod(modFit_lgn_dc, expInd=expInd, excType=excType, normType=1, lossType=lossType, lgnFrontEnd=lgnFrontEnd, newMethod=newMethod)
  mod_V1_f1  = mrpt.sfNormMod(modFit_V1_f1, expInd=expInd, excType=excType, normType=2, lossType=lossType, lgnFrontEnd=0, newMethod=newMethod)
  mod_LGN_f1 = mrpt.sfNormMod(modFit_lgn_f1, expInd=expInd, excType=excType, normType=1, lossType=lossType, lgnFrontEnd=lgnFrontEnd, newMethod=newMethod)

  # get the correct trInf
  trInf_dc, gt_resp_DC = mrpt.process_data(expInfo, expInd=expInd, respMeasure=0); 
  trInf_f1, gt_resp_F1 = mrpt.process_data(expInfo, expInd=expInd, respMeasure=1); 
  val_trials = trInf_dc['num']; # these are the indices of valid, original trials

  # get the corresponding DC & F1 responses
  resp_V1_dc  = mod_V1_dc.forward(trInf_dc, respMeasure=0).detach().numpy();
  resp_LGN_dc = mod_LGN_dc.forward(trInf_dc, respMeasure=0).detach().numpy();
  resp_V1_f1  = mod_V1_f1.forward(trInf_f1, respMeasure=1).detach().numpy();
  resp_LGN_f1 = mod_LGN_f1.forward(trInf_f1, respMeasure=1).detach().numpy();

# organize the responses!

# ---- DATA (Do, regardless of fitBase)
# now get the mask+base response (f1 at base TF)
respMatrixDC, respMatrixF1 = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0); # i.e. get the base response for F1
# and get the mask only response (f1 at mask TF)
respMatrixDC_onlyMask, respMatrixF1_onlyMask, respMatrixDC_onlyMask_all, respMatrixF1_onlyMask_all = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1, returnByTr=1); # i.e. get the maskONLY response
# and get the mask+base response (but f1 at mask TF)
_, respMatrixF1_maskTf = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1); # i.e. get the maskONLY response

if fitBase is not None:
   # ---- V1 model responses
  respMatrix_V1_dc, respMatrix_V1_f1, respMatrix_V1_dc_all, respMatrix_V1_f1_all = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0, dc_resp=resp_V1_dc, f1_base=resp_V1_f1[:,baseInd], f1_mask=resp_V1_f1[:,maskInd], val_trials=val_trials, returnByTr=1); # i.e. get the base response for F1
  # and get the mask only response (f1 at mask TF)
  respMatrix_V1_dc_onlyMask, respMatrix_V1_f1_onlyMask, respMatrix_V1_dc_onlyMask_all, respMatrix_V1_f1_onlyMask_all = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1, dc_resp=resp_V1_dc, f1_base=resp_V1_f1[:,baseInd], f1_mask=resp_V1_f1[:,maskInd], val_trials=val_trials, returnByTr=1); # i.e. get the maskONLY response
  # and get the mask+base response (but f1 at mask TF)
  _, respMatrix_V1_f1_maskTf, _, respMatrix_V1_f1_maskTf_all = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1, dc_resp=resp_V1_dc, f1_base=resp_V1_f1[:,baseInd], f1_mask=resp_V1_f1[:,maskInd], val_trials=val_trials, returnByTr=1); # i.e. get the maskONLY response
  # ---- LGN model responses
  respMatrix_LGN_dc, respMatrix_LGN_f1, respMatrix_LGN_dc_all, respMatrix_LGN_f1_all = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0, dc_resp=resp_LGN_dc, f1_base=resp_LGN_f1[:,baseInd], f1_mask=resp_LGN_f1[:,maskInd], val_trials=val_trials, returnByTr=1); # i.e. get the base response for F1
  # and get the mask only response (f1 at mask TF)
  respMatrix_LGN_dc_onlyMask, respMatrix_LGN_f1_onlyMask, respMatrix_LGN_dc_onlyMask_all, respMatrix_LGN_f1_onlyMask_all = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1, dc_resp=resp_LGN_dc, f1_base=resp_LGN_f1[:,baseInd], f1_mask=resp_LGN_f1[:,maskInd], val_trials=val_trials, returnByTr=1); # i.e. get the maskONLY response
  # and get the mask+base response (but f1 at mask TF)
  _, respMatrix_LGN_f1_maskTf, _, respMatrix_LGN_f1_maskTf_all = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1, dc_resp=resp_LGN_dc, f1_base=resp_LGN_f1[:,baseInd], f1_mask=resp_LGN_f1[:,maskInd], val_trials=val_trials, returnByTr=1); # i.e. get the maskONLY response

  if whichPlots <= 0:

    ##############
    # Raw, i.e. trial-by-trial, responses
    ##############

    data_respFixed = [gt_resp_DC.copy(), gt_resp_F1.copy()];
    # let's correct the (F1) responses to have a "zero" F1 amplitude when the contrast for that component (mask or base) is 0%
    data_respFixed[1][trInf_f1['con'][:,maskInd]==0, maskInd] = 1e-10 # force F1 ~ 0 if con of that stim is 0
    data_respFixed[1][trInf_f1['con'][:,baseInd]==0, baseInd] = 1e-10 # force F1 ~ 0 if con of that stim is 0

    mod_resp = [[resp_V1_dc, resp_V1_f1], [resp_LGN_dc, resp_LGN_f1]]; # first V1 model [dc, f1]; then LGN model [dc, f1]

    for mod in range(nMods):
      for respMeasure, curr_data in enumerate(data_respFixed): # two resp measures - DC [0], and F1 [1]
        curr_mod = mod_resp[mod][respMeasure];
        respStr = hf_sf.get_resp_str(respMeasure);

        if respMeasure == 0:
            f, ax = plt.subplots(figsize=(30, 15))

            plt.subplot(1,3,1)
            plt.plot(curr_data, color=typeClrs[0], alpha=0.5)
            # plt.plot(output[:,].detach(), 'k')
            plt.plot(curr_mod, color=typeClrs[0], alpha=0.5)
            max_resp = np.maximum(np.max(curr_data), np.max(curr_mod));
            plt.ylim([-0.1, 1.1*max_resp]);

            plt.subplot(1,3,2)
            plt.plot(curr_data - curr_mod, 'b')
            plt.ylim([-1.1*max_resp, 1.1*max_resp]);

            plt.subplot(1,3,3)
            # data vs model (sorted)
            plt.title('Sorted responses')
            plt.plot(sorted(curr_data), label='data', color=typeClrs[0])
            plt.plot(sorted(curr_mod), label='model', color=typeClrs[1])
            plt.legend();

        if respMeasure == 1: # i.e. F1
            f, ax = plt.subplots(figsize=(30,15))

            plt.subplot(2,3,1) # top row will be mask
            plt.plot(curr_data[:,maskInd], color=typeClrs[0], alpha=0.5)
            plt.plot(curr_mod[:,maskInd],  color=typeClrs[1], alpha=0.5)
            max_resp = np.maximum(np.max(curr_data[:,maskInd]), np.max(curr_mod[:,maskInd]));
            plt.ylim([-0.1, 1.1*max_resp]);
            plt.title('mask F1')

            plt.subplot(2,3,2)
            plt.title('Difference (data-mod)')
            plt.plot(curr_data[:,maskInd] - curr_mod[:,maskInd], 'b')
            plt.ylim([-1.1*max_resp, 1.1*max_resp]);

            plt.subplot(2,3,3)
            # data vs model (just mask F1)
            plt.title('mask TF resp')
            plt.plot(sorted(curr_data[:,maskInd]), label='data', color=typeClrs[0])
            plt.plot(sorted(curr_mod[:,maskInd]), label='model', color=typeClrs[1])
            plt.legend();

            plt.subplot(2,3,4) # bottom row will be base
            plt.title('Difference (data-mod)')
            plt.plot(curr_data[:,baseInd], 'k', alpha=0.5)
            plt.plot(curr_mod[:,baseInd], 'r', alpha=0.5)
            max_resp = np.maximum(np.max(curr_data[:,baseInd]), np.max(curr_mod[:,baseInd]));
            plt.ylim([-0.1, 1.1*max_resp]);
            plt.title('base F1')

            plt.subplot(2,3,5)
            plt.plot(curr_data[:,baseInd] - curr_mod[:,baseInd], 'b')

            plt.subplot(2,3,6)
            # data vs model (just base F1)
            plt.title('base TF resp')
            plt.plot(sorted(curr_mod[:,baseInd]), label='model')
            plt.plot(sorted(curr_data[:,baseInd]), label='data')
            plt.legend();

        # NOW save it!
        sns.despine(offset=10);

        saveName = "/cell_%03d_%s_%s.pdf" % (cellNum, respStr, modStrs[mod])
        full_save = os.path.dirname(str(save_loc + 'core_diag/sorted/'));
        if not os.path.exists(full_save):
            os.makedirs(full_save);
        pdfSv = pltSave.PdfPages(full_save + saveName);
        pdfSv.savefig(f)
        plt.close(f)
        pdfSv.close()


    ##############
    # Raw, i.e. trial-by-trial, responses
    ##############
    maskCons, maskSfs = expInfo['maskCon'], expInfo['maskSF'];
    nCons, nSfs = len(maskCons), len(maskSfs);

    datAll = [respMatrixDC_onlyMask_all, respMatrixF1_onlyMask_all]
    modAll = [[respMatrix_V1_dc_onlyMask_all, respMatrix_V1_f1_onlyMask_all], [respMatrix_LGN_dc_onlyMask_all, respMatrix_LGN_f1_onlyMask_all]]; # first V1 mod [dc, f1]; then LGN [dc, f1]

    for mod in range(nMods):
      for respMeasure, curr_data in enumerate(datAll): # two resp measures - DC [0], and F1 [1]
        curr_mod = modAll[mod][respMeasure];
        respStr = hf_sf.get_resp_str(respMeasure);

        # we'll go down, with the 1st column for SF, 2nd column for con
        nRows = np.maximum(nCons, nSfs);
        f, ax = plt.subplots(nRows, 2, figsize=(20*2, 15*nRows), sharey=True);

        for fixCon, nVals in enumerate([nSfs, nCons]): # if 0, we fix SF; if 1, we fix CON
          for whichInd in range(nVals):

            if fixCon:
                ax[whichInd, fixCon].plot(curr_mod[whichInd,:,:].flatten(), 'ro', label='mod', alpha=0.3)
                ax[whichInd, fixCon].plot(curr_data[whichInd,:,:].flatten(), 'ko', label='dat', alpha=0.3)
                # plot mean of data?
                [ax[whichInd, fixCon].plot(20*i+0, np.mean(curr_data[whichInd,i,0:10]), 'k>') for i in range(nVals)]
            else:
                ax[whichInd, fixCon].plot(curr_mod[:, whichInd,:].flatten(), 'ro', label='mod', alpha=0.3)
                ax[whichInd, fixCon].plot(curr_data[:, whichInd,:].flatten(), 'ko', label='dat', alpha=0.3)
                # plot mean of data?
                [ax[whichInd, fixCon].plot(20*i+0, np.mean(curr_data[i,whichInd,0:10]), 'k>') for i in range(nVals)]
            # relabel xaxes
            if fixCon:
                lbls = expInfo['maskSF'];
                ax[whichInd, fixCon].set_xlabel('SF (c/deg)')
                ax[whichInd, fixCon].set_title('Con fixed at %d%%' % (100*expInfo['maskCon'][whichInd]))
            else:
                lbls = 100*expInfo['maskCon'];
                ax[whichInd, fixCon].set_xlabel('Contrast')
                ax[whichInd, fixCon].set_title('SF fixed at %.1f' % (expInfo['maskSF'][whichInd]))

            tickLocs = [20*i+5 for i in range(nVals)]
            tickLbls = ['%.1f' % lbls[i] for i in range(nVals)]

            ax[whichInd, fixCon].set_xticks(tickLocs);
            ax[whichInd, fixCon].set_xticklabels(tickLbls);
            ax[whichInd, fixCon].legend()

        #sns.despine(offset=8)
        saveName = "/cell_%03d_%s_%s.pdf" % (cellNum, respStr, modStrs[mod])
        full_save = os.path.dirname(str(save_loc + 'core_diag/organized/'));
        if not os.path.exists(full_save):
            os.makedirs(full_save);
        pdfSv = pltSave.PdfPages(full_save + saveName);
        pdfSv.savefig(f)
        plt.close(f)
        pdfSv.close()

# This, we can do regardless of fitBase
if whichPlots >= 0:
  ##############
  # Response adjustment plots (only if plotPhase, i.e. whichPlots >= 0)
  ##############
  ### We'll go through and get all trials of the same condition...
  # fixed
  dir = -1;
  stimDur = 1
  conDig = 2
  maskInd, baseInd = hf_sf.get_mask_base_inds();

  if whichPlots >= 2: # e.g. if it's 2
    allOnsets = hf.np_smart_load(data_loc + 'onset_transients.npy'); # here's the set of all onset transients
    onsetKey = (onsetDur, halfWidth);
    onsetTransient = allOnsets[cellNum-1][onsetKey]['transient'];
    str_onset = '_wOnset';
    str_onsetPrms = '_%03d_%03d' % (onsetDur, halfWidth);
    nrow = 2;
  else:
    onsetTransient = None;
    str_onset = '';
    str_onsetPrms = '';
    nrow = 1;

  # Gather all possible stimulus conditions
  maskSfs = expInfo['maskSF'];
  maskCons = np.round(expInfo['maskCon'], conDig);
  baseSf = expInfo['baseSF']; # TODO: This assumes only one base condition (i.e. sfBB_core); adapt for sfBB_var*
  baseCon = np.round(expInfo['baseCon'], conDig);

  for maskOn in range(2):
    for baseOn in range(2):
      if (maskOn + baseOn) == 0: # these are just blank trials...
        continue;

      if maskOn == 1: # if the mask is on, we'll need to specify which trials
        allSfs = maskSfs;
        allCons = maskCons;
      else:
        allSfs = [0]; allCons = [0]; # the values won't matter, anyway, so just pass in [0], [0]
      for whichSf, maskSf in enumerate(allSfs):
        for whichCon, maskCon in enumerate(allCons):
          # Choose which condition to consider
          val_trials, stimPh, stimTf = hf_sf.get_valid_trials(expInfo, maskOn, baseOn, whichCon, whichSf, returnStimConds=1);
          # Then get the vec avg'd responses, phase information, etc
          vec_avgs, vec_byTrial, rel_amps, _, _, _ = hf_sf.get_vec_avg_response(expInfo, val_trials, dir=dir, stimDur=stimDur);
          # - unpack the vec_avgs, and per trial
          mean_r, mean_phi = vec_avgs[0], vec_avgs[1];
          resp_amp, phase_rel_stim = vec_byTrial[0], vec_byTrial[1];
          # -- average the uncorrected amplitudes...
          uncorr_r = np.mean(rel_amps, axis=0);
          # - and unpack the stimTF...(NOTE: suptitle call assumes only one TF value for mask, base, respectively)
          maskTf = stimTf[:, maskInd];
          baseTf = stimTf[:, baseInd];

          max_r = 10*np.ceil(np.max(resp_amp)/10) # round to the nearest multiple of 10

          ncol = 2; # always 2 columns, but nrow is specified with the onset transient...
          f, ax = plt.subplots(nrow, ncol, figsize=(ncol*20, nrow*20));

          f.suptitle('mask (%d @ %d Hz, %.2f cpd, %d%%); base (%d @ %d Hz, %.2f cpd, %d%%)' % (maskOn, np.unique(maskTf), np.unique(maskSf)[0], np.unique(np.round(100*maskCon))[0], 
                                                                                            baseOn, np.unique(baseTf), np.unique(baseSf)[0], np.unique(np.round(100*baseCon))[0]));

          # MASK (will do base component after)
          plt.subplot(nrow,ncol,1, projection='polar');
          [plt.plot([0, np.deg2rad(phi)], [0, r], 'o--k', alpha=0.3) for phi,r in zip(phase_rel_stim[:, maskInd], resp_amp[:, maskInd])]
          curr_r, curr_phi = mean_r[maskInd], mean_phi[maskInd]
          plt.plot([0, np.deg2rad(curr_phi)], [0, curr_r], 'o-k', label=r'$ mu(r,\phi) = (%.1f, %.0f)$ vs $r_0 = %.1f$' % (curr_r, curr_phi, uncorr_r[maskInd]))
          plt.title('Mask -- Corrected phase')
          plt.ylim([0, max_r])
          plt.legend(fontsize='medium');

          # BASE
          plt.subplot(nrow,ncol,2, projection='polar');
          [plt.plot([0, np.deg2rad(phi)], [0, r], 'o--k', alpha=0.3) for phi,r in zip(phase_rel_stim[:, baseInd], resp_amp[:, baseInd])]
          curr_r, curr_phi = mean_r[baseInd], mean_phi[baseInd]
          plt.plot([0, np.deg2rad(curr_phi)], [0, curr_r], 'o-k', label=r'$ mu(r,\phi) = (%.1f, %.0f)$ vs. $r_0 = %.1f$' % (curr_r, curr_phi, uncorr_r[baseInd]))
          plt.title('Base -- Corrected phase')
          plt.ylim([0, max_r])
          plt.legend(fontsize='medium');

          if onsetTransient is not None:
            vec_avgs, vec_byTrial, rel_amps, _, _, _ = hf_sf.get_vec_avg_response(expInfo, val_trials, dir=dir, stimDur=stimDur, onsetTransient=onsetTransient);
            # - unpack the vec_avgs, and per trial
            mean_r, mean_phi = vec_avgs[0], vec_avgs[1];
            resp_amp, phase_rel_stim = vec_byTrial[0], vec_byTrial[1];
            # -- average the uncorrected amplitudes...
            uncorr_r = np.mean(rel_amps, axis=0);

            # MASK (will do base component after)
            plt.subplot(nrow,ncol,3, projection='polar');
            [plt.plot([0, np.deg2rad(phi)], [0, r], 'o--k', alpha=0.3) for phi,r in zip(phase_rel_stim[:, maskInd], resp_amp[:, maskInd])]
            curr_r, curr_phi = mean_r[maskInd], mean_phi[maskInd]
            plt.plot([0, np.deg2rad(curr_phi)], [0, curr_r], 'o-k', label=r'$ mu(r,\phi) = (%.1f, %.0f)$ vs $r_0 = %.1f$' % (curr_r, curr_phi, uncorr_r[maskInd]))
            plt.title('Mask -- Transient-correction (onset=%03dms, halfWidth=%03dms)' % (onsetDur, halfWidth))
            plt.ylim([0, max_r])
            plt.legend(fontsize='medium');

            # BASE
            plt.subplot(nrow,ncol,4, projection='polar');
            [plt.plot([0, np.deg2rad(phi)], [0, r], 'o--k', alpha=0.3) for phi,r in zip(phase_rel_stim[:, baseInd], resp_amp[:, baseInd])]
            curr_r, curr_phi = mean_r[baseInd], mean_phi[baseInd]
            plt.plot([0, np.deg2rad(curr_phi)], [0, curr_r], 'o-k', label=r'$ mu(r,\phi) = (%.1f, %.0f)$ vs. $r_0 = %.1f$' % (curr_r, curr_phi, uncorr_r[baseInd]))
            plt.title('Base -- Transient-correction')
            plt.ylim([0, max_r])
            plt.legend(fontsize='medium');

          saveName = "/cell_%03d_phaseCorr_sf%03d_con%03d.pdf" % (cellNum, np.int(100*np.unique(maskSf)), np.int(100*np.unique(maskCon)))

          if maskOn and baseOn:
            subdir = 'both';
          elif maskOn:
            subdir = 'maskOnly';
          elif baseOn:
            subdir = 'baseOnly';

          full_save = os.path.dirname(str(save_loc + 'phase_corr%s/%s/cell_%03d%s/%s/' % (str_onset, expName, cellNum, str_onsetPrms, subdir)));
          if not os.path.exists(full_save):
              os.makedirs(full_save);
          print('Saving %s' % (full_save + saveName));
          pdfSv = pltSave.PdfPages(full_save + saveName);
          pdfSv.savefig(f)
          plt.close(f)
          pdfSv.close()

          ## Yes, still part of the same outer loop (didn't want to disturb the above plotting)
          if whichPlots >= 3: # i.e. if it's greater than 2, then we'll plot PSTH
            # get the mask/base TF values and cycle duration (in mS)
            maskTf = np.unique(expInfo['trial']['tf'][maskInd,:])[0]
            baseTf = np.unique(expInfo['trial']['tf'][baseInd,:])[0]
            cycleDur_mask = 1e3/maskTf # guaranteed only one TF value
            cycleDur_base = 1e3/baseTf # guaranteed only one TF value

            # get the spikes
            msTenthToS = 1e-4; # the spike times are in 1/10th ms, so multiply by 1e-4 to convert to S
            spikeTimes = [expInfo['spikeTimes'][trNum]*msTenthToS for trNum in val_trials]

            # -- compute PSTH
            psth, bins = hf.make_psth(spikeTimes, stimDur=stimDur)
            # -- compute FFT (manual), then start to unpack - [matrix, coeffs, spectrum, amplitudes]
            man_fft = [hf.manual_fft(psth_curr, tfs=np.array([int(np.unique(maskTf)), int(np.unique(baseTf))]), onsetTransient=onsetTransient, stimDur=stimDur) for psth_curr in psth]
            man_fft_noTransient = [hf.manual_fft(psth_curr, tfs=np.array([int(np.unique(maskTf)), int(np.unique(baseTf))]), onsetTransient=None, stimDur=stimDur) for psth_curr in psth]
            # Now, plot!
            nPlots = len(psth);
            nrow = int(np.floor(np.sqrt(nPlots))); ncol = int(np.ceil(nPlots/nrow));
            fPsth, axPsth = plt.subplots(nrow, ncol, figsize=(15*ncol, nrow*8));
            psth_slide, bins_slide = hf.make_psth_slide(spikeTimes, binWidth=halfWidth*1e-3)

            max_rate = np.max(psth_slide);

            for ii, psth_curr in enumerate(psth_slide): # yes, it's a loop, but w/e...
              # First, plot the (smoothed) psth
              plot_ind = np.unravel_index(ii, (nrow, ncol));
              axPsth[plot_ind].set_ylim([0, 1.2*max_rate]);
              axPsth[plot_ind].plot(1e3*bins_slide, psth_curr);
              # Next, plot the manual FFT "fits" - first with, then without the transient
              curr_fft = man_fft[ii];
              curr_fft_noTransient = man_fft_noTransient[ii];
              axPsth[plot_ind].plot(1e3*bins_slide[0:-1], np.matmul(curr_fft[0], curr_fft[1]), 'k--', label='FFT++')
              axPsth[plot_ind].plot(1e3*bins_slide[0:-1], np.matmul(curr_fft_noTransient[0], curr_fft_noTransient[1]), 'b--', label='FFT')
              # and print the coefficients (DC, mask, base)
              axPsth[plot_ind].text(cycleDur_mask*1.2, 1.125*max_rate, 'FFT: DC %.1f, mask %.1f, base %.2f' % (*curr_fft[3], ))
              axPsth[plot_ind].text(cycleDur_mask*1.2, 1.025*max_rate, 'FFT++: DC %.1f, mask %.1f, base %.2f' % (*curr_fft_noTransient[3], ))
              # - and plot the length of one mask/base cycle
              axPsth[plot_ind].plot([0, cycleDur_mask], [1.125*max_rate, 1.125*max_rate], 'k-', label='Mask cycle')
              axPsth[plot_ind].plot([0, cycleDur_base], [1.025*max_rate, 1.025*max_rate], 'r-', label='Base cycle')
              if ii == 0: # only need the labels/legend once...
                axPsth[plot_ind].set_ylabel('Spike rate');
                axPsth[plot_ind].set_xlabel('Time (ms)');
              if ii+1 == len(psth_slide): # i.e. we're at the last plot -- let's make the legend in an otherwise empty subplot...
                plot_ind = np.unravel_index(ii+1, (nrow, ncol));
                axPsth[plot_ind].plot(1e3*bins_slide[0:-1], np.matmul(curr_fft[0], curr_fft[1]), 'k--', label='FFT++')
                axPsth[plot_ind].plot(1e3*bins_slide[0:-1], np.matmul(curr_fft_noTransient[0], curr_fft_noTransient[1]), 'b--', label='FFT')
                axPsth[plot_ind].plot([0, cycleDur_mask], [1.125*max_rate, 1.125*max_rate], 'k-', label='Mask cycle')
                axPsth[plot_ind].plot([0, cycleDur_base], [1.025*max_rate, 1.025*max_rate], 'r-', label='Base cycle')
                axPsth[plot_ind].legend(fontsize='large');


            sns.despine(offset=3);

            full_save = os.path.dirname(str(save_loc + 'PSTH%s/%s/cell_%03d%s/%s/' % (str_onset, expName, cellNum, str_onsetPrms, subdir)));
            saveName = "/cell_%03d_PSTH_sf%03d_con%03d.pdf" % (cellNum, np.int(100*np.unique(maskSf)), np.int(100*np.unique(maskCon)))

            if not os.path.exists(full_save):
                os.makedirs(full_save);
            print('Saving %s' % (full_save + saveName));
            pdfSv = pltSave.PdfPages(full_save + saveName);
            pdfSv.savefig(fPsth)
            plt.close(fPsth)
            pdfSv.close()

