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

if len(sys.argv) > 7:
  fixRespExp = float(sys.argv[7]);
  if fixRespExp <= 0: # this is the code to not fix the respExp
    fixRespExp = None;
else:
  fixRespExp = None; # default (see modCompare.ipynb for details)

if len(sys.argv) > 8:
  respVar = int(sys.argv[8]);
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
   # ---- DATA
  # now get the mask+base response (f1 at base TF)
  respMatrixDC, respMatrixF1 = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0); # i.e. get the base response for F1
  # and get the mask only response (f1 at mask TF)
  respMatrixDC_onlyMask, respMatrixF1_onlyMask, respMatrixDC_onlyMask_all, respMatrixF1_onlyMask_all = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1, returnByTr=1); # i.e. get the maskONLY response
  # and get the mask+base response (but f1 at mask TF)
  _, respMatrixF1_maskTf = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1); # i.e. get the maskONLY response

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
