import torch
import torch.nn as nn
from torch.utils import data as torchdata

import numpy as np

import helper_fcns as hf
import helper_fcns_sfBB as hf_sfBB
import time
import datetime
import sys, os
import warnings
from functools import partial

import pdb

torch.autograd.set_detect_anomaly(True)

#########
### Some global things...
#########
torch.set_num_threads(1) # to reduce CPU usage - 20.01.26
force_earlyNoise = None; # if None, allow it as parameter; otherwise, force it to this value; used 0 for 210308-210315; None for 210321
recenter_norm = 1;
_schedule = True; # use scheduler or not??? True or False
fall2020_adj = 1; # 210121, 210206, 210222, 210226, 210304, 210308/11/12/14, 210321
spring2021_adj = 1; # further adjustment to make scale a sigmoid rather than abs; 210222
if fall2020_adj:
  globalMin = 1e-10 # what do we "cut off" the model response at? should be >0 but small
  #globalMin = 1e-1 # what do we "cut off" the model response at? should be >0 but small
else:
  globalMin = 1e-6 # what do we "cut off" the model response at? should be >0 but small
modRecov = 0;
# --- for parameters that are transformed with sigmoids, what's the scalar in front of the sigmoid??
### WARNING: If you adjust _sigmoidRespExp, adjust the corresponding line in sfNormMod.forward (search for _sigmoidRespExp, used in ratio = ...)
_sigmoidRespExp = None; # 3 or None, as of 21.03.14
### WARNING: See above
_sigmoidSigma = 5; # if None, then we just use raw value; otherwise, we'll do a sigmoid transform
_sigmoidScale = 10
_sigmoidDord = 5;
_sigmoidGainNorm = 5;
# --- and a flag for whether or not to include the LGN filter for the gain control
_LGNforNorm = 0;
### force_full datalist?
expDir = sys.argv[2];
force_full = 0 if expDir == 'V1_BB/' else 1;

try:
  cellNum = int(sys.argv[1]);
except:
  cellNum = np.nan;
try:
  dataListName = hf.get_datalist(expDir, force_full=force_full); # argv[2] is expDir
except:
  dataListName = None;

### Helper -- from Billy
def _cast_as_tensor(x, device='cpu', dtype=torch.float32):
    # needs to be float32 to work with the Hessian calculations
    #return torch.tensor(x, dtype=dtype, device=device) # per bill broderick
    return torch.as_tensor(x, dtype=dtype, device=device) # updated for expected lower CPU usage

def _cast_as_param(x, requires_grad=True):
    return torch.nn.Parameter(_cast_as_tensor(x), requires_grad=requires_grad)

### Helper -- other

## rvc
def get_rvc_model(params, cons):
  ''' simply return the rvc model used in the fits (type 0; should be used only for LGN)
      --- from Eq. 3 of Movshon, Kiorpes, Hawken, Cavanaugh; 2005
  '''
  b = params[0]; k = params[1]; c0 = params[2];

  return torch.add(b, torch.mul(k, torch.log(1+torch.div(cons, c0))));

## sf tuning
def flexible_Gauss(params, stim_sf, minThresh=0.1, sigmoidValue=_sigmoidSigma):
    respFloor       = params[0];
    respRelFloor    = params[1];
    sfPref          = params[2];
    sigmaLow        = params[3] if sigmoidValue is None else torch.mul(_cast_as_tensor(sigmoidValue), torch.sigmoid(params[3]));
    sigmaHigh       = params[4] if sigmoidValue is None else torch.mul(_cast_as_tensor(sigmoidValue), torch.sigmoid(params[4]));

    # Tuning function
    sf0   = torch.div(stim_sf, sfPref);
    sigma = sigmaLow.repeat(sf0.shape)
    #sigma = torch.full_like(sf0, _cast_as_tensor(sigmaLow.item())); # was sigmaLow.detach
    whereSigHigh = torch.where(sf0>1);
    sigma[whereSigHigh] = sigmaHigh;

    shape = torch.exp(-torch.div(torch.pow(torch.log(sf0),2), 2*torch.pow(sigma,2)))
                
    return torch.max(_cast_as_tensor(minThresh), respFloor + respRelFloor*shape);

def DoGsach(gain_c, r_c, gain_s, r_s, stim_sf):
  ''' Difference of gaussians as described in Sach's thesis
  [0] gain_c    - gain of the center mechanism
  [1] r_c       - radius of the center
  [2] gain_s    - gain of surround mechanism
  [3] r_s       - radius of surround
  '''
  dog = gain_c*np.pi*torch.pow(r_c,2)*torch.exp(-torch.pow(stim_sf*np.pi*r_c,2)) - gain_s*np.pi*torch.pow(r_s,2)*torch.exp(-torch.pow(stim_sf*np.pi*r_s,2));

  return dog;

def DiffOfGauss(gain, f_c, gain_s, j_s, stim_sf):
  ''' Difference of gaussians - as formulated in Levitt et al, 2001
  gain      - overall gain term
  f_c       - characteristic frequency of the center, i.e. freq at which response is 1/e of maximum
  gain_s    - relative gain of surround (e.g. gain_s of 0.5 says peak surround response is half of peak center response
  j_s       - relative characteristic freq. of surround (i.e. char_surround = f_c * j_s)
  '''

  ctr = torch.mul(gain, torch.exp(-torch.pow(stim_sf/f_c, 2)));
  sur = torch.mul(gain*gain_s, torch.exp(-torch.pow(stim_sf/(f_c*j_s),2)));
  dog = torch.sub(ctr, sur);

  return dog;

def get_descrResp(params, stim_sf, DoGmodel, minThresh=0.1):
  # returns only pred_spikes
  if DoGmodel == 0:
    #pred_spikes = flexible_Gauss(params, stim_sf=stim_sf, minThresh=minThresh);
    pred_spikes = flexible_Gauss(params, stim_sf=stim_sf, minThresh=minThresh);
  elif DoGmodel == 1:
    pred_spikes = DoGsach(*params, stim_sf=stim_sf);
  elif DoGmodel == 2:
    pred_spikes = DiffOfGauss(*params, stim_sf=stim_sf);
  return pred_spikes;

def organize_mean_perCond(trInf, resp):
  ''' Given trInf and resp, organize the mean per condition
      Design this to work for all experiments...
  '''
  nConds = 0;
  means = np.nan * np.zeros_like(resp);
  conRounds = np.round(trInf['con'], 3);
  unique_conSet = np.unique(conRounds, axis=0);
  sfRounds = np.round(trInf['sf'], 3);
  unique_sfSet = np.unique(sfRounds, axis=0);
  for conSet in unique_conSet:
    con_rows = np.where((conRounds == conSet).all(axis=1))[0];
    if conSet[0] == 0: # i.e. if the first stimulus is blank, then the first SF doesn't matter
      # Yes, this is just written for expInd == -1 but could be generalized to avoid needing an if statement...to do later on
      unique_sfsubSet = np.unique(sfRounds[:, 1:]);
      for sfSet in unique_sfsubSet:
        sf_rows = np.where((sfRounds[:, 1:] == sfSet).all(axis=1))[0]
        to_mean = np.intersect1d(con_rows, sf_rows);
        if len(to_mean) == 0:
          continue;
        means[to_mean, :] = np.mean(resp[to_mean, :], axis=0)
        nConds += 1;
    else:
      for sfSet in unique_sfSet:
        sf_rows = np.where((sfRounds == sfSet).all(axis=1))[0];
        to_mean = np.intersect1d(con_rows, sf_rows);
        if len(to_mean) == 0:
          continue;
        means[to_mean, :] = np.mean(resp[to_mean, :], axis=0)
        nConds += 1;
  #print('# conditions: %d' % nConds);
  return means;

## FFT
def fft_amplitude(fftSpectrum, stimDur):
    ''' given an fftSpectrum (and assuming all other normalization has taken place), we double the non-DC frequencies and return
        only the DC and positive frequencies; we also convert these values into rates (i.e. spikes or power per second)

        normalization: since this code is vectorized, the length of each signal passed into np.fft.fft is 1
        i.e. an array whose length is one, with the array[0] having length stimDur/binWidth
        Thus, the DC amplitude is already correct, i.e. not in need of normalization by nSamples

        But, remember that for a real signal like this, non-DC amplitudes need to be doubled - we take care of that here       
    '''
    # Note that fftSpectrum will be [nTr x nFr]
    nyquist = [np.int32(x.shape[1]/2) for x in fftSpectrum];
    correctFFT = [];
    for i, spect in enumerate(fftSpectrum):
      allFFT = spect[:, 0:nyquist[i]+1]; # include nyquist 
      # since we cannot modify in-place, we simply double all of the non-DC amplitudes, then grab the unmodified DC, and create a new tensor accordingly
      nonDCvals = 2*allFFT[:, 1:nyquist[i]+1];
      currFFT = torch.cat((allFFT[:,0].unsqueeze(dim=1), nonDCvals), dim=1) # will be [nTr x nyquist+1]
      # note the divison by stimDur; our packaging of the psth when we call np.fft.fft means that the baseline is each trial
      # is one second; i.e. amplitudes are rates IFF stimDur = 1; here, we divide by stimDur to ensure all trials/psth
      # are true rates
      currFFT = torch.div(currFFT, _cast_as_tensor(stimDur));
      correctFFT.append(currFFT);
    
    return correctFFT;   

def spike_fft(psth, tfs = None, stimDur = None, binWidth=1e-3, inclPhase=0):
    ''' given a psth (and optional list of component TFs), compute the fourier transform of the PSTH
        if the component TFs are given, return the FT power at the DC, and at all component TFs
        NOTE: spectrum, rel_amp are rates (spks/s)
              full_fourier is unprocessed, in that regard
        
        normalization: since this code is vectorized, the length of each signal passed into np.fft.fft is 1
        i.e. an array whose length is one, with the array[0] having length stimDur/binWidth
        Thus, the DC amplitude is already correct, i.e. not in need of normalization by nSamples
        But, remember that for a real signal like this, non-DC amplitudes need to be doubled - we take care of that here 

    '''

    full_fourier = [torch.rfft(x, signal_ndim=1, onesided=False) for x in psth];
    epsil = 1e-10;
    if inclPhase:
      # NOTE: I have checked that the amplitudes (i.e. just "R" in polar coordinates, 
      # -- computed as sqrt(x^2 + y^2)) are ...
      # -- equivalent when derived from with_phase as in spectrum, below
      with_phase = fft_amplitude(full_fourier, stimDur); # passing in while still keeping the imaginary component (so that we can back out phase)

    full_fourier = [torch.sqrt(epsil + torch.add(torch.pow(x[:,:,0], 2), torch.pow(x[:,:,1], 2))) for x in full_fourier]; # just get the amplitude
    spectrum = fft_amplitude(full_fourier, stimDur);

    if tfs is not None:
      try:
        if 'torch' in str(type(tfs)):
          tf_as_ind = _cast_as_tensor(hf.tf_to_ind(tfs.numpy(), stimDur), dtype=torch.long); # if 1s, then TF corresponds to index; if stimDur is 2 seconds, then we can resolve half-integer frequencies -- i.e. 0.5 Hz = 1st index, 1 Hz = 2nd index, ...; CAST to integer
        else:
          tf_as_ind = _cast_as_tensor(hf.tf_to_ind(tfs, stimDur), dtype=torch.long); # if 1s, then TF corresponds to index; if stimDur is 2 seconds, then we can resolve half-integer frequencies -- i.e. 0.5 Hz = 1st index, 1 Hz = 2nd index, ...; CAST to integer
        if len(spectrum[0]) == len(tf_as_ind): # i.e. we have separate tfs for each trial
          spec_curr = spectrum[0];
          rel_amp = [torch.gather(spec_curr, dim=1, index=tf_as_ind)];
          #rel_amp = [torch.stack([spec_curr[i][tf_as_ind[i]] for i in range(len(tf_as_ind))])]; # DEPRECATED --> SLOW!
        else: # i.e. it's just a fixed set of TFs that applies for all trials (expInd=-1)
          rel_amp = [spectrum[i][:, tf_as_ind[i]] for i in range(len(tf_as_ind))];
      except:
        warnings.warn('In spike_fft: if accessing power at particular frequencies, you must also include the stimulation duration!');
        rel_amp = [];
    else:
      rel_amp = [];

    if inclPhase:
      return spectrum, rel_amp, full_fourier, with_phase;
    else:
      return spectrum, rel_amp, full_fourier;

### Datawrapper/loader

def process_data(coreExp, expInd=-1, respMeasure=0, respOverwrite=None, whichTrials=None):
  ''' Process the trial-by-trial stimulus information for ease of use with the model
      Specifically, we stack the stimuli to be [nTr x nStimComp], where 
      - [:,0] is base, [:,1] is mask, respectively for sfBB
      - If respOverwrite is not None, it will overwrite the responses from coreExp (only used for expInd>=1)
  
      Note: for expInd>=1, pass in X['sfm']['exp']['trial']
  '''
  ### TODO: Make sure you are consistent with returning either rates or counts (I think we should do counts, at least for DC)
  trInf = dict();

  ### first, find out which trials we should be analyzing
  if expInd == -1: # i.e. sfBB
    trialInf = coreExp['trial'];
    if whichTrials is None: # if whichTrials is None, then we're using ALL non-blank trials (i.e. fitting 100% of data)
      whichTrials = np.where(np.logical_or(trialInf['maskOn'], trialInf['baseOn']))[0]
    if respMeasure == 0:
      resp = np.expand_dims(coreExp['spikeCounts'][whichTrials], axis=1); # make (nTr, 1)
    elif respMeasure == 1: # then we're getting F1 -- first at baseTF, then maskTF
      # NOTE: CORRECTED TO MASK, then BASE on 20.11.15
      # -- the tranpose turns it from [2, nTr] to [nTr, 2], but keeps [:,0] as mask; [:,1] as base
      resp = np.vstack((coreExp['f1_mask'][whichTrials], coreExp['f1_base'][whichTrials])).transpose();
      # NOTE: Here, if spikeCount == 0 for that trial, then f1_mask/base will be NaN -- replace with zero
      to_repl = np.where(coreExp['spikeCounts'][whichTrials] == 0)[0];
      resp[to_repl, :] = 0;
  # --- first step, but for older experiments    
  elif expInd == 1: # i.e. sfMix original
    trialInf = coreExp;
    if whichTrials is None: # if whichTrials is None, then we're using ALL non-blank trials (i.e. fitting 100% of data)
      mask = np.ones_like(coreExp['spikeCount'], dtype=bool); # i.e. true
      # and get rid of orientation tuning curve trials
      oriBlockIDs = np.hstack((np.arange(131, 155+1, 2), np.arange(132, 136+1, 2))); # +1 to include endpoint like Matlab
      oriInds = np.empty((0,));
      for iB in oriBlockIDs:
          indCond = np.where(coreExp['blockID'] == iB);
          if len(indCond[0]) > 0:
              oriInds = np.append(oriInds, indCond);
      mask[oriInds.astype(np.int64)] = False;
      # and also blank/NaN trials
      whereOriNan = np.where(np.isnan(coreExp['ori'][0]))[0];
      mask[whereOriNan] = False;
      whichTrials = np.where(mask)[0];
      # now, finally get ths responses
      if respOverwrite is not None:
        resp = respOverwrite;
      else:
        if respMeasure == 0:
          resp = coreExp['spikeCount'].astype(int); # cast as int, since some are as uint
        else:
          resp = coreExp['f1'];
        resp = np.expand_dims(resp, axis=1); # expand dimensions to make it (nTr, 1)
      resp = resp[whichTrials];
  elif expInd >= 2: # the updated sfMix experiments
    trialInf = coreExp;
    if whichTrials is None:
      # start with all trials...
      mask = np.ones_like(coreExp['spikeCount'], dtype=bool); # i.e. true
      # BUT, if we pass in trialSubset, then use this as our mask (i.e. overwrite the above mask)
      whichTrials = np.where(~np.isnan(np.sum(coreExp['ori'], 0)))[0];
      # TODO -- put in the proper responses...
      if respOverwrite is not None:
        resp = respOverwrite;
      else:
        if respMeasure == 0:
          resp = coreExp['spikeCount'].astype(int); # cast as int, since some are as uint
        elif respMeasure == 1:
          resp = coreExp['f1'];
        resp = np.expand_dims(resp, axis=1);
      resp = resp[whichTrials];

  # then, process the raw data such that trInf is [nTr x nComp]
  # for expInd == -1, this means mask [ind=0], then base [ind=1]
  trInf['num'] = whichTrials;
  trInf['ori'] = _cast_as_tensor(np.transpose(np.vstack(trialInf['ori']), (1,0))[whichTrials, :])
  trInf['tf'] = _cast_as_tensor(np.transpose(np.vstack(trialInf['tf']), (1,0))[whichTrials, :])
  trInf['ph'] = _cast_as_tensor(np.transpose(np.vstack(trialInf['ph']), (1,0))[whichTrials, :])
  trInf['sf'] = _cast_as_tensor(np.transpose(np.vstack(trialInf['sf']), (1,0))[whichTrials, :])
  trInf['con'] = _cast_as_tensor(np.transpose(np.vstack(trialInf['con']), (1,0))[whichTrials, :])
  resp = _cast_as_tensor(resp);

  return trInf, resp;

class dataWrapper(torchdata.Dataset):
    def __init__(self, expInfo, expInd=-1, respMeasure=0, device='cpu', whichTrials=None, respOverwrite=None):
        # if respMeasure == 0, then we're getting DC; otherwise, F1
        # respOverwrite means overwrite the responses; used only for expInd>=0 for now

        super().__init__();
        trInf, resp = process_data(expInfo, expInd, respMeasure, whichTrials=whichTrials, respOverwrite=respOverwrite)

        self.trInf = trInf;
        self.resp = resp;
        self.device = device;
        self.expInd = expInd;
        self.respMeasure = respMeasure
        
    def get_single_item(self, idx):
        # NOTE: This assumes that trInf['ori', 'tf', etc...] are already [nTr, nStimComp]
        feature = dict();
        feature['ori'] = self.trInf['ori'][idx, :]
        feature['tf'] = self.trInf['tf'][idx, :]
        feature['sf'] = self.trInf['sf'][idx, :]
        feature['con'] = self.trInf['con'][idx, :]
        feature['ph'] = self.trInf['ph'][idx, :]
        feature['num'] = idx; # which trials are part of this
        
        target = dict();
        try:
          target['resp'] = self.resp[idx];
          #target['resp'] = self.resp[idx, :];
        except:
          print('resp [rm=%d] has shape:' % self.respMeasure);
          print(self.resp.shape);

        if self.expInd == -1:
          maskInd, baseInd = hf_sfBB.get_mask_base_inds();
          target['maskCon'] = self.trInf['con'][idx, maskInd]
          target['baseCon'] = self.trInf['con'][idx, baseInd]
        else:
          target['cons'] = self.trInf['con'][idx, :];
        
        return (feature, target);

    def __getitem__(self, idx):
        return self.get_single_item(idx)

    def __len__(self):
        return len(self.resp)

### The model
class sfNormMod(torch.nn.Module):
  # inherit methods/fields from torch.nn.Module()

  def __init__(self, modParams, expInd=-1, excType=2, normType=1, lossType=1, lgnFrontEnd=0, newMethod=0, lgnConType=1, applyLGNtoNorm=1, device='cpu', pSfBound=14.9, pSfBound_low=0.1, fixRespExp=False):

    super().__init__();

    ### meta/fit parameters
    self.expInd = expInd;
    self.excType = excType
    self.normType = normType;
    self.lossType = lossType;
    self.lgnFrontEnd = lgnFrontEnd;
    self.lgnConType = lgnConType;
    self.applyLGNtoNorm = applyLGNtoNorm; # do we also apply the LGN front-end to the gain control tuning? Default is 1 for backwards compatability, but should be 0 (i.e., DON'T; per Eero, 21.02.26)
    self.device = device;
    self.newMethod = newMethod;
    self.maxPrefSf = _cast_as_tensor(pSfBound); # don't allow the max prefSf to exceed this value (will enforce via sigmoid)
    self.minPrefSf = _cast_as_tensor(pSfBound_low); # don't allow the prefSf to go below this value (will avoid div by 0)

    ### all modparams
    self.modParams = modParams;

    ### now, establish the parameters to optimize
    # Get parameter values
    nParams = hf.nParamsByType(self.normType, excType, lgnFrontEnd);

    # handle the possibility of a multi fitting first
    if self.lgnFrontEnd == 99:
      self.mWeight = _cast_as_param(modParams[nParams-1]);
    elif self.lgnFrontEnd == 1 and self.lgnConType == 1:
      self.mWeight = _cast_as_param(modParams[-1]);
    elif self.lgnConType == 2: # FORCE at 0.5
       self.mWeight = _cast_as_tensor(0); # then it's NOT a parameter, and is fixed at 0, since as input to sigmoid, this gives 0.5
    elif self.lgnConType == 3: # still not a parameter, but use the value in modParams; this case shouldn't really come up...
      self.mWeight = _cast_as_tensor(modParams[-1]);
    elif self.lgnConType == 4: # FORCE at -1, i.e. all parvo
      self.mWeight = _cast_as_tensor(-np.Inf); # then it's NOT a parameter, and is fixed at -Inf, since as input to sigmoid, this gives 0 (i.e. all parvo)
    # make sure mWeight is 0 if LGN is not on (this will also ensure it's defined)
    if lgnFrontEnd<=0: # i.e. no LGN, then overwrite this!
      self.mWeight = _cast_as_tensor(-np.Inf); # not used anyway!
      
    self.prefSf = _cast_as_param(modParams[0]);
    if self.excType == 1:
      self.dordSp = _cast_as_param(modParams[1]);
    if self.excType == 2:
      self.sigLow = _cast_as_param(modParams[1]);
      highInd = -1-np.sign(lgnFrontEnd);
      self.sigHigh = _cast_as_param(modParams[highInd]);

    # Other (nonlinear) model components
    self.sigma    = _cast_as_param(modParams[2]); # normalization constant
    #self.respExp  = _cast_as_tensor(1); # response exponent -- fixed at one (not a free param)
    self.respExp  = _cast_as_param(modParams[3]) if fixRespExp is None else _cast_as_tensor(modParams[3]); # response exponent
    self.scale    = _cast_as_param(modParams[4]); # response scalar

    # Noise parameters
    if force_earlyNoise is None:
      self.noiseEarly = _cast_as_param(modParams[5]);   # early additive noise
    else:
      self.noiseEarly = _cast_as_tensor(force_earlyNoise); # early additive noise - fixed value, i.e. NOT optimized in this case
    self.noiseLate  = _cast_as_param(modParams[6]);  # late additive noise
    if self.lossType == 3:
      self.varGain    = _cast_as_param(modParams[7]);  # multiplicative noise
    else:
      self.varGain    = _cast_as_tensor(modParams[7]);  # NOT optimized in this case

    ### Normalization parameters
    normParams = hf.getNormParams(modParams, normType);
    if self.normType == 1:
      self.inhAsym = _cast_as_tensor(normParams); # then it's not really meant to be optimized, should be just zero
      self.gs_mean = None; self.gs_std = None; # replacing the "else" in commented out 'if normType == 2 or normType == 4' below
    elif self.normType == 2 or self.normType == 5:
      self.gs_mean = _cast_as_param(normParams[0]);
      self.gs_std  = _cast_as_param(normParams[1]);
      if self.normType == 5:
        self.gs_gain = _cast_as_param(normParams[2]);
      else:
        self.gs_gain = None;
    elif self.normType == 3:
      # sigma calculation
      self.offset_sigma = _cast_as_param(normParams[0]);  # c50 filter will range between [v_sigOffset, 1]
      self.stdLeft      = _cast_as_param(normParams[1]);  # std of the gaussian to the left of the peak
      self.stdRight     = _cast_as_param(normParams[2]); # '' to the right '' 
      self.sfPeak       = _cast_as_param(normParams[3]); # where is the gaussian peak?
    elif self.normType == 4: # two-halved Gaussian...
      self.gs_mean = _cast_as_param(normParams[0]);
      self.gs_std = _cast_as_param(normParams[1]);
    else:
      self.inhAsym = _cast_as_param(normParams);

    ### LGN front parameters
    self.LGNmodel = 2; # parameterized as DiffOfGauss (not DoGsach)
    # prepopulate DoG parameters -- will overwrite, if needed
    self.M_k = _cast_as_tensor(1); # gain of 1
    self.M_fc = _cast_as_tensor(3); # char. freq of 3
    self.M_ks = _cast_as_tensor(0.3); # surround gain rel. to center
    self.M_js = _cast_as_tensor(0.4); # relative char. frequency of surround
    self.P_k = _cast_as_tensor(1); # gain of 1 [we handle M vs. P sensitivity in the RVC]
    self.P_fc = _cast_as_tensor(9); # char. freq of 9 [3x is a big discrepancy]
    self.P_ks = _cast_as_tensor(0.5); # surround gain rel. to center
    self.P_js = _cast_as_tensor(0.4); # relative char. freq of surround
    if self.lgnFrontEnd == 2:
      self.M_fc = _cast_as_tensor(6); # different variant (make magno f_c=6, not 3)
    elif self.lgnFrontEnd == 99: # 99 is code for fitting an LGN front end which is common across all cells in the dataset...
      # parameters are passed as [..., m_fc, p_fc, m_ks, p_ks, m_js, p_js]
      self.M_fc = _cast_as_param(self.modParams[-6]);
      self.M_ks = _cast_as_param(self.modParams[-4]);
      self.M_js = _cast_as_param(self.modParams[-2]);
      ## TODO: Will need to update P_fc in the code rather than rely on this one-time writing of self.P_fc (since self.M_fc will update)
      self.P_fc = torch.mul(self.M_fc, _cast_as_param(self.modParams[-5]));
      self.P_ks = _cast_as_param(self.modParams[-3]);
      self.P_js = _cast_as_param(self.modParams[-1]);
    # specify rvc parameters (not true "parameters", i.e. not optimized)
    if self.lgnFrontEnd > 0:
      self.rvcMod = 0; # Tony formulation of RVC
      self.rvc_m = _cast_as_tensor([0, 12.5, 0.05]); # magno has lower gain, c50
      self.rvc_p = _cast_as_tensor([0, 17.5, 0.50]);
      # --- and pack the DoG parameters we specified above
      self.dog_m = [self.M_k, self.M_fc, self.M_ks, self.M_js] 
      self.dog_p = [self.P_k, self.P_fc, self.P_ks, self.P_js]
    ### END OF INIT

  #######
  def print_params(self, transformed=1):
    # return a list of the parameters
    print('\n********MODEL PARAMETERS********');
    print('prefSf: %.2f' % (self.minPrefSf.item() + self.maxPrefSf.item()*torch.sigmoid(self.prefSf).item())); # was just self.prefSf.item()
    if self.excType == 1:
      dord = torch.mul(_cast_as_tensor(_sigmoidDord), torch.sigmoid(self.dordSp)) if transformed else self.dordSp.item();
      print('deriv. order: %.2f' % dord);
    elif self.excType == 2:
      print('sigma l|r: %.2f|%.2f' % (self.sigLow.item(), self.sigHigh.item()));
    if self.lgnFrontEnd > 0:
      mWt = torch.sigmoid(self.mWeight).item() if transformed else self.mWeight.item();
      print('mWeight: %.2f (orig %.2f)' % (mWt, self.mWeight.item()));
    scale = torch.mul(_cast_as_tensor(_sigmoidScale), torch.sigmoid(self.scale)).item() if transformed else self.scale.item();
    print('scalar|early|late: %.3f|%.3f|%.3f' % (scale, self.noiseEarly.item(), self.noiseLate.item()));
    print('norm. const.: %.2f' % self.sigma.item());
    if self.normType == 2 or self.normType == 5:
      normMn = torch.exp(self.gs_mean).item() if transformed else self.gs_mean.item();
      print('tuned norm mn|std: %.2f|%.2f' % (normMn, self.gs_std.item()));
      if self.normType == 5:
        print('\tAnd the norm gain (transformed|untransformed) is: %.2f|%.2f' % (torch.mul(_cast_as_tensor(_sigmoidGainNorm), torch.sigmoid(self.gs_gain)).item(), self.gs_gain.item()));
    print('still applying the LGN filter for the gain control' if self.applyLGNtoNorm else 'No LGN for GC')
    #print('still applying the LGN filter for the gain control') if self.applyLGNtoNorm else print('No LGN for GC')
    print('********END OF MODEL PARAMETERS********\n');

    return None;

  def return_params(self):
    # return a list of the parameters
    if self.normType == 1:
        if self.excType == 1:
          param_list = [self.prefSf.item(), self.dordSp.item(), self.sigma.item(), self.respExp.item(), self.scale.item(), self.noiseEarly.item(), self.noiseLate.item(), self.varGain.item(), self.inhAsym.item(), self.mWeight.item()];
        elif self.excType == 2:
          param_list = [self.prefSf.item(), self.sigLow.item(), self.sigma.item(), self.respExp.item(), self.scale.item(), self.noiseEarly.item(), self.noiseLate.item(), self.varGain.item(), self.inhAsym.item(), self.sigHigh.item(), self.mWeight.item()];
    elif self.normType == 2:
        if self.excType == 1:
          param_list = [self.prefSf.item(), self.dordSp.item(), self.sigma.item(), self.respExp.item(), self.scale.item(), self.noiseEarly.item(), self.noiseLate.item(), self.varGain.item(), self.gs_mean.item(), self.gs_std.item(), self.mWeight.item()];
        elif self.excType == 2:
          param_list = [self.prefSf.item(), self.sigLow.item(), self.sigma.item(), self.respExp.item(), self.scale.item(), self.noiseEarly.item(), self.noiseLate.item(), self.varGain.item(), self.gs_mean.item(), self.gs_std.item(), self.sigHigh.item(), self.mWeight.item()];
    elif self.normType == 5:
        if self.excType == 1:
          param_list = [self.prefSf.item(), self.dordSp.item(), self.sigma.item(), self.respExp.item(), self.scale.item(), self.noiseEarly.item(), self.noiseLate.item(), self.varGain.item(), self.gs_mean.item(), self.gs_std.item(), self.gs_gain.item(), self.mWeight.item()];
        elif self.excType == 2:
          param_list = [self.prefSf.item(), self.sigLow.item(), self.sigma.item(), self.respExp.item(), self.scale.item(), self.noiseEarly.item(), self.noiseLate.item(), self.varGain.item(), self.gs_mean.item(), self.gs_std.item(), self.gs_gain.item(), self.sigHigh.item(), self.mWeight.item()];
    if self.lgnFrontEnd == 0: # then we'll trim off the last constraint, which is mWeight bounds (and the last param, which is mWeight)
      param_list = param_list[0:-1];

    return param_list

  def simpleResp_matMul(self, trialInf, stimParams = [], sigmoidSigma=_sigmoidSigma, preCompOri=None):
    # returns object with simpleResp and other things
    # --- Created 20.10.12 --- provides ~4x speed up compared to SFMSimpleResp() without need to explicit parallelization
    # --- Updated 20.10.29 --- created new method 

    # simpleResp_matMul       Computes response of simple cell for sfmix experiment

    # simpleResp_matMul(varargin) returns a complex cell response for the
    # mixture stimuli used in sfMix. The cell's receptive field is the n-th
    # derivative of a 2-D Gaussian that need not be circularly symmetric.

    # NOTE: For first pass, not building in make_own_stim, since this will be for
    # - optimizing the model, not for evaluating at arbitrary stimuli

    # Get spatial coordinates
    xCo = 0; # in visual degrees, centered on stimulus center
    yCo = 0; # in visual degrees, centered on stimulus center

    # Store some results in M

    # Pre-allocate memory
    z             = trialInf;
    nStimComp     = hf.get_exp_params(self.expInd).nStimComp;
    nFrames       = hf.num_frames(self.expInd);
    try:
      nTrials = len(z['num']);
    except:
      nTrials = len(z['con'][0]); 
    
    ####
    # Set stim parameters
    stimTf = _cast_as_tensor(z['tf'], self.device);
    stimCo = _cast_as_tensor(z['con'], self.device);
    stimSf = _cast_as_tensor(z['sf'], self.device);
    stimOr = _cast_as_tensor((np.pi/180) * z['ori'], self.device);
    stimPh = _cast_as_tensor((np.pi/180) * z['ph'], self.device);

    ### LGN filtering stage
    ### Assumptions: No interaction between SF/con -- which we know is not true...
    # - first, SF tuning: model 2 (Tony)
    if self.lgnFrontEnd > 0:
      resps_m = get_descrResp(self.dog_m, stimSf, self.LGNmodel, minThresh=globalMin)
      resps_p = get_descrResp(self.dog_p, stimSf, self.LGNmodel, minThresh=globalMin)
      # -- make sure we normalize by the true max response:
      sfTest = _cast_as_tensor(np.geomspace(0.1, 15, 1000));
      max_m = torch.max(get_descrResp(self.dog_m, sfTest, self.LGNmodel, minThresh=globalMin));
      max_p = torch.max(get_descrResp(self.dog_p, sfTest, self.LGNmodel, minThresh=globalMin));
      # -- then here's our selectivity per component for the current stimulus
      selSf_m = torch.div(resps_m, max_m);
      selSf_p = torch.div(resps_p, max_p);
      # - then RVC response: # ASSUMES rvcMod 0 (Movshon)
      # --- the following commented out lines show how we could evaluate the rvc at one contrast for all components -->
      # -----> HOWEVER, this is how we factor in contrast, so likely this will be useful only for eval. SF at diff. con, if we introduce LGN shifts [22.10.03]
      #scm = torch.max(stimCo, axis=1)[0];
      #selCon_m = get_rvc_model(self.rvc_m, scm).unsqueeze(dim=1);
      #selCon_p = get_rvc_model(self.rvc_p, scm).unsqueeze(dim=1);
      selCon_m = get_rvc_model(self.rvc_m, stimCo); # could evaluate at torch.max(stimCo,axis=1)[0] rather than stimCo, i.e. highest grating con, not per grating
      selCon_p = get_rvc_model(self.rvc_p, stimCo);

      if self.lgnConType == 1: # DEFAULT
        # -- then here's our final responses per component for the current stimulus
        # ---- NOTE: The real mWeight will be sigmoid(mWeight), such that it's bounded between 0 and 1
        lgnSel = torch.add(torch.mul(torch.sigmoid(self.mWeight), torch.mul(selSf_m, selCon_m)), torch.mul(1-torch.sigmoid(self.mWeight), torch.mul(selSf_p, selCon_p)));
      elif self.lgnConType == 2 or self.lgnConType == 3 or self.lgnConType == 4:
        # -- Unlike the above (default) case, we don't allow for a separate M & P RVC - instead we just take the average of the two
        avgWt = torch.sigmoid(self.mWeight);
        selCon_avg = avgWt*selCon_m + (1-avgWt)*selCon_p;
        lgnSel = torch.add(torch.mul(torch.sigmoid(self.mWeight), torch.mul(selSf_m, selCon_avg)), torch.mul(1-torch.sigmoid(self.mWeight), torch.mul(selSf_p, selCon_avg)));

    if self.excType == 1:
      # Compute spatial frequency tuning - Deriv. order Gaussian
      sfRel = torch.div(stimSf, self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.prefSf));
      effDord = torch.mul(_cast_as_tensor(_sigmoidDord), torch.sigmoid(self.dordSp));
      s     = torch.pow(stimSf, effDord) * torch.exp(-effDord/2 * torch.pow(sfRel, 2));
      sMax  = torch.pow(self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.prefSf), effDord) * torch.exp(-effDord/2);
      selSf   = torch.div(s, sMax);
    elif self.excType == 2:
      selSf = flexible_Gauss([0,1,self.minPrefSf + self.maxPrefSf*torch.sigmoid(self.prefSf),self.sigLow,self.sigHigh], stimSf, minThresh=0, sigmoidValue=sigmoidSigma);
 
    # II. Phase, space and time
    if preCompOri is None:
      omegaX = torch.mul(stimSf, torch.cos(stimOr)); # the stimulus in frequency space
      omegaY = torch.mul(stimSf, torch.sin(stimOr));
    else: # preCompOri is the same for all trials/comps --> cos(stimOr) is [0], sin(-) is [1]
      omegaX = torch.mul(stimSf, preCompOri[0]); # the stimulus in frequency space
      omegaY = torch.mul(stimSf, preCompOri[1]);
    omegaT = stimTf;

    P = torch.empty((nTrials, nFrames, nStimComp, 3)); # nTrials x nFrames for number of frames x nStimComp x [two for x and y coordinate, one for time]
    P[:,:,:,0] = torch.full((nTrials, nFrames, nStimComp), 2*np.pi*xCo);  # P is the matrix that contains the relative location of each filter in space-time (expressed in radians)
    P[:,:,:,1] = torch.full((nTrials, nFrames, nStimComp), 2*np.pi*yCo);  # P(:,0) and p(:,1) describe location of the filters in space

    # NEW: 20.07.16 -- why divide by 2 for the LGN stage? well, if selectivity is at peak for M and P, sum will be 2 (both are already normalized) // could also just sum...
    if self.lgnFrontEnd > 0:
      selSi = torch.mul(selSf, lgnSel); # filter sensitivity for the sinusoid in the frequency domain
    else:
      selSi = selSf;

    # Use the effective number of frames displayed/stimulus duration
    # phase calculation -- 
    stimFr = torch.div(torch.arange(nFrames), float(nFrames));
    phOffset = torch.div(stimPh, torch.mul(2*np.pi, stimTf));
    # fast way?
    P3Temp = torch.add(phOffset.unsqueeze(-1), stimFr.unsqueeze(0).unsqueeze(0)).permute(0,2,1); # result is [nTrials x nFrames x nStimComp], so transpose
    P[:,:,:,2]  = 2*np.pi*P3Temp; # P(:,2) describes relative location of the filters in time.

    # per LCV code: preallocation and then filling in is much more efficient than using stack
    omegas = torch.empty((*omegaX.shape,3), device=omegaX.device);
    omegas[..., 0] = omegaX;
    omegas[..., 1] = omegaY;
    omegas[..., 2] = omegaT;
    #omegas = torch.stack((omegaX, omegaY, omegaT),axis=-1); # make this [nTr x nSC x nFr x 3]
    dotprod = torch.einsum('ijkl,ikl->ijk',P,omegas); # dotproduct over the "3" to get [nTr x nSC x nFr]
    # - AxB where A is the overall stimulus selectivity ([nTr x nStimComp]), B is Fourier ph/space calc. [nTr x nFr x nStimComp]
    pt1 = torch.mul(selSi, stimCo); 
    # as of 20.10.14, torch doesn't handle complex numbers
    # since we're just computing e^(iX), we can simply code the real (cos(x)) and imag (sin(x)) parts separately
    # -- by virtue of e^(iX) = cos(x) + i*sin(x) // Euler's identity
    realPart = torch.cos(dotprod);
    imagPart = torch.sin(dotprod);
    # per LCV code: preallocation and then filling in is much more efficient than using stack
    realImag = torch.empty((*realPart.shape,2), device=realPart.device);
    realImag[...,0] = realPart;
    realImag[...,1] = imagPart;
    # NOTE: here, I use the term complex to denote that it is a complex number NOT
    # - that it reflects a complex cell (i.e. this is still a simple cell response)
    rComplex = torch.einsum('ij,ikjz->ikz', torch.mul(selSi,stimCo), realImag) # mult. to get [nTr x nFr x 2] response
    # The above line takes care of summing over stimulus components

    # four filters placed in quadrature (only if self.newMethod == 0, which is default)

    # Store response in desired format - which is actually [nFr x nTr], so transpose it!
    if self.newMethod == 1:
      respSimple1 = rComplex[...,0]; # we'll keep the half-wave rectification for the end...
      return torch.transpose(respSimple1, 0, 1);
    else: # Old approach, in which we return the complex response
      # the remaining filters in quadrature
      respSimple1 = torch.max(_cast_as_tensor(globalMin), rComplex[...,0]); # half-wave rectification,...
      respSimple2 = torch.max(_cast_as_tensor(globalMin), torch.mul(_cast_as_tensor(-1),rComplex[...,0]));
      respSimple3 = torch.max(_cast_as_tensor(globalMin), rComplex[...,1]);
      respSimple4 = torch.max(_cast_as_tensor(globalMin), torch.mul(_cast_as_tensor(-1),rComplex[...,1]));

      # if channel is tuned, it is phase selective...
      # NOTE: 19.05.14 - made response always complex...(wow)! See git for previous version
      respComplex = torch.pow(respSimple1, 2) + torch.pow(respSimple2, 2) \
          + torch.pow(respSimple3, 2) + torch.pow(respSimple4, 2);
      respAvg = torch.div(respComplex, 4);
      respComp = torch.sqrt(respAvg); # div by 4 to avg across all filters

      return torch.transpose(respComp, 0, 1);

  def genNormWeightsSimple(self, trialInf, recenter_norm=recenter_norm, threshWeights=1e-3):
    ''' simply evaluates the usual normalization weighting but at the frequencies of the stimuli directly
    i.e. in effect, we are eliminating the bank of filters in the norm. pool
        --- if threshWeights is None, then we won't threshold the norm. weights; 
        --- if it's not None, we do max(threshWeights, calculatedWeights)
    '''

    sfs = _cast_as_tensor(trialInf['sf']); # [nComps x nTrials]
    cons = _cast_as_tensor(trialInf['con']); # [nComps x nTrials]
    consSq = np.square(cons);

    # apply LGN stage -
    # -- NOTE: previously, we applied equal M and P weight, since this is across a population of neurons, not just the one one neuron under consideration
    if self.lgnFrontEnd > 0 and self.applyLGNtoNorm:
      resps_m = get_descrResp(self.dog_m, sfs, self.LGNmodel, minThresh=globalMin)
      resps_p = get_descrResp(self.dog_p, sfs, self.LGNmodel, minThresh=globalMin)
      # -- make sure we normalize by the true max response:
      sfTest = _cast_as_tensor(np.geomspace(0.1, 15, 1000));
      max_m = torch.max(get_descrResp(self.dog_m, sfTest, self.LGNmodel, minThresh=globalMin));
      max_p = torch.max(get_descrResp(self.dog_p, sfTest, self.LGNmodel, minThresh=globalMin));
      # -- then here's our selectivity per component for the current stimulus
      selSf_m = torch.div(resps_m, max_m);
      selSf_p = torch.div(resps_p, max_p);
      # - then RVC response: # ASSUMES rvcMod 0 (Movshon)
      selCon_m = get_rvc_model(self.rvc_m, cons);
      selCon_p = get_rvc_model(self.rvc_p, cons);
      # -- then here's our final responses per component for the current stimulus
      if self.lgnConType == 1: # DEFAULT
        # -- then here's our final responses per component for the current stimulus
        # ---- NOTE: The real mWeight will be sigmoid(mWeight), such that it's bounded between 0 and 1
        lgnStage = torch.add(torch.mul(torch.sigmoid(self.mWeight), torch.mul(selSf_m, selCon_m)), torch.mul(1-torch.sigmoid(self.mWeight), torch.mul(selSf_p, selCon_p)));
      elif self.lgnConType == 2 or self.lgnConType == 3 or self.lgnConType == 4:
        # -- Unlike the above (default) case, we don't allow for a separate M & P RVC - instead we just take some average in-between of the two (depending on lgnConType)
        selCon_avg = torch.sigmoid(self.mWeight)*selCon_m + (1-torch.sigmoid(self.mWeight))*selCon_p;
        lgnStage = torch.add(torch.mul(torch.sigmoid(self.mWeight), torch.mul(selSf_m, selCon_avg)), torch.mul(1-torch.sigmoid(self.mWeight), torch.mul(selSf_p, selCon_avg)));
    else:
      lgnStage = torch.ones_like(sfs);

    if self.gs_mean is None or self.gs_std is None: # we assume inhAsym is 0
      self.inhAsym = _cast_as_tensor(0);
      new_weights = 1 + self.inhAsym*(torch.log(sfs) - torch.mean(torch.log(sfs)));
      new_weights = torch.mul(lgnStage, new_weights);
    elif self.normType == 2 or self.normType == 5:
      # Relying on https://pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal.log_prob
      log_sfs = torch.log(sfs);
      weight_distr = torch.distributions.normal.Normal(self.gs_mean, self.gs_std)
      new_weights = torch.exp(weight_distr.log_prob(log_sfs));
      gain_curr = torch.mul(_cast_as_tensor(_sigmoidGainNorm), torch.sigmoid(self.gs_gain)) if self.normType == 5 else _cast_as_tensor(1);
      new_weights = torch.mul(gain_curr, torch.mul(lgnStage, new_weights));
      if recenter_norm: # we'll recenter this weighted normalization around zero
        normMin, normMax = torch.min(new_weights), torch.max(new_weights)
        centerVal = (normMax-normMin)/2
        toAdd = 1-centerVal-normMin; # to make the center of these weights at 1
        new_weights = toAdd + new_weights
        if threshWeights is not None:
          new_weights = torch.max(_cast_as_tensor(threshWeights), new_weights);
        
    return new_weights;

  def SimpleNormResp(self, trialInf, trialArtificial=None, recenter_norm=recenter_norm):

    if trialArtificial is not None:
      trialInf = trialArtificial;
    else:
      trialInf = trialInf;
    consSq = torch.pow(_cast_as_tensor(trialInf['con']), 2);
    # cons (and wghts) will be (nComps x nTrials)
    wghts = self.genNormWeightsSimple(trialInf, recenter_norm=recenter_norm);

    # now put it all together
    resp = torch.mul(wghts, consSq);
    respPerTr = torch.sqrt(resp.sum(1)); # i.e. sum over components, then sqrt

    return respPerTr; # will be [nTrials] -- later, will ensure right output size during operation

  def respPerCell(self, trialInf, debug=0, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm, preCompOri=None):
    # excitatory filter, first
    simpleResp = self.simpleResp_matMul(trialInf, sigmoidSigma=sigmoidSigma, preCompOri=preCompOri);
    normResp = self.SimpleNormResp(trialInf, recenter_norm=recenter_norm); # [nFrames x nTrials]
    if self.newMethod == 1:
      Lexc = simpleResp; # [nFrames x nTrials]
      Linh = normResp; # un-normalized...
    else:
      Lexc = torch.div(simpleResp, torch.max(simpleResp)); # [nFrames x nTrials]
      Linh = torch.div(normResp, torch.max(normResp)); # normalize the normResp...

    # the below line assumes normType != 3 (which was only briefly used/explored, anyway...)
    sigmaFilt = torch.pow(torch.pow(_cast_as_tensor(10), self.sigma), 2); # i.e. square the normalization constant

    if debug:
      return Lexc, Linh, sigmaFilt;

    numerator     = torch.add(self.noiseEarly, Lexc); # [nFrames x nTrials]
    denominator   = torch.pow(sigmaFilt + torch.pow(Linh, 2), 0.5); # nTrials
    rawResp       = torch.div(numerator, denominator.unsqueeze(0)); # unsqueeze(0) to account for the missing leading dimension (nFrames)
    # half-wave rectification??? we add a squaring after the max, then respExp will do what it does...
    #ratio         = torch.pow(torch.pow(torch.max(_cast_as_tensor(globalMin), rawResp), 2), self.respExp);
    # just rectify, forget the squaring
    # -- this line if we're using sigmoid-transformed response exponent (bounded between [1,1+_sigmoidRespExp], currently [1,4]
    #ratio         = torch.pow(torch.max(_cast_as_tensor(globalMin), rawResp), 1+torch.mul(_cast_as_tensor(_sigmoidRespExp), torch.sigmoid(self.respExp)));
    # -- otherwise, this line
    ratio         = torch.pow(torch.max(_cast_as_tensor(globalMin), rawResp), self.respExp);
    # just rectify, don't even add the power
    #ratio = torch.max(_cast_as_tensor(globalMin), rawResp);
    # we're now skipping the averaging across frames...
    if self.newMethod == 1:
      if fall2020_adj:
        if spring2021_adj:
          respModel     = torch.max(_cast_as_tensor(globalMin), torch.add(self.noiseLate, torch.mul(torch.mul(_cast_as_tensor(_sigmoidScale), torch.sigmoid(self.scale)), ratio))); # why 10 as the scalar? From multiple fits, the value is never over 1 or 2, so 10 gives the parameter enough operating range
        else:
          respModel     = torch.max(_cast_as_tensor(globalMin), torch.add(self.noiseLate, torch.mul(torch.abs(self.scale), ratio)));
      else:
        respModel     = torch.add(self.noiseLate, torch.mul(torch.abs(self.scale), ratio));
      return torch.transpose(respModel, 1, 0);
    else:
      meanRate      = ratio.mean(0);
      respModel     = torch.add(self.noiseLate, torch.mul(torch.abs(self.scale), meanRate));
      return respModel; # I don't think we need to transpose here...

  def forward(self, trialInf, respMeasure=0, returnPsth=0, debug=0, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm, preCompOri=None): # expInd=-1 for sfBB
    # respModel is the psth! [nTr x nFr]
    respModel = self.respPerCell(trialInf, debug, sigmoidSigma=sigmoidSigma, recenter_norm=recenter_norm, preCompOri=preCompOri);

    if debug:
      return respModel

    stimDur = hf.get_exp_params(self.expInd).stimDur
    nFrames = respModel.shape[1]/stimDur; # convert to frames per second...
    # We bifurcate here based on sfBB experiment or not - why? I originally designed the code with only the sfBB experiments in mind
    # - and it takes advantage of the TF for all (2) stimulus components being the same for all trials
    # - for sfMix experiments, we have to handle it differently, but spike_fft works accordingly
    if self.expInd == -1:
      # then, get the base & mask TF
      maskInd, baseInd = hf_sfBB.get_mask_base_inds();
      maskTf, baseTf = trialInf['tf'][0, maskInd], trialInf['tf'][0, baseInd] # relies on tf being same for all trials (i.e. maskTf always same, baseTf always same)!
      tfAsInts = np.array([int(maskTf), int(baseTf)]);
      # important to transpose the respModel before passing in to spike_fft
      amps, rel_amps, full_fourier = spike_fft([respModel], tfs=[tfAsInts], stimDur=stimDur, binWidth=1.0/nFrames)
    else:
      tfAsInts = trialInf['tf']; # does not actually need to be integer values...
      # important to transpose the respModel before passing in to spike_fft
      amps, rel_amps, full_fourier = spike_fft([respModel], tfs=tfAsInts, stimDur=stimDur, binWidth=1.0/nFrames)

    if returnPsth == 1:
      if respMeasure == 1: # i.e. F1
        return rel_amps[0], respModel;
      else:
        return amps[0][:,0], respModel; # i.e. just the DC, position zero
    else:
      if respMeasure == 1: # i.e. F1
        return rel_amps[0];
      else:
        return amps[0][:,0]; # i.e. just the DC, position zero

### End of class (sfNormMod)
    
def loss_sfNormMod(respModel, respData, lossType=1, debug=0, nbinomCalc=2, varGain=None):
  # nbinomCalc, varGain used only in lossType == 3

  if lossType == 1: # sqrt
      lsq = torch.pow(torch.sign(respModel)*torch.sqrt(torch.abs(respModel)) - torch.sign(respData)*torch.sqrt(torch.abs(respData)), 2);

      per_cond = lsq;
      NLL = torch.mean(lsq);

  if lossType == 2: # poiss TODO FIX TODO
      poiss_loss = torch.nn.PoissonNLLLoss(log_input=False);
      if debug == 1: # only eval if debug, since this is slow
        per_cond = _cast_as_tensor([poiss_loss(x,y) for x,y in zip(respModel, respData)]);
      NLL = poiss_loss(respModel, respData); # previously was respData, respModel

  if lossType == 3:
      # varGain, respData, respMeans
      # -- all_counts is the spike count from every trial
      # -- count_mean is averaged across condition
      # - How does it work? Mu will need to be broadcast to be the same length as all_counts
      # - p is similarly broadcast, while r is just one value
      mu = torch.max(_cast_as_tensor(.1), respModel); # The predicted mean spike count
      # -- sigmoid(varGain) to ensure it's non-negative
      var = mu + (torch.sigmoid(varGain)*torch.pow(mu, 2)); # The corresponding variance of the spike count
      # Note: Two differeing versions of r - the first (doesn't use var) is from Robbe's code shared through Hasse/Mariana
      # -- the second is from the code that was in early code of my V1 model as written by Robbe
      if nbinomCalc == 1:
          r  = 1/varGain;
      elif nbinomCalc == 2:
          r  = torch.pow(mu, 2) / (var - mu); # The parameters r and p of the negative binomial distribution
      pSucc  = r / (r + mu);
      p = 1-pSucc; # why? Well, compared to scipy.stats.nbinom, the "p" here is for failure, not success, so it should be 1-p
      nbinomDistr = torch.distributions.negative_binomial.NegativeBinomial(r,p);
      # -- Evaluate the model
      # ---- but first, make any negative values just greater than zero (if F1 responses)
      if 'flo' in str(respData.dtype):
        respData = torch.max(_cast_as_tensor(1e-6), respData);
      llh = nbinomDistr.log_prob(respData); # The likelihood for each pass under the doubly stochastic model
      per_cond = -llh;
      NLL = torch.mean(-llh);

  if debug:
    return NLL, per_cond;
  else:
    return NLL;

### Now, actually do the optimization!

### Now, the optimization
# - what to specify...
#def setParams():
#  ''' Set the parameters of the model '''

### 22.10.01 --> max_epochs was 15000
### --- temporarily, reduce to make faster
# ---- previous to 22.10.03, batch_size=3000 (trials)
def setModel(cellNum, expDir=-1, excType=1, lossType=1, fitType=1, lgnFrontEnd=0, lgnConType=1, applyLGNtoNorm=1, max_epochs=1000, learning_rate=0.001, batch_size=256, scheduler=True, initFromCurr=0, kMult=0.1, newMethod=0, fixRespExp=None, trackSteps=True, fL_name=None, respMeasure=0, vecCorrected=0, whichTrials=None, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm, to_save=True, pSfBound=15, pSfFloor=0.1, allCommonOri=True, rvcName = 'rvcFitsHPC_220928', rvcMod=1, rvcDir=1, returnOnlyInits=False): # learning rate 0.04 on 22.10.01 (0.15 seems too high - 21.01.26); was 0.10 on 21.03.31;
  '''
  # --- max_epochs usually 7500; learning rate _usually_ 0.04-0.05
  # --- to_save should be set to False if calling setModel in parallel!
  '''
  global dataListName
  global force_full
  
  ### Load the cell, set up the naming
  ########
  # Load cell
  ########
  loc_base = os.getcwd() + '/';
  loc_data = loc_base + expDir + 'structures/';

  if 'pl1465' in loc_base:
    loc_str = 'HPC';
  else:
    loc_str = '';
  #loc_str = 'HPC';

  if fL_name is None: # otherwise, it's already defined...
    if modRecov == 1:
      fL_name = 'mr_fitList%s_190516cA' % loc_str
    else:
      if excType == 1: #dG for derivative of Gaussian
        #fL_name = 'fitList%s_pyt_201017' % (loc_str); # pyt for pytorch
        #fL_name = 'fitList%s_pyt_210226_dG' % (loc_str); # pyt for pytorch - 2x2 matrix of fit type (all with at least SOME LGN front-end)
        fL_name = 'fitList%s_pyt_210308_dG' % (loc_str); # pyt for pytorch
        if recenter_norm:
          #fL_name = 'fitList%s_pyt_210312_dG' % (loc_str); # pyt for pytorch
          #fL_name = 'fitList%s_pyt_210314%s_dG' % (loc_str, rExpStr); # pyt for pytorch
          fL_name = 'fitList%s_pyt_210321_dG' % (loc_str); # pyt for pytorch
          if force_full:
            fL_name = 'fitList%s_pyt_210331_dG' % (loc_str); # pyt for pytorch
        #fL_name = 'fitList%s_pyt_210304_dG' % (loc_str); # pyt for pytorch; FULL datalists for V1_orig, altExpl
      elif excType == 2:
        #fL_name = 'fitList%s_pyt_201107' % (loc_str); # pyt for pytorch
        #fL_name = 'fitList%s_pyt_210121' % (loc_str); # pyt for pytorch - lgn flat vs. V1 weight
        #fL_name = 'fitList%s_pyt_210226' % (loc_str); # pyt for pytorch - 2x2 matrix of fit type (all with at least SOME LGN front-end)
        if sigmoidSigma is None:
          fL_name = 'fitList%s_pyt_210308' % (loc_str); # pyt for pytorch - 2x2 matrix of fit type (all with at least SOME LGN front-end)
        else:
          fL_name = 'fitList%s_pyt_210310' % (loc_str); # pyt for pytorch - 2x2 matrix of fit type (all with at least SOME LGN front-end)
        # overwriting...
        if recenter_norm:
          #fL_name = 'fitList%s_pyt_210312' % (loc_str); # pyt for pytorch
          #fL_name = 'fitList%s_pyt_210314%s' % (loc_str, rExpStr); # pyt for pytorch
          fL_name = 'fitList%s_pyt_210321' % (loc_str);
          if force_full:
            fL_name = 'fitList%s_pyt_210331' % (loc_str); # pyt for pytorch
    # TEMP: Just overwrite any of the above with this name
    fL_name = 'fitList%s_pyt_221007f_noRE' % loc_str;

  todoCV = 1 if whichTrials is not None else 0;

  fitListName = hf.fitList_name(base=fL_name, fitType=fitType, lossType=lossType, lgnType=lgnFrontEnd, lgnConType=lgnConType, vecCorrected=vecCorrected, CV=todoCV);
  # get the name for the stepList name, regardless of whether or not we keep this now
  stepListName = str(fitListName.replace('.npy', '_details.npy'));

  print('\nFitList: %s [expDir is %s]' % (fitListName, expDir));

  # Load datalist, then specific cell
  try:
    dataList = hf.np_smart_load(str(loc_data + dataListName));
  except:
    dataListName = hf.get_datalist(expDir, force_full=force_full);
    dataList = hf.np_smart_load(str(loc_data + dataListName));
  dataNames = dataList['unitName'];
  print('loading data structure from %s...' % loc_data);
  try:
    expInd = hf.exp_name_to_ind(dataList['expType'][cellNum-1]);
  except:
    if expDir == 'V1_BB/':
      expInd = -1; # for sfBB
    elif expDir == 'V1_orig/':
      expInd = 1;
  print('expInd is %d' % expInd);
  # - then cell
  if expInd == -1:
    S = hf.np_smart_load(str(loc_data + dataNames[cellNum-1] + '_sfBB.npy')); # why -1? 0 indexing...
  else:
    S = hf.np_smart_load(str(loc_data + dataNames[cellNum-1] + '_sfm.npy')); # why -1? 0 indexing...
  if expInd == -1:
    expInfo = S['sfBB_core'];
  else:
    expInfo = S['sfm']['exp']['trial'];

  respOverwrite = None; # default to None, but if vecCorrected and expInd != -1, then we will specify
  print('respMeasure, vecCorrected: %d, %d' % (respMeasure, vecCorrected));
  if respMeasure == 1 and expInd!=1: # we cannot do F1 on V1_orig
    # NOTE: For F1, we keep responses per component, and zero-out the blanks later on
    if vecCorrected: # then do vecF1 correction
      if expInd == -1:
        # Overwrite f1 spikes
        vec_corr_mask, vec_corr_base = hf_sfBB.adjust_f1_byTrial(expInfo);
        expInfo['f1_mask'] = vec_corr_mask;
        expInfo['f1_base'] = vec_corr_base;
      else:
        respOverwrite = hf.adjust_f1_byTrial(expInfo, expInd);
    else: # then do phAmp correction here!
      if expInd == -1:
        # Overwrite f1 spikes
        #### FIX THIS --> make sure you get trial-by-trial, phAmp corrected for sfBB
        vec_corr_mask, vec_corr_base = hf_sfBB.adjust_f1_byTrial(expInfo);
        expInfo['f1_mask'] = vec_corr_mask;
        expInfo['f1_base'] = vec_corr_base;
      else: # and the same for the "main" experiment
        rvcFits = hf.get_rvc_fits(loc_data, expInd, cellNum, rvcName=rvcName, rvcMod=rvcMod, direc=rvcDir, vecF1=0);
        respOverwrite, whichMeasure_out = hf.get_adjusted_spikerate(expInfo, cellNum, expInd, loc_data, rvcName=rvcFits, rvcMod=-1, baseline_sub=False, return_measure=True, vecF1=0, force_f1=True, returnByComp=True); # if we're here, then we get F1 regardless of f1f0
        # however, if whichMeasure is D.C., then we quit!
        if whichMeasure_out==0: # i.e. dc
          print('Cell %d failed in getting F1 responses' % cellNum);
          return [], []; # this one failed --> could not get DC
  elif respMeasure == 1 and expInd==1:
    if to_save:
      sys.exit('Cannot run F1 model analysis on V1_orig/ experiment - exiting!');
    else:
      return [], []; # return two blank placeholders so that the parallelization can move on

  try:
    trInf, resp = process_data(expInfo, expInd=expInd, respMeasure=respMeasure, whichTrials=whichTrials, respOverwrite=respOverwrite);
  except:
    if to_save:
      raise Exception("Could not process_data in mrpt.setModel --> cell %d, respMeasure %d" % (cellNum, respMeasure))
    else:
      return [], [];
  # we zero out the blanks later on for all other loss types, but do it here otherwise
  if lossType == 3:
    if respMeasure == 1:
      blanks = np.where(trInf['con']==0); # then we'll need to zero-out the blanks
      resp[blanks] = 1e-6; # TODO: should NOT be magic number (replace with global_min?)
    orgMeans = _cast_as_tensor(organize_mean_perCond(trInf, resp));
 
  respStr = hf_sfBB.get_resp_str(respMeasure);

  if os.path.isfile(loc_data + fitListName):
    fitList = hf.np_smart_load(str(loc_data + fitListName));
    try:
      curr_fit = fitList[cellNum-1][respStr];
      curr_params = curr_fit['params'];
      # Run the model, evaluate the loss to ensure we have a valid parameter set saved -- otherwise, we'll generate new parameters
      testModel = sfNormMod(curr_params, expInd=expInd, excType=excType, normType=fitType, lossType=lossType, lgnConType=lgnConType, newMethod=newMethod, lgnFrontEnd=lgnFrontEnd, applyLGNtoNorm=applyLGNtoNorm)
      trInfTemp, respTemp = process_data(expInfo, expInd, respMeasure, respOverwrite=respOverwrite) # warning: added respOverwrite here; also add whichTrials???
      predictions = testModel.forward(trInfTemp, respMeasure=respMeasure);
      if testModel.lossType == 3:
        loss_test = loss_sfNormMod(_cast_as_tensor(predictions.flatten()), _cast_as_tensor(respTemp.flatten()), testModel.lossType, varGain=testModel.varGain)
      else:
        loss_test = loss_sfNormMod(_cast_as_tensor(predictions.flatten()), _cast_as_tensor(respTemp.flatten()), testModel.lossType)
      if np.isnan(loss_test.item()):
        initFromCurr = 0; # then we've saved bad parameters -- force new ones!
    except:
      initFromCurr = 0; # force the old parameters
  else:
    initFromCurr = 0;
    fitList = dict();
    curr_fit = dict();

  ### set parameters
  # --- first, estimate prefSf, normConst if possible (TODO); inhAsym, normMean/Std
  prefSfEst_goal = np.random.uniform(0.3, 2); # this is the value we want AFTER taking the sigmoid (and applying the upper bound)
  sig_inv_input = (pSfFloor+prefSfEst_goal)/pSfBound;
  prefSfEst = -np.log((1-sig_inv_input)/sig_inv_input)
  normConst = -2; # per Tony, just start with a low value (i.e. closer to linear)
  if fitType == 1:
    inhAsym = 0;
  if fitType == 2 or fitType == 5:
    # see modCompare.ipynb, "Smarter initialization" for details
    normMean = np.random.uniform(0.75, 1.25) * np.log10(prefSfEst_goal) if initFromCurr==0 else curr_params[8]; # start as matched to excFilter
    normStd = np.random.uniform(0.3, 2) if initFromCurr==0 else curr_params[9]; # start at high value (i.e. broad)
    if fitType == 5:
      normGain = np.random.uniform(-3, -1) if initFromCurr == 0 else curr_params[10]; # will be a sigmoid-ed value...
  # --- then, set up each parameter
  pref_sf = float(prefSfEst) if initFromCurr==0 else curr_params[0];
  if excType == 1:
    dOrd_preSigmoid = np.random.uniform(1, 2.5)
    dOrdSp = -np.log((_sigmoidDord-dOrd_preSigmoid)/dOrd_preSigmoid) if initFromCurr==0 else curr_params[1];
  elif excType == 2:
    if _sigmoidSigma is None:
      sigLow = np.random.uniform(0.1, 0.3) if initFromCurr==0 else curr_params[1];
      # - make sigHigh relative to sigLow, but bias it to be lower, i.e. narrower
      sigHigh = sigLow*np.random.uniform(0.5, 1.25) if initFromCurr==0 else curr_params[-1-np.sign(lgnFrontEnd)]; # if lgnFrontEnd == 0, then it's the last param; otherwise it's the 2nd to last param
    else: # this is from modCompare::smarter initialization, assuming _sigmoidSigma = 5
      sigLow = np.random.uniform(-2, 0.5);
      sigHigh = np.random.uniform(-2, 0.5);
  normConst = normConst if initFromCurr==0 else curr_params[2];
  if fixRespExp is not None:
    respExp = fixRespExp; # then, we set it to this value and make it a tensor (rather than parameter)
  else:
    respExp = np.random.uniform(1.5, 2.5) if initFromCurr==0 else curr_params[3];
  if newMethod == 0:
    # easier to start with a small scalar and work up, rather than work down
    respScalar = np.random.uniform(200, 700) if initFromCurr==0 else curr_params[4];
    noiseEarly = -1 if initFromCurr==0 else curr_params[5]; # 02.27.19 - (dec. up. bound to 0.01 from 0.1)
    noiseLate = 1e-1 if initFromCurr==0 else curr_params[6];
  else: # see modCompare.ipynb, "Smarter initialization" for details
    if respMeasure == 0: # slightly different range of successfully-fit respScalars for DC vs. F1 fits 
      if spring2021_adj:
        respScalar = np.random.uniform(-7, -2) if initFromCurr==0 else curr_params[4];
      else:
        respScalar = np.power(10, np.random.uniform(-1, 0)) if initFromCurr==0 else curr_params[4];
      if force_earlyNoise is None:
        noiseLate = np.random.uniform(-0.4, 0.4) if initFromCurr==0 else curr_params[6];
        noiseEarly = np.random.uniform(-0.5, 0.1) if initFromCurr==0 else curr_params[5]; # 02.27.19 - (dec. up. bound to 0.01 from 0.1)
      else: # again, from modCompare::smarter initialization, with force_noiseEarly = 0
        noiseLate = np.random.uniform(-2, 0.4) if initFromCurr==0 else curr_params[6];
        noiseEarly = force_earlyNoise if initFromCurr==0 else curr_params[6];
    elif respMeasure == 1:
      if spring2021_adj:
        respScalar = np.random.uniform(-12, -4) if initFromCurr==0 else curr_params[4];
      else:
        respScalar = np.power(10, np.random.uniform(-2.5, -0.5)) if initFromCurr==0 else curr_params[4];
      if force_earlyNoise is None:
        noiseLate = np.random.uniform(0, 1) if initFromCurr==0 else curr_params[6];
        noiseEarly = np.random.uniform(-0.5, 0.5) if initFromCurr==0 else curr_params[5]; # 02.27.19 - (dec. up. bound to 0.01 from 0.1)
      else: # again, from modCompare::smarter initialization, with force_noiseEarly = 0
        noiseLate = np.random.uniform(0, 3) if initFromCurr==0 else curr_params[6];
        noiseEarly = force_earlyNoise if initFromCurr==0 else curr_params[6];
  varGain = np.random.uniform(0.01, 1) if initFromCurr==0 else curr_params[7];
  if lgnFrontEnd > 0:
    # Now, the LGN weighting 
    mWt_preSigmoid = np.random.uniform(0.25, 0.75); # this is what we want the real/effective mWeight initialization to be
    mWeight = -np.log((1-mWt_preSigmoid)/mWt_preSigmoid) if initFromCurr==0 else curr_params[-1];
  else:
    mWeight = -99; # just a "dummy" value

  # --- finally, actually create the parameter list
  if fitType == 1:
    if excType == 1:
      param_list = (pref_sf, dOrdSp, normConst, respExp, respScalar, noiseEarly, noiseLate, varGain, inhAsym, mWeight);
    elif excType == 2:
      param_list = (pref_sf, sigLow, normConst, respExp, respScalar, noiseEarly, noiseLate, varGain, inhAsym, sigHigh, mWeight);
  elif fitType == 2:
    if excType == 1:
      param_list = (pref_sf, dOrdSp, normConst, respExp, respScalar, noiseEarly, noiseLate, varGain, normMean, normStd, mWeight);
    elif excType == 2:
      param_list = (pref_sf, sigLow, normConst, respExp, respScalar, noiseEarly, noiseLate, varGain, normMean, normStd, sigHigh, mWeight);
  elif fitType == 5:
    ### TODO: make this less redundant???
    if excType == 1:
      param_list = (pref_sf, dOrdSp, normConst, respExp, respScalar, noiseEarly, noiseLate, varGain, normMean, normStd, normGain, mWeight);
    elif excType == 2:
      param_list = (pref_sf, sigLow, normConst, respExp, respScalar, noiseEarly, noiseLate, varGain, normMean, normStd, normGain, sigHigh, mWeight);
  if lgnFrontEnd == 0: # then we'll trim off the last constraint, which is mWeight bounds (and the last param, which is mWeight)
    param_list = param_list[0:-1];   

  ### define model, grab training parameters
  model = sfNormMod(param_list, expInd, excType=excType, normType=fitType, lossType=lossType, newMethod=newMethod, lgnFrontEnd=lgnFrontEnd, lgnConType=lgnConType, applyLGNtoNorm=applyLGNtoNorm, pSfBound=pSfBound, pSfBound_low=pSfFloor)

  training_parameters = [p for p in model.parameters() if p.requires_grad]
  model.print_params(); # optionally, nicely print the initial parameters...

  if returnOnlyInits: # 22.10.06 --> make an option to just use this for returning the initial parameters
    return param_list;

  ###  data wrapping
  dw = dataWrapper(expInfo, respMeasure=respMeasure, expInd=expInd, respOverwrite=respOverwrite); # respOverwrite defined above (None if DC or if expInd=-1)
  exp_length = expInfo['num'][-1]; # longest trial
  print('cell %d: rem. is %d [last iteration]' % (cellNum, np.mod(exp_length,batch_size)));
  dl_shuffle = batch_size<2000 # i.e. if batch_size<2000, then shuffle!
  dl_droplast = bool(np.mod(exp_length,batch_size)<10) # if the last iteration will have fewer than 10 trials, drop it!
  dataloader = torchdata.DataLoader(dw, batch_size, shuffle=dl_shuffle, drop_last=dl_droplast)

  ### then set up the optimization
  optimizer = torch.optim.Adam(training_parameters, amsgrad=True, lr=learning_rate, ) # amsgrad is variant of opt
  # - and the LR scheduler, if applicable
  if scheduler:
    # value of 0.5 per Billy (21.02.09); patience is # of epochs before we start to reduce the LR
    LR_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                              factor=0.3, patience=np.maximum(10, int(max_epochs/10))); # factor was 0.5 when l.r. was 0.10; 0.15 when lr was 0.20
                                                              #factor=0.3, patience=np.maximum(8, int(max_epochs/10))); # factor was 0.5 when l.r. was 0.10; 0.15 when lr was 0.20
                                                              #factor=0.3, patience=int(max_epochs/15)); # factor was 0.5 when learning rate was 0.10; 0.15 when lr was 0.20

  # - then data
  # - predefine some arrays for tracking loss
  loss_history = []
  start_time = time.time()
  time_history = []
  model_history = []
  hessian_history = []

  if len(np.unique(trInf['ori']))==1: # then we can pre-compute the ori!
    only_ori = np.unique(trInf['ori']);
    preCompOri = [torch.cos(_cast_as_tensor((np.pi/180)*only_ori)), torch.sin(_cast_as_tensor((np.pi/180)*only_ori))]
  else:
    preCompOri = None;
  first_pred = model.forward(trInf, respMeasure=respMeasure, preCompOri=preCompOri);

  #import cProfile
  #cProfile.runctx('model.simpleResp_matMul(trInf, preCompOri=preCompOri)', {'model':model}, locals())
  #cProfile.runctx('model.forward(trInf, respMeasure=respMeasure, preCompOri=preCompOri)', {'model':model}, locals())

  accum = np.nan; # keep track of accumulator (will be replaced with zero at first step)
  for t in range(max_epochs):
      optimizer.zero_grad() # reset the gradient for each epoch!

      loss_history.append([])
      time_history.append([])

      for bb, (feature, target) in enumerate(dataloader):
          predictions = model.forward(feature, respMeasure=respMeasure, preCompOri=preCompOri)
          if respMeasure == 1: # figure out which stimulus components were blank for the given trials
            if expInd == -1:
              maskInd, baseInd = hf_sfBB.get_mask_base_inds();
              target['resp'][target['maskCon']==0,0] = 1e-6 # force F1 ~ 0 if con of that stim is 0
              target['resp'][target['baseCon']==0,1] = 1e-6 # force F1 ~ 0 if con of that stim is 0
              predictions[target['maskCon']==0,0] = 1e-6 # force F1 ~ 0 if con of that stim is 0
              predictions[target['baseCon']==0,1] = 1e-6 # force F1 ~ 0 if con of that stim is 0
            else:
              blanks = np.where(target['cons']==0);
              target['resp'][blanks] = 1e-6; # force F1 ~ 0 if con of that component is 0
              predictions[blanks] = 1e-6; # force F1 ~ 0 if con of that component is 0
          target = target['resp'].flatten(); # since it's [nTr, 1], just make it [nTr] (if respMeasure == 0)
          predictions = predictions.flatten(); # either [nTr, nComp] to [nComp*nTr] or [nTr,1] to [nTr]
          if model.lossType == 3:
            loss_curr = loss_sfNormMod(predictions, target, model.lossType, varGain=model.varGain)
          else:
            loss_curr = loss_sfNormMod(predictions, target, model.lossType)

          if np.mod(t,500)==0: # and bb==0:
              if bb == 0:
                now = datetime.datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print('\n****** STEP %d [%s] [t=%s] [prev loss: %.3f] *********' % (t, respStr, current_time, accum))
                #print('\nTARGET, then predictions, finally loss');
                #print(target[0:20]);
                #print(predictions[0:20]);
                #print(loss_sfNormMod(predictions[0:20], target[0:20], model.lossType, debug=1)[1]);
                #print('\n****** STEP %d [%s] [prev loss: %.2f] *********' % (t, respStr, accum))
                accum = 0;
              prms = model.named_parameters()
              #[print(x, '\n') for x in prms];
              curr_loss = loss_curr.item();
              accum += curr_loss;
              #print('\t%.3f' % curr_loss);
              #print('\t%.3f' % loss_curr.item());
              #print(loss_curr.item())
              #print(loss_curr.grad)

          loss_history[t].append(loss_curr.item())
          time_history[t].append(time.time() - start_time)
          if np.isnan(loss_curr.item()) or np.isinf(loss_curr.item()):
            if to_save: # we raise an exception here and then try again.
              raise Exception("Loss is nan or inf on epoch %s, batch %s!" % (t, 0))
            else: # otherwise, it's assumed that we're running this in parallel, so let's just give up on this cell!
              return [], []; # we'll just save empty lists...

          loss_curr.backward(retain_graph=True)
          optimizer.step()
          if scheduler:
            LR_scheduler.step(loss_curr.item());

      model.eval()
      model.train()

  ##############
  #### OPTIM IS DONE ####
  ##############
  # Most importantly, get the optimal parameters
  opt_params = model.return_params(); # check this...
  curr_resp = model.forward(dw.trInf, respMeasure=respMeasure, preCompOri=preCompOri);
  gt_resp = _cast_as_tensor(dw.resp);
  # fix up responses if respMeasure == 1 (i.e. if mask or base con is 0, match the data & model responses...)
  if respMeasure == 1:
    if expInd == -1:
      maskInd, baseInd = hf_sfBB.get_mask_base_inds();
      curr_resp[dw.trInf['con'][:,maskInd]==0, maskInd] = 1e-6 # force F1 ~= 0 if con of that stim is 0
      curr_resp[dw.trInf['con'][:,baseInd]==0, baseInd] = 1e-6 # force F1 ~= 0 if con of that stim is 0
      gt_resp[dw.trInf['con'][:,maskInd]==0, maskInd] = 1e-6 # force F1 ~= 0 if con of that stim is 0
      gt_resp[dw.trInf['con'][:,baseInd]==0, baseInd] = 1e-6 # force F1 ~= 0 if con of that st
    else:
      blanks = np.where(dw.trInf['con']==0);
      curr_resp[blanks] = 1e-6; # force F1 ~ 0 if con of that component is 0
      gt_resp[blanks] = 1e-6; # force F1 ~ 0 if con of that component is 0

  if model.lossType == 3:
    NLL = loss_sfNormMod(curr_resp.flatten(), gt_resp.flatten(), model.lossType, varGain=model.varGain).detach().numpy();
  else:
    NLL = loss_sfNormMod(curr_resp.flatten(), gt_resp.flatten(), model.lossType).detach().numpy();

  ## we've finished optimization, so reload again to make sure that this NLL is better than the currently saved one
  ## -- why do we have to do it again here? We may be running multiple fits for the same cells at the same and we want to make sure that if one of those has updated, we don't overwrite that opt. if it's better
  currNLL = 1e7;
  if os.path.exists(loc_data + fitListName) and to_save: # otherwise, no need to reload...
    fitList = hf.np_smart_load(str(loc_data + fitListName));
  try: # well, even if fitList loads, we might not have currNLL, so we have to have an exception here
    currNLL = fitList[cellNum-1][respStr]['NLL']; # exists - either from real fit or as placeholder
  except:
    pass; # we've already defined the currNLL...

  try:
    nll_history = fitList[cellNum-1][respStr]['nll_history'];
  except:
    nll_history = np.array([]);

  ### SAVE: Now we save the results, including the results of each step, if specified
  print('...finished. New NLL (%.2f) vs. previous NLL (%.2f)' % (NLL, currNLL)); 
  # reload fitlist in case changes have been made with the file elsewhere!
  if os.path.exists(loc_data + fitListName) and to_save:
    fitList = hf.np_smart_load(str(loc_data + fitListName));
  # else, nothing to reload!!!
  # but...if we reloaded fitList and we don't have this key (cell) saved yet, recreate the key entry...
  # TODO: Make this smarter for doing cross-validation...

  # curr_fit will slot into fitList[cellNum-1][respStr]

  # now, if the NLL is now the best, update this
  if NLL < currNLL:
    curr_fit = dict();
    curr_fit['NLL'] = NLL;
    curr_fit['params'] = opt_params;
    # NEW: Also save *when* this most recent fit was made (19.02.04); and nll_history below
    curr_fit['time'] = datetime.datetime.now();
    # NEW: Also also save entire loss/optimizaiotn structure
    optInfo = dict();
    optInfo['call'] = optimizer;
    optInfo['epochs'] = max_epochs;
    optInfo['batch_size'] = batch_size;
    optInfo['learning_rate'] = learning_rate;
    optInfo['shuffle'] = dl_shuffle;
    optInfo['dropLast'] = dl_droplast;
    optInfo['init_params'] = param_list; # should be helpful to know starting values
    curr_fit['opt'] = optInfo;
    curr_fit['nll_history'] = np.append(nll_history, NLL);
  else:
    print('new NLL not less than currNLL, not saving result, but updating overall fit list (i.e. tracking each fit)');

  if to_save:
    if cellNum-1 not in fitList:
      print('cell did not exist yet');
      fitList[cellNum-1] = dict();
      fitList[cellNum-1][respStr] = dict();
    elif respStr not in fitList[cellNum-1]:
      print('%s did not exist yet' % respStr);
      fitList[cellNum-1][respStr] = dict();
    else:
      print('we will be overwriting %s (if updating)' % respStr);
    # again, this is under to_save; otherwise, we're just going to return curr_fit
    fitList[cellNum-1][respStr] = curr_fit;
    np.save(loc_data + fitListName, fitList);

  if os.path.exists(loc_data + stepListName):
    try:
      stepList = hf.np_smart_load(str(loc_data + stepListName));
      curr_steplist = stepList[cellNum-1][respStr];
    except: # if the file is corrupted in some way...
      stepList = dict();
      curr_steplist = dict();
  else:
    stepList = dict();
    curr_steplist = dict();

  # now the step list, if needed
  if trackSteps and NLL < currNLL:
    curr_steplist = dict(); # again, will slot into stepList[cellNum-1][respStr]
    curr_steplist['loss'] = loss_history;
    curr_steplist['time'] = time_history;

    if to_save:
      if cellNum-1 not in stepList:
        print('[steplist] cell did not exist yet');
        stepList[cellNum-1] = dict();
        stepList[cellNum-1][respStr] = dict();
      elif respStr not in stepList[cellNum-1]:
        print('%s did not exist yet' % respStr);
        stepList[cellNum-1][respStr] = dict();
      else:
        print('we will be overwriting %s (if updating)' % respStr);
      stepList[cellNum-1][respStr] = curr_steplist;
      np.save(loc_data + stepListName, stepList);

  if to_save:
    return NLL, opt_params; #, loss_history; # LAST RETURN ITEM IS TEMPORARY 22.10.04
  else:
    return curr_fit, curr_steplist;

#############


if __name__ == '__main__':

    if len(sys.argv) < 9:
      print('uhoh...you need 9 arguments here'); # and one is the script itself...
      print('See this file (setModel) or batchFitUnix.sh for guidance');
      exit();

    cellNum      = int(sys.argv[1]);
    expDir       = sys.argv[2];
    excType      = int(sys.argv[3]);
    lossType     = int(sys.argv[4]);
    fitType      = int(sys.argv[5]);
    lgnFrontOn   = int(sys.argv[6]);
    initFromCurr = int(sys.argv[7]);
    trackSteps   = int(sys.argv[8]);

    if len(sys.argv) > 9:
      kMult = float(sys.argv[9]);
    else:
      kMult = 0.10; # default (see modCompare.ipynb for details)

    if len(sys.argv) > 10:
      newMethod = float(sys.argv[10]);
    else:
      newMethod = 0; # default

    if len(sys.argv) > 11:
      vecCorrected = int(sys.argv[11]);
    else:
      vecCorrected = 0;

    if len(sys.argv) > 12:
      fixRespExp = float(sys.argv[12]);
      if fixRespExp <= 0: # this is the code to not fix the respExp
        fixRespExp = None;
    else:
      fixRespExp = None; # default (see modCompare.ipynb for details)

    if len(sys.argv) > 13: # what model of contrast do we apply for the LGN (separate for M & P or a joint one?)
      lgnConType = int(sys.argv[13]);
      if lgnConType <= 0:
        lgnConType = 1;
    else:
      lgnConType = 1;

    if len(sys.argv) > 14:
      toPar = int(sys.argv[14]); # 1 for True, 0 for False
      # Then, create the pool paramemeter globally (i.e. how many CPUs?)
      if toPar == 1:
        nCpu = mp.cpu_count()
    else:
      toPar = False;

    #setModel(cellNum, expDir, excType, lossType, fitType, lgnFrontOn, lgnConType=lgnConType, applyLGNtoNorm=_LGNforNorm, initFromCurr=initFromCurr, kMult=kMult, fixRespExp=fixRespExp, trackSteps=trackSteps, respMeasure=1, newMethod=newMethod, vecCorrected=vecCorrected, scheduler=_schedule); # try an F1 (use for debugging)
    #import cProfile
    #cProfile.runctx('oyvey=setModel(cellNum, expDir, excType, lossType, fitType, lgnFrontOn, lgnConType=lgnConType, applyLGNtoNorm=_LGNforNorm, initFromCurr=initFromCurr, kMult=kMult, fixRespExp=fixRespExp, trackSteps=trackSteps, respMeasure=1, newMethod=newMethod, vecCorrected=vecCorrected, scheduler=_schedule)', {'setModel':setModel}, locals())
    #oyvey=setModel(cellNum, expDir, excType, lossType, fitType, lgnFrontOn, lgnConType=lgnConType, applyLGNtoNorm=_LGNforNorm, initFromCurr=initFromCurr, kMult=kMult, fixRespExp=fixRespExp, trackSteps=trackSteps, respMeasure=1, newMethod=newMethod, vecCorrected=vecCorrected, scheduler=_schedule)
    #pdb.set_trace();
  
    start = time.process_time();
    dcOk = 0; f1Ok = 0 if (expDir == 'V1/' or expDir == 'V1_BB/') else 1; # i.e. we don't bother fitting F1 if fit is from V1_orig/ or altExp/
    nTry = 10; # 30
    if cellNum >= 0:
      while not dcOk and nTry>0:
        try:
          setModel(cellNum, expDir, excType, lossType, fitType, lgnFrontOn, lgnConType=lgnConType, applyLGNtoNorm=_LGNforNorm, initFromCurr=initFromCurr, kMult=kMult, fixRespExp=fixRespExp, trackSteps=trackSteps, respMeasure=0, newMethod=newMethod, vecCorrected=vecCorrected, scheduler=_schedule); # first do DC
          dcOk = 1;
          print('passed with nTry = %d' % nTry);
        except Exception as e:
          print(e)
          pass;
        nTry -= 1;
      # now, do F1 fits

      nTry=10; #30; # reset nTry...
      while not f1Ok and nTry>0:
        try:
          setModel(cellNum, expDir, excType, lossType, fitType, lgnFrontOn, lgnConType=lgnConType, applyLGNtoNorm=_LGNforNorm, initFromCurr=initFromCurr, kMult=kMult, fixRespExp=fixRespExp, trackSteps=trackSteps, respMeasure=1, newMethod=newMethod, vecCorrected=vecCorrected, scheduler=_schedule); # then F1
          f1Ok = 1;
          print('passed with nTry = %d' % nTry);
        except Exception as e:
          print(e)
          pass;
        nTry -= 1;

    elif cellNum == -1:
      loc_base = os.getcwd() + '/'; # ensure there is a "/" after the final directory
      loc_data = loc_base + expDir + 'structures/';
      dataList = hf.np_smart_load(str(loc_data + dataListName));
      dataNames = dataList['unitName'];
      cellNums = np.arange(1, 1+len(dataNames));

      from functools import partial
      import multiprocessing as mp
      nCpu = 20; # mp.cpu_count()-1; # heuristics say you should reqeuest at least one fewer processes than their are CPU
      print('***cpu count: %02d***' % nCpu);

      # do f1 here?
      sm_perCell = partial(setModel, expDir=expDir, excType=excType, lossType=lossType, fitType=fitType, lgnFrontEnd=lgnFrontOn, lgnConType=lgnConType, applyLGNtoNorm=_LGNforNorm, initFromCurr=initFromCurr, kMult=kMult, fixRespExp=fixRespExp, trackSteps=trackSteps, respMeasure=1, newMethod=newMethod, vecCorrected=vecCorrected, scheduler=_schedule, to_save=False);
      with mp.Pool(processes = nCpu) as pool:
        smFits_f1 = pool.map(sm_perCell, cellNums); # use starmap if you to pass in multiple args
        pool.close();

      # First, DC? (should only do DC or F1?)
      sm_perCell = partial(setModel, expDir=expDir, excType=excType, lossType=lossType, fitType=fitType, lgnFrontEnd=lgnFrontOn, lgnConType=lgnConType, applyLGNtoNorm=_LGNforNorm, initFromCurr=initFromCurr, kMult=kMult, fixRespExp=fixRespExp, trackSteps=trackSteps, respMeasure=0, newMethod=newMethod, vecCorrected=vecCorrected, scheduler=_schedule, to_save=False);
      #sm_perCell(1); # use this to debug...
      with mp.Pool(processes = nCpu) as pool:
        smFits_dc = pool.map(sm_perCell, cellNums); # use starmap if you to pass in multiple args
        pool.close();

      ### do the saving HERE!
      todoCV = 0; #  1 if whichTrials is not None else 0;
      loc_str = 'HPC' if 'pl1465' in loc_data else '';
      fL_name = 'fitList%s_pyt_221007f%s' % (loc_str, '_noRE' if fixRespExp is not None else ''); # figure out how to pass the name into setModel, too, so names are same regardless of call?
      fitListName = hf.fitList_name(base=fL_name, fitType=fitType, lossType=lossType, lgnType=lgnFrontOn, lgnConType=lgnConType, vecCorrected=vecCorrected, CV=todoCV)
      if os.path.isfile(loc_data + fitListName):
        print('reloading fit list...');
        fitListNPY = hf.np_smart_load(loc_data + fitListName);
      else:
        fitListNPY = dict();
      # and load fit details
      fitDetailsName = fitListName.replace('.npy', '_details.npy');
      if os.path.isfile(loc_data + fitListName):
        print('reloading fit list...');
        fitDetailsNPY = hf.np_smart_load(loc_data + fitListName);
      else:
        fitDetailsNPY = dict();

      # now, iterate through and fit!
      respStr_dc = hf_sfBB.get_resp_str(0);
      respStr_f1 = hf_sfBB.get_resp_str(1);
      for (iii, currFit_dc), currFit_f1 in zip(enumerate(smFits_dc), smFits_f1):
        fitListNPY[iii] = dict();
        # dc
        fitListNPY[iii][respStr_dc] = currFit_dc[0];
        # f1
        fitListNPY[iii][respStr_f1] = currFit_f1[0];
        try: # try the details
          fitDetailsNPY[iii] = dict();
          fitDetailsNPY[iii][respStr_dc] = currFit_dc[1];
          fitDetailsNPY[iii][respStr_f1] = currFit_f1[1];
        except:
          pass;
      # --- finally, save
      np.save(loc_data + fitListName, fitListNPY)
      np.save(loc_data + fitDetailsName, fitDetailsNPY)

    enddd = time.process_time();
    print('Took %d minutes -- dc %d || f1 %d' % ((enddd-start)/60, dcOk, f1Ok));
