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

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import seaborn as sns

torch.autograd.set_detect_anomaly(True)

# Some global things...
fall2020_adj = 1;
if fall2020_adj:
  globalMin = 1e-10 # what do we "cut off" the model response at? should be >0 but small
  #globalMin = 1e-1 # what do we "cut off" the model response at? should be >0 but small
else:
  globalMin = 1e-6 # what do we "cut off" the model response at? should be >0 but small
modRecov = 0;

try:
  cellNum = int(sys.argv[1]);
except:
  cellNum = np.nan;
try:
  dataListName = hf.get_datalist(sys.argv[2]); # argv[2] is expDir
except:
  dataListName = None;

### Helper -- from Billy
def _cast_as_tensor(x, device='cpu', dtype=torch.float32):
    # needs to be float32 to work with the Hessian calculations
    return torch.tensor(x, dtype=dtype, device=device)

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
def flexible_Gauss(params, stim_sf, minThresh=0.1):
    respFloor       = params[0];
    respRelFloor    = params[1];
    sfPref          = params[2];
    sigmaLow        = params[3];
    sigmaHigh       = params[4];

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
    nyquist = [np.int(x.shape[1]/2) for x in fftSpectrum];
    correctFFT = [];
    for i, spect in enumerate(fftSpectrum):
      # -- removed torch.abs(spect[...]), since we already square the coefficients in spike_fft
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


def spike_fft(psth, tfs = None, stimDur = None, binWidth=1e-3):
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
    full_fourier = [torch.sqrt(epsil + torch.add(torch.pow(x[:,:,0], 2), torch.pow(x[:,:,1], 2))) for x in full_fourier]; # just get the amplitude
    #spectrum = full_fourier; # bypassing this func for now...
    spectrum = fft_amplitude(full_fourier, stimDur);

    if tfs is not None:
      try:
        tf_as_ind = _cast_as_tensor(hf.tf_to_ind(tfs, stimDur), dtype=torch.long); # if 1s, then TF corresponds to index; if stimDur is 2 seconds, then we can resolve half-integer frequencies -- i.e. 0.5 Hz = 1st index, 1 Hz = 2nd index, ...; CAST to integer
        rel_amp = [spectrum[i][:, tf_as_ind[i]] for i in range(len(tf_as_ind))];
      except:
        warnings.warn('In spike_fft: if accessing power at particular frequencies, you must also include the stimulation duration!');
        rel_amp = [];
    else:
      rel_amp = [];

    return spectrum, rel_amp, full_fourier;

### Datawrapper/loader

def process_data(coreExp, expInd, respMeasure=0, respOverwrite=None):
  ''' Process the trial-by-trial stimulus information for ease of use with the model
      Specifically, we stack the stimuli to be [nTr x nStimComp], where 
      - [:,0] is base, [:,1] is mask, respectively for sfBB
  '''
  trInf = dict();

  ### first, process the raw data such that trInf is [nTr x nComp]
  if expInd == -1: # i.e. sfBB
    trialInf = coreExp['trial'];
    whereNotBlank = np.where(np.logical_or(trialInf['maskOn'], trialInf['baseOn']))[0]
    if respMeasure == 0:
      resp = np.expand_dims(coreExp['spikeCounts'][whereNotBlank], axis=1); # make (nTr, 1)
    elif respMeasure == 1: # then we're getting F1 -- first at baseTF, then maskTF
      # NOTE: CORRECTED TO MASK, then BASE on 20.11.15
      # -- the tranpose turns it from [2, nTr] to [nTr, 2], but keeps [:,0] as mask; [:,1] as base
      resp = np.vstack((coreExp['f1_mask'][whereNotBlank], coreExp['f1_base'][whereNotBlank])).transpose();
  elif expInd >= 0: # i.e. sfMix*
    trialInf = coreExp['sfm']['exp']['trial'];
    whereNotBlank = np.where(~np.isnan(np.sum(trialInf['ori'], 0)))[0];
    # TODO -- put in the proper responses...
    if respOverwrite is not None:
      spikes = respOverwrite;
    else:
      if respMeasure == 0:
        spikes = trialInf['spikeCount'];
      elif respMeasure == 1:
        spikes = trialInf['f1'];
    resp = spikes[whereNan];

  # mask, then base
  trInf['num'] = whereNotBlank;
  trInf['ori'] = np.transpose(np.vstack(trialInf['ori']), (1,0))[whereNotBlank, :]
  trInf['tf'] = np.transpose(np.vstack(trialInf['tf']), (1,0))[whereNotBlank, :]
  trInf['ph'] = np.transpose(np.vstack(trialInf['ph']), (1,0))[whereNotBlank, :]
  trInf['sf'] = np.transpose(np.vstack(trialInf['sf']), (1,0))[whereNotBlank, :]
  trInf['con'] = np.transpose(np.vstack(trialInf['con']), (1,0))[whereNotBlank, :]

  return trInf, resp;

class dataWrapper(torchdata.Dataset):
    def __init__(self, expInfo, expInd=-1, respMeasure=0, device='cpu'):
        # if respMeasure == 0, then we're getting DC; otherwise, F1
        # respOverwrite means overwrite the responses; used only for expInd>=0 for now

        super().__init__();
        trInf, resp = process_data(expInfo, expInd, respMeasure)

        self.trInf = trInf;
        self.resp = resp;
        self.device = device;
        
    def get_single_item(self, idx):
        # NOTE: This assumes that trInf['ori', 'tf', etc...] are already [nTr, nStimComp]
        feature = dict();
        feature['ori'] = _cast_as_tensor(self.trInf['ori'][idx, :])
        feature['tf'] = _cast_as_tensor(self.trInf['tf'][idx, :])
        feature['sf'] = _cast_as_tensor(self.trInf['sf'][idx, :])
        feature['con'] = _cast_as_tensor(self.trInf['con'][idx, :])
        feature['ph'] = _cast_as_tensor(self.trInf['ph'][idx, :])
        feature['num'] = self.trInf['ori'].shape[0] # num is the # of trials included here...
        
        target = dict();
        target['resp'] = _cast_as_tensor(self.resp[idx, :]);
        maskInd, baseInd = hf_sfBB.get_mask_base_inds();
        target['maskCon'] = _cast_as_tensor(self.trInf['con'][idx, maskInd])
        target['baseCon'] = _cast_as_tensor(self.trInf['con'][idx, baseInd])
        
        return (feature, target);

    def __getitem__(self, idx):
        return self.get_single_item(idx)

    def __len__(self):
        return len(self.resp)

### The model
class sfNormMod(torch.nn.Module):
    # inherit methods/fields from torch.nn.Module()

  def __init__(self, modParams, expInd=-1, excType=2, normType=1, lossType=1, lgnFrontEnd=0, newMethod=0, device='cpu'):

    super().__init__();

    ### meta/fit parameters
    self.expInd = expInd;
    self.excType = excType
    self.normType = normType;
    self.lossType = lossType;
    self.lgnFrontEnd = lgnFrontEnd;
    self.device = device;
    self.newMethod = newMethod;

    ### all modparams
    self.modParams = modParams;

    ### now, establish the parameters to optimize
    # Get parameter values
    nParams = hf.nParamsByType(self.normType, excType, lgnFrontEnd);

    # handle the possibility of a multi fitting first
    if self.lgnFrontEnd == 99:
      self.mWeight = _cast_as_param(modParams[nParams-1]);
    elif self.lgnFrontEnd > 0:
      self.mWeight = _cast_as_param(modParams[-1]);
    else:
      self.mWeight = _cast_as_tensor(modParams[-1]); # then it's NOT a parameter
    self.pWeight = 1-self.mWeight;

    self.prefSf = _cast_as_param(modParams[0]);
    if self.excType == 1:
      self.dordSp = _cast_as_param(modParams[1]);
    if self.excType == 2:
      self.sigLow = _cast_as_param(modParams[1]);
      highInd = -1-np.sign(lgnFrontEnd);
      self.sigHigh = _cast_as_param(modParams[highInd]);

    # Other (nonlinear) model components
    self.sigma    = _cast_as_param(modParams[2]); # normalization constant
    self.respExp  = _cast_as_param(modParams[3]); # response exponent
    self.scale    = _cast_as_param(modParams[4]); # response scalar

    # Noise parameters
    self.noiseEarly = _cast_as_param(modParams[5]);   # early additive noise
    self.noiseLate  = _cast_as_param(modParams[6]);  # late additive noise
    if self.lossType == 3:
      self.varGain    = _cast_as_param(modParams[7]);  # multiplicative noisew
    else:
      self.varGain    = _cast_as_tensor(modParams[7]);  # multiplicative noisew

    ### Normalization parameters
    normParams = hf.getNormParams(modParams, normType);
    if self.normType == 1:
      self.inhAsym = _cast_as_tensor(normParams); # then it's not really meant to be optimized, should be just zero
      self.gs_mean = None; self.gs_std = None; # replacing the "else" in commented out 'if normType == 2 or normType == 4' below
    elif self.normType == 2:
      self.gs_mean = _cast_as_param(normParams[0]);
      self.gs_std  = _cast_as_param(normParams[1]);
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
    self.LGNmodel = 2;
    # prepolate DoG parameters -- will overwrite, if needed
    self.M_k = _cast_as_tensor(1);
    self.M_fc = _cast_as_tensor(3);
    self.M_ks = _cast_as_tensor(0.3);
    self.M_js = _cast_as_tensor(0.4);
    self.P_k = _cast_as_tensor(1);
    self.P_fc = _cast_as_tensor(9);
    self.P_ks = _cast_as_tensor(0.5);
    self.P_js = _cast_as_tensor(0.4);
    if self.lgnFrontEnd == 2:
      self.M_fc = _cast_as_tensor(6);
    elif self.lgnFrontEnd == 99: # 99 is code for fitting an LGN front end which is common across all cells in the dataset...
      # parameters are passed as [..., m_fc, p_fc, m_ks, p_ks, m_js, p_js]
      self.M_fc = _cast_as_param(self.modParams[-6]);
      self.M_ks = _cast_as_param(self.modParams[-4]);
      self.M_js = _cast_as_param(self.modParams[-2]);
      self.P_fc = torch.mul(self.M_fc, _cast_as_param(self.modParams[-5]));
      self.P_ks = _cast_as_param(self.modParams[-3]);
      self.P_js = _cast_as_param(self.modParams[-1]);
    # specify rvc parameters (not true "parameters", i.e. not optimized)
    if self.lgnFrontEnd > 0:
      self.rvcMod = 0;
      self.rvc_m = _cast_as_tensor([0, 12.5, 0.05]);
      self.rvc_p = _cast_as_tensor([0, 17.5, 0.50]);
      self.dog_m = [self.M_k, self.M_fc, self.M_ks, self.M_js]
      self.dog_p = [self.P_k, self.P_fc, self.P_ks, self.P_js]
    ### END OF INIT

  #######
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

    if self.lgnFrontEnd == 0: # then we'll trim off the last constraint, which is mWeight bounds (and the last param, which is mWeight)
      param_list = param_list[0:-1];

    return param_list

  def simpleResp_matMul(self, trialInf, stimParams = []):
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
      selCon_m = get_rvc_model(self.rvc_m, stimCo);
      selCon_p = get_rvc_model(self.rvc_p, stimCo);
      # -- then here's our final responses per component for the current stimulus
      lgnSel = torch.add(torch.mul(self.mWeight, torch.mul(selSf_m, selCon_m)), torch.mul(self.pWeight, torch.mul(selSf_p, selCon_p)));

    if self.excType == 1:
      # Compute spatial frequency tuning - Deriv. order Gaussian
      sfRel = torch.div(stimSf, self.prefSf);
      s     = torch.pow(stimSf, self.dordSp) * torch.exp(-self.dordSp/2 * torch.pow(sfRel, 2));
      sMax  = torch.pow(self.prefSf, self.dordSp) * torch.exp(-self.dordSp/2);
      selSf   = torch.div(s, sMax);
    elif self.excType == 2:
      selSf = flexible_Gauss([0,1,self.prefSf,self.sigLow,self.sigHigh], stimSf, minThresh=0);
 
    # II. Phase, space and time
    omegaX = torch.mul(stimSf, torch.cos(stimOr)); # the stimulus in frequency space
    omegaY = torch.mul(stimSf, torch.sin(stimOr));
    omegaT = stimTf;

    P = torch.empty((nTrials, nFrames, nStimComp, 3)); # nTrials x nFrames for number of frames x nStimComp x [two for x and y coordinate, one for time]
    P[:,:,:,0] = torch.full((nTrials, nFrames, nStimComp), 2*np.pi*xCo);  # P is the matrix that contains the relative location of each filter in space-time (expressed in radians)
    P[:,:,:,1] = torch.full((nTrials, nFrames, nStimComp), 2*np.pi*yCo);  # P(:,0) and p(:,1) describe location of the filters in space

    # NEW: 20.07.16 -- why divide by 2 for the LGN stage? well, if selectivity is at peak for M and P, sum will be 2 (both are already normalized) // could also just sum...
    if self.lgnFrontEnd > 0:
      selSi = torch.mul(selSf, lgnSel); # filter sensitivity for the sinusoid in the frequency domain
    else:
      selSi = selSf;
    #selSi[torch.where(torch.isnan(selSi))] = 0;

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
    rComplex = torch.einsum('ij,ikjz->ikz', torch.mul(selSi,stimCo), realImag) # mult. to get [nTr x nFr x 2] response
    #rComplex[torch.where(torch.isnan(rComplex))] = 1e-4; # avoid NaN
    # The above line takes care of summing over stimulus components

    # four filters placed in quadrature (only if self.newMethod == 0, which is default)

    # Store response in desired format - which is actually [nFr x nTr], so transpose it!
    if self.newMethod == 1:
      respSimple1 = rComplex[...,0]; # we'll keep the half-wave rectification for the end...
      return torch.transpose(respSimple1, 0, 1);
    else:
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
      respSimple = torch.sqrt(respAvg); # div by 4 to avg across all filters

      return torch.transpose(respSimple, 0, 1);

  def genNormWeightsSimple(self, trialInf):
    ''' simply evaluates the usual normalization weighting but at the frequencies of the stimuli directly
    i.e. in effect, we are eliminating the bank of filters in the norm. pool
    '''

    sfs = _cast_as_tensor(trialInf['sf']); # [nComps x nTrials]
    cons = _cast_as_tensor(trialInf['con']); # [nComps x nTrials]
    consSq = np.square(cons);

    # apply LGN stage, if specified - we apply equal M and P weight, since this is across a population of neurons, not just the one one uron under consideration
    if self.lgnFrontEnd > 0:
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
      lgnStage = torch.add(torch.mul(self.mWeight, torch.mul(selSf_m, selCon_m)), torch.mul(self.pWeight, torch.mul(selSf_p, selCon_p)));
    else:
      lgnStage = torch.ones_like(sfs);

    if self.gs_mean is None or self.gs_std is None: # we assume inhAsym is 0
      self.inhAsym = _cast_as_tensor(0);
      new_weights = 1 + self.inhAsym*(torch.log(sfs) - torch.mean(torch.log(sfs)));
      new_weights = torch.mul(lgnStage, new_weights);
    elif self.normType == 2:
      # Relying on https://pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal.log_prob
      log_sfs = torch.log(sfs);
      weight_distr = torch.distributions.normal.Normal(self.gs_mean, self.gs_std)
      new_weights = torch.exp(weight_distr.log_prob(log_sfs)); 
      new_weights = torch.mul(lgnStage, new_weights);

    return new_weights;

  def SimpleNormResp(self, trialInf, trialArtificial=None):

    if trialArtificial is not None:
      trialInf = trialArtificial;
    else:
      trialInf = trialInf;
    consSq = torch.pow(_cast_as_tensor(trialInf['con']), 2);
    # cons (and wghts) will be (nComps x nTrials)
    wghts = self.genNormWeightsSimple(trialInf);

    # now put it all together
    resp = torch.mul(wghts, consSq);
    respPerTr = torch.sqrt(resp.sum(1)); # i.e. sum over components, then sqrt

    return respPerTr; # will be [nTrials] -- later, will ensure right output size during operation

  def respPerCell(self, trialInf, debug=0):
    # excitatory filter, first
    simpleResp = self.simpleResp_matMul(trialInf);
    normResp = self.SimpleNormResp(trialInf); # [nFrames x nTrials]
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
    ratio         = torch.pow(torch.max(_cast_as_tensor(globalMin), rawResp), self.respExp);
    # just rectify, don't even add the power
    #ratio = torch.max(_cast_as_tensor(globalMin), rawResp);
    # we're now skipping the averaging across frames...
    if self.newMethod == 1:
      if fall2020_adj:
        #respModel     = torch.max(_cast_as_tensor(globalMin), torch.add(self.noiseLate, torch.pow(torch.mul(self.scale, ratio), self.respExp)));
        respModel     = torch.max(_cast_as_tensor(globalMin), torch.add(self.noiseLate, torch.mul(self.scale, ratio)));
      else:
        respModel     = torch.add(self.noiseLate, torch.mul(self.scale, ratio));
      return torch.transpose(respModel, 1, 0);
    else:
      meanRate      = ratio.mean(0);
      respModel     = torch.add(self.noiseLate, torch.mul(self.scale, meanRate));
      return respModel; # I don't think we need to transpose here...

  def forward(self, trialInf, respMeasure=0, returnPsth=0, expInd=-1, debug=0): # expInd=-1 for sfBB
    # respModel is the psth! [nTr x nFr]
    respModel = self.respPerCell(trialInf, debug);

    if debug:
      return respModel

    # then, get the base & mask TF
    maskInd, baseInd = hf_sfBB.get_mask_base_inds();
    maskTf, baseTf = trialInf['tf'][0, maskInd], trialInf['tf'][0, baseInd] # relies on tf being same for all trials (i.e. maskTf always same, baseTf always same)!
    tfAsInts = np.array([int(maskTf), int(baseTf)]);
    stimDur = hf.get_exp_params(expInd).stimDur
    nFrames = len(respModel)/stimDur;
    # important to transpose the respModel before passing in to spike_fft
    amps, rel_amps, full_fourier = spike_fft([respModel], tfs=[tfAsInts], stimDur=stimDur, binWidth=1.0/nFrames)

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
    
def loss_sfNormMod(respModel, respData, lossType=1):

  if lossType == 1: # sqrt
      #mask = (~torch.isnan(respModel)) & (~torch.isnan(respData));
      #lsq = torch.pow(torch.sign(respModel[mask])*torch.sqrt(torch.abs(respModel[mask])) - torch.sign(respData[mask])*torch.sqrt(torch.abs(respData[mask])), 2);
      
      lsq = torch.pow(torch.sign(respModel)*torch.sqrt(torch.abs(respModel)) - torch.sign(respData)*torch.sqrt(torch.abs(respData)), 2);

      NLL = torch.mean(lsq);

  if lossType == 2: # poiss TODO FIX TODO
      poiss_loss = torch.nn.PoissonNLLLoss(log_input=False);
      NLL = poiss_loss(respModel, respData); # previously was respData, respModel

  return NLL;

### Now, actually do the optimization!

### Now, the optimization
# - what to specify...
#def setParams():
#  ''' Set the parameters of the model '''

def setModel(cellNum, expDir=-1, excType=1, lossType=1, fitType=1, lgnFrontEnd=0, max_epochs=500, learning_rate=0.001, batch_size=200, initFromCurr=0, kMult=0.1, newMethod=0, fixRespExp=None, trackSteps=True, fL_name=None, respMeasure=0):

  ### Load the cell, set up the naming
  ########
  # Load cell
  ########
  loc_base = os.getcwd() + '/'; # ensure there is a "/" after the final directory
  loc_data = loc_base + expDir + 'structures/';

  if 'pl1465' in loc_base:
    loc_str = 'HPC';
  else:
    loc_str = '';
  if fL_name is None: # otherwise, it's already defined...
    if modRecov == 1:
      fL_name = 'mr_fitList%s_190516cA' % loc_str
    else:
      if excType == 1:
        fL_name = 'fitList%s_pyt_200417' % (loc_str); # pyt for pytorch
      elif excType == 2:
        fL_name = 'fitList%s_pyt_200507' % (loc_str); # pyt for pytorch

  if lossType == 4: # chiSq...
    fL_name = '%s%s' % (fL_name, hf.chiSq_suffix(kMult));

  if fixRespExp is not None:
    fL_name = '%s_re%d' % (fL_name, np.round(fixRespExp*10)); # suffix to indicate that the response exponent is fixed...

  if lgnFrontEnd == 1:
    fL_name = '%s_LGN' % fL_name # implicit "a" at the end of LGN...
  if lgnFrontEnd == 2:
    fL_name = '%s_LGNb' % fL_name
  if lgnFrontEnd == 99:
    fL_name = '%s_jLGN' % fL_name

  fitListName = hf.fitList_name(base=fL_name, fitType=fitType, lossType=lossType);
  # get the name for the stepList name, regardless of whether or not we keep this now
  stepListName = str(fitListName.replace('.npy', '_details.npy'));

  print('\nFitList: %s' % fitListName);

  # Load datalist, then specific cell
  dataList = hf.np_smart_load(str(loc_data + dataListName));
  dataNames = dataList['unitName'];
  print('loading data structure from %s...' % loc_data);
  try:
    expInd = hf.exp_name_to_ind(dataList[expType][cellNum-1]);
  except:
    expInd = -1; # for sfBB
  # - then cell
  S = hf.np_smart_load(str(loc_data + dataNames[cellNum-1] + '_sfBB.npy')); # why -1? 0 indexing...
  expInfo = S['sfBB_core']; # TODO: generalize...
  trInf, resp = process_data(expInfo, expInd=expInd, respMeasure=respMeasure); 


  respStr = hf_sfBB.get_resp_str(respMeasure);
  if os.path.isfile(loc_data + fitListName):
    fitList = hf.np_smart_load(str(loc_data + fitListName));
    try:
      curr_params = fitList[cellNum-1][respStr]['params'];
      # Run the model, evaluate the loss to ensure we have a valid parameter set saved -- otherwise, we'll generate new parameters
      testModel = sfNormMod(curr_params, expInd=expInd, excType=excType, normType=fitType, lossType=lossType, newMethod=newMethod, lgnFrontEnd=lgnFrontEnd)
      trInf, resp = process_data(expInfo, expInd, respMeasure)
      predictions = testModel.forward(trInf, respMeasure=respMeasure);
      loss_test = loss_sfNormMod(_cast_as_tensor(predictions.flatten()), _cast_as_tensor(resp.flatten()), testModel.lossType)
      if np.isnan(loss_test.item()):
        initFromCurr = 0; # then we've saved bad parameters -- force new ones!
    except:
      initFromCurr = 0; # force the old parameters
  else:
    fitList = dict();

  ### set parameters
  # --- first, estimate prefSf, normConst if possible (TODO); inhAsym, normMean/Std
  prefSfEst = 1;
  normConst = -2;
  if fitType == 1:
    inhAsym = 0;
  if fitType == 2:
    normMean = np.log10(prefSfEst) if initFromCurr==0 else curr_params[8]; # start as matched to excFilter
    normStd = 0.5 if initFromCurr==0 else curr_params[9]; # start at high value (i.e. broad)

  # --- then, set up each parameter
  pref_sf = float(prefSfEst) if initFromCurr==0 else curr_params[0];
  if excType == 1:
    dOrdSp = np.random.uniform(1, 3) if initFromCurr==0 else curr_params[1];
  elif excType == 2:
    sigLow = np.random.uniform(0.1, 0.3) if initFromCurr==0 else curr_params[1];
    # - make sigHigh relative to sigLow, but bias it to be lower, i.e. narrower
    sigHigh = sigLow*np.random.uniform(0.5, 1.25) if initFromCurr==0 else curr_params[-1-np.sign(lgnFrontEnd)]; # if lgnFrontEnd == 0, then it's the last param; otherwise it's the 2nd to last param
  normConst = normConst if initFromCurr==0 else curr_params[2];
  respExp = np.random.uniform(1.5, 2.5) if initFromCurr==0 else curr_params[3];
  if newMethod == 0:
    # easier to start with a small scalar and work up, rather than work down
    respScalar = np.random.uniform(200, 700) if initFromCurr==0 else curr_params[4];
    noiseEarly = -1 if initFromCurr==0 else curr_params[5]; # 02.27.19 - (dec. up. bound to 0.01 from 0.1)
  else:
    respScalar = np.random.uniform(0.01, 0.05) if initFromCurr==0 else curr_params[4];
    noiseEarly = 1e-3 if initFromCurr==0 else curr_params[5]; # 02.27.19 - (dec. up. bound to 0.01 from 0.1)
  noiseLate = 1e-1 if initFromCurr==0 else curr_params[6];
  varGain = np.random.uniform(0.1, 1) if initFromCurr==0 else curr_params[7];
  if lgnFrontEnd > 0:
    # Now, the LGN weighting 
    mWeight = np.random.uniform(0.25, 0.75) if initFromCurr==0 else curr_params[-1];
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
  if lgnFrontEnd == 0: # then we'll trim off the last constraint, which is mWeight bounds (and the last param, which is mWeight)
    param_list = param_list[0:-1];   

  ### define model, grab training parameters
  model = sfNormMod(param_list, expInd, excType=excType, normType=fitType, lossType=lossType, newMethod=newMethod, lgnFrontEnd=lgnFrontEnd)

  training_parameters = [p for p in model.parameters() if p.requires_grad]

  ###  data wrapping
  dw = dataWrapper(expInfo, respMeasure=respMeasure);
  dataloader = torchdata.DataLoader(dw, batch_size)

  ### then set up the optimization
  # optimizer = torch.optim.SGD(training_parameters, lr=learning_rate)
  optimizer = torch.optim.Adam(training_parameters, amsgrad=True, lr=learning_rate, )
  # - then data
  # - predefine some arrays for tracking loss
  loss_history = []
  start_time = time.time()
  time_history = []
  model_history = []
  hessian_history = []

  first_pred = model(trInf, respMeasure=respMeasure);
  for t in range(max_epochs):
      optimizer.zero_grad()

      loss_history.append([])
      time_history.append([])

      for bb, (feature, target) in enumerate(dataloader):
          predictions = model.forward(feature, respMeasure=respMeasure)
          if respMeasure == 1:
              maskInd, baseInd = hf_sfBB.get_mask_base_inds();
              target['resp'][target['maskCon']==0,0] = 1e-6 # force F1 ~= 0 if con of that stim is 0
              target['resp'][target['baseCon']==0,1] = 1e-6 # force F1 ~= 0 if con of that stim is 0
              predictions[target['maskCon']==0,0] = 1e-6 # force F1 ~= 0 if con of that stim is 0
              predictions[target['baseCon']==0,1] = 1e-6 # force F1 ~= 0 if con of that stim is 0
          target = target['resp'].flatten(); # since it's [nTr, 1], just make it [nTr]
          predictions = predictions.flatten(); # either [nTr,2] to [2*nTr] or [nTr,1] to [nTr]
          loss_curr = loss_sfNormMod(predictions, target, model.lossType)

          if np.mod(t,100)==0 and bb==0:
              print('\n****** STEP %d *********' % t)
              prms = model.named_parameters()
              #[print(x, '\n') for x in prms];
              print(loss_curr.item())
              #print(loss_curr.grad)

          loss_history[t].append(loss_curr.item())
          time_history[t].append(time.time() - start_time)
          if np.isnan(loss_curr.item()) or np.isinf(loss_curr.item()):
              # we raise an exception here and then try again.
              raise Exception("Loss is nan or inf on epoch %s, batch %s!" % (t, 0))

  #         loss_curr.backward()
          loss_curr.backward(retain_graph=True)
          optimizer.step()

      model.eval()
      model.train()

  ##############
  #### OPTIM IS DONE ####
  ##############
  # Most importantly, get the optimal parameters
  opt_params = model.return_params(); # check this...
  curr_resp = model.forward(dw.trInf, respMeasure=respMeasure);
  gt_resp = _cast_as_tensor(dw.resp);
  NLL = loss_sfNormMod(curr_resp, gt_resp, model.lossType).detach().numpy();

  ## we've finished optimization, so reload again to make sure that this  NLL is better than the currently saved one
  ## -- why do we have to do it again here? We may be running multiple fits for the same cells at the same and we want to make sure that if one of those has updated, we don't overwrite that opt. if it's better
  currNLL = 1e7;
  if os.path.exists(loc_data + fitListName):
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
  if os.path.exists(loc_data + fitListName):
    fitList = hf.np_smart_load(str(loc_data + fitListName));
  # else, nothing to reload!!!
  # but...if we reloaded fitList and we don't have this key (cell) saved yet, recreate the key entry...
  if cellNum-1 not in fitList:
    print('cell did not exist yet');
    fitList[cellNum-1] = dict();
    fitList[cellNum-1][respStr] = dict();
  elif respStr not in fitList[cellNum-1]:
    print('%s did not exist yet' % respStr);
    fitList[cellNum-1][respStr] = dict();
  else:
    print('we will be overwriting %s (if updating)' % respStr);
  # now, if the NLL is now the best, update this
  if NLL < currNLL:
    fitList[cellNum-1][respStr]['NLL'] = NLL;
    fitList[cellNum-1][respStr]['params'] = opt_params;
    # NEW: Also save *when* this most recent fit was made (19.02.04); and nll_history below
    fitList[cellNum-1][respStr]['time'] = datetime.datetime.now();
    # NEW: Also also save entire loss/optimization structure
    optInfo = dict();
    optInfo['call'] = optimizer;
    optInfo['epochs'] = max_epochs;
    optInfo['batch_size'] = batch_size;
    optInfo['learning_rate'] = learning_rate;
    fitList[cellNum-1][respStr]['opt'] = optInfo;
  else:
    print('new NLL not less than currNLL, not saving result, but updating overall fit list (i.e. tracking each fit)');
  fitList[cellNum-1][respStr]['nll_history'] = np.append(nll_history, NLL);
  np.save(loc_data + fitListName, fitList);
  # now the step list, if needed
  if trackSteps and NLL < currNLL:
    if os.path.exists(loc_data + stepListName):
      stepList = hf.np_smart_load(str(loc_data + stepListName));
    else:
      stepList = dict();
    stepList[cellNum-1] = dict();
    stepList[cellNum-1][respStr] = dict();
    stepList[cellNum-1][respStr]['loss'] = loss_history;
    stepList[cellNum-1][respStr]['time'] = time_history;
    np.save(loc_data + stepListName, stepList);

  return NLL, opt_params;

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
      rvcMod = float(sys.argv[11]);
    else:
      rvcMod = 1; # default (naka-rushton)

    if len(sys.argv) > 12:
      fixRespExp = float(sys.argv[12]);
      if fixRespExp <= 0: # this is the code to not fix the respExp
        fixRespExp = None;
    else:
      fixRespExp = None; # default (see modCompare.ipynb for details)

    if len(sys.argv) > 13:
      toPar = int(sys.argv[13]); # 1 for True, 0 for False
      # Then, create the pool paramemeter globally (i.e. how many CPUs?)
      if toPar == 1:
        nCpu = mp.cpu_count()
    else:
      toPar = False;

    start = time.process_time();
    if cellNum >= 0:
      setModel(cellNum, expDir, excType, lossType, fitType, lgnFrontOn, initFromCurr=initFromCurr, kMult=kMult, fixRespExp=fixRespExp, trackSteps=trackSteps, respMeasure=0, newMethod=newMethod); # first do DC
      setModel(cellNum, expDir, excType, lossType, fitType, lgnFrontOn, initFromCurr=initFromCurr, kMult=kMult, fixRespExp=fixRespExp, trackSteps=trackSteps, respMeasure=1, newMethod=newMethod); # then F1

    elif cellNum == -1:
      loc_base = os.getcwd() + '/'; # ensure there is a "/" after the final directory
      loc_data = loc_base + expDir + 'structures/';
      dataList = hf.np_smart_load(str(loc_data + dataListName));
      dataNames = dataList['unitName'];
      cellNums = np.arange(1, 1+len(dataNames));
      # SETMODEL_JOINT DOES NOT EXIST TODO
      #setModel_joint(cellNums, expDir, lossType, fitType, initFromCurr, trackSteps=trackSteps, modRecov=modRecov, kMult=kMult, rvcMod=rvcMod, excType=excType, lgnFrontEnd=lgnFrontOn, fixRespExp=fixRespExp, toPar=toPar);

    enddd = time.process_time();
    print('Took %d time -- NO par!!!' % (enddd-start));
