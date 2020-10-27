import torch
import torch.nn as nn
from torch.utils import data as torchdata

import numpy as np

import helper_fcns as hf
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import seaborn as sns

torch.autograd.set_detect_anomaly(True)

#
globalMin = 1e-4 # what do we "cut off" the model response at? should be >0 but small

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
  b = params[0], k = params[1], c0 = params[2];

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
    pred_spikes, _ = DoGsach(*params, stim_sf=stim_sf);
  elif DoGmodel == 2:
    pred_spikes, _ = DiffOfGauss(*params, stim_sf=stim_sf);
  return pred_spikes;

### Datawrapper/loader

def process_data(coreExp, expInd, respMeasure=0, respOverwrite=None):
  trInf = dict();

  ### first, process the raw data such that trInf is [nTr x nComp]
  if expInd == -1: # i.e. sfBB
    trialInf = coreExp['trial'];
    whereNotBlank = np.where(np.logical_or(trialInf['maskOn'], trialInf['baseOn']))[0]
    if respMeasure == 0:
      resp = coreExp['spikeCounts'][whereNotBlank];
    elif respMeasure == 1: # then we're getting F1 -- first at baseTF, then maskTF
      resp = np.vstack((coreExp['f1_base'], coreExp['f1_mask'])).transpose().shape;
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

  trInf['num'] = whereNotBlank;
  trInf['ori'] = np.transpose(np.vstack(trialInf['ori']), (1,0))[whereNotBlank, :]
  trInf['tf'] = np.transpose(np.vstack(trialInf['tf']), (1,0))[whereNotBlank, :]
  trInf['ph'] = np.transpose(np.vstack(trialInf['ph']), (1,0))[whereNotBlank, :]
  trInf['sf'] = np.transpose(np.vstack(trialInf['sf']), (1,0))[whereNotBlank, :]
  trInf['con'] = np.transpose(np.vstack(trialInf['con']), (1,0))[whereNotBlank, :]

  return trInf, resp;

class dataWrapper(torchdata.Dataset):
    def __init__(self, trInf, resp, device='cpu'):
        # if respMeasure == 0, then we're getting DC; otherwise, F1
        # respOverwrite means overwrite the responses; used only for expInd>=0 for now

        super().__init__();

        self.trInf = trInf;
        self.resp = resp;
        self.device = device;
        
    def get_single_item(self, idx):
        # NOTE: THis assumes that trInf['ori', 'tf', etc...] are already [nTr, nStimComp]
        feature = dict();
        feature['ori'] = _cast_as_tensor(self.trInf['ori'][idx, :])
        feature['tf'] = _cast_as_tensor(self.trInf['tf'][idx, :])
        feature['sf'] = _cast_as_tensor(self.trInf['sf'][idx, :])
        feature['con'] = _cast_as_tensor(self.trInf['con'][idx, :])
        feature['ph'] = _cast_as_tensor(self.trInf['ph'][idx, :])
        feature['num'] = trInf['ori'].shape[0] # num is the # of trials included here...
        
        target = _cast_as_tensor(self.resp[idx]);
        
        return (feature, target.to(self.device));

    def __getitem__(self, idx):
        return self.get_single_item(idx)

    def __len__(self):
        return len(self.resp)

### The model
class sfNormMod(torch.nn.Module):
    # inherit methods/fields from torch.nn.Module()

  def __init__(self, modParams, expInd, excType=2, normType=1, lossType=1, lgnFrontEnd=0, device='cpu'):

    super().__init__();

    ### meta/fit parameters
    self.expInd = expInd;
    self.excType = excType
    self.normType = normType;
    self.lossType = lossType;
    self.lgnFrontEnd = lgnFrontEnd;
    self.device = device;

    ### all modparams
    self.modParams = modParams;

    ### now, establish the parameters to optimize
    # Get parameter values
    nParams = hf.nParamsByType(normType, excType, lgnFrontEnd);

    # handle the possibility of a multi fitting first
    if lgnFrontEnd == 99:
      self.mWeight = _cast_as_param(modParams[nParams-1]);
    else:
      self.mWeight = _cast_as_param(modParams[-1]);

    self.prefSf = _cast_as_param(modParams[0]);
    if excType == 1:
      self.dordSp = _cast_as_param(modParams[1]);
    if excType == 2:
      self.sigLow = _cast_as_param(modParams[1]);
      highInd = -1-np.sign(lgnFrontEnd);
      self.sigHigh = _cast_as_param(modParams[highInd]);

    # Other (nonlinear) model components
    self.sigma    = torch.pow(_cast_as_tensor(10), _cast_as_param(modParams[2])); # normalization constant
    self.respExp  = _cast_as_param(modParams[3]); # response exponent
    self.scale    = _cast_as_param(modParams[4]); # response scalar

    # Noise parameters
    self.noiseEarly = _cast_as_param(modParams[5]);   # early additive noise
    self.noiseLate  = _cast_as_param(modParams[6]);  # late additive noise
    self.varGain    = _cast_as_param(modParams[7]);  # multiplicative noise

    ### Normalization parameters
    normParams = hf.getNormParams(modParams, normType);
    if normType == 1:
      self.inhAsym = _cast_as_tensor(normParams); # then it's not really meant to be optimized, should be just zero
      self.gs_mean = None; self.gs_std = None; # replacing the "else" in commented out 'if normType == 2 or normType == 4' below
    elif normType == 2:
      self.gs_mean = _cast_as_param(normParams[0]);
      self.gs_std  = _cast_as_param(normParams[1]);
    elif normType == 3:
      # sigma calculation
      self.offset_sigma = _cast_as_param(normParams[0]);  # c50 filter will range between [v_sigOffset, 1]
      self.stdLeft      = _cast_as_param(normParams[1]);  # std of the gaussian to the left of the peak
      self.stdRight     = _cast_as_param(normParams[2]); # '' to the right '' 
      self.sfPeak       = _cast_as_param(normParams[3]); # where is the gaussian peak?
    elif normType == 4: # two-halved Gaussian...
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
    self.M_fc = _cast_as_tensor(9);
    self.M_ks = _cast_as_tensor(0.5);
    self.M_js = _cast_as_tensor(0.4);
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

    ### END OF INIT

  #######
  def simpleResp_matMul(self, trialInf, stimParams = []):
    # returns object with simpleResp and other things
    # --- Created 20.10.12 --- provides ~4x speed up compared to SFMSimpleResp() without need to explicit parallelization

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
    if self.lgnFrontEnd > 0: # params, stim_sf, DoGmodel, minThresh=0.1):
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
      lgnSel = torch.add(torch.mul(self.mWeight, torch.mul(selSf_m, selCon_m)), torch.mul(pWeight, torch.mul(selSf_p, selCon_p)));

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

    # four filters placed in quadrature
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

    # Store response in desired format - which is actually [nFr x nTr], so transpose it!
    #interim = torch.transpose(respSimple, 0, 1);
    #interim[torch.where(torch.isnan(interim))] = 0;
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
      max_m = np.max(get_descrResp(self.dog_m, sfTest, self.LGNmodel, minThresh=globalMin));
      max_p = np.max(get_descrResp(self.dog_p, sfTest, self.LGNmodel, minThresh=globalMin));
      # -- then here's our selectivity per component for the current stimulus
      selSf_m = np.divide(resps_m, max_m);
      selSf_p = np.divide(resps_p, max_p);
      # - then RVC response: # ASSUMES rvcMod 0 (Movshon)
      selCon_m = get_rvc_model(self.rvc_m, stimCo);
      selCon_p = get_rvc_model(self.rvc_p, stimCo);
      # -- then here's our final responses per component for the current stimulus
      lgnSel = torch.add(torch.mul(self.mWeight, torch.mul(selSf_m, selCon_m)), torch.mul(pWeight, torch.mul(selSf_p, selCon_p)));
    else:
      lgnStage = torch.ones_like(sfs);

    if self.gs_mean is None or self.gs_std is None: # we assume inhAsym is 0
      self.inhAsym = _cast_as_tensor(0);
      new_weights = 1 + self.inhAsym*(torch.log(sfs) - torch.mean(torch.log(sfs)));
      new_weights = torch.mul(lgnStage, new_weights);
    elif normType == 2:
      # Relying on https://pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal.log_prob
      log_sfs = torch.log(sfs);
      weight_distr = torch.distributions.normal.Normal(self.gs_mean, self.gs_std)
      new_weights = torch.exp(weight_distr.log_prop(log_sfs)); 
      new_weights = torch.mult(lgnStage, new_weights);

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

  def respPerCell(self, trialInf):
    # excitatory filter, first
    Lexc = self.simpleResp_matMul(trialInf); # [nFrames x nTrials]
    #perTrial = torch.clamp(resps.mean(0), min=globalMin); # nF

    Linh = self.SimpleNormResp(trialInf); # [nFrames x nTrials]
    # the below line assumes normType != 3 (which was only briefly tried, anyway...)
    sigmaFilt = torch.pow(self.sigma, 2); # i.e. square the normalization constant

    #import pdb
    #pdb.set_trace();

    numerator     = torch.add(self.noiseEarly, Lexc); # [nFrames x nTrials]
    denominator   = torch.pow(sigmaFilt + pow(Linh, 2), 0.5); # nTrials
    rawResp       = torch.div(numerator, denominator.unsqueeze(0)); # unsqueeze(0) to account for the missing leading dimension (nFrames)
    ratio         = torch.pow(torch.max(_cast_as_tensor(globalMin), rawResp), self.respExp);
    meanRate      = ratio.mean(0);
    respModel     = torch.add(self.noiseLate, torch.mul(self.scale, meanRate));

    return respModel;

  def forward(self, trialInf):
    respModel = self.respPerCell(trialInf);

    return respModel;

### End of class (sfNormMod)
    
def loss_sfNormMod(respModel, respData, lossType=1):

  if lossType == 1: # sqrt
      #mask = (~torch.isnan(respModel)) & (~torch.isnan(respData));
      #lsq = torch.pow(torch.sign(respModel[mask])*torch.sqrt(torch.abs(respModel[mask])) - torch.sign(respData[mask])*torch.sqrt(torch.abs(respData[mask])), 2);
      
      lsq = torch.pow(torch.sign(respModel)*torch.sqrt(torch.abs(respModel)) - torch.sign(respData)*torch.sqrt(torch.abs(respData)), 2);

      NLL = torch.mean(lsq);

  if lossType == 2: # poiss
      poiss_llh = numpy.log(poisson.pmf(spikeCount[mask], respModel[mask]));
      nll_notSum = poiss_llh;
      NLL = torch.mean(-poiss_llh);

  return NLL;

### Now, actually do the optimization!

### Now, the optimization
# - what to specify...
def setParams():
  ''' Set the parameters of the model '''

def setModel(max_epochs=1e3, learning_rate=1e-2, batch_size=200):
 
  ### set parameters
  # --- first, estimate prefSf, normConst if possible (TODO); inhAsym, normMean/Std
  prefSfEst = 1;
  normConst = -2;
  if fitType == 1:
    inhAsym = 0;
  if fitType == 2:
    normMean = np.log10(prefSfEst) if initFromCurr==0 else curr_params[8]; # start as matched to excFilter
    normStd = 1.5 if initFromCurr==0 else curr_params[9]; # start at high value (i.e. broad)

  # --- then, set up each parameter
  pref_sf = float(prefSfEst) if initFromCurr==0 else curr_params[0];
  if excType == 1:
    dOrdSp = np.random.uniform(1, 3) if initFromCurr==0 else curr_params[1];
  elif excType == 2:
    sigLow = np.random.uniform(1, 4) if initFromCurr==0 else curr_params[1];
    sigHigh = np.random.uniform(0.1, 2) if initFromCurr==0 else curr_params[-1-numpy.sign(lgnFrontEnd)]; # if lgnFrontEnd == 0, then it's the last param; otherwise it's the 2nd to last param
  normConst = normConst if initFromCurr==0 else curr_params[2];
  respExp = np.random.uniform(1.5, 2.5) if initFromCurr==0 else curr_params[3];
  # easier to start with a small scalar and work up, rather than work down
  respScalar = np.random.uniform(0.05, 0.25) if initFromCurr==0 else curr_params[4];
  noiseEarly = np.random.uniform(0.001, 0.01) if initFromCurr==0 else curr_params[5]; # 02.27.19 - (dec. up. bound to 0.01 from 0.1)
  noiseLate = np.random.uniform(0.1, 1) if initFromCurr==0 else curr_params[6];
  varGain = np.random.uniform(0.1, 1) if initFromCurr==0 else curr_params[7];
  if lgnFrontEnd > 0:
    # Now, the LGN weighting 
    mWeight = np.random.uniform(0.25, 0.75) if initFromCurr==0 else curr_params[-1];
  else:
    mWeight = np.nan

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

  ### define model, grab training parameters
  model = mrpt.sfNormMod(param_list, expInd, excType=excType, normType=normType, lossType=lossType, lgnFrontEnd=lgnFrontEnd)

  training_parameters = [p for p in model.parameters() if p.requires_grad]

  ###  data wrapping
  dw = dataWrapper(expInfo);
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

  first_pred = model(trInf);
  # print(first_pred)

  for t in range(max_epochs):
      optimizer.zero_grad()

      loss_history.append([])
      time_history.append([])

      for bb, (feature, target) in enumerate(dataloader):
          predictions = model(feature)
          loss_curr = mrpt.loss_sfNormMod(predictions, target, model.lossType)

          if np.mod(t,100)==0 and bb==0:
              print('\n****** STEP %d *********' % t)
              prms = model.named_parameters()
              [print(x, '\n') for x in prms];
              print(loss_curr.item())
              print(loss_curr.grad)

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
