import torch
import torch.nn as nn
import numpy as np

import helper_fcns as hf
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import seaborn as sns

### Helper -- from Billy
def _cast_as_tensor(x, device='cpu'):
    # needs to be float32 to work with the Hessian calculations
    return torch.tensor(x, dtype=torch.float32, device=device)

def _cast_as_param(x, requires_grad=True, device='cpu'):
    return torch.nn.Parameter(_cast_as_tensor(x), requires_grad=requires_grad, device=device)

### Helper -- other
def flexible_Gauss(params, stim_sf, minThresh=0.1):
    respFloor       = params[0];
    respRelFloor    = params[1];
    sfPref          = params[2];
    sigmaLow        = params[3];
    sigmaHigh       = params[4];

    # Tuning function
    sf0   = torch.div(stim_sf, sfPref);
    sigma = torch.full_like(sf0, sigmaLow.detach());
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

### The model
class sfNormMod(torch.nn.Module):
    # inherit methods/fields from torch.nn.Module()

  def __init__(self, modParams, excType=2, normType=1, lossType=1, lgnFrontEnd=0, device='cpu'):

    super().__init__();

    ### meta/fit parameters
    self.excType = excType
    self.normType = normType;
    self.lossType = lossType;
    self.lgnFrontEnd = lgnFrontEnd;

    ### now, establish the parameters to optimize
    # Get parameter values
    nParams = hf.nParamsByType(normType, excType, lgnFrontEnd);

    # handle the possibility of a multi fitting first
    if lgnFrontEnd == 99:
      mWeight = _cast_as_param(modParams[nParams-1], device=device);
    else:
      mWeight = _cast_as_param(modParams[-1], device=device);

    self.prefSf = _cast_as_param(modParams[0], device=device);
    if excType == 1:
      self.dordSp = _cast_as_param(modParams[1], device=device);
      self.excChannel = {'pref': self.prefSf, 'dordSp': self.dordSp, 'mWeight': self.mWeight};
    if excType == 2:
      self.sigLow = _cast_as_param(modParams[1], device=device);
      highInd = -1-torch.sign(lgnFrontEnd);
      self.sigHigh = _cast_as_param(modParams[highInd], device=device);
      excChannel = {'pref': self.prefSf, 'sigLow': self.sigLow, 'sigHigh': self.sigHigh, 'mWeight': self.mWeight};

    # Other (nonlinear) model components
    sigma    = torch.pow(_cast_as_tensor(10), _cast_as_param(modParams[2], device=device)); # normalization constant
    respExp  = _cast_as_param(modParams[3], device=device); # response exponent
    scale    = _cast_as_param(modParams[4], device=device); # response scalar

    # Noise parameters
    noiseEarly = _cast_as_param(modParams[5], device=device);   # early additive noise
    noiseLate  = _cast_as_param(modParams[6], device=device);  # late additive noise
    varGain    = _cast_as_param(modParams[7], device=device);  # multiplicative noise

    ### Normalization parameters
    normParams = hf.getNormParams(modParams, normType);
    if normType == 1:
      inhAsym = _cast_as_param(normParams, device=device);
      gs_mean = None; gs_std = None; # replacing the "else" in commented out 'if normType == 2 or normType == 4' below
    elif normType == 2:
      gs_mean = _cast_as_param(normParams[0], device=device);
      gs_std  = _cast_as_param(normParams[1], device=device);
    elif normType == 3:
      # sigma calculation
      offset_sigma = _cast_as_param(normParams[0], device=device);  # c50 filter will range between [v_sigOffset, 1]
      stdLeft      = _cast_as_param(normParams[1], device=device);  # std of the gaussian to the left of the peak
      stdRight     = _cast_as_param(normParams[2], device=device); # '' to the right '' 
      sfPeak       = _cast_as_param(normParams[3], device=device); # where is the gaussian peak?
    elif normType == 4: # two-halved Gaussian...
      gs_mean = _cast_as_param(normParams[0], device=device);
      gs_std = _cast_as_param(normParams[1], device=device);
    else:
      inhAsym = _cast_as_param(normParams, device=device);

    ### END OF INIT

  #######
  def simpleResp_matMul(self, trNum, trialInf, stimParams = []):
  #def SFMSimpleResp_matMul(S, channel, stimParams = [], expInd = 1, trialInf = None, excType = 1, lgnFrontEnd = 0, allParams=None):
    # returns object with simpleResp and other things
    # --- Created 20.10.12 --- provides ~4x speed up compared to SFMSimpleResp() without need to explicit parallelization

    # simpleResp_matMul       Computes response of simple cell for sfmix experiment

    # simpleResp_matMul(varargin) returns a complex cell response for the
    # mixture stimuli used in sfMix. The cell's receptive field is the n-th
    # derivative of a 2-D Gaussian that need not be circularly symmetric.

    # NOTE: For first pass, not building in make_own_stim, since this will be for
    # - optimizing the model, not for evaluating at arbitrary stimuli

    np = numpy;

    # Load the data structure
    T = S['sfm'];
    if trialInf is None: # otherwise, we've passed in explicit trial information!
      trialInf = T['exp']['trial'];

    # Get preferred stimulus values
    prefSf = channel['pref']['sf'];                              # in cycles per degree
    # CHECK LINE BELOW
    prefTf = round(torch.nanmean(trialInf['tf'][0]));     # in cycles per second

    # Get directional selectivity - removed 7/18/17

    # Get derivative order in space and time
    dOrdTi = channel['dord']['ti'];
    if excType == 1:
      dOrdSp = channel['dord']['sp'];
    # (or, if needed get sigmaLow/High)
    if excType == 2:
      sigLow = channel['sigLow'];
      sigHigh = channel['sigHigh'];
    # unpack the LGN weights, but we'll only use if lgnFrontEnd > 0
    mWeight = channel['mWeight'];
    pWeight = 1-mWeight;

    # Get aspect ratio in space - removed 7/18/17

    # Get spatial coordinates
    xCo = 0; # in visual degrees, centered on stimulus center
    yCo = 0; # in visual degrees, centered on stimulus center

    # Store some results in M
    M = dict();
    pref = dict();
    dord = dict();
    pref.setdefault('sf', prefSf);
    pref.setdefault('tf', prefTf);
    pref.setdefault('xCo', xCo);
    pref.setdefault('yCo', yCo);
    if excType == 1:
      dord.setdefault('sp', dOrdSp);
    dord.setdefault('ti', dOrdTi);
    
    M.setdefault('pref', pref);
    M.setdefault('dord', dord);
    if excType == 2:
      M.setdefault('sig', (sigLow, sigHigh));

    # Pre-allocate memory
    z             = trialInf;
    nStimComp     = hf.get_exp_params(expInd).nStimComp;
    nFrames       = hf.num_frames(expInd);
    try:
      nTrials = len(z['num']);
    except:
      nTrials = len(z['con'][0]); 
    
    # set it zero
    M['simpleResp'] = torch.zeros((nFrames, nTrials));

    DoGmodel = 2;
    if lgnFrontEnd == 1:
      dog_m = [1, 3, 0.3, 0.4]; # k, f_c, k_s, j_s
      dog_p = [1, 9, 0.5, 0.4];
    elif lgnFrontEnd == 2:
      dog_m = [1, 6, 0.3, 0.4]; # k, f_c, k_s, j_s
      dog_p = [1, 9, 0.5, 0.4];
    elif lgnFrontEnd == 99: # 99 is code for fitting an LGN front end which is common across all cells in the dataset...
      # parameters are passed as [..., m_fc, p_fc, m_ks, p_ks, m_js, p_js]
      dog_m = [1, allParams[-6], allParams[-4], allParams[-2]];
      dog_p = [1, allParams[-6]*allParams[-5], allParams[-3], allParams[-1]];
    if lgnFrontEnd > 0:
      # specify rvc parameters
      rvcMod = 0;
      params_m = [0, 12.5, 0.05];
      params_p = [0, 17.5, 0.50];
      # save everything for use later
      M['dogModel'] = 2;
      M['dog_m'] = dog_m;
      M['dog_p'] = dog_p;
      M['rvcModel'] = rvcMod;
      M['rvc_m'] = params_m;
      M['rvc_p'] = params_p;

    ####
    # Set stim parameters
    stimOr = _cast_as_tensor((np.pi/180) * np.transpose(np.vstack(z['ori']), (1,0)), device) # in radians
    stimTf = _cast_as_tensor(np.transpose(np.vstack(z['tf']), (1,0)), device) # in radians
    stimCo = _cast_as_tensor(np.transpose(np.vstack(z['con']), (1,0)), device) # in radians
    stimPh = _cast_as_tensor((np.pi/180) * np.transpose(np.vstack(z['ph']), (1,0)), device) # in radians
    stimSf = _cast_as_tensor(np.transpose(np.vstack(z['sf']), (1,0)), device) # in radians

    # I. Orientation, spatial frequency and temporal frequency
    # Compute orientation tuning - removed 17.18.7

    ### LGN filtering stage
    ### Assumptions: No interaction between SF/con -- which we know is not true...
    # - first, SF tuning: model 2 (Tony)
    if lgnFrontEnd > 0:
      resps_m = hf.get_descrResp(dog_m, stimSf, DoGmodel, minThresh=0.1)
      resps_p = hf.get_descrResp(dog_p, stimSf, DoGmodel, minThresh=0.1)
      # -- make sure we normalize by the true max response:
      sfTest = torch.geomspace(0.1, 10, 1000);
      max_m = torch.max(hf.get_descrResp(dog_m, sfTest, DoGmodel, minThresh=0.1));
      max_p = torch.max(hf.get_descrResp(dog_p, sfTest, DoGmodel, minThresh=0.1));
      # -- then here's our selectivity per component for the current stimulus
      selSf_m = torch.div(resps_m, max_m);
      selSf_p = torch.div(resps_p, max_p);
      # - then RVC response: # rvcMod 0 (Movshon)
      rvc_mod = hf.get_rvc_model();
      selCon_m = rvc_mod(*params_m, stimCo)
      selCon_p = rvc_mod(*params_p, stimCo)
      # -- then here's our final responses per component for the current stimulus
      lgnSel = torch.add(torch.mul(mWeight, torch.mul(selSf_m, selCon_m)), torch.mul(pWeight, torch.mul(selSf_p, selCon_p)));

    if excType == 1:
      # Compute spatial frequency tuning - Deriv. order Gaussian
      sfRel = torch.div(stimSf, prefSf);
      s     = torch.pow(stimSf, dOrdSp) * torch.exp(-dOrdSp/2 * torch.pow(sfRel, 2));
      sMax  = torch.pow(prefSf, dOrdSp) * torch.exp(-dOrdSp/2);
      selSf   = torch.div(s, sMax);
    elif excType == 2:
      selSf = hf.flexible_Gauss_np([0,1,prefSf,sigLow,sigHigh], stimSf, minThresh=0);

    # Compute temporal frequency tuning - removed 19.05.13

    # II. Phase, space and time
    omegaX = torch.mul(stimSf, torch.cos(stimOr)); # the stimulus in frequency space
    omegaY = torch.mul(stimSf, torch.sin(stimOr));
    omegaT = stimTf;

    P = torch.empty((nTrials, nFrames, nStimComp, 3)); # nTrials x nFrames for number of frames x nStimComp x [two for x and y coordinate, one for time]
    P[:,:,:,0] = torch.full((nTrials, nFrames, nStimComp), 2*np.pi*xCo);  # P is the matrix that contains the relative location of each filter in space-time (expressed in radians)
    P[:,:,:,1] = torch.full((nTrials, nFrames, nStimComp), 2*np.pi*yCo);  # P(:,0) and p(:,1) describe location of the filters in space

    respSimple = torch.zeros((nTrials,nFrames,));

    # NEW: 20.07.16 -- why divide by 2 for the LGN stage? well, if selectivity is at peak for M and P, sum will be 2 (both are already normalized) // could also just sum...
    if lgnFrontEnd > 0:
      selSi = torch.mul(selSf, lgnSel); # filter sensitivity for the sinusoid in the frequency domain
    else:
      selSi = selSf;

    # Use the effective number of frames displayed/stimulus duration
    # phase calculation -- 
    stimFr = torch.div(torch.arange(nFrames), float(nFrames));
    phOffset = torch.div(stimPh, torch.mul(2*torch.pi, stimTf));
    # slow way?
    phOffsetTile = torch.transpose(torch.tile(phOffset, [nFrames, 1, 1]), [1,0,2]); # output is [nFrames x nTrials x nStimComp], so trans.
    stimPosTile = torch.transpose(torch.tile(stimFr, [nTrials, nStimComp, 1]), [0,2,1]); # output is [nTrials x nStimComp x nFrames], so trans.
    P3slow = torch.add(phOffsetTile, stimPosTile);
    # fast way?
    P3Temp = torch.transpose(torch.add.outer(phOffset, stimFr), [0,2,1]); # result is [nTrials x nFrames x nStimComp], so transpose
    P[:,:,:,2]  = 2*np.pi*P3Temp; # P(:,2) describes relative location of the filters in time.

    omegas = torch.stack((omegaX, omegaY, omegaT),axis=-1); # make this [nTr x nSC x nFr x 3]
    dotprod = torch.einsum('ijkl,ikl->ijk',P,omegas); # dotproduct over the "3" to get [nTr x nSC x nFr]
    # - AxB where A is the overall stimulus selectivity ([nTr x nStimComp]), B is Fourier ph/space calc. [nTr x nFr x nStimComp]
    rComplex = torch.einsum('ij,ikj->ik', torch.mul(selSi,stimCo), torch.exp(torch.mul(1j,dotprod))) # mult. to get [nTr x nFr] response
    # The above line takes care of summing over stimulus components

    # four filters placed in quadrature
    respSimple1 = torch.max(0, rComplex.real); # half-wave rectification,...
    respSimple2 = torch.max(0, torch.mul(-1,rComplex.real));
    respSimple3 = torch.max(0, rComplex.imag);
    respSimple4 = torch.max(0, torch.mul(-1,rComplex.imag));

    # if channel is tuned, it is phase selective...
    # NOTE: 19.05.14 - made response always complex...(wow)! See git for previous version
    respComplex = torch.pow(respSimple1, 2) + torch.pow(respSimple2, 2) \
        + torch.pow(respSimple3, 2) + torch.pow(respSimple4, 2); 
    respSimple = torch.sqrt(torch.div(respComplex, 4)); # div by 4 to avg across all filters

    # Store response in desired format - which is actually [nFr x nTr], so transpose it!
    M['simpleResp'] = torch.transpose(respSimple);
        
    return M;


#######

  def forward(self, stimSf):
    # WORKS!

    sfRel = torch.div(stimSf, self.prefSf);
    # - set the sigma appropriately, depending on what the stimulus SF is                                                                                                      
    sigma = torch.mul(self.sigLow, torch.ones_like(sfRel, requires_grad=True));
    sigma[[x for x in range(len(sfRel)) if sfRel[x] > 1]] = self.sigHigh;
#         - now, compute the responses (automatically normalized, since max gaussian value is 1...) 
    exp_numer = torch.pow(torch.log(sfRel), 2);
    exp_denom = torch.mul(2, torch.pow(sigma, 2));
    s = torch.exp(torch.mul(-1, torch.div(exp_numer, exp_denom)));

    return torch.clamp(s, min=1e-4);

### End of class (sfNormMod)
    
def loss_sfNormMod(respModel, respData, lossType=1):

  if lossType == 1:
      lsq = torch.pow(torch.sign(respModel)*torch.sqrt(torch.abs(respModel)) - torch.sign(respData)*torch.sqrt(torch.abs(respData)), 2);

      NLL = torch.mean(lsq);

  return NLL;
