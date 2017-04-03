% SFMGIVEBOF   Computes the negative log likelihood for the LN-LN model

function [NLL, respModel, E] = SFMGiveBof_linear(params, structureSFM)

    % 01 = preferred direction of motion (degrees)
    % 02 = preferred spatial frequency   (cycles per degree)
    % 03 = aspect ratio 2-D Gaussian
    % 04 = derivative order in space
    % 05 = directional selectivity
    % 06 = gain inhibitory channel
    % 07 = normalization constant        (log10 basis)
    % 08 = response exponent
    % 09 = response scalar
    % 10 = early additive noise
    % 11 = late additive noise
    % 12 = variance of response gain    
    % 13 = asymmetry suppressive signal 

T = structureSFM.sfm;

% Get parameter values
% Excitatory channel
excChannel.pref.or = params(1);
excChannel.pref.sf = params(2);
excChannel.arat.sp = params(3);
excChannel.dord.sp = params(4);
excChannel.dord.ti = 0.25;                                                 % derivative order in the temporal domain, d = 0.25 ensures broad tuning for temporal frequency
excChannel.ds      = params(5);
% excChannel.ecc     = params(13);
% excChannel.wMagno  = params(14);

% Inhibitory channel
inhChannel.gain = params(6);
inhChannel.asym = 0; % used to be param, but we are NOT dealing with inhibition here, so set value to zero

% Other (nonlinear) model components
sigma    = 10^params(7);                                                   % normalization constant
respExp  = params(8);                                                      % response exponent
scale    = params(9);                                                      % response scalar

% Noise parameters
noiseEarly = params(10);                                                   % early additive noise
noiseLate  = params(11);                                                   % late additive noise
varGain    = params(12);                                                   % multiplicative noise


%% Evaluate prior on response exponent -- corresponds loosely to the measurements in Priebe et al. (2004)
% priorExp = lognpdf(respExp, 1.15, 0.3);
% NLLExp   = -log(priorExp);
NLLExp = 0; % set to zero because we don't use respExp, therefore don't want the prior to influence!

%% Compute weights for suppressive signals
for iP = 1:numel(T.mod.normalization.pref.sf)
    inhWeight{iP} = 1 + inhChannel.asym*(log(T.mod.normalization.pref.sf{iP}) - mean(log(T.mod.normalization.pref.sf{iP})));
end
inhWeightMat = repmat([inhWeight{1}, inhWeight{2}], [numel(T.exp.trial.num) 1 120]);

%% Evaluate sfmix experiment
for iR = 1:numel(structureSFM.sfm)
    T = structureSFM.sfm(iR);

    % Get simple cell response for excitatory channel
   [E] = SFMSimpleResp('cellStructure', structureSFM, 'channel', excChannel, 'expRun', iR);  
%     [E] = retinaSimpleResp('cellStructure', structureSFM, 'channel', excChannel, 'expRun', iR);  
    
    % Extract simple cell response (half-rectified linear filtering)
    Lexc = E.simpleResp;
    
    % Get inhibitory response (pooled responses of complex cells tuned to wide range of spatial frequencies, square root to bring everything in linear contrast scale again)
    Linh = squeeze(sqrt(sum(inhWeightMat.*T.mod.normalization.normResp, 2)))';
    
    % Compute full model response (the normalization signal is the same as the subtractive suppressive signal)
    numerator     = noiseEarly + Lexc; % + inhChannel.gain*Linh;
    denominator   = sigma.^2; %+ Linh;
%    ratio         = max(0, numerator./denominator).^respExp;
    ratio 	  = max(0, numerator);
    meanRate      = mean(ratio);
    respModel{iR} = noiseLate + scale*meanRate;
    
    % Get predicted spike count distributions
    mu  = max(.01, T.exp.trial.duration.*respModel{iR});                   % The predicted mean spike count
    var = mu + (varGain*(mu.^2));                                          % The corresponding variance of the spike count
    r   = (mu.^2)./(var - mu);                                             % The parameters r and p of the negative binomial distribution
    p   = r./(r + mu);
    
    % Evaluate the model
    llh = nbinpdf(T.exp.trial.spikeCount, r, p);                           % The likelihood for each pass under the doubly stochastic model
    
    NLLtempSFM(iR) = sum(-log(llh));                                       % The negative log-likelihood of the whole data-set      
end


%% Combine dataand prior
NLL = sum(NLLtempSFM) + NLLExp;

