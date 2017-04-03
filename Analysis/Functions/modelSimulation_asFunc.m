function [S] = modelSimulation_asFunc(cellNumber, sf_c, stim_repeats, linear)

%% constants

if ~exist('sf_c', 'var')
    n_sf_steps = 11;
    sf_c = logspace(log10(0.3), log10(10), n_sf_steps); % 0.3 and 10 cpd are experimental bounds for sf_c 
end

if ~exist('stim_repeats', 'var')
    stim_repeats = 10; % to match experimental conditions
end

if ~exist('linear', 'var')
    linear = 0; % assume we're fitting full model
end

numFamiles = 5;
numCons = 2;

%% Set paths and get cell
currentPath  = strcat('/e/3.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv/Analysis/Scripts');
loadPath     = strcat('/e/3.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv/Analysis/Structures');
functionPath = strcat('/e/3.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv/Analysis/Functions');

% Load data files
cd(loadPath)
load('dataList');
loadName = [N.unitName{cellNumber}, '_sfm'];
load(loadName);
cd(currentPath)

% Get model parameters if they exits
if linear == 1 && isfield(S.sfm.mod, 'fit_lin')
    params = S.sfm(1).mod.fit_lin.params;       
elseif linear == 0 && isfield(S.sfm.mod, 'fit')
    params = S.sfm(1).mod.fit.params;
end

%% Model parameters

% excChannel
excChannel.pref.or = params(1);
excChannel.pref.sf = params(2);
excChannel.arat.sp = params(3);
excChannel.dord.sp = params(4);
excChannel.dord.ti = 0.25;                                                 % derivative order in the temporal domain, d = 0.25 ensures broad tuning for temporal frequency
excChannel.ds      = params(5);

% Inhibitory channel
inhChannel.gain = params(6);
inhChannel.asym = 0; %??? this used by be params(13). Be careful if you bring this back.

% Other (nonlinear) model components
sigma    = 10^params(7);                                                   % normalization constant
respExp  = params(8);                                                      % response exponent
scale    = params(9);                                                      % response scalar

% Noise parameters
noiseEarly = params(10);                                                   % early additive noise
noiseLate  = params(11);                                                   % late additive noise
varGain    = params(12);                                                   % multiplicative noise

iR = 1; % ASSUMPTION: We're just assuming that only one experiment was run, or at least that we just grab the first one

%% compute full response

clear respModel; % start fresh

for family = 1 : numFamiles
    
    fprintf('\nfamily %d:\n\t', family);
    
    clear stimParams;
    stimParams.stimFamily = family;   
    stimparams.repeats = stim_repeats;
    
    for con = 1 : numCons

        fprintf('contrast %d - ', con);
        
        stimParams.conLevel = con;
        
        for sf_i = 1 : numel(sf_c)
                        
            stimParams.sf_c = sf_c(sf_i);
            
            % linear filter
            E = SFMSimpleResp('cellStructure', S, 'channel', excChannel, 'expRun', iR, 'stimParams', stimParams); 

            % Extract simple cell response (half-rectified linear filtering)
            Lexc = E.simpleResp;

            % Compute suppressive signals

            [norm_signal] = GetNormResp(S, stimParams);
            
            for iP = 1:numel(norm_signal.pref.sf)
                inhWeight{iP} = 1 + inhChannel.asym*(log(norm_signal.pref.sf{iP}) - mean(log(norm_signal.pref.sf{iP})));
            end
            inhWeightMat = repmat([inhWeight{1}, inhWeight{2}], [stimparams.repeats 1 120]); % number of trials!
            
            % Get inhibitory response (pooled responses of complex cells tuned to wide range of spatial frequencies, square root to bring everything in linear contrast scale again)
            Linh = squeeze(sqrt(sum(inhWeightMat.*norm_signal.normResp, 2)))';

            % Compute full model response (the normalization signal is the same as the subtractive suppressive signal)
            numerator     = noiseEarly + Lexc + inhChannel.gain*Linh;
            denominator   = sigma.^2 + Linh;
            ratio         = max(0, numerator./denominator).^respExp;
            meanRate      = mean(ratio);

            respModel{family}{con}(sf_i) = mean(noiseLate + scale*meanRate);
            
            % [sf_i X repeats]
            respModel_full{family}{con}(sf_i, 1:stimparams.repeats) = noiseLate + scale*meanRate;
            
            
        end
        
    end
    
end

%% now save

fprintf('\n');

S.sfm.mod.sim.resp_full = respModel_full;
S.sfm.mod.sim.resp = respModel;
S.sfm.mod.sim.sf_c = sf_c;

cd(loadPath)
save(loadName, 'S'); % save it again!

