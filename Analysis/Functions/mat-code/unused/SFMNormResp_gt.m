function M = SFMNormResp_gt(unitName, varargin)

% SFNNormResp       Computes normalization response for sfMix experiment
%
% SFMNormResp(unitName, varargin) returns a simulated V1-normalization
% response to the mixture stimuli used in sfMix. The normalization pool
% consists of spatially distributed filters, tuned for orientation, spatial 
% frequency, and temporal frequency. The tuning functions decribe responses
% of spatial filters obtained by taking a d-th order derivative of a 
% 2-D Gaussian. The pool includes both filters that are broadly and 
% narrowly tuned for spatial frequency. The filters are chosen such that 
% their summed responses approximately tile the spatial frequency domain
% between 0.3 c/deg and 10 c/deg.

% 1/23/17 - Edited, like SFMSimpleResp, to allow for making own stimuli

% 1/25/17 - Allowed 'S' to be passed in by checking if unitName is a string
%       or not (if ischar...)
%   Discovery that rComplex = line is QUITE SLOW. Need to speed up...
%   So, decompose that operation into [static] + [dynamic]
%   where static is the part of computation that doesn't change with frame
%   and dynamic does, meaning only the latter needs to computed repeatedly!

% 1/30/17 - Allow for passing in of specific trials from which to draw phase/tf

% Get input values from varargin or assign default values
loadPath = GetNamedInput(varargin, 'loadPath', pwd);
normPool = GetNamedInput(varargin, 'normPool', pwd);
stimParams = GetNamedInput(varargin, 'stimParams', pwd);

make_own_stim = 0;
if ~strcmpi(stimParams, pwd) % i.e. if we actually have stimParams
    make_own_stim = 1;
end

% If unitName is char/string, load the data structure
% Otherwise, we assume that we already are passing loaded data structure
if ischar(unitName)
    cd(loadPath)
    loadName = [unitName, '_sfm'];
    load(loadName);
    fprintf('losing time...\n');
else
    S = unitName;
end

if make_own_stim
    if ~isfield(stimParams, 'template')
        stimParams.template = S;
    end
    
    if isfield(stimParams, 'trial_used')
        stimParams.template.trial_used = stimParams.trial_used;
    end
    
    if ~isfield(stimParams, 'repeats')
        stimParams.repeats = 10; % why 10? To match experimental
    end
end

for iR = 1:numel(S.sfm)
    
    T = S.sfm(iR);
        
    % Get filter properties in spatial frequency domain
    for iB = 1:numel(normPool.n)
        prefSf{iB} = logspace(log10(.1), log10(30), normPool.nUnits{iB});
        gain{iB}   = normPool.gain{iB};
    end
       
    % Get filter properties in direction of motion and temporal frequency domain
    prefOr = (pi/180)*unique(T.exp.trial.ori{2}(isfinite(T.exp.trial.ori{2})));     % in radians
    prefTf = round(nanmean(T.exp.trial.tf{1}));                            % in cycles per second
    
    % Compute spatial coordinates filter centers (all in phase, 4 filters per period)
    stimSi = T.exp.size;                                                   % in visual degrees
    stimSc = 1.75;                                                         % in cycles per degree, this is approximately center frequency of stimulus distribution
    nCycle = stimSi*stimSc;
    radius = sqrt((ceil(4*nCycle)^2)/pi);
    vec    = -ceil(radius):1:ceil(radius);
    xTemp  = .25/stimSc*repmat(vec', [1, length(vec)]);
    yTemp  = .25/stimSc*repmat(vec, [length(vec), 1]);
    ind    = sign(stimSi/2 - sqrt(xTemp.^2 + yTemp.^2));
    xCo    = xTemp(ind > 0)';                                              % in visual degrees, centered on stimulus center
    yCo    = yTemp(ind > 0)';                                              % in visual degrees, centered on stimulus center
    
    % Store some results in M
    M          = struct;
    M.pref.or  = prefOr;
    M.pref.sf  = prefSf;
    M.pref.tf  = prefTf;
    M.pref.xCo = xCo;
    M.pref.yCo = yCo;
    
    % Pre-allocate memory
    z          = T.exp.trial;
    nSf        = 0;
    for iS = 1:numel(prefSf);
        nSf = nSf + numel(prefSf{iS});
    end
    if make_own_stim == 1
        nTrials = stimParams.repeats; % keep consistent with 10 repeats per stim. condition
    else
        nTrials  = numel(z.num);
    end
    M.normResp = zeros(nTrials, nSf, 120);
    
    
    % Compute normalization response for all trials
    if ~make_own_stim
        disp(['Computing normalization response for ', unitName, ' ...']);
    end
        
    for p = 1:nTrials
        
        if round(p/156) == p/156; % 156 is from Robbe...Comes from 10 repeats --> 1560(ish) total trials
            fprintf('\n normalization response computed for %d of %d repeats...', p/156, round(nTrials/156));
        end
        
        % Set stim parameters
        if make_own_stim == 1
            % why the two blanks in the middle? Those are for more manual
            % control of stimuli, used for debugging only
           
            % So... If we want to pass in specific trials, check that those
            % trials exist
            % if there are enough for the trial 'p' we are at now, then
            % grab that one; otherwise get the first
            if isfield(stimParams, 'trial_used')
                if numel(stimParams.trial_used) >= p
                    stimParams.template.trial_used = stimParams.trial_used(p);
                else
                    stimParams.template.trial_used = stimParams.trial_used(1);
                end
            end
            
            [stimOr, stimTf, stimCo, stimPh, stimSf, trial_used(p)] = makeStimulus(stimParams.stimFamily, stimParams.conLevel, stimParams.sf_c, [], [], stimParams.template);
        else
            for iC = 1:9
                stimOr(iC) = z.ori{iC}(p) * pi/180;                                % in radians
                stimTf(iC) = z.tf{iC}(p);                                          % in cycles per second
                stimCo(iC) = z.con{iC}(p);                                         % in Michelson contrast
                stimPh(iC) = z.ph{iC}(p) * pi/180;                                 % in radians
                stimSf(iC) = z.sf{iC}(p);                                          % in cycles per degree
            end
        end

        % I. Orientation, spatial frequency and temporal frequency         % matrix size: 9 x nFilt (i.e., number of stimulus components by number of orientation filters)
        for iB = 1:numel(normPool.n)
            sfRel     = repmat(stimSf', [1 numel(prefSf{iB})])./repmat(prefSf{iB}, [9 1]);
            s         = repmat(stimSf', [1 numel(prefSf{iB})]).^normPool.n{iB} .* exp(-normPool.n{iB}/2 * sfRel.^2);
            sMax      = repmat(prefSf{iB}, [9 1]).^normPool.n{iB} .* exp(-normPool.n{iB}/2);
            selSf{iB} = gain{iB} * s./sMax;
        end
        
        % Spatial frequency
        selOr = ones(9, 1);                                                % all stimulus components of the spatial frequency mixtures were shown at the cell's preferred direction of motion

        % Compute temporal frequency tuning
        dOrdTi = 0.25;                                                     % derivative order in the temporal domain, d = 0.25 ensures broad tuning for temporal frequency
        tfRel  = stimTf./prefTf;
        t      = stimTf.^dOrdTi .* exp(-dOrdTi/2 * tfRel.^2);
        tMax   = prefTf.^dOrdTi .* exp(-dOrdTi/2);
        tNl    = t'/tMax;
        selTf  = tNl;

        
        % II. Phase, space and time
        omegaX = stimSf.*cos(stimOr);                                      % the stimulus in frequency space
        omegaY = stimSf.*sin(stimOr);
        omegaT = stimTf;
        
        P(:,1) = 2*pi*repmat(xCo', [120 1]);                               % P is the matrix that contains the relative location of each filter in space-time (expressed in radians)
        P(:,2) = 2*pi*repmat(yCo', [120 1]);                               % P(:,1) and p(:,2) describe location of the filters in space
                
        % Pre-allocate some variables
        respComplex = zeros(nSf, length(xCo), 120);
        selSfVec    = [selSf{1}, selSf{2}];
        
        countz = 0;
        
        for iF = 1:nSf
            linR1 = zeros(120*length(xCo), 9);                             % pre-allocation
            linR2 = zeros(120*length(xCo), 9);
            linR3 = zeros(120*length(xCo), 9);
            linR4 = zeros(120*length(xCo), 9);
            computeSum = 0;                                                % important constant: if stimulus contrast or filter sensitivity equals zero there is no point in computing the response
            
            for c = 1:9                                                    % there are up to nine stimulus components
                selSi = selOr(c)*selSfVec(c,iF)*selTf(c);                  % filter sensitivity for the sinusoid in the frequency domain
                
                if (selSi ~= 0 && stimCo(c) ~= 0)
                    computeSum = 1;
                    
                    % The number of frames displayed/stimulus duration
                    stimPos = (0:119)/120 + stimPh(c)/(2*pi*stimTf(c));    % 120 frames + the appropriate phase-offset
                    P3Temp  = (repmat(stimPos, [length(xCo) 1]));
                    P(:,3)  = 2*pi*P3Temp(:);                              % P(:,3) describes relative location of the filters in time.
                    
                    % This line is slow. How to speed up?
                    rComplex = selSi*stimCo(c)*exp(1i*P*[omegaX(c) omegaY(c) omegaT(c)]');
                    
                    linR1(:,c) = real(rComplex);                           % four filters placed in quadrature
                    linR2(:,c) = -1*real(rComplex);
                    linR3(:,c) = imag(rComplex);
                    linR4(:,c) = -1*imag(rComplex);
                end
            end
            
            if computeSum == 1
                respSimple1 = max(0, sum(linR1, 2)).^2;                    % superposition and half-squaring,...
                respSimple2 = max(0, sum(linR2, 2)).^2;
                respSimple3 = max(0, sum(linR3, 2)).^2;
                respSimple4 = max(0, sum(linR4, 2)).^2;                    % followed by summation over filter phase,...
                
                respComplex(iF,:,:) = reshape(respSimple1 + respSimple2 + respSimple3 + respSimple4, [length(xCo), 120]);
            end            
        end
        
        % integration over space (compute average response across space, normalize by number of spatial frequency channels)
        respInt = squeeze(mean((1/numel(normPool.n))*respComplex, 2));

        % square root to bring everything in linear contrast scale again
        M.normResp(p,:,:) = respInt;   
    end
    
    if ~make_own_stim
        fprintf('\n \n \n \n \n \n');
    end
    
    M.trial_used = trial_used;
    
    % if you make/use your own stimuli, just return the output, M;
    % otherwise, save the responses
    if ~make_own_stim 
         % Save the simulated normalization response in the unit's structure
        S.sfm(iR).mod.normalization = M;
        save([loadPath,'/',loadName],'S');
    end
end

end




%% Functions used in the main script
%% GetNamedInput
function y = GetNamedInput(C, varName, varDefault)
% looks for the string varName in varargin, and returns the following entry
% in varargin. If varName is named more than once, a cell array is
% returned. If it is not found, varDefault is returned.

y = varDefault;

k = 0;
for i = 1:(length(C)-1)
    if strcmpi(C{i}, varName)
        k = k+1;                                                           % increment k every time the varName is found in varargin
        if k > 1
            y{k} = C{i+1};
        else
            y = C{i+1};
        end
    end
end
end
%%

