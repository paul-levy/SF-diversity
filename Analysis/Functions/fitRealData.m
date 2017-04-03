function fitRealData(iU, linear)

% fitRealData    Fits an LN-LN model to cell responses elicited by gratings
% and mixture stimuli. The fitting algorithm is a Nelder-Mead simplex. The 
% model is fit with a multistart procedure and semi-randomized starting 
% values. All texpat and spatial frequency data are fitted

% force respExp = 2

%%

if nargin() == 1
  linear = 0;
end

%%
% Set paths [local]
% currentPath  = strcat('/e/3.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv/Analysis/Scripts');
% loadPath     = strcat('/e/3.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv/Analysis/Structures');
% functionPath = strcat('/e/3.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv/Analysis/Functions');

% Cluster paths
 currentPath  = strcat('/home/pl1465/modelCluster/sfDiv/Analysis/Scripts');
 loadPath     = strcat('/home/pl1465/modelCluster/sfDiv/Analysis/Structures');
 functionPath = strcat('/home/pl1465/modelCluster/sfDiv/Analysis/Functions');

% Set constants
nMultiStarts = 10;

% Now fit LN-LN model
for iR = 1:nMultiStarts
    
    % Clear memory
    clear bestNLL;
    clear S;
    
    % Set fit options
    if (iR/2 == round(iR/2))
        options = optimset('Display', 'iter', 'Maxiter', 20, 'MaxFuneval', 1000, 'Algorithm', 'sqp');
    else
        options = optimset('Display', 'iter', 'Maxiter', 20, 'MaxFuneval', 1000, 'Algorithm', 'interior-point');
    end
    
    % Load data files
    cd(loadPath)
    load('dataList');    

    % Loads S
    loadNameSfm = [N.unitName{iU}, '_sfm'];
    load(loadNameSfm);
    cd(currentPath)

    % Some useful values
    oriPref = unique(S.sfm.exp.trial.ori{2}(isfinite(S.sfm.exp.trial.ori{2})));
    sfPref  = unique(S.sfm.exp.trial.sf{1}(S.sfm.exp.trial.con{1} == .01));
    rMax    = max(S.sfm.exp.oriRateMean);
    
    
    %% Constraints are set in the form of lower and upper parameter bounds
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
    
    lowerBound     = zeros(12,1);   upperBound     = zeros(12,1);    
    lowerBound(1)  = -180;          upperBound(1)  = 440;                  
    lowerBound(2)  = .05;           upperBound(2)  = 15;                  
    lowerBound(3)  = .1;            upperBound(3)  = 4;                    
    lowerBound(4)  = .1;            upperBound(4)  = 6;                    
    lowerBound(5)  = 0;             upperBound(5)  = 1;                    
    lowerBound(6)  = -1;            upperBound(6)  = 0;                   
    lowerBound(7)  = -3;            upperBound(7)  = 1;                
    lowerBound(8)  = 2;             upperBound(8)  = 2;                 
%     lowerBound(8)  = 1;             upperBound(8)  = 10;                 
    lowerBound(9)  = 1e-3;          upperBound(9)  = 1e9;                  
    lowerBound(10) = 0;             upperBound(10) = 1;                
    lowerBound(11) = 0;             upperBound(11) = 100;                
    lowerBound(12) = 10^-3;         upperBound(12) = 10^1;                
    lowerBound(13) = -.35;          upperBound(13) = .35;                
    
    
    % Define the objective function, set the startvalues, perform the fit
    fprintf('\n \n')
    disp(['Fitting model for ', N.unitName{iU}])
    cd(functionPath)

    if linear == 1
    	obFun = @(params) SFMGiveBof_linear(params, S);
    else
    	obFun = @(params) SFMGiveBof(params, S);
    end

    try
        if linear == 1
            bestNLL    = S.sfm(1).mod.fit_lin.NLL;                                 % Check for previously saved fit outcome
            bestParams = reshape(S.sfm(1).mod.fit_lin.params, [1 13]);
        else
            bestNLL    = S.sfm(1).mod.fit.NLL;                                 % Check for previously saved fit outcome
            bestParams = reshape(S.sfm(1).mod.fit.params, [1 13]);
        end
        fprintf('Current best fit is %0.5g \n', bestNLL);
        
        if iR == 1
            if linear == 1
              startvalues = reshape(S.sfm(1).mod.fit_lin.params, [1 13]);
            else
              startvalues = reshape(S.sfm(1).mod.fit.params, [1 13]);
            end
        else
            startvalues = [bestParams(1:2), bestParams(3:end).*(.5+rand(size(bestParams(3:end))))];
        end
    catch
        disp('Previous fit statistics not found.');
        startvalues = [oriPref sfPref 2 2 0.5 -0.05 0 3.0 rMax*10 0.1 0.1 0.1 0];
    end
    
    %%%%%%%%%%
%     startvalues = [352.5 4 2 1 1 0 -.5 4 500 0 .01 .25 0];
    %%%%%%%%%%
%     startvalues = min(upperBound' - 10^-4, max(lowerBound' + 10^-4, startvalues));
    
    try [modelParams, NLL] = fmincon(obFun, startvalues, [], [], [], [], lowerBound, upperBound, [], options);
        
        modelParams(1) = 180*(1+sawtooth(pi/180 * modelParams(1)));        % Ensure that preferred orientation is expressed between 0 and 360
        
        % Store outcome if better than previous best outcome
        if (exist('bestNLL', 'var') && exist('NLL', 'var'))
            if (NLL < bestNLL)
                disp('Better fit found!');
                if linear == 1
                          S.sfm(1).mod.fit_lin.NLL    = NLL;
                          S.sfm(1).mod.fit_lin.params = modelParams;
                else
                          S.sfm(1).mod.fit.NLL    = NLL;
                          S.sfm(1).mod.fit.params = modelParams;
                end                

                cd(loadPath)
                save(loadNameSfm, 'S');
                disp(['Saving fit model for ', N.unitName{iU}])
            else
                disp('No better fit found...');
            end
        else
            if linear == 1
                  S.sfm(1).mod.fit_lin.NLL    = NLL;
                  S.sfm(1).mod.fit_lin.params = modelParams;
            else
                  S.sfm(1).mod.fit.NLL    = NLL;
                  S.sfm(1).mod.fit.params = modelParams;
            end            

            cd(loadPath)
            save(loadNameSfm, 'S');
            disp(['Saving fit model for ', N.unitName{iU}])
        end
        
    catch
        disp('error, fit failed')
    end
end

cd(currentPath)

