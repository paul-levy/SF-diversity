function [M, M_gt] = GetNormResp(iU, stimParams)

% GETNORMRESP    Runs the code that computes the response of the
% normalization pool for the recordings in the SfDiv project.

% Robbe Goris, 10-30-2015

% Edit - Paul Levy, 1/23/17 to give option 2nd parameter for passing in own
% stimuli (see SFMNormResp for more details)
% 1/25/17 - Allowed 'S' to be passed in by checking if unitName is numeric
% or not (if isnumeric...)

%%
% Set paths [cluster]
 currentPath  = strcat('/home/pl1465/modelCluster/sfDiv/Analysis/Scripts');
 loadPath     = strcat('/home/pl1465/modelCluster/sfDiv/Analysis/Structures');
 functionPath = strcat('/home/pl1465/modelCluster/sfDiv/Analysis/Functions');

clear normPool;

M = [];
M_gt = [];

% Set paths [local]
% currentPath  = strcat('/e/3.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv/Analysis/Scripts');
% loadPath     = strcat('/e/3.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv/Analysis/Structures');
% functionPath = strcat('/e/3.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv/Analysis/Functions');

% Set characteristics normalization pool
normPool.n{1} = .75;                                                       % The exponents of the filters used to approximately tile the spatial frequency domain
normPool.n{2} = 1.5;                                                       % The pool includes broad and narrow filters

normPool.nUnits{1} = 12;                                                   % The number of cells in the broad pool
normPool.nUnits{2} = 15;                                                   % The number of cells in the narrow pool

normPool.gain{1} = .57;                                                    % The gain of the linear filters in the broad pool
normPool.gain{2} = .614;                                                   % The gain of the linear filters in the narrow pool

%%
cd(loadPath)
load('dataList');

% if iU is a number, then we load that cell's unitName
% otherwise, we assume that iU is actually the cell structure already!
%   in that case, SFMNormResp has been modified to handle this
if isnumeric(iU)
    cd(functionPath)
    unitName = N.unitName{iU};
else
    unitName = iU;
end
    
if exist('stimParams', 'var')
    cd(functionPath);
    M = SFMNormResp(unitName, 'loadPath', loadPath, 'normPool', normPool, 'stimParams', stimParams);
    
     %if debugging
%     if isfield(M, 'trials_used')
%         stimParams.trials_used = M.trials_used;
%     end
%     M_gt = SFMNormResp_gt(unitName, 'loadPath', loadPath, 'normPool', normPool, 'stimParams', stimParams);
else
    M = SFMNormResp_gt(unitName, 'loadPath', loadPath, 'normPool', normPool);
end

cd(currentPath)
end


