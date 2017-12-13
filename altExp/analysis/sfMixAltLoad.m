function sfMixAltLoad(unitLabels, varargin)

% sfMixAltLoad       Sorting of sfMixAlt xml-files.
%
%   sfMixAltLoad(unitLabels) loads the xml-files associated with the Expo
%   program used for the sfMixAlt experiment. The input unitLabels is either
%   a string specifying the label of a single unit (e.g. 'm621r47') or a
%   cell array of strings specifying multiple units (e.g. unitLabels =
%   {'m621r47', 'm630r05', 'm620r67'}). Essential information for each
%   experiment is organized into a single structure for each unit. Single
%   unit structures are saved to individual files (e.g. 'm621r47_tp.mat').
%
%   sfMixAltLoad(unitLabels, 'loadPath', path) specifies the directory from
%   which to load xml files. Default is the current directory.
%
%   sfMixAltLoad(unitLabels, 'savePath', path) specifies the directory to
%   which resulting structures will be saved (one file per unit). Default
%   is to match loadPath.
%
%   sfMixAltLoad(unitLabels, 'processRepeats', 1) enables processing multiple
%   runs of the sfMix program. All runs are saved as fields with different
%   subscripts in the single unit structure. Default is to process only the
%   version that was run last.
%
%   v1.0 2015: Robbe Goris
%   
%   v2.0, 12/13/17: Paul Levy
%   Adjusted for the new sfMixAlt experiment, which uses a new set of
%   stimuli
%
%#ok<*NASGU>
%%

% addpath with expo resources
addpath(genpath('/u/vnl/matlab/ExpoMatlab/'));

supportedPrograms = {'sfMixAlt'};
supportedTypes    = {'sfm'};


%% Parse inputs, perform initial operations
% Make 'unitLabels' into cell array if single string:
if ~iscell(unitLabels), unitLabels = {unitLabels}; end

% Get unit info from label string
nUnits = length(unitLabels);

% Get input values from varargin or assign default values
loadPath         = GetNamedInput(varargin, 'loadPath', pwd);
savePath         = GetNamedInput(varargin, 'savePath', loadPath);
processRepeats   = GetNamedInput(varargin, 'processRepeats', 0);
selectedPrograms = GetNamedInput(varargin, 'prog', supportedPrograms);

if ~iscell(selectedPrograms)
    selectedPrograms = {selectedPrograms}; 
end


% Loop through cells
for iU = 1:nUnits
    
    % Initialize structure S to hold single unit
    S = struct();
    
    % Specify the cell's save name
    saveName = [unitLabels{iU}, '_sfm'];
    
    % Get some information about the experiments run for this cell
    xmlList = dir([loadPath, '/', unitLabels{iU}, '*xml']);
    nFiles  = size(xmlList,1);
    
    if nFiles>0
        disp(['Loading XML data for ', unitLabels{iU}, ' ...']);
    else
        disp(['No XML files for ', unitLabels{iU}, ' found in ', loadPath]);
    end
    
    progNames = regexprep({xmlList(:).name}, '.+\[(\w+)\].+', '$1');
    
    if ~processRepeats && length(unique(progNames)) < length(progNames)                                         % more than one of a given supported type found
        FileNDX = sort(cellfun(@(x) find(ismember(progNames,x),1,'last'), unique(progNames)));
    else
        FileNDX = 1:length(progNames);
    end
    
    fprintf('Found (%d) programs and (%d) multiple run(s). Analyzing (%d) files.\n',...
        nFiles,length(progNames)-length(unique(progNames)),length(FileNDX));

    % Check all these experiments
    for iF = FileNDX
        
        % search filename for something between two brackets and return it.
        progName = progNames{iF};
        
        % Check that the program is supported and has been selected for analysis
        matchesSupported = ismember(supportedPrograms, progName);                                               % is progName a member of supportedPrograms?
        matchesSelected  = ismember(selectedPrograms, progName);                                                % is progName a member of  selectedPrograms?
        
        if any(matchesSelected) && any(matchesSupported)
            expType = supportedTypes{matchesSupported};
            fprintf('[%s]',expType);
            [s] = AnalyzeExperiment(loadPath, expType, xmlList(iF).name);
            expStruct.exp = s;
            if isfield(S, expType)
                S.(expType) = [S.(expType), expStruct];
            else
                S.(expType) = expStruct;
            end
        end
    end
    fprintf('\n');
    
    % Add this unit to structure array for output
    S.unitLabel = unitLabels{iU};
    S = orderfields(S);   
    
    % Save the sorted DIS in the designated folder
    save([savePath,'/',saveName],'S');
end
end
%%
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
        k = k+1;                                                                                                % increment k every time the varName is found in varargin
        if k > 1
            y{k} = C{i+1};
        else
            y = C{i+1};
        end
    end
end
end
%%
%% AnalyzeExperiment
function [s] = AnalyzeExperiment(loadPath, expType, fileName)

% use to get the value of an attribute ("x"; e.g. contrast, orientation) for
% each component of a given trial ("tr")
get_comp = @(x, tr) cellfun(@(y) y(tr), x); % get components by trial
nGrats = 7;

% Load the appropriate file, no output is printed to the screen
DIS = ReadExpoXML([loadPath, '/', fileName], 0, 0);

% Get the block IDs of the matrix
mBlockIDs = DIS.matrix.MatrixBaseID + (0:DIS.matrix.NumOfBlocks-1);

% Get the indices of all valid passes
ValidPassIDs  = DIS.passes.IDs(ismember(DIS.passes.BlockIDs, mBlockIDs));                                       % indexed at 0
ValidBlockIDs = DIS.passes.BlockIDs(ismember(DIS.passes.BlockIDs, mBlockIDs));                                  % indexed at 0

StimNDXs      = logical(GetEvents(DIS, ValidPassIDs, 'DKL Texture', '', 0, 'Contrast') > 0);                    % index of ValidPasses
StimPassIDs   = ValidPassIDs(StimNDXs);                                                                         % indexed at 0
StimBlockIDs  = unique(ValidBlockIDs(StimNDXs));                                                                % indexed at 0
nStimBlockIDs = length(StimBlockIDs);

BlankNDXs      = logical(GetEvents(DIS, ValidPassIDs, 'DKL Texture', '', 0, 'Contrast') == 0);                  % index of ValidPasses
BlankPassIDs   = ValidPassIDs(BlankNDXs);                                                                       % indexed at 0
BlankBlockIDs  = unique(ValidBlockIDs(BlankNDXs));                                                              % indexed at 0
nBlankBlockIDs = length(BlankBlockIDs);
nBlankPasses   = sum(BlankNDXs);

% Get the stimulus location and other parameters of interest
xPosVals = unique(GetEvents(DIS, StimPassIDs, 'Surface', '', 0, 'X Position', 'deg'))';
yPosVals = unique(GetEvents(DIS, StimPassIDs, 'Surface', '', 0, 'Y Position', 'deg'))';
sizeVals = unique(GetEvents(DIS, StimPassIDs, 'Surface', '', 0, 'Width', 'deg'))';

% Get the latency of the neuron in this experiment
latency = GetLatency(DIS, 10:150, mBlockIDs);

% Get the stimulus durations and spike times
duration   = GetDurations(DIS, ValidPassIDs, 'sec');                                                            % durations only for valid trials
spikeTimes = GetSpikeTimes(DIS, 0, ValidPassIDs, .001*latency, .001*latency, 0, 'sec');                         % spikeTimes only for valid trials
% 0.001*latency to convert seconds (from milliseconds)
tfPass     = GetEvents(DIS, ValidPassIDs, 'Surface', '', 0, 'Drift Rate', 'cyc/sec');

trial.num           = 1:length(ValidPassIDs);                                                                   % 1 to the number of valid trials
trial.blockID       = ValidBlockIDs;                                                                            % block ids of valid trials
trial.duration      = duration;
trial.spikeTimes    = spikeTimes;
trial.spikeCount    = GetSpikeCounts(spikeTimes, 0, duration, 'sec', 'impulses');                               % spikeCounts for valid trials
trial.f1            = SpikeTrainFT(spikeTimes, tfPass, trial.duration);                                         % response modulation at fundamental Fourier component
trial.f1(BlankNDXs) = NaN;

% Get the spatial frequency, temporal frequency, orientation, phase, and contrast for each stimulus component
for iC = 1:nGrats
    trial.sf{iC}             = reshape(GetEvents(DIS, ValidPassIDs, 'DKL Texture', '', iC-1, 'Spatial Frequency X', 'cyc/deg'),  1, []);
    trial.tf{iC}             = reshape(GetEvents(DIS, ValidPassIDs, 'Surface',     '', iC-1, 'Drift Rate'),  1, []);
    trial.ori{iC}            = reshape(GetEvents(DIS, ValidPassIDs, 'Surface',     '', iC-1, 'Orientation'), 1, []);
    trial.ph{iC}             = reshape(GetEvents(DIS, ValidPassIDs, 'Surface',     '', iC-1, 'Phase'),  1, []);
    trial.ori{iC}(BlankNDXs) = NaN;
    trial.tf{iC}(BlankNDXs)  = NaN;
    trial.ph{iC}(BlankNDXs)  = NaN;
    trial.sf{iC}(BlankNDXs)  = NaN;
    
    temp.con{iC}            = reshape(GetEvents(DIS, ValidPassIDs, 'DKL Texture', '', iC-1, 'Contrast'),    1, []);
    temp.opa{iC}            = reshape(GetEvents(DIS, ValidPassIDs, 'DKL Texture', '', iC-1, 'Opacity'),    1, []);
    temp.con{iC}(BlankNDXs) = NaN;
    temp.opa{iC}(BlankNDXs) = NaN;
end

trial.con{7} = temp.con{7}.*temp.opa{7};
trial.con{6} = temp.con{6}.*temp.opa{6}.*(1-temp.opa{7});
trial.con{5} = temp.con{5}.*temp.opa{5}.*(1-temp.opa{7}).*(1-temp.opa{6});
trial.con{4} = temp.con{4}.*temp.opa{4}.*(1-temp.opa{7}).*(1-temp.opa{6}).*(1-temp.opa{5});
trial.con{3} = temp.con{3}.*temp.opa{3}.*(1-temp.opa{7}).*(1-temp.opa{6}).*(1-temp.opa{5}).*(1-temp.opa{4});
trial.con{2} = temp.con{2}.*temp.opa{2}.*(1-temp.opa{7}).*(1-temp.opa{6}).*(1-temp.opa{5}).*(1-temp.opa{4}).*(1-temp.opa{3});
trial.con{1} = temp.con{1}.*temp.opa{1}.*(1-temp.opa{7}).*(1-temp.opa{6}).*(1-temp.opa{5}).*(1-temp.opa{4}).*(1-temp.opa{3}).*(1-temp.opa{2});

% Compute the spontaneous discharge
blankDurMean = mean(trial.duration(BlankNDXs));
sponRateMean = mean(trial.spikeCount(BlankNDXs)./trial.duration(BlankNDXs));
sponRateVar  = var(trial.spikeCount(BlankNDXs)./trial.duration(BlankNDXs));

% Analyze responses for main experiment
nTrials = trial.num(end);
total_con = NaN(nTrials, 1);
num_comps = NaN(nTrials, 1);
cent_sf = NaN(nTrials, 1);

for tr = 1 : nTrials
    
    currCons = get_comp(trial.con, tr);
    trial.total_con(tr) = sum(currCons);
    trial.num_comps(tr) = sum(currCons>0);
    trial.cent_sf(tr) = trial.sf{1}(tr);
    
end

% % Analyze the stimulus-driven responses for the spatial frequency mixtures
% for iE = 1:2
%     for iW = 1:5
% 
%         StimBlockIDs  = ((iW-1)*(DIS.matrix.Dimensions{2}.Size*DIS.matrix.Dimensions{3}.Size)+1)+(iE-1):2:((iW)*(DIS.matrix.Dimensions{2}.Size*DIS.matrix.Dimensions{3}.Size)-5)+(iE-1);
%         nStimBlockIDs = length(StimBlockIDs);
%         
%         % Initialize Variables
%         sfVals{iW}{iE}          = nan(1, nStimBlockIDs);
%         nRepeats{iW}{iE}        = nan(1, nStimBlockIDs);
%         durMean{iW}{iE}         = nan(1, nStimBlockIDs);
%         rateMean{iW}{iE}        = nan(1, nStimBlockIDs);
%         rateVar{iW}{iE}         = nan(1, nStimBlockIDs);
%         modRatioGeoMean{iW}{iE} = nan(1, nStimBlockIDs);
% 
%         iC = 0;
% 
%         for iB = StimBlockIDs
%             indCond = find(trial.blockID == iB);
%             
%             if ~isempty(indCond)
%                 iC = iC+1;
%                 
%                 sfVals{iW}{iE}(iC)   = unique(GetEvents(DIS, ValidPassIDs(indCond), 'DKL Texture', '', 0, 'Spatial Frequency X', 'cyc/deg'))';
%                 nRepeats{iW}{iE}(iC) = length(indCond);
%                 durMean{iW}{iE}(iC)  = mean(trial.duration(indCond));
%                 rateMean{iW}{iE}(iC) = mean(trial.spikeCount(indCond)./trial.duration(indCond));
%                 rateVar{iW}{iE}(iC)  = var(trial.spikeCount(indCond)./trial.duration(indCond));
%                                 
%                 f0                          = trial.spikeCount(indCond)./trial.duration(indCond);
%                 f1                          = 2*abs(trial.f1(indCond));
%                 modRatio                    = (f1)./(f0 - sponRateMean);                                                         % Only f0 is corrected for spontaneous discharge
%                 modRatioGeoMean{iW}{iE}(iC) = geomean(modRatio(modRatio > 0));       
%                 
%                 conProfile{iW}{iE}(1,iC) = unique(trial.con{1}(indCond));
%                 conProfile{iW}{iE}(2,iC) = unique(trial.con{2}(indCond));
%                 conProfile{iW}{iE}(3,iC) = unique(trial.con{3}(indCond));
%                 conProfile{iW}{iE}(4,iC) = unique(trial.con{4}(indCond));
%                 conProfile{iW}{iE}(5,iC) = unique(trial.con{5}(indCond));
%                 conProfile{iW}{iE}(6,iC) = unique(trial.con{6}(indCond));
%                 conProfile{iW}{iE}(7,iC) = unique(trial.con{7}(indCond));
%                 conProfile{iW}{iE}(8,iC) = unique(trial.con{8}(indCond));
%                 conProfile{iW}{iE}(9,iC) = unique(trial.con{9}(indCond));
%             end            
%         end
%     end
% end
% 
% 
% % Analyze the stimulus-driven responses for the orientation tuning curve
% oriBlockIDs = [131:2:155, 132:2:136];
% iC = 0;
% 
% for iB = oriBlockIDs
%     indCond = find(trial.blockID == iB);
%     
%     if ~isempty(indCond)
%         iC              = iC+1;
%         oriVals(iC)     = unique(trial.ori{1}(indCond));
%         oriRepeats(iC)  = length(indCond);
%         oriDurMean(iC)  = mean(trial.duration(indCond));
%         oriRateMean(iC) = mean(trial.spikeCount(indCond)./trial.duration(indCond));
%         oriRateVar(iC)  = var(trial.spikeCount(indCond)./trial.duration(indCond));
%         
%         f0                     = trial.spikeCount(indCond)./trial.duration(indCond);
%         f1                     = 2*abs(trial.f1(indCond));
%         modRatio               = (f1)./(f0 - sponRateMean);                                                         % Only f0 is corrected for spontaneous discharge
%         oriModRatioGeoMean(iC) = geomean(modRatio(modRatio > 0));
%     end
% end
% 
% % Analyze the stimulus-driven responses for the contrast response function
% conBlockIDs = [138:2:156];
% iC = 0;
% 
% for iB = conBlockIDs
%     indCond = find(trial.blockID == iB);
%     
%     if ~isempty(indCond)
%         iC              = iC+1;
%         conVals(iC)     = unique(trial.con{1}(indCond));
%         conRepeats(iC)  = length(indCond);
%         conDurMean(iC)  = mean(trial.duration(indCond));
%         conRateMean(iC) = mean(trial.spikeCount(indCond)./trial.duration(indCond));
%         conRateVar(iC)  = var(trial.spikeCount(indCond)./trial.duration(indCond));
%         
%         f0                     = trial.spikeCount(indCond)./trial.duration(indCond);
%         f1                     = 2*abs(trial.f1(indCond));
%         modRatio               = (f1)./(f0 - sponRateMean);                                                         % Only f0 is corrected for spontaneous discharge
%         conModRatioGeoMean(iC) = geomean(modRatio(modRatio > 0));
%     end
% end

% Store statistics in the structure
s.fileName      = fileName;
s.xPos          = xPosVals;
s.yPos          = yPosVals;
s.size          = sizeVals;
s.latency       = latency;
s.nBlankPasses  = nBlankPasses;
% s.nStimPasses   = nRepeats;
s.blankDuration = blankDurMean;
% s.stimDuration  = durMean;
% s.sponRateMean  = sponRateMean;
% s.sponRateVar   = sponRateVar;
% s.sf            = sfVals;
% s.conProfile    = conProfile;
% s.sfRateMean    = rateMean;
% s.sfRateVar     = rateVar;
% s.sfModRatio    = modRatioGeoMean;
% s.con           = conVals;
% s.conRateMean   = conRateMean;
% s.conRateVar    = conRateVar;
% s.conModRatio   = conModRatioGeoMean;
s.trial         = trial;
end
%%



%% SpikeTrainFT
function Y = SpikeTrainFT(Tau, f, T)
% Fourier transfom of an idealized spike train of delta functions
% at times tau (seconds), evaluated at frequencies in f (Hz)
% Division by duration T give units of impulses /second.

if iscell(Tau)
    for i = 1:numel(Tau)
        tau = Tau{i}(:);
        Y(:,i) = sum(exp(-2*pi*sqrt(-1).* tau*f(i)) ,1)./T(i);
    end
else
    tau = Tau(:);
    Y(:,1) = sum(exp(-2*pi*sqrt(-1).* tau*f) ,1)./T;
end
end
%%


