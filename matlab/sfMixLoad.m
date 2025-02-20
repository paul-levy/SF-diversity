function sfMixLoad(unitLabels, varargin)

% sfMixLGNLoad       Sorting of sfMixLGN xml-files.
%
%   sfMixLGNLoad(unitLabels) loads the xml-files associated with the Expo
%   program used for the sfMix* series of experiments. The input unitLabels is either
%   a string specifying the label of a single unit (e.g. 'm621r47') or a
%   cell array of strings specifying multiple units (e.g. unitLabels =
%   {'m621r47', 'm630r05', 'm620r67'}). Essential information for each
%   experiment is organized into a single structure for each unit. Single
%   unit structures are saved to individual files (e.g. 'm621r47_tp.mat').
%
%   sfMixLGNLoad(unitLabels, 'loadPath', path) specifies the directory from
%   which to load xml files. Default is the current directory.
%
%   sfMixLGNLoad(unitLabels, 'savePath', path) specifies the directory to
%   which resulting structures will be saved (one file per unit). Default
%   is to match loadPath.
%
%   sfMixLGNLoad(unitLabels, 'getFullWave', 0/1) specifies whether or not
%   to grab the entire saved waveform for storing in the data structure -
%   default is 0
% 
%   sfMixLGNLoad(unitLabels, 'nGrats', integer) specifies how many gratings
%   are in the stimulus set - must specify (default is -1)
%
%   sfMixLGNLoad(unitLabels, 'processRepeats', 1) enables processing multiple
%   runs of the sfMix program. All runs are saved as fields with different
%   subscripts in the single unit structure. Default is to process only the
%   version that was run last.
%
%   sfMixLGNLoad(unitLabels, 'programIn', name) allows you to specify
%   which version of the sfMix series to access.
% 
%   sfMixLGNLoad(unitLabels, 'spkOffset', timeInSec) allows you to specify
%   if the spike times in expo were somehow offset relative to the frames
%
%   sfMixLGNLoad(unitLabels, 'glxStruct', glxStruct) allows you to pass in
%   a struct with needed fields to run alignExpoGLX (align sorted spike times
%   with expo pass times)
% 
%   v1.0 2015: Robbe Goris
%   
%   v2.0, 12/13/17: Paul Levy
%   Adjusted for the new sfMixAlt experiment, which uses a new set of
%   stimuli
%  
%   v2.1, 8/10/18: Paul Levy
%   Adjusted for the sfMixLGN experiment (m675), which is a slightly adjusted
%   subset of the stimulus set for sfMixAlt. Only the first and third
%   dispsersions are used; the spacing between sfCenters and the
%   "reference/center" SF are determined by cell; and the temporal
%   frequencies are not chosen from a Gaussian, but instead are in integer
%   steps away from a chosen center - each SF appears only at one TF, which
%   is within +/- 2 of the chosen tfCenter
%
%   v2.2, 01/26/19: Paul Levy
%   Generalized for arbitrary sfMix* experiment, though we now don't
%   organize the response average/stds by condition. Instead, we simply 
%   organize the stimulus parameters for each trial, etc 
%
%   v2.3, 03/20/19: Paul Levy
%   added spikeGLX option to get/align sorted times (from mountainsort) into expo structure
%#ok<*NASGU>
%%

% addpath with expo resources
basePath = pwd;
addpath(genpath([basePath '/../ExpoAnalysisTools'])); % newest version (from github)
% above is old version
% addpath(genpath('/u/vnl/matlab/ExpoMatlab/')); % can use/uncomment if on
% CNS/VNL machine (e.g. Gutrune, Zemina)

supportedTypes    = {'sfm'};


%% Parse inputs, perform initial operations
% Make 'unitLabels' into cell array if single string:
if ~iscell(unitLabels), unitLabels = {unitLabels}; end

% Get unit info from label string
nUnits = length(unitLabels);

% Get input values from varargin or assign default values
loadPath          = GetNamedInput(varargin, 'loadPath', pwd);
savePath          = GetNamedInput(varargin, 'savePath', loadPath);
getFullWave       = GetNamedInput(varargin, 'getFullWave', 0);
% nGrats            = GetNamedInput(varargin, 'nGratings', -1);
supportedPrograms = GetNamedInput(varargin, 'programIn', 'sfMix');
spkOffset         = GetNamedInput(varargin, 'spkOffset', 0);
glxStruct           = GetNamedInput(varargin, 'glxStruct', -1);
processRepeats    = GetNamedInput(varargin, 'processRepeats', 0);
selectedPrograms  = GetNamedInput(varargin, 'prog', supportedPrograms);

if ~iscell(selectedPrograms)
    selectedPrograms = {selectedPrograms}; 
end

% Loop through cells
for iU = 1:nUnits
    
    % Initialize structure S to hold single unit
    S = struct();
    
    % Specify the cell's save name
    if getFullWave
        saveName = [unitLabels{iU}, '_sfm_fullWave'];
    else
        saveName = [unitLabels{iU}, '_sfm'];
    end
    
    % Get some information about the experiments run for this cell
    xmlList = dir([loadPath, '/', unitLabels{iU}, '*xml']); % '#*xml'
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

    toSave = 0;

    % Check all these experiments
    for iF = FileNDX
        
        % search filename for something between two brackets and return it.
        progName = progNames{iF};
        
        % Check that the program is supported and has been selected for analysis
        matchesSupported = ismember(supportedPrograms, progName);                                               % is progName a member of supportedPrograms?
        matchesSelected  = ismember(selectedPrograms, progName);                                                % is progName a member of selectedPrograms?
        
        if any(matchesSelected) && any(matchesSupported)
            toSave = 1; % save only if it matches!
            expType = supportedTypes{matchesSupported};
            fprintf('\t[%s] - found!',expType);
            [s] = AnalyzeExperiment(loadPath, expType, xmlList(iF).name, getFullWave, spkOffset, glxStruct);
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
    if toSave
        save([savePath,'/',saveName],'S');
    end
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
function [s] = AnalyzeExperiment(loadPath, expType, fileName, getFullWave, spkOffset, glxStruct)

% use to get the value of an attribute ("x"; e.g. contrast, orientation) for
% each component of a given trial ("tr")
get_comp = @(x, tr) cellfun(@(y) y(tr), x); % get components by trial

% Load the appropriate file, no output is printed to the screen
DIS = ReadExpoXML([loadPath, '/', fileName], getFullWave, 0);

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

% Get the stimulus durations, spike times, and full waveform, if specified
duration   = GetDurations(DIS, ValidPassIDs, 'sec');                                                            % durations only for valid trials
spikeTimes = GetSpikeTimes(DIS, 0, ValidPassIDs, .001*latency + spkOffset, .001*latency + spkOffset, 0, 'sec');                          % spikeTimes only for valid trials
if getFullWave
    fullWaves  = GetWaveforms(DIS,0 , ValidPassIDs, .001*latency, .001*latency, 0);
else
    fullWaves = [];
end
% get sorted spike times, if present
if isa(glxStruct, 'struct')
  fprintf('Getting mountainsort spike times\n');
  [clusid, spikes_by_clus, classification] = alignExpoGLX(DIS, glxStruct.msFolder, [glxStruct.msFolder glxStruct.filename], glxStruct.expoName, glxStruct.firingsName, glxStruct.metricsName);
  spks_GLX.times = spikes_by_clus;
  spks_GLX.IDs   = clusid;
  spks_GLX.classification = classification; 
end

% 0.001*latency to convert seconds (from milliseconds)
tfPass     = GetEvents(DIS, ValidPassIDs, 'Surface', '', 0, 'Drift Rate', 'cyc/sec');

trial.num           = 1:length(ValidPassIDs);                                                                   % 1 to the number of valid trials
trial.blockID       = ValidBlockIDs;                                                                            % block ids of valid trials
trial.duration      = duration;
trial.spikeTimes    = spikeTimes;
trial.spikeCount    = GetSpikeCounts(spikeTimes, 0, duration, 'sec', 'impulses');                               % spikeCounts for valid trials
trial.fullWaves     = fullWaves;
if isa(glxStruct, 'struct')
  trial.spikeTimesGLX = spks_GLX;
end
trial.f1            = SpikeTrainFT(spikeTimes, tfPass, trial.duration);                                         % response modulation at fundamental Fourier component
trial.f1(BlankNDXs) = NaN;

% Get the spatial frequency, temporal frequency, orientation, phase, and contrast for each stimulus component
% determine the number of gratings in this experiment
nGrats = 0;
% NOTE: This will elicit "warning", since we try more gratings than there
% are, but in GetEvents, this "warning" is just a disp, so we cannot toggle
% it on/off
while true
    fprintf('nGrats: %d', nGrats);
    if unique(GetEvents(DIS, ValidPassIDs, 'DKL Texture', '', nGrats, 'Contrast')) == 0
        break;
    else
        nGrats = nGrats+1;
    end
end

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

mult = 1; % cumulative multiplier
for i = nGrats:-1:1
    trial.con{i} = temp.con{i}.*temp.opa{i}.*mult;
    mult = mult .* (1-temp.opa{i});
end
% As an explicit example of the above, this is for 5 components/gratings
% trial.con{5} = temp.con{5}.*temp.opa{5};
% trial.con{4} = temp.con{4}.*temp.opa{4}.*(1-temp.opa{5});
% trial.con{3} = temp.con{3}.*temp.opa{3}.*(1-temp.opa{5}).*(1-temp.opa{4});
% trial.con{2} = temp.con{2}.*temp.opa{2}.*(1-temp.opa{5}).*(1-temp.opa{4}).*(1-temp.opa{3});
% trial.con{1} = temp.con{1}.*temp.opa{1}.*(1-temp.opa{5}).*(1-temp.opa{4}).*(1-temp.opa{3}).*(1-temp.opa{2});

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


