%% pass in a spike time offset if analyzing files run with neuropixel probe/spikeGLX
spkOffset = 0.135; % from Manu, offset is 135 ms, roughly...

%% for one cell at a time

%%% set up the glx structure, if grabbing sorted spikes
%baseDir = pwd;
%baseDir = strrep(baseDir, '/matlab', '/');
%disp(baseDir);
%expDir  = 'V1/recordings';
%expName = 'm676l01#109/';
%progName = 'sfMixLGN';

%msFolder = [baseDir expDir '/sorted/' expName];
%filename = 'sfMixHalfInt.dat';
%expoName = [baseDir expDir strrep(expName, '/', '[sfMixHalfInt].xml')];

% set up the GLX structure
%firingsName = 'firings_180_209.curated.mda';
%metricsName = 'cluster_metrics_180_209.json';

%glx.msFolder = msFolder;
%glx.filename = filename;
%glx.expoName = expoName;
%glx.firingsName = firingsName;
%glx.metricsName = metricsName;

%% then call the function that does the work!
%sfMixLoad(strrep(expName, '/', ''), 'loadPath', [baseDir expDir], 'savePath', [baseDir strrep(expDir, 'recordings', 'structuresTest')], 'prog', progName);
%sfMixLoad(strrep(expName, '#95/', ''), 'loadPath', [baseDir expDir], 'savePath', [baseDir strrep(expDir, 'recordings', 'structuresTest')], 'prog', progName);
% sfMixLoad(strrep(expName, '#13/', ''), 'loadPath', [baseDir expDir], 'savePath', [baseDir strrep(expDir, 'recordings', 'structures')], 'spkOffset', spkOffset, 'glxStruct', glx, 'prog', progName);

%% or, run down here for looped sfMixLoad (i.e. for multiple cells at a time)
% ******** NOTE: if you wanted to change the file names, go to convertMatToPy.ipynb first and convert the raw file names %


currDir = pwd;
baseDir = strrep(currDir, '/matlab', '/');
expDir  = 'V1/recordings';
progName = 'sfMixHalfInt';

fileBase = '/m681*xml'; 
valFiles = dir([baseDir expDir fileBase]);
for i = 1:length(valFiles)
  currFile = valFiles(i).name;
  leftBrace = find(currFile == '[');
  rightBrace = find(currFile == ']');
  cellName = currFile(1:leftBrace-1);
  progName = currFile(leftBrace+1:rightBrace-1);

  fprintf('converting %s (prog: %s)\n', cellName, progName);

  sfMixLoad(cellName, 'loadPath', [baseDir expDir], 'savePath', [baseDir strrep(expDir, 'recordings', 'structures')], 'prog', progName);
end

